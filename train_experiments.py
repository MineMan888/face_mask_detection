"""
Script unico per addestrare e confrontare sistematicamente i modelli YOLO nano
(yolo11n, yolo12n, yolo26n) su 3 configurazioni di iperparametri ottimizzate
per il dataset di riconoscimento mascherine.

Esempi di utilizzo:
  # Esegue tutte le 9 combinazioni (3 modelli x 3 configurazioni) + valutazione test
  python train_experiments.py

  # Stampa il piano degli esperimenti senza avviare il training
  python train_experiments.py --dry-run

  # Esegue solo un modello specifico su tutte e 3 le configurazioni
  python train_experiments.py --models yolo26n

  # Esegue una configurazione specifica su tutti e 3 i modelli
  python train_experiments.py --configs cfg2_finegrained_balanced

  # Riprende una sessione interrotta saltando i run che hanno già best.pt
  python train_experiments.py --skip-existing
"""

import argparse
import csv
import gc
import time
from pathlib import Path

import torch
from ultralytics import YOLO


# =============================================================================
# 1. MODELLI NANO DA CONFRONTARE
# =============================================================================
MODELS: dict[str, str] = {
    "yolo11n": "yolo11n.pt",
    "yolo12n": "yolo12n.pt",
    "yolo26n": "yolo26n.pt",
}


# =============================================================================
# 2. LE 3 CONFIGURAZIONI DI TRAINING
# =============================================================================
# Parametri comuni a tutti gli esperimenti per garantire un confronto equo
COMMON_PARAMS: dict = {
    "data": "data.yaml",
    "save": True,
    "exist_ok": True,
    "plots": True,
    "deterministic": True,
}

# Classi di interesse specifico per il report comparativo (da data.yaml)
FFP_CLASSES: dict[int, str] = {
    14: "FFP1",
    15: "FFP2",
    16: "FFP3",
}
# Classi con 0-1 istanze in validation/test (escluse nella metrica mAP50_no_rare)
RARE_OR_EMPTY_CLASS_IDS: set[int] = {8, 10, 17}  # Hats, Sunglasses, Other - Unknown

CONFIGS: dict[str, dict] = {
    # -------------------------------------------------------------------------
    # Config 1: Baseline ottimizzata (AdamW + Cosine LR)
    # Corregge i limiti della configurazione iniziale (optimizer="auto" che
    # ignorava lr0, decadimento lineare ed early stopping prima di close_mosaic).
    # -------------------------------------------------------------------------
    "cfg1_adamw_cos": {
        "description": "Baseline pulita: AdamW + Cosine LR + close_mosaic effettivo (640px)",
        "params": {
            "epochs": 100,
            "patience": 25,
            "imgsz": 640,
            "batch": 16,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "warmup_epochs": 5.0,
            "mosaic": 1.0,
            "mixup": 0.0,
            "close_mosaic": 15,
        },
    },
    # -------------------------------------------------------------------------
    # Config 2: Fine-grained augmentation + bilanciamento classi (cls & cls_pw)
    # Mirata al collo di bottiglia del dataset (confusione FFP1/FFP2/FFP3):
    # aumenta il peso della classification loss (cls=1.0), attiva il
    # bilanciamento inverso moderato (cls_pw=0.35), riduce il mosaic e aggiunge
    # leggere rotazioni della testa preservando i dettagli cromatici.
    # -------------------------------------------------------------------------
    "cfg2_finegrained_balanced": {
        "description": "Fine-grained & Class Balancing: cls=1.0, cls_pw=0.35, aug mirate (640px)",
        "params": {
            "epochs": 120,
            "patience": 30,
            "imgsz": 640,
            "batch": 16,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "cos_lr": True,
            "weight_decay": 0.001,
            "warmup_epochs": 5.0,
            "cls": 1.0,
            "cls_pw": 0.35,
            "mosaic": 0.7,
            "mixup": 0.0,
            "close_mosaic": 20,
            "degrees": 10.0,
            "hsv_h": 0.01,
            "hsv_s": 0.5,
            "hsv_v": 0.4,
        },
    },
    # -------------------------------------------------------------------------
    # Config 3: Alta risoluzione (800px) + Multi-Scale + MuSGD
    # Sfrutta le immagini ad alta risoluzione del dataset (1000-1500px) per
    # preservare scritte/valvole/cuciture delle FFP, gestisce i ritagli piccoli
    # con multi_scale=0.2 e testa l'ottimizzatore MuSGD (nativo di Ultralytics
    # 8.4/YOLO26, compatibile anche con YOLO11 e YOLO12) con batch=8 e nbs=64
    # per mantenere invariato il batch effettivo via gradient accumulation.
    # -------------------------------------------------------------------------
    "cfg3_highres_musgd": {
        "description": "High-Res 800px + Multi-Scale (0.2) + MuSGD + Class Balancing",
        "params": {
            "epochs": 120,
            "patience": 30,
            "imgsz": 800,
            "batch": 8,
            "nbs": 64,
            "multi_scale": 0.2,
            "optimizer": "MuSGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.9,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "warmup_epochs": 5.0,
            "cls": 1.0,
            "cls_pw": 0.35,
            "mosaic": 0.7,
            "mixup": 0.0,
            "close_mosaic": 20,
            "degrees": 10.0,
            "hsv_h": 0.01,
            "hsv_s": 0.5,
            "hsv_v": 0.4,
        },
    },
}


SUMMARY_FIELDS: list[str] = [
    "run_name",
    "model",
    "config",
    "status",
    "train_minutes",
    "val_mAP50",
    "val_mAP50_95",
    "val_mAP50_no_rare",
    "val_precision",
    "val_recall",
    "val_FFP1_AP50",
    "val_FFP2_AP50",
    "val_FFP3_AP50",
    "val_FFP_mean_AP50",
    "test_mAP50",
    "test_mAP50_95",
    "test_mAP50_no_rare",
    "test_precision",
    "test_recall",
    "test_FFP1_AP50",
    "test_FFP2_AP50",
    "test_FFP3_AP50",
    "test_FFP_mean_AP50",
    "best_weights",
]


def extract_metrics(metrics_obj, prefix: str) -> dict[str, float | str]:
    """Estrae metriche globali, mAP50 senza classi rare e AP50 su FFP1/2/3."""
    if metrics_obj is None or not hasattr(metrics_obj, "box"):
        return {}

    box = metrics_obj.box
    out: dict[str, float | str] = {
        f"{prefix}_mAP50": round(float(box.map50), 5),
        f"{prefix}_mAP50_95": round(float(box.map), 5),
        f"{prefix}_precision": round(float(box.mp), 5),
        f"{prefix}_recall": round(float(box.mr), 5),
    }

    per_class_ap50: dict[int, float] = {}
    ap50_arr = box.ap50
    for idx, cls_id in enumerate(box.ap_class_index):
        if idx < len(ap50_arr):
            per_class_ap50[int(cls_id)] = float(ap50_arr[idx])

    non_rare_vals = [
        val for cid, val in per_class_ap50.items() if cid not in RARE_OR_EMPTY_CLASS_IDS
    ]
    if non_rare_vals:
        out[f"{prefix}_mAP50_no_rare"] = round(sum(non_rare_vals) / len(non_rare_vals), 5)

    ffp_vals = []
    for cid, cname in FFP_CLASSES.items():
        if cid in per_class_ap50:
            val = per_class_ap50[cid]
            out[f"{prefix}_{cname}_AP50"] = round(val, 5)
            ffp_vals.append(val)
    if ffp_vals:
        out[f"{prefix}_FFP_mean_AP50"] = round(sum(ffp_vals) / len(ffp_vals), 5)

    return out


def upsert_summary_csv(csv_path: Path, row: dict) -> None:
    """Aggiorna o inserisce una riga nel CSV riepilogativo (chiave: run_name)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_name: dict[str, dict] = {}

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            for existing in csv.DictReader(f):
                rows_by_name[existing["run_name"]] = existing

    clean_row = {field: row.get(field, "") for field in SUMMARY_FIELDS}
    rows_by_name[clean_row["run_name"]] = clean_row

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for r in rows_by_name.values():
            writer.writerow({field: r.get(field, "") for field in SUMMARY_FIELDS})


def print_summary_table(csv_path: Path) -> None:
    """Stampa a console una tabella compatta dei risultati salvati."""
    if not csv_path.exists():
        return

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return

    header = (
        f"{'Run':<36} | {'Val mAP50':>9} | {'Val noRare':>10} | {'Val FFP':>7} | "
        f"{'Test mAP50':>10} | {'Test FFP':>8} | {'Min':>5}"
    )
    print("\n" + "=" * len(header))
    print("RIEPILOGO ESPERIMENTI")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['run_name']:<36} | "
            f"{str(r.get('val_mAP50', '')):>9} | "
            f"{str(r.get('val_mAP50_no_rare', '')):>10} | "
            f"{str(r.get('val_FFP_mean_AP50', '')):>7} | "
            f"{str(r.get('test_mAP50', '')):>10} | "
            f"{str(r.get('test_FFP_mean_AP50', '')):>8} | "
            f"{str(r.get('train_minutes', '')):>5}"
        )
    print("=" * len(header))
    print(f"CSV completo salvato in: {csv_path}\n")


def cleanup_gpu() -> None:
    """Libera la VRAM tra un esperimento e l'altro."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_experiments(args: argparse.Namespace) -> None:
    project_dir = Path(args.project).resolve()
    data_path = str(Path(COMMON_PARAMS["data"]).resolve())
    summary_csv = project_dir / "experiments_summary.csv"

    plan = [
        (model_key, cfg_key)
        for model_key in args.models
        for cfg_key in args.configs
    ]

    print(f"\nPiano esperimenti ({len(plan)} run totali):")
    for i, (m_key, c_key) in enumerate(plan, start=1):
        print(
            f"  [{i}/{len(plan)}] {m_key} ({MODELS[m_key]}) x {c_key} "
            f"-> {CONFIGS[c_key]['description']}"
        )

    if args.dry_run:
        print("\nModalità --dry-run attiva: nessun addestramento avviato.")
        return

    for idx, (model_key, cfg_key) in enumerate(plan, start=1):
        weights_file = MODELS[model_key]
        cfg_params = CONFIGS[cfg_key]["params"]
        run_name = f"mask_{model_key}_{cfg_key}"
        best_weights_path = project_dir / run_name / "weights" / "best.pt"

        print("\n" + "#" * 80)
        print(f"[{idx}/{len(plan)}] AVVIO: {run_name}")
        print(f"Modello: {weights_file} | Config: {CONFIGS[cfg_key]['description']}")
        print("#" * 80)

        row: dict = {
            "run_name": run_name,
            "model": model_key,
            "config": cfg_key,
            "status": "ok",
            "train_minutes": "",
            "best_weights": str(best_weights_path),
        }

        val_metrics = None
        test_metrics = None

        try:
            if args.skip_existing and best_weights_path.exists():
                print(f"[SKIP] Pesi già presenti in {best_weights_path}, salto il training.")
                row["status"] = "skipped_train"
                best_model = YOLO(str(best_weights_path))
                val_metrics = best_model.val(
                    data=data_path,
                    split="val",
                    imgsz=cfg_params["imgsz"],
                    batch=cfg_params["batch"],
                    device=args.device,
                    project=str(project_dir),
                    name=f"{run_name}_val",
                    exist_ok=True,
                )
            else:
                train_kwargs = {
                    **COMMON_PARAMS,
                    **cfg_params,
                    "data": data_path,
                    "device": args.device,
                    "seed": args.seed,
                    "project": str(project_dir),
                    "name": run_name,
                }
                t0 = time.time()
                model = YOLO(weights_file)
                val_metrics = model.train(**train_kwargs)
                elapsed_min = round((time.time() - t0) / 60.0, 2)
                row["train_minutes"] = elapsed_min
                print(f"[OK] Training {run_name} completato in {elapsed_min} min.")

                if getattr(model, "trainer", None) and getattr(model.trainer, "best", None):
                    best_weights_path = Path(model.trainer.best)
                    row["best_weights"] = str(best_weights_path)

                del model
                cleanup_gpu()

                best_model = (
                    YOLO(str(best_weights_path)) if best_weights_path.exists() else None
                )

            if not args.no_test and best_model is not None:
                print(f"[TEST] Valutazione di {best_weights_path} sullo split 'test'...")
                test_metrics = best_model.val(
                    data=data_path,
                    split="test",
                    imgsz=cfg_params["imgsz"],
                    batch=cfg_params["batch"],
                    device=args.device,
                    project=str(project_dir),
                    name=f"{run_name}_test",
                    exist_ok=True,
                )

            if best_model is not None:
                del best_model

            row.update(extract_metrics(val_metrics, "val"))
            row.update(extract_metrics(test_metrics, "test"))

        except Exception as exc:
            print(f"[ERRORE] Esperimento {run_name} fallito: {exc}")
            row["status"] = f"error: {type(exc).__name__}"
        finally:
            upsert_summary_csv(summary_csv, row)
            cleanup_gpu()

    print_summary_table(summary_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark unificato YOLO11n / YOLO12n / YOLO26n su 3 configurazioni."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        default=list(MODELS.keys()),
        help="Modelli nano da addestrare (default: tutti e 3).",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS.keys()),
        default=list(CONFIGS.keys()),
        help="Configurazioni da eseguire (default: tutte e 3).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device di training (es. '0' per GPU NVIDIA, 'cpu' per CPU).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed fisso per riproducibilità tra modelli e configurazioni.",
    )
    parser.add_argument(
        "--project",
        default="runs/detect/experiments",
        help="Cartella radice in cui salvare i run e il CSV riepilogativo.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Salta il training se weights/best.pt esiste già (esegue comunque val/test).",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Salta la valutazione finale sullo split di test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra il piano degli esperimenti senza avviare il training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiments(parse_args())
