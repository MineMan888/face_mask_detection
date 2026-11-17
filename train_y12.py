from ultralytics import YOLO

def train_mask_detector():
    # 1. Carica il modello pre-addestrato yolo11n.pt (sfrutta il Transfer Learning)
    model = YOLO("yolo12n.pt")

    # 2. Avvia l'addestramento con i parametri ottimizzati
    results = model.train(
        data="data.yaml",       # File di configurazione del dataset
        epochs=75,              # Numero totale di epoche (da 50 a 100 è l'ideale per iniziare)
        imgsz=640,              # Risoluzione delle immagini (standard 640x640)
        batch=16,               # Batch size (riduci a 8 o 4 se riscontri errori di memoria VRAM)
        device=0,               # 0 per usare la GPU NVIDIA, oppure 'cpu'
        patience=15,            # Early stopping: interrompe se la validazione non migliora per 15 epoche
        save=True,              # Salva i checkpoint e i pesi migliori (best.pt)
        save_txt=True,           # Salva le coordinate dei box
        name="mask_yolo12n",    # Sottocartella specifica per questo esperimento
        
        # --- Parametri avanzati per affinare la precisione ---
        lr0=0.01,               # Learning rate iniziale
        mosaic=1.0,             # Data augmentation: combina 4 immagini in una per simulare contesti affollati
        mixup=0.1,              # Data augmentation: fonde leggermente due immagini per regolarizzare il training
        close_mosaic=10         # Disattiva il mosaic nelle ultime 10 epoche per stabilizzare il fine-tuning
    )

    print("Addestramento completato con successo!")

if __name__ == "__main__":
    train_mask_detector()