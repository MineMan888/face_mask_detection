from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data = "data.yaml",
        epochs = 50,
        imgsz = 640,
        batch = 16,
        device = 0,
        patience = 10
    )

if __name__ == "__main__":
    main()