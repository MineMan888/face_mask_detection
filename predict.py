from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train-3/weights/best.pt")

results = model.predict(
    source="dataset/images/testing",
    conf = 0.5,
    save = True,
    project = "runs/detect",
    name = "test_results"
    )

cv2.destroyAllWindows()