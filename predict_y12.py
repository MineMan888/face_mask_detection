from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/mask_yolo12n/weights/best.pt")

results = model.predict(
    source="dataset/images/testing",
    conf = 0.5,
    save = True,
    project = "runs/detect",
    name = "test_results_y12"
    )

cv2.destroyAllWindows()