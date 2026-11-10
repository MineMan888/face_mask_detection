from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train-2/weights/best.pt")

results = model.predict(
    source="datasets/wv/images/test",
    conf = 0.5,
    save = True,
    project = "runs/detect",
    name = "test_results"
    )

for r in results:
    im_array = r.plot()
    cv2.imshow("Rilevamento Mascherine", im_array)
    cv2.waitKey(0)
cv2.destroyAllWindows()