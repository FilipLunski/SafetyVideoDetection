
import cv2
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolov8n-pose.pt")

video = cv2.VideoCapture(0)

box_annotator = sv.BoxAnnotator()

while True:
    success, frame = video.read()

    result = model(frame)
    detections = sv.Detections.from_yolov5(result)
    frame = box_annotator(frame, detections)

    cv2.imshow("Video", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()