from ultralytics import YOLO
import cv2


model = YOLO("weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ لا يمكن فتح الكاميرا")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ فشل قراءة الإطار من الكاميرا")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Sign Language Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
