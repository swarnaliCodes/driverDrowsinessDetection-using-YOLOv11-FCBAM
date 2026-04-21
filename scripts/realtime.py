import cv2
from ultralytics import YOLO
from collections import deque
import threading
import winsound
import time

model = YOLO("best.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

class_names = ["eyeclosed", "yawn"]

WINDOW_SIZE = 120
FPS_ASSUMED = 30
WINDOW_SECONDS = WINDOW_SIZE / FPS_ASSUMED

W_PERCLOS = 0.7
W_YAWN = 0.3
DROWSY_SCORE_THRESHOLD = 0.5
STABILITY_SECONDS = 2

eye_status_window = deque(maxlen=WINDOW_SIZE)
yawn_window = deque(maxlen=WINDOW_SIZE)

alert_active = False
drowsy_start_time = None

def beep_alert():
    winsound.Beep(1000, 800)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, conf=0.25, verbose=False)

    eye_closed_detected = 0
    yawn_detected = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = f"{class_names[cls_id]} {conf:.2f}"
            color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if class_names[cls_id] == "eyeclosed":
                eye_closed_detected = 1
            if class_names[cls_id] == "yawn":
                yawn_detected = 1

    eye_status_window.append(eye_closed_detected)
    yawn_window.append(yawn_detected)

    if len(eye_status_window) < WINDOW_SIZE:
        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    perclos = sum(eye_status_window) / WINDOW_SIZE
    yawns_in_window = sum(yawn_window)
    yawn_rate = yawns_in_window / WINDOW_SECONDS

    drowsy_score = (W_PERCLOS * perclos) + (W_YAWN * yawn_rate)

    current_time = time.time()

    if drowsy_score > DROWSY_SCORE_THRESHOLD:
        if drowsy_start_time is None:
            drowsy_start_time = current_time

        if current_time - drowsy_start_time >= STABILITY_SECONDS:
            status = "DROWSY"
            color = (0, 0, 255)

            if not alert_active:
                alert_active = True
                threading.Thread(target=beep_alert, daemon=True).start()
        else:
            status = "MONITORING"
            color = (0, 255, 255)
    else:
        status = "ALERT"
        color = (0, 255, 0)
        alert_active = False
        drowsy_start_time = None

    cv2.putText(frame, f"Status: {status}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Yawn Rate: {yawn_rate:.2f}/s", (30, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, f"Score: {drowsy_score:.2f}", (30, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
