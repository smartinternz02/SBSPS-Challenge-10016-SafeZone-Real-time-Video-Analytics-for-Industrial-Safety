import cv2
import torch
import winsound
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open camera
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Detect objects using YOLOv5
        results = model(frame)

        # Process detection results
        detections = results.pred[0]

        # Initialize variables to track safety equipment detection
        safety_equipment_detected = False

        # Draw bounding boxes on frame and check for safety equipment detection
        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if cls == 0:  # Class index for person
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            elif cls == 1:  # Class index for safety equipment
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                safety_equipment_detected = True

        # Check if safety equipment is not detected and display alert
        if not safety_equipment_detected:
            alert_message = "ALERT: No safety equipment detected!"
            cv2.putText(frame, alert_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            winsound.Beep(1000, 1000)  # Beep sound (frequency, duration)

        # ... (rest of your frame generation logic)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('newindex.html')

@app.route('/video_feed')
def video_feed_route():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
