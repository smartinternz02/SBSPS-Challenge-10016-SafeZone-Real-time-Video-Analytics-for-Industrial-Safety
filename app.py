import cv2
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv5 model
model = YOLO('C:\\ipynb_yolov5\\results_yolov8n_100e\\kaggle\\working\\runs\\detect\\train\\weights\\best.pt')

def camera_generator():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()

        if not ret:
            break

        # Resize the frame to a smaller size
        new_width = 940  # Adjust this value to the desired width
        new_height = 600  # Adjust this value to the desired height
        frame = cv2.resize(frame, (new_width, new_height))
        # Save the frame as a temporary image file
        cv2.imwrite('temp_frame.jpg', frame)

        # Predict using the model
        results = model.predict(source='temp_frame.jpg', save=True)

        # Draw bounding boxes on the frame
        #for det in results.pred[0]:
        #    x1, y1, x2, y2, conf, cls = det.tolist()
        #    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        #    class_name = model.names[int(cls)]
        #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #    cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1 - 10),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
