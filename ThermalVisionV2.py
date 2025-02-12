# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:20:34 2024

@author: mohan
"""

import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, render_template

camera = cv2.VideoCapture(0)
app = Flask(__name__)

def draw_text(img, text, font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0), font_scale=3, font_thickness=2, text_color=(0, 0, 0), text_color_bg=(0, 0, 0)):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return text_size

def generate_frames():
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    model = YOLO(r"D:\Academics\RAwork\Thermal Vision\ThermVIsionModel.pt")
    while True:
        success, frame = camera.read()
        if not success:
            print("Camera not found!")
            break
        results = model(source=frame, conf=0.7)
        for result in results:
            boxes = result.boxes.xyxy.tolist()
            class_ids = result.boxes.cls.cpu().tolist()
            for box, class_id in zip(boxes, class_ids):
                x1, y1, x2, y2 = box
                draw_text(frame, f"{model.names[class_id]}", font_scale=1, pos=(int(x1), int(y1) - 5), text_color=(0, 0, 0), text_color_bg=(0, 204, 0))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 204, 0), 2)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = np.array(imgencode)
        string_data = data.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + string_data + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002)
