import cv2
import time
from yolov8 import YOLOv8
import sys

try:
    import os 
    os.makedirs("output")
except:
    pass

cap = cv2.VideoCapture("test_inputs/test_video_2.mp4")
# Initialize YOLOv7 model
model_path = "models/yolov8n-face-lindevs.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)

if len(sys.argv)==1:
    print("Please provide the number of frames to run the model on\nUsage: python video_object_detection.py <number_of_frames>")
    sys.exit(0)

count = 0
while cap.isOpened():
    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue
    if count == int(sys.argv[1]):
        break
    # Update object localizer
    boxes, scores, class_ids = yolov8_detector(frame)
    post_process_start_time = time.time()
    combined_img = yolov8_detector.draw_detections(frame)
    post_process_end_time   = time.time()
    print("post process time : ",(post_process_end_time-post_process_start_time))

    count += 1

