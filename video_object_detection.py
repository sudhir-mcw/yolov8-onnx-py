import cv2
import time
from yolov8 import YOLOv8
from statistics import mean
import sys
import codecs
import json 

def profiler():
    pre_process_cpu_times = []
    post_process_cpu_times = []
    count = 0
    for file in os.listdir(os.path.join(os.getcwd(),"exp-logs/prepare_input")):
        file = os.path.join(os.getcwd(),"exp-logs/prepare_input",file)
        with codecs.open(file, 'r', encoding='utf-8',
                 errors='replace') as fdata:
            json_data = json.load(fdata, strict=False)
        
        trace_events = json_data["traceEvents"]
        entry_count = 0
        for event in trace_events:
            if " prepare_input" in event.get("name"):
                pre_process_cpu_times.append(int(event.get("dur")))
                entry_count+=1
        count+=1
    count   =   0
    for file in os.listdir(os.path.join(os.getcwd(),"exp-logs/process_output")):

        file = os.path.join(os.getcwd(),"exp-logs/process_output",file)
        with codecs.open(file, 'r', encoding='utf-8',
                 errors='replace') as fdata:
            json_data = json.load(fdata, strict=False)
        
        trace_events = json_data["traceEvents"]
        entry_count = 0
        for event in trace_events:
            if "process_output" in event.get("name"):
                post_process_cpu_times.append(int(event.get("dur")))
                entry_count+=1

        count+=1

    return mean(pre_process_cpu_times)/1000, mean(post_process_cpu_times)/1000

try:
    import os 
    os.rmdir(os.path.join(os.getcwd(),"exp-logs"))    
except:
    print('no logs to remove')
    pass



if __name__ == "__main__":

    cap = cv2.VideoCapture("test_inputs/test_video_2.mp4")
    model_path = "models/yolov8n-face-lindevs-fixed.onnx"
    yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)
    MAX_FRAMES=int(sys.argv[1])
    for i in range(0,5):
        count = 0
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
            except Exception as e:
                print(e)
                continue
            if count == MAX_FRAMES:
                break
            boxes, scores, class_ids = yolov8_detector(frame)
            count += 1
    preprocess_time, postprocess_time = profiler()
    print("\nPreprocess time per frame: ", (preprocess_time)," ms")
    print("\nPost process time per frame: ", (postprocess_time)," ms")





