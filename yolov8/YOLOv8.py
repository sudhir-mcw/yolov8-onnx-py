import time
import cv2
import numpy as np
import onnxruntime
import torch
from yolov8.utils import xywh2xyxy, draw_detections, multiclass_nms

class YOLOv8:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.frame_count   = 0
        self.preprocess_time = []
        self.postprocess_time = []
        self.resize_time = []
	# Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()


    def detect_objects(self, image):
        prepare_input_start_time = time.time()
        input_tensor = self.prepare_input(image)
        prepare_input_end_time   =  time.time()
        print("pre process time : ",(prepare_input_end_time-prepare_input_start_time))
        outputs = self.inference(input_tensor)
        post_process_start_time = time.time()
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        post_process_end_time = time.time()
        print("post process time : ",(post_process_end_time-post_process_start_time))
       
    
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.resize(image, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        self.frame_count+=1
    
        return  input_tensor

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        predictions = torch.from_numpy(output[0]).squeeze(0).transpose(0,1)
        scores      = torch.max(predictions[:,4:],dim=1)[0]
        predictions = predictions[scores > self.conf_threshold]
        scores      = scores[scores > self.conf_threshold]
        if scores.numel()   == 0:
            return [],[],[]
        class_ids   =   torch.argmax(predictions[:,4:],dim=1)
        predictions =   predictions.detach().numpy()
        boxes       =   self.extract_boxes(predictions)
        class_ids   =   class_ids.detach().numpy()
        scores      =   scores.detach().numpy()
        indices     = multiclass_nms(boxes,scores,class_ids,self.iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        detection = draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)
        return detection

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

