import time
import cv2
import numpy as np
import onnxruntime
import torch
from yolov8.utils import xywh2xyxy, draw_detections, multiclass_nms
from torchvision import transforms

PROFILE = True

'''
    profiling function uses torch.prolfer.profile(),
    logs to a file name with functions name  
    usage : 
    @profile
    def fun_to_profile():
        pass
'''
def profile(func):
    def profiler(*args, **kwargs):
        if not PROFILE:
            return func(*args, **kwargs)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./exp-logs/{func.__name__}'),
            profile_memory=True,
            with_stack=True
            ) as prof:
            generated_ids = func(*args, **kwargs)
        return generated_ids
 
    return profiler

class YOLOv8:
    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.frame_count   = 0
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
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs,image)

        return self.boxes, self.scores, self.class_ids
    
    @profile
    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = transforms.ToTensor()(image)
        input_img = transforms.Resize((self.input_width, self.input_height))(input_img)
        input_tensor = input_img.unsqueeze(0)
        input_tensor = input_tensor.float()
        self.frame_count+=1    
        return  input_tensor.detach().numpy()

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        return outputs
    
    @profile
    def process_output(self, output,image):
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
        draw_detections(image,boxes[indices], scores[indices], class_ids[indices])
        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
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
        try:
            self.input_height = int(self.input_shape[2])
            self.input_width = int(self.input_shape[3])
        except ValueError:
            print("Unable to convert Height and width to integers")

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

