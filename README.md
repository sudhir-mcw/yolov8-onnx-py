# Introduction 
- This project detects faces in images, videos, and webcam feeds.
-  It utilizes an ONNX model for inference.
-  The project contains a test video and image present inside the test_inputs folder.
- All the outputs produced are dumped to output/ folder.

# Prequisites
- OS: Linux/Unix/ Windows

- Python : 3.10 \
**Use VENV to create an environment**
```
python -m venv myenv 
source myenv/bin/activate
``` 
# Requirements

* Check the **requirements.txt** file.
* For ONNX, if you have a NVIDIA GPU, then install the **onnxruntime-gpu** use 
```pip install onnxruntime-gpu```
, otherwise use the **onnxruntime** library use ```pip install onnxruntime```.

# How to run 
1. Install the requirements
```
pip install -r requirments.txt
```
2. Download the model file and save it in models/ folder if not present already.
```
wget https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx
mkdir models
mv yolov8n-face-lindevs.onnx models
```
3. To run inference on image input run 
```
python image_object_detection.py
```
4. To run inference on video input run 
```
python video_object_detection.py
```
5. To run inference on webcam as input run 
```
python webcam_object_detection.py
```
Note: webcam mode works with imshow so need display port inorder to run webcam_object_detection.py


