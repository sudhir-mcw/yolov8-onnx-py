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
1. Clone the repository
```
git clone https://github.com/sudhir-mcw/yolov8-onnx-py
cd yolov8-onnx-py
```
2. Install the requirements
```
pip install -r requirments.txt
```
3. Download the model file and save it in models/ folder if not present already.
```
cd yolov8-onnx-py
wget https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx
mkdir models
mv yolov8n-face-lindevs.onnx models
```
Convert the onnx model input dimension from variable to fixed 
```
python onnx_convert_fixed_dims.py <path_to_onnx_model> <destination_path>
```
Example:
```
python onnx_convert_fixed_dims.py ./models/yolov8n-face-lindevs.onnx ./models/
```
4. To run inference on video input run 
```
python video_object_detection.py <no_of_frames>
```
Example:
```
python video_object_detection.py 100
```

