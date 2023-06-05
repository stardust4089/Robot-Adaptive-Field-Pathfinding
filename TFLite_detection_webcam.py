######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
# class VideoStream:
    # """Camera object that controls video streaming from the Picamera"""
    # def __init__(self,resolution=(640,480),framerate=30):
    #     # Initialize the PiCamera and the camera image stream
    #     self.stream = cv2.VideoCapture(0)
    #     ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    #     ret = self.stream.set(3,resolution[0])
    #     ret = self.stream.set(4,resolution[1])
            
    #     # Read first frame from the stream
    #     (self.grabbed, self.frame) = self.stream.read()

	# # Variable to control when the camera is stopped
    #     self.stopped = False

    # def start(self):
	# # Start the thread that reads frames from the video stream
    #     Thread(target=self.update,args=()).start()
    #     return self

    # def update(self):
    #     # Keep looping indefinitely until the thread is stopped
    #     while True:
    #         # If the camera is stopped, stop the thread
    #         if self.stopped:
    #             # Close camera resources
    #             self.stream.release()
    #             return

    #         # Otherwise, grab the next frame from the stream
    #         (self.grabbed, self.frame) = self.stream.read()

    # def read(self):
	# # Return the most recent frame
    #     return self.frame

    # def stop(self):
	# # Indicate that the camera and thread should be stopped
    #     self.stopped = True

MODEL_NAME = "bumper_detect_m3.tflite"
GRAPH_NAME = "bumper_detect_m3.tflite"
LABELMAP_NAME = "labelmap.txt"
min_conf_threshold = float(0.45)
resW, resH = '852x480'.split('x')
imW, imH = int(resW), int(resH)
use_TPU = False

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    import tensorflow as tf
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = "C:/Users/Clementine/Documents/Robotics Programming/A_star_idea/bumper_detect_m3.tflite"

# Path to label map file
PATH_TO_LABELS = "C:/Users/Clementine/Documents/Robotics Programming/A_star_idea/labelmap.txt"

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = tf.lite.Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = tf.lite.Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

frame_to_return = None

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
t1 = cv2.getTickCount()

class boxObj:
    def __init__(self, box, label):
        self.box = box
        self.label = label

def run_model(frame):
    global frame_to_return, frame_rate_calc
    # Initialize video stream
    time.sleep(1)

    # Grab frame from video stream
    frame1 = frame

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
    
    box_objs = []
    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
            box_objs += [boxObj([xmin, ymin, xmax, ymax], object_name)]


    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    frame_to_return = frame

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
    return frame_to_return, box_objs

def get_frame():
    return frame_to_return

def set_resolution(width, height):
    global imW, imH
    imW = width
    imH = height
    
def get_fps():
    return frame_rate_calc