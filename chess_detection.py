import cv2
import numpy as np
from jetson_inference import detectnet
from jetson_utils import videoSource, videoOutput, cudaFromNumpy,

# parse cli arguments

#createinput from camera(try csi://0 first) and videooutput

#process cuda image to contain roi with format (gray8)

#convert opencv image to cudaimage

#detect object with model

#stream output to via webrtc or standardoutput
