#!/usr/bin/env python3

import cv2
import numpy as np

from jetson_inference import detectNet
from jetson_utils import (cudaAllocMapped, cudaConvertColor, cudaFromNumpy, videoOutput, videoSource, cudaImage, cudaMemcpy, cudaDeviceSynchronize)

def main():

    net = detectNet("/home/kian/jetson-inference/python/training/detection/ssd/models/ssd-mobilenet.onnx", labels="/home/kian/jetson-inference/python/training/detection/ssd/models/labels.txt", input_blob="input_0", output_cvg="scores", output_bbox="boxes")
    cam = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1640, height=(int)1232,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    display = videoOutput("webrtc://@:8554/stream")

    ret, frame = cam.read()
    M = calcTransformation(frame)

    while cam.isOpened():
        ret, cv_image = cam.read()
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        cv_img = cv2.warpPerspective(cv_image, M, (1640, 1232))

        converted_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        converted_img = cudaFromNumpy(cv_img)
        
        cuda_rgb = cudaAllocMapped(width=converted_img.width, height=converted_img.height, format='rgb8')
        cudaConvertColor(converted_img, cuda_rgb)
        cuda_img = cudaAllocMapped(width=converted_img.width, height=converted_img.height, format='gray8')
        cudaConvertColor(cuda_rgb, cuda_img)
        detections = net.Detect(converted_img)
        print("List of detected objects: {}".format(detections))
        display.Render(converted_img)

def calcTransformation(frame):

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    pt1x = int(input('X-Coordinate of point 1:'))
    pt1y = int(input('Y-Coordinate of point 1:'))
    pt2x = int(input('X-Coordinate of point 2:'))
    pt2y = int(input('Y-Coordinate of point 2:'))
    pt3x = int(input('X-Coordinate of point 3:'))
    pt3y = int(input('Y-Coordinate of point 3:'))
    pt4x = int(input('X-Coordinate of point 4:'))
    pt4y = int(input('Y-Coordinate of point 4:'))

    originCornerPoints = np.float32([[pt1x,pt1y], [pt2x, pt2y], [pt3x, pt3y], [pt4x, pt4y]])
    transformedCornerPoints = np.float32([[0, 0], [1640, 0], [0, 1232], [1640, 1232]])

    M = cv2.getPerspectiveTransform(originCornerPoints, transformedCornerPoints)
    
    return M

if(__name__=="__main__"):
    main()
