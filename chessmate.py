#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import sys
import copy
import os

from CustomExceptions import IndexException
from chessgame import Game

from jetson_inference import detectNet
from jetson_utils import (cudaAllocMapped, cudaConvertColor, cudaFromNumpy, videoOutput, videoSource, cudaImage, cudaMemcpy, cudaDeviceSynchronize)
from pynput import keyboard

key = ""

def main():
    
    #creating possibility to define path to object detection model and label file
    parser = argparse.ArgumentParser(description="Detecting chess pieces from a live match and documenting the moves taken")
    parser.add_argument("--modelpath", nargs='?', default="/home/kian/jetson-inference/python/training/detection/ssd/models/model3/ssd-mobilenet.onnx")
    parser.add_argument("--labelpath", nargs='?', default="/home/kian/jetson-inference/python/training/detection/ssd/models/mymodels/labels.txt")
    parser.add_argument("--headless", nargs='?', default="")
    parser.add_argument("--videomode", nargs='?', type=int, default=1, help="determine resolution. 1:1640x1232, 2:1920x1080 (default 1)")
    parser.add_argument("--notationdir", nargs='?', default=".", description="Directory to save the notation file"))
    parser.add_argument("--white", nargs='?', default="", description="Player playing the white pieces")
    parser.add_argument("--black", nargs='?', default="", description="Player playing the black pieces")
    args = parser.parse_args(sys.argv[1:])

    resolution_width = 1640
    resolution_height = 1232

    if(args.videomode==2):
        resolution_width = 1920
        resolution_height = 1080
   
    #Use chrome browser to display stream. Hiding of IP for WebRTC must be disabled
    if(args.headless=="yes"):
        display = videoOutput("webrtc://@:8554/stream", ["--headless"])
    else:
        display = videoOutput("webrtc://@:8554/stream")
    
    notation_dir = args.notationdir

    os.chdir(notation_dir)

    net = detectNet(modelpath, labels=labelpath, input_blob="input_0", output_cvg="scores", output_bbox="boxes")
    net.SetConfidenceThreshold(0.15)
    net.SetClusteringThreshold(0.33)
    
    cam = cv2.VideoCapture(f"nvarguscamerasrc ! video/x-raw(memory:NVMM), width={resolution_width}, height={resolution_height},format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
   
    ret, frame = cam.read()
    M = calcTransformation(frame, resolution_width, resolution_height)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    game = Game()

    while cam.isOpened():
        global key
        ret, cv_image = cam.read()
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        cv_img = cv2.warpPerspective(cv_image, M, (resolution_width, resolution_height))

        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        converted_img = cudaFromNumpy(cv_img)


        detections = net.Detect(converted_img)
        display.Render(converted_img)
        if(key in ["m","d", "n"]):
            notation = game.update_board(detections, key)
            key = ""

    cam.release()

def calcTransformation(frame, resolution_width, resolution_height):
        '''
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        pt1x = int(input('X-Coordinate of point 1:'))
        pt1y = int(input('Y-Coordinate of point 1:'))
        pt2x = int(input('X-Coordinate of point 2:'))
        pt2y = int(input('Y-Coordinate of point 2:'))
        pt3x = int(input('X-Coordinate of point 3:'))
        pt3y = int(input('Y-Coordinate of point 3:'))
        pt4x = int(input('X-Coordinate of point 4:'))
        pt4y = int(input('Y-Coordinate of point 4:'))
        '''
        pt1x = 113
        pt1y = 321
        pt2x = 1051
        pt2y = 340
        pt3x = 95
        pt3y = 1259
        pt4x = 1030
        pt4y = 1278

        originCornerPoints = np.float32([[pt1x,pt1y], [pt2x, pt2y], [pt3x, pt3y], [pt4x, pt4y]])
        transformedCornerPoints = np.float32([[0, 0], [resolution_width, 0], [0, resolution_height], [resolution_width, resolution_height]])

        M = cv2.getPerspectiveTransform(originCornerPoints, transformedCornerPoints)

        return M

def on_press(event_key):
    global key
    try:
        key =event_key.char

    except AttributeError:
        key = event_key

if(__name__=="__main__"):
    main()
