#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import sys
import copy

from CustomExceptions import IndexException
from chessgame import Game

from jetson_inference import detectNet
from jetson_utils import (cudaAllocMapped, cudaConvertColor, cudaFromNumpy, videoOutput, videoSource, cudaImage, cudaMemcpy, cudaDeviceSynchronize)
from pynput import keyboard

key = ""

def main():
    
    #creating possibility to define path to object detection model and label file
    parser = argparse.ArgumentParser(description="Detecting chess pieces from a live match and documenting the moves taken")
    parser.add_argument("--modelpath", nargs='?', default="")
    parser.add_argument("--labelpath", nargs='?', default="")
    parser.add_argument("--headless", nargs='?', default="")
    parser.add_argument("--videomode", nargs='?', type=int, default=1, help="determine resolution. 1:1640x1232, 2:1920x1080 (default 1)")
    args = parser.parse_args(sys.argv[1:])

    resolution_width = 1640
    resolution_height = 1232

    #if modelpath and labelpath are not given fall back to default values. Otherwise
    #adopt cli arguments
    if(args.modelpath == ""):
        modelpath = "/home/kian/jetson-inference/python/training/detection/ssd/models/model3/ssd-mobilenet.onnx"
    else:
        modelpath = args.modelpath

    if(args.labelpath == ""):
        labelpath = "/home/kian/jetson-inference/python/training/detection/ssd/models/mymodels/labels.txt"
    else:
        labelpath = args.labelpath

    if(args.videomode==2):
        resolution_width = 1920
        resolution_height = 1080
   
    #Use chrome browser to display stream. Hiding of IP for WebRTC must be disabled
    if(args.headless=="yes"):
        display = videoOutput("webrtc://@:8554/stream", ["--headless"])
    else:
        display = videoOutput("webrtc://@:8554/stream")


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
        print(f"Key that was pressed: {key}")
        ret, cv_image = cam.read()
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        cv_img = cv2.warpPerspective(cv_image, M, (resolution_width, resolution_height))

        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        converted_img = cudaFromNumpy(cv_img)


#       cuda_rgb = cudaAllocMapped(width=converted_img.width, height=converted_img.height, format='rgb8')
#       cuda_img = cudaAllocMapped(width=converted_img.width, height=converted_img.height, format='gray8')
        #cudaConvertColor(cuda_rgb, cuda_img)
        detections = net.Detect(converted_img)
        display.Render(converted_img)
        #cv2.imshow('CV2', cv_img)
        #k = cv2.waitKey(500)
        #if counter%50==0:
        #    update_board(detections)
        #if k==-1:
        #    continue
        #if k == ord('d'):
        #    update_board(detections)
        #print("Loops durchlaufen: {}\nTaste: {}".format(counter, k))
        #counter += 1

    cam.release()

def on_press(event_key):
    global key
    try:
        print(f"alphanumeric key {event_key.char} pressed")
        key =event_key.char

    except AttributeError:
        print(f"special key {event_key} pressed")
        key = event_key

if(__name__=="__main__"):
    main()
