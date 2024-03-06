#!/usr/bin/env python3.6
''' /usr/local/bin/image_capture is a hard link from ~/chessmate/image_capture.py
to make it globally accessible as an executable
'''

import cv2
import sys
import argparse
import os
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt

def main():

    # Creating possibility to define an output path for captured images
    parser = argparse.ArgumentParser(description="capturing images from csi camera module")
    parser.add_argument("outputpath", nargs='?', default="")
    args = parser.parse_args(sys.argv[1:])

    # If output path doesn't exist or is not empty exit application
    print(args.outputpath)
    if not os.path.exists(args.outputpath) and not args.outputpath=="":
        print("Output directory {} does not exist.".format(args.outputpath))
        sys.exit()
    
    os.chdir(args.outputpath)
    cam=cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1640, height=(int)1232,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    
    print(cam.isOpened())
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab initial frame. Fix camera!")
        sys.exit()
    
    M = calcTransformation(frame)   

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame.")
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        dst = cv2.warpAffine(frame, M, (1640, 1232))
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        #_, threshFrame = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        cv2.imshow("test", gray)

        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape pressed, closing...")
            break
        elif k%256 == 32:
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, gray)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

def calcTransformation(frame):
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    plt.imshow(frame)
    plt.show()
    pt1x = int(input('X-Coordinate of point 1:'))
    pt1y = int(input('Y-Coordinate of point 1:'))
    pt2x = int(input('X-Coordinate of point 2:'))
    pt2y = int(input('Y-Coordinate of point 2:'))
    pt3x = int(input('X-Coordinate of point 3:'))
    pt3y = int(input('Y-Coordinate of point 3:'))

    originCornerPoints = np.float32([[pt1x,pt1y], [pt2x, pt2y], [pt3x, pt3y]])
    transformedCornerPoints = np.float32([[0, 0], [1640, 0], [0, 1232]])
    

    M = cv2.getAffineTransform(originCornerPoints, transformedCornerPoints)

    return M


if __name__ == "__main__":
    main()
