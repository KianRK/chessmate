#!/usr/bin/env python3

import cv2
import sys
import argparse
from pathlib import Path
import os

# Add output path and gray scaling to image before saving

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

    cam=cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1640,\
            height=(int)1232,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv \
            ! video/x-raw, format=(string)BGRx ! videoconvert !  appsink")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame.")
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            print("Escape pressed, closing...")
            break
        elif k%256 == 32:
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
