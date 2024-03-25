#!/usr/bin/env python3

import cv2
import numpy as np

from jetson_inference import detectNet
from jetson_utils import (cudaAllocMapped, cudaConvertColor, cudaFromNumpy, videoOutput, videoSource, cudaImage, cudaMemcpy, cudaDeviceSynchronize)

def main():
    
    #creating a dictionary to retrieve piece description from detected class id
    chess_piece_dict={2:"w_King",3:"w_Queen",4:"w_Bishop",5:"w_Knight",6:"w_Rook",7:"w_Pawn",8:"b_King", 9:"b_Queen", 10:"b_Bishop",11:"b_Knight",12:"b_Rook",13:"b_Pawn"}
    
    #creating dictionaries for the vertical chess board files, since they need to be converted to letters
    vert_file_dict = {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h"}

    #creating a numpy array representing the current state of the chess board. Initialized with the state of a beginning chess match, assuming white is at the bottom
    board = np.array([[6,5,4,3,2,4,5,6],[7,7,7,7,7,7,7,7],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[13,13,13,13,13,13,13,13],[12,11,10,9,8,10,11,12]], np.int16)

    net = detectNet("/home/kian/jetson-inference/python/training/detection/ssd/models/mymodels/ssd-mobilenet.onnx", labels="/home/kian/jetson-inference/python/training/detection/ssd/models/mymodels/labels.txt", input_blob="input_0", output_cvg="scores", output_bbox="boxes")
    cam = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1640, height=(int)1232,format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
    display = videoOutput("webrtc://@:8554/stream")
    #Konfiguration von Confidence und Clustering ueber Aufrufparameter implementieren
    net.SetConfidenceThreshold(0.25)
    net.SetClusteringThreshold(0.33)
    ret, frame = cam.read()
    M = calcTransformation(frame)
    while cam.isOpened():
        ret, cv_image = cam.read()
        cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        cv_img = cv2.warpPerspective(cv_image, M, (1640, 1232))

        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        converted_img = cudaFromNumpy(cv_img)
        
        cuda_rgb = cudaAllocMapped(width=converted_img.width, height=converted_img.height, format='rgb8')
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

def update_board(detections):
    for piece in detections:
        x = piece.Center[0]
        y = piece.Center[1]
        i,j = determine_board_position(x,y)
        if(board[i][j] != piece.ClassID):
            document_move(piece.ClassID, i, j)


#method to retrieve the field on which a chess piece is standing on as a list index
def determine_board_position(x, y):
    width = 1640/8
    height = 1232/8
    horizontal = x//width
    #opencv counting y coordinate ascending from top to bottom
    vertical = 7 - y//height
    
    return horizontal, vertical
    
def document_move(class_id, i, j, capture):
    piece_name = chess_piece_dict[class_id]
    piece_letter = piece_name[2]
    column = vert_file_dict[i]
    row = "{}".format(j+1)
    if(capture):
        notation = piece_letter + "x" + column + row
    else:
        notation = piece_letter + column + row

if(__name__=="__main__"):
    main()
