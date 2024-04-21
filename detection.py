#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import sys
import copy

from CustomExceptions import IndexException

from jetson_inference import detectNet
from jetson_utils import (cudaAllocMapped, cudaConvertColor, cudaFromNumpy, videoOutput, videoSource, cudaImage, cudaMemcpy, cudaDeviceSynchronize)
from pynput import keyboard


class Game():

    self.board = np.array([[5,4,3,2,1,3,4,5],[6,6,6,6,6,6,6,6],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[12,12,12,12,12,12,12,12],[11,10,9,8,7,9,10,11]], np.int16)

    self.chess_piece_dict={1:"King_w",2:"Queen_w",3:"Bishop_w",4:"N_Knight_w",5:"Rook_w",6:"Pawn_w",7:"King_b",8:"Queen_b",9:"Bishop_b",10:"N_Knight_b",11:"Rook_b",12:"Pawn_b"}

    self.vert_file_dict = {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h"}

    self.white_piece_numbers = [1,2,3,4,5,6]
    self.black_piece_numbers = [7,8,9,10,11,12]

    self.white_kings_field = "e2"
    self.black_kings_field = "e8"

    self.key = ""


    #method to retrieve the field on which a chess piece is standing on as a list index
    def determine_board_position(self, x, y):
        width = 1640/8
        height = 1232/8
        horizontal = x//width 
        vertical = y//height
        
        return int(horizontal), int(vertical)

    def determine_origin(self, prev_board, new_board, landing_row, landing_columnn):
            #Iterate over the board. 
            for k in range(8):
                for l in range(8):
                        
                #If field in new_board does not store same piece as prev_board and field 
                #is not the landing field, it should be the origin field
                    if(prev_board[k][l] != new_board[k][l] and k!=landing_row and l!=landing_column):
                        origin_row = k
                        origin_column = l

            return origin_row, origin_column

    def document_move(self, moved_piece, origin_row_index, origin_column_index, landing_row_index, landing_column_index, capture, check_given):
        piece_name = moved_piece
        piece_letter = piece_name[0]
        origin_column = vert_file_dict[origin_column_index]
        origin_row = "{}".format(origin_row_index+1)
        landing_column = vert_file_dict[landing_column_index]
        landing_row = "{}".format(landing_row_index+1)
        if(capture):
            notation = piece_letter + origin_column + origin_row + "x" + landing_column + landing_row
        else:
            notation = piece_letter + origin_column + origin_row + "-" + landing_column + landing_row

        if(check_given):
            notation += "#"

        return notation

    def document_en_passant(self, origin_row_index, origin_column_index, landing_row_index, landing_column_index, check_given):
        origin_column = vert_file_dict[origin_column_index]
        origin_row = "{}".format(origin_row_index+1)
        landing_column = vert_file_dict[landing_column_index]
        landing_row = "{}".format(landing_row_index+1)

        notation = origin_column + origin_row + "x" + landing_column + landing_row + " e.p."

        if(check_given):
            notation = origin_column + origin_row + "x" + landing_column + landing_row + "#" + " e.p."

        return notation

    def document_pawn_promotion(self, moved_piece, origin_row_index, origin_column_index, landing_row_index, landing_column_index, check_given):
        
        origin_column = vert_file_dict[origin_column_index]
        origin_row = "{}".format(origin_row_index+1)
        landing_column = vert_file_dict[landing_column_index]
        landing_row = "{}".format(landing_row_index+1)

        if(origin_column!=landing_column):
            notation = origin_column + origin_row + "x" + landing_column + landing_row + moved_piece[0]

        else:
            notation = origin_column + origin_row + "-" + landing_column + landing_row + moved_piece[0]

        if(check_given):
            notation += "#"

        return notation

    def reachable_by_pawn(self, color, board, prev_board, row_index, column_index):

        reachable_fields = []

        if(color == "w"):
            # If field in front is empty pawn can move to field
            if(board[row_index+1][column_index]==0):
                field = get_field_string(row_index+1, column_index)
                reachable_fields.append(field)
            # If pawn is not in "A" file and front right field as enemy piece it can be captured
            try:
                if(board[row_index+1][column_index+1] in black_piece_numbers):
                    field = get_field_string(row_index+1, column_index+1)
                    reachable_fields.append(field)
            except (KeyError, IndexError) as e:
                pass

            # If pawn is not in "A" file and front left field as enemy piece it can be captured
            try:
                if(board[row_index+1][column_index-1] in black_piece_numbers):
                    field = get_field_string(row_index+1, column_index-1)
                    reachable_fields.append(field)
            except (KeyError, IndexError) as e:
                pass
            
            # En passant only possible for white pawn in 5th row 
            if(row_index==4):
                try:
                    # Check if black pawn recently moved to left field
                    if(board[row_index][column_index-1]==12 and prev_board[row_index][column_index-1]==0):
                        field = get_field_string(row_index+1, column_index-1)
                        reachable_fields.append(field)
                except (KeyError, IndexError) as e:
                    pass

                try:
                    # Check if black pawn recently moved to right field
                    if(board[row_index][column_index+1]==12 and prev_board[row_index][column_index+1]!=12):
                        field = get_field_string(row_index+1, column_index+1)
                        reachable_fields.append(field)
                except (KeyError, IndexError) as e:
                    pass
       
        # Analog logic as for white
        if(color == "b"):
            if(board[row_index-1][column_index]==0):
                field = get_field_string(row_index-1, column_index)
                reachable_fields.append(field)

            try:
                if(board[row_index-1][column_index-1] in white_piece_numbers):
                    field = get_field_string(row_index-1, column_index-1)
                    reachable_fields.append(field)
            except (KeyError, IndexError) as e:
                pass

            try:
                if(board[row_index-1][column_index+1] in white_piece_numbers):
                    field = get_field_string(row_index-1, column_index+1)
                    reachable_fields.append(field)
            except (KeyError, IndexError) as e:
                pass
            
            if(row_index==3):
                try:
                    if(board[row_index][column_index-1]==6 and prev_board[row_index][column_index-1]!=6):
                        field = get_field_string(row_index-1, column_index-1)
                        reachable_fields.append(field)
                except (KeyError, IndexError) as e:
                    pass

                try:
                    if(board[row_index][column_index+1]==6 and prev_board[row_index][column_index+1]!=6):
                        field = get_field_string(row_index-1, column_index+1)
                        reachable_fields.append(field)
                except (KeyError, IndexError) as e:
                    pass

        return reachable_fields

    def reachable_by_king(self, color, board, prev_board, row_index, column_index):
        
        reachable_fields = []

        if(color=="w"):
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if(i == 0 and j == 0):
                        continue
                    try:
                        if(board[row_index+i][column_index+j] in black_piece_numbers or board[row_index+i][column_index+j]==0):
                            field = get_field_string(row_index+i, column_index+j)
                            reachable_fields.append(field)
                    except (KeyError,IndexError, IndexException) as e:
                        pass
        
        if(color=="b"):
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if(i == 0 and j == 0):
                        continue
                    try:
                        if(board[row_index+i][column_index+j] in white_piece_numbers or board[row_index+i][column_index+j]==0):
                            field = get_field_string(row_index+i, column_index+j)
                            reachable_fields.append(field)
                    except (KeyError, IndexError, IndexException) as e:
                        pass

        return reachable_fields


    def reachable_by_bishop_or_diagonal(self, color, board, prev_board, row_index, column_index):
        
        reachable_fields = []

        if(color=="w"):
            for i in [-1, 1]:
                for j in [-1, 1]:
                    row_counter = row_index + i
                    column_counter = column_index + j
                    try:
                        while(board[row_counter][column_counter] not in white_piece_numbers):
                        
                            field = get_field_string(row_counter, column_counter)
                            reachable_fields.append(field)
                            if(board[row_counter][column_counter] in black_piece_numbers):
                                break
                            row_counter += i
                            column_counter += j

                    except (KeyError, IndexError, IndexException) as e:
                        pass
        
        if(color=="b"):
            print(f"row")
            for i in [-1, 1]:
                for j in [-1, 1]:
                    row_counter = row_index + i
                    column_counter = column_index + j
                    try:
                        while(board[row_counter][column_counter] not in black_piece_numbers):
                            field = get_field_string(row_counter, column_counter)
                            reachable_fields.append(field)
                            if(board[row_counter][column_counter] in white_piece_numbers):
                                break
                            row_counter += i
                            column_counter += j

                    except (KeyError, IndexError, IndexException) as e:
                        pass

        return reachable_fields


    def reachable_by_rook_or_straight(self, color, board, prev_board, row_index, column_index):
        
        reachable_fields = []

        if(color=="w"):
            for i in [-1, 1]:
                row_counter = row_index + i
                try:
                    while(board[row_counter][column_index] not in white_piece_numbers):
                        field = get_field_string(row_counter, column_index)
                        reachable_fields.append(field)
                        if(board[row_counter][column_index] in black_piece_numbers):
                            break
                        row_counter += i

                except (KeyError, IndexError, IndexException) as e:
                    pass
            
                column_counter = column_index + i
                try:
                    while(board[row_index][column_counter] not in white_piece_numbers):
                        field = get_field_string(row_index, column_counter)
                        reachable_fields.append(field)
                        if(board[row_index][column_counter] in black_piece_numbers):
                            break
                        column_counter += i

                except (KeyError, IndexError, IndexException) as e:
                    pass

        if(color=="b"):
            for i in [-1, 1]:
                row_counter = row_index + i
                try:
                    while(board[row_counter][column_index] not in black_piece_numbers):
                        field = get_field_string(row_counter, column_index)
                        reachable_fields.append(field)
                        if(board[row_counter][column_index] in white_piece_numbers):
                            break
                        row_counter += i

                except (KeyError, IndexError, IndexException) as e:
                    pass
            
                column_counter = column_index + i
                try:
                    while(board[row_index][column_counter] not in black_piece_numbers):
                        field = get_field_string(row_index, column_counter)
                        reachable_fields.append(field)
                        if(board[row_index][column_counter] in white_piece_numbers):
                            break
                        column_counter += i

                except (KeyError, IndexError, IndexException) as e:
                    pass

        return reachable_fields


    def reachable_by_queen(self, color, board, prev_board, row_index, column_index):

        diagonal_fields = reachable_by_bishop_or_diagonal(color, board, None, row_index, column_index)

        straight_fields = reachable_by_rook_or_straight(color, board, None, row_index, column_index)

        reachable_fields = diagonal_fields + straight_fields

        return reachable_fields


    def reachable_by_knight(self, color, board, prev_board, row_index, column_index):
        
        reachable_fields = []

        if(color=="w"):
            for i in [-2,2]:
                for j in [-1, 1]:
                    try:
                        row_index1 = row_index + i
                        column_index1 = column_index +j
                        if(board[row_index1][column_index1] not in white_piece_numbers):
                            field = get_field_string(row_index1, column_index1)
                            reachable_fields.append(field)
                    except (KeyError, IndexError, IndexException) as e:
                        pass

                    try:
                        row_index2 = row_index + j
                        column_index2 = column_index +i
                        if(board[row_index2][column_index2] not in white_piece_numbers):
                            field = get_field_string(row_index2, column_index2)
                            reachable_fields.append(field)
                    except (KeyError, IndexError, IndexException) as e:
                        pass
        
        if(color=="b"):
            for i in [-2,2]:
                for j in [-1, 1]:
                    try:
                        row_index1 = row_index + i
                        column_index1 = column_index + j
                        if(board[row_index1][column_index1] not in black_piece_numbers):
                            field = get_field_string(row_index1, column_index1)
                            reachable_fields.append(field)
                    except (KeyError, IndexError, IndexException) as e:
                        pass

                    try:
                        row_index2 = row_index + j
                        column_index2 = column_index + i
                        if(board[row_index2][column_index2] not in black_piece_numbers):
                            field = get_field_string(row_index2, column_index2)
                            reachable_fields.append(field)
                    except (KeyError, IndexError, IndexException) as e:
                        pass


        return reachable_fields

    def get_field_string(self, row_index, column_index):

        if(row_index < 0 or row_index > 7 or column_index < 0 or column_index > 7):
                raise IndexException(f"One of indexes row_index: {row_index} and column_index: {column_index} is smaller than zero or lager than 7")
        row = f"{row_index+1}"
        column = vert_file_dict[column_index]

        return column+row

    def get_all_reachable_fields(self, board, prev_board):
        reachable_by_white = []
        reachable_by_black = []
        for row_index in range(8):
            for column_index in range(8):
                if board[row_index][column_index] in chess_piece_dict:
                    color = chess_piece_dict[board[row_index][column_index]][-1]
                    first_letter = chess_piece_dict[board[row_index][column_index]][0]
                    reachable_fields = reachable_field_function_dict[first_letter](color, board, prev_board, row_index, column_index)
                    reachable_by_white.extend(reachable_fields) if color=="w" else reachable_by_black.extend(reachable_fields)

        return reachable_by_white, reachable_by_black

    def check_for_check(self, reachable_by_white, reachable_by_black , white_kings_field, black_kings_field):
        check_given = black_kings_field in reachable_by_white if color == "w" else white_kings_field in reachable_by_black

        return check_given


    reachable_field_function_dict = {"K":reachable_by_king,"Q":reachable_by_queen,"B":reachable_by_bishop_or_diagonal,"N":reachable_by_knight,"R":reachable_by_rook_or_straight,"P":reachable_by_pawn}


    def calcTransformation(self, frame, resolution_width, resolution_height):

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

    def update_board(self, detections):
        moved_piece = ""
        reachable_by_white = []
        reachable_by_black = []
        landing_column_index = 0
        landing_row_index = 0
        origin_column_index = 0
        origin_row_index = 0
        capture = 0
        new_board = np.zeros((8,8), np.int16)

        for piece in detections:

            #Get the detected pieces box horizontal and vertical center and retrieve according field
            x = piece.Center[0]
            y = piece.Center[1]
            row_index,column_index = determine_board_position(x,y)

            if(piece.ClassID%6==1):
                if(piece.ClassID==1):
                    white_kings_field = get_field_string(row_index, column_index)
                if(piece.ClassID==7):
                    black_kings_field = get_field_string(row_index, column_index)

            #Flag the field of detected piece as updated
            new_board[row_index][column_index] = piece.ClassID

            #If the currently stored piece is not the same as the detected piece, store the information for documentation.

            # ROCHADE IST AUSNAHME
            if(board[row_index][column_index] != piece.ClassID):
                moved_piece = chess_piece_dict[piece.ClassID]
                landing_row_index = row_index
                landing_column_index = column_index
            
            #If the field has not been empty previously, flag the move as a capture
                if(board[row_index][column_index] in chess_piece_dict):
                    capture = 1

        reachable_by_white, reachable_by_black = get_all_reachable_fields(new_board, board)

        check_given = check_for_check(reachable_by_white, reachable_by_black, moved_piece[-1])

        origin_row_index, origin_column_index = determine_origin(board, new_board, landing_row_index, landing_column_index)

        notation = document_move(moved_piece, origin_row_index, origin_column_index, landing_row_index, landing_row_index, capture, check_given)

        board = new_board

        global key 
        key = ""

    def on_press(self, event_key):
        global key
        try:
            print('alphanumeric key {0} pressed'.format(event_key.char))
            key = event_key.char
        except AttributeError:
            print('special key {0} pressed'.format(event_key))
            key = event_key

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

    net = detectNet(modelpath, labels=labelpath, input_blob="input_0", output_cvg="scores", output_bbox="boxes")
    cam = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width={}, height={},format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink".format(resolution_width, resolution_height))
   
   #Use chrome browser to display stream. Hiding of IP for WebRTC must be disabled
    if(args.headless=="yes"):
        display = videoOutput("webrtc://@:8554/stream", ["--headless"])
    else:
        display = videoOutput("webrtc://@:8554/stream")

    net.SetConfidenceThreshold(0.15)
    net.SetClusteringThreshold(0.33)
    ret, frame = cam.read()
    M = calcTransformation(frame, resolution_width, resolution_height)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while cam.isOpened():
        global key
#        print(f"Key that was pressed: {key}")
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

if(__name__=="__main__"):
    main()
