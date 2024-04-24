#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
import sys
import copy

from CustomExceptions import IndexException

from math import copysign
from jetson_inference import detectNet
from jetson_utils import (cudaAllocMapped, cudaConvertColor, cudaFromNumpy, videoOutput, videoSource, cudaImage, cudaMemcpy, cudaDeviceSynchronize)
from pynput import keyboard


class Game():
        
    def __init__(self):
        self.board = np.array([[5,4,3,2,1,3,4,5],[6,6,6,6,6,6,6,6],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[12,12,12,12,12,12,12,12],[11,10,9,8,7,9,10,11]], np.int16)
        
        self.board_history = []

        self.new_board = np.zeros((8,8), np.int16)
        
        self.chess_piece_dict={1:"King_w",2:"Queen_w",3:"Bishop_w",4:"N_Knight_w",5:"Rook_w",6:"Pawn_w",7:"King_b",8:"Queen_b",9:"Bishop_b",10:"N_Knight_b",11:"Rook_b",12:"Pawn_b"}

        self.vert_file_dict = {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h"}

        self.reachable_field_function_dict = {"K":self.reachable_by_king,"Q":self.reachable_by_queen,"B":self.reachable_by_bishop_or_diagonal,"N":self.reachable_by_knight,"R":self.reachable_by_rook_or_straight,"P":self.reachable_by_pawn}
        
        self.document_move_function_dict = {"c":self.document_castle,"e":self.document_en_passant,"p":self.document_pawn_promotion}

        self.white_piece_numbers = [1,2,3,4,5,6]
        self.black_piece_numbers = [7,8,9,10,11,12]

        self.white_kings_field = "e2"
        self.white_kings_row_index = 0
        self.white_kings_column_index = 4

        self.black_kings_field = "e8"
        self.black_kings_row_index = 7
        self.black_kings_column_index = 4

        self.white_can_castle = True
        self.black_can_castle = True
    
    def update_board(self, detections, key):
        moved_piece = ""
        moved_color = ""
        reachable_by_white = []
        protected_by_white = []
        reachable_by_black = []
        protected_by_black = []
        fields_of_pieces_giving_check = []
        landing_column_index = 0
        landing_row_index = 0
        origin_column_index = 0
        origin_row_index = 0
        castle = False
        capture = False
        checkmate = False
        en_passant = False
        pawn_promotion = False

        for piece in detections:

            #Get the detected pieces box horizontal and vertical center and retrieve according field
            x = piece.Center[0]
            y = piece.Center[1]
            row_index, column_index = determine_board_position(x,y)

            if(piece.ClassID%6==1):
                if(piece.ClassID==1):
                    self.white_kings_row_index = row_index
                    self.white_kings_column_index = column_index
                    self.white_kings_field = self.get_field_string(row_index, column_index)
                if(piece.ClassID==7):
                    self.black_kings_row_index = row_index
                    self.black_kings_column_index = column_index
                    self.black_kings_field = self.get_field_string(row_index, column_index)
            # Update the piece on the field
            self.new_board[row_index][column_index] = piece.ClassID

            #If the currently stored piece is not the same as the detected piece, store the information for documentation.

            if(self.board[row_index][column_index] != piece.ClassID):
                moved_piece = self.chess_piece_dict[piece.ClassID]
                landing_row_index = row_index
                landing_column_index = column_index
                moved_color = moved_piece[-1]
            
            #If the field has not been empty previously, flag the move as a capture
                if(self.board[row_index][column_index] in self.chess_piece_dict):
                    capture = True

        if(moved_piece[0]="P"):
            en_passant = self.check_for_en_passant(landing_row_index, landing_column_index)
        elif(landing_row_index%7 == 0):
            pawn_promotion = self.check_for_pawn_promotion(landing_row_index, landing_column_index)

        origin_row_index, origin_column_index = determine_origin(landing_row_index, landing_column_index)

        reachable_by_white, reachable_by_black, protected_by_white, protected_by_black, fields_of_pieces_giving_check, path_to_king = self.get_all_reachable_fields()
        
        check_given = len(field_of_piecex_giving_check > 0)

        '''if(check_given):
            if(moved_color == "w"):
                checkmate = check_for_mate(reachable_by_white, reachable_by_black, protected_by_black, fields_of_pieces_giving_check, path_to_king, moved_color)
            else:
                checkmate = check_for_mate(reachable_by_white, reachable_by_black, protected_by_white, fields_of_pieces_giving_check, path_to_king, moved_color)

        else:
            stalemate = len(reachable_by_white) == 0 if moved_color == "w" else len(reachable_by_black) == 0
'''

        if(moved_piece[0] == "K" or moved_piece[0] == "R"):
            castle = self.check_for_castle(moved_piece[-1])

        if(castle):
            if(moved_color == "w"):
                self.white_can_castle = False
            else:
                self.black_can_castle = False

        notation = document_move(moved_piece, origin_row_index, origin_column_index, landing_row_index, landing_row_index, capture, check_given, castle, en_passant, pawn_promotion, key)

        self.board_history.append(self.board)
        
        self.board = self.new_board
 
        key = ""



    def document_move(self, moved_piece, origin_row_index, origin_column_index, landing_row_index, landing_column_index, capture, castle, check_given, en_passant, pawn_promotion, key):

        if(castle):
            notation = self.document_castle(moved_piece[-1], check_given)
            print(notation)
            return notation

        if(en_passant):
            notation = self.document_en_passant(origin_row_index, origin_column_index, landing_row_index, landing_column_index, check_given)
            print(notation)
            return notation

        if(pawn_promotion):
            notation = self.document_pawn_promotion(moved_piece, origin_row_index, origin_column_index, landing_row_index, landing_column_index, check_given)
            print(notation)
            return notation

        piece_letter = moved_piece[0]
        origin_column = self.vert_file_dict[origin_column_index]
        origin_row = "{}".format(origin_row_index+1)
        landing_column = self.vert_file_dict[landing_column_index]
        landing_row = "{}".format(landing_row_index+1)

        if(capture):

            notation = piece_letter + origin_column + origin_row + "x" + landing_column + landing_row
        else:
            notation = piece_letter + origin_column + origin_row + "-" + landing_column + landing_row
        
        if(key=="m"):
            notation += "#"
            print(notation)
            return notation

        if(key=="d"):
            notation = "="
            print(notation)
            return notation

        if(check_given):
            notation += "#"

        print(notation)
        return notation

    def document_en_passant(self, origin_row_index, origin_column_index, landing_row_index, landing_column_index, check_given):
        origin_column = self.vert_file_dict[origin_column_index]
        origin_row = "{}".format(origin_row_index+1)
        landing_column = self.vert_file_dict[landing_column_index]
        landing_row = "{}".format(landing_row_index+1)

        notation = origin_column + origin_row + "x" + landing_column + landing_row + " e.p."

        if(check_given):
            notation = origin_column + origin_row + "x" + landing_column + landing_row + "#" + " e.p."

        return notation

    def document_pawn_promotion(self, moved_piece, origin_row_index, origin_column_index, landing_row_index, landing_column_index, check_given):
        
        origin_column = self.vert_file_dict[origin_column_index]
        origin_row = "{}".format(origin_row_index+1)
        landing_column = self.vert_file_dict[landing_column_index]
        landing_row = "{}".format(landing_row_index+1)

        if(origin_column!=landing_column):
            notation = origin_column + origin_row + "x" + landing_column + landing_row + moved_piece[0]

        else:
            notation = origin_column + origin_row + "-" + landing_column + landing_row + moved_piece[0]

        if(check_given):
            notation += "#"

        return notation

    def document_castle(self, color, check_given):

        if(color=="w"):
            notation = "0-0" if(self.new_board[0][5]!=self.board[0][5]) else "0-0-0"
        
        else:
            notation = "0-0" if(self.new_board[7][5]!=self.board[7][5]) else "0-0-0"

        if(check_given):
            notation += "#"

        return notation

    def reachable_by_pawn(self, color, row_index, column_index):

        reachable_fields = []
        protected_fields = []

        if(color == "w"):
            # If field in front is empty pawn can move to field
            if(self.new_board[row_index+1][column_index]==0):
                field =self.get_field_string(row_index+1, column_index)
                reachable_fields.append(field)
            # If pawn is not in "A" file and front right field as enemy piece it can be captured
            try:
                for i in [-1,1]:
                    field =self.get_field_string(row_index+1, column_index+i)
                    if(self.new_board[row_index+1][column_index+i] in self.black_piece_numbers):
                        reachable_fields.append(field)
                    elif(self.new_board[row_index+1][column_index+i] in self.white_piece_numbers):
                        protected_fields.append(field)
            except (KeyError, IndexError, IndexException) as e:
                pass

            
            # En passant only possible for white pawn in 5th row 
            if(row_index==4):
                try:
                    for i in [-1,1]:
                        # Check if black pawn recently moved to right field
                        if(self.new_board[row_index][column_index+i]==12 and self.board[row_index][column_index+1]!=12):
                            field =self.get_field_string(row_index+1, column_index+i)
                            reachable_fields.append(field)
                except (KeyError, IndexError, IndexException) as e:
                    pass
       
        # Analog logic as for white
        if(color == "b"):
            if(self.new_board[row_index-1][column_index]==0):
                field =self.get_field_string(row_index-1, column_index)
                reachable_fields.append(field)
 
            for i in [-1,1]:
                try:
                    field =self.get_field_string(row_index-1, column_index+i)
                    if(self.new_board[row_index-1][column_index+1] in self.white_piece_numbers):
                        reachable_fields.append(field)
                    elif(self.new_board[row_index-1][column_index+i] in self.black_piece_numbers):
                        protected_fields.append(field)
                except (KeyError, IndexError, IndexException) as e:
                    pass
            
            if(row_index==3):
                for i in [-1,1]:
                    try:
                        if(self.new_board[row_index][column_index+i]==6 and self.board[row_index][column_index+i]!=6):
                            field =self.get_field_string(row_index-1, column_index+i)
                            reachable_fields.append(field)
                    except (KeyError, IndexError, IndexException) as e:
                        pass
            
        return reachable_fields, protected_fields

    def reachable_by_king(self, color, row_index, column_index):
        
        reachable_fields = []
        protected_fields = []

        if(color=="w"):
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if(i == 0 and j == 0):
                        continue
                    try:
                        field =self.get_field_string(row_index+i, column_index+j)
                        if(self.new_board[row_index+i][column_index+j] not in self.white_piece_numbers):
                            reachable_fields.append(field)
                        else:
                            protected_fields.append(field)

                    except (KeyError,IndexError, IndexException) as e:
                        pass
        
        if(color=="b"):
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if(i == 0 and j == 0):
                        continue
                    try:
                        field =self.get_field_string(row_index+i, column_index+j)
                        if(self.new_board[row_index+i][column_index+j] not in self.black_piece_numbers):
                            reachable_fields.append(field)
                        else:
                            protected_fields.append(field)
                    except (KeyError, IndexError, IndexException) as e:
                        pass
    
        return reachable_fields, protected_fields


    def reachable_by_bishop_or_diagonal(self, color, row_index, column_index):
        
        reachable_fields = []
        protected_fields = []

        if(color=="w"):
            for i in [-1, 1]:
                for j in [-1, 1]:
                    row_counter = row_index + i
                    column_counter = column_index + j
                    try:
                        while(self.new_board[row_counter][column_counter] == 0):     
                            field =self.get_field_string(row_counter, column_counter)
                            reachable_fields.append(field)
                            row_counter += i
                            column_counter += j
                    
                        
                        field =self.get_field_string(row_counter, column_counter)
                        
                        if(self.new_board[row_counter][column_counter] in self.black_piece_numbers):
                            reachable_fields.append(field)
                        else:
                            protected_fields.append(field)


                    except (KeyError, IndexError, IndexException) as e:
                        pass
        
        if(color=="b"):
            for i in [-1, 1]:
                for j in [-1, 1]:
                    row_counter = row_index + i
                    column_counter = column_index + j
                    try:
                        while(self.new_board[row_counter][column_counter] == 0):
                            field =self.get_field_string(row_counter, column_counter)
                            reachable_fields.append(field)
                            row_counter += i
                            column_counter += j

                        field =self.get_field_string(row_counter, column_counter)
                        
                        if(self.new_board[row_counter][column_counter] in self.white_piece_numbers):
                            reachable_fields.append(field)
                        else:
                            protected_fields.append(field)

                    except (KeyError, IndexError, IndexException) as e:
                        pass

        return reachable_fields, protected_fields


    def reachable_by_rook_or_straight(self, color, row_index, column_index):
        
        reachable_fields = []
        protected_fields = []

        if(color=="w"):
            for i in [-1, 1]:
                row_counter = row_index + i
                try:
                    while(self.new_board[row_counter][column_index] == 0):
                        field =self.get_field_string(row_counter, column_index)
                        reachable_fields.append(field)
                        row_counter += i
                        
                    field =self.get_field_string(row_counter, column_index)
                    if(self.new_board[row_counter][column_index] in self.black_piece_numbers):
                        reachable_fields.append(field)
                    else:
                        protected_fields.append(field)


                except (KeyError, IndexError, IndexException) as e:
                    pass
            
                column_counter = column_index + i
                try:
                    while(self.new_board[row_index][column_counter] == 0):
                        field =self.get_field_string(row_index, column_counter)
                        reachable_fields.append(field)
                        column_counter += i
                    
                    field =self.get_field_string(row_index, column_counter)
                    if(self.new_board[row_index][column_counter] in self.black_piece_numbers):
                        reachable_fields.append(field)
                    else:
                        protected_fields.append(field)

                except (KeyError, IndexError, IndexException) as e:
                    pass

        if(color=="b"):
            for i in [-1, 1]:
                row_counter = row_index + i
                try:
                    while(self.new_board[row_counter][column_index] == 0):
                        field =self.get_field_string(row_counter, column_index)
                        reachable_fields.append(field)
                        row_counter += i
                        
                    field =self.get_field_string(row_counter, column_index)
                    if(self.new_board[row_counter][column_index] in self.white_piece_numbers):
                        reachable_fields.append(field)
                    else:
                        protected_fields.append(field)

                except (KeyError, IndexError, IndexException) as e:
                    pass
            
                column_counter = column_index + i
                try:
                    while(self.new_board[row_index][column_counter] == 0):
                        field =self.get_field_string(row_index, column_counter)
                        reachable_fields.append(field)
                        column_counter += i
                    
                    field =self.get_field_string(row_index, column_counter)
                    if(self.new_board[row_index][column_counter] in self.white_piece_numbers):
                        reachable_fields.append(field)
                    else:
                        protected_fields.append(field)


                except (KeyError, IndexError, IndexException) as e:
                    pass

        return reachable_fields, protected_fields


    def reachable_by_queen(self, color, row_index, column_index):

        diagonal_fields, protected_diagonal = self.reachable_by_bishop_or_diagonal(color, row_index, column_index)

        straight_fields, protected_straight = self.reachable_by_rook_or_straight(color, row_index, column_index)

        reachable_fields = diagonal_fields + straight_fields
        protected_fields = protected_diagonal + protected_straight

        return reachable_fields, protected_fields


    def reachable_by_knight(self, color, row_index, column_index):
        
        reachable_fields = []
        protected_fields = []

        if(color=="w"):
            for i in [-2,2]:
                for j in [-1, 1]:
                    try:
                        row_index1 = row_index + i
                        column_index1 = column_index +j
                        field =self.get_field_string(row_index1, column_index1)
                        if(self.new_board[row_index1][column_index1] not in self.white_piece_numbers):
                            reachable_fields.append(field)
                        else:
                            protected_fields.append(field)


                    except (KeyError, IndexError, IndexException) as e:
                        pass

                    try:
                        row_index2 = row_index + j
                        column_index2 = column_index +i
                        field =self.get_field_string(row_index2, column_index2)
                        if(self.new_board[row_index2][column_index2] not in self.white_piece_numbers):
                            reachable_fields.append(field)
                        else:
                            protected_fields.append(field)
                    except (KeyError, IndexError, IndexException) as e:
                        pass
        
        if(color=="b"):
            for i in [-2,2]:
                for j in [-1, 1]:
                    try:
                        row_index1 = row_index + i
                        column_index1 = column_index + j
                        field =self.get_field_string(row_index1, column_index1)
                        if(self.new_board[row_index1][column_index1] not in self.black_piece_numbers):
                            reachable_fields.append(field)
                        else:
                            protected_fields.append(field)
                    except (KeyError, IndexError, IndexException) as e:
                        pass

                    try:
                        row_index2 = row_index + j
                        column_index2 = column_index + i
                        field =self.get_field_string(row_index2, column_index2)
                        if(self.new_board[row_index2][column_index2] not in self.black_piece_numbers):
                            reachable_fields.append(field)
                        else:
                            protected_fields.append(field)
                    except (KeyError, IndexError, IndexException) as e:
                        pass


        return reachable_fields, protected_fields

    def get_field_string(self, row_index, column_index):

        if(row_index < 0 or row_index > 7 or column_index < 0 or column_index > 7):
                raise IndexException(f"One of indexes row_index: {row_index} and column_index: {column_index} is smaller than zero or lager than 7")
        row = f"{row_index+1}"
        column = self.vert_file_dict[column_index]

        return column+row

    def get_all_reachable_fields(self):
        reachable_by_white = []
        reachable_by_black = []
        protected_by_white = []
        protected_by_black = []
        fields_of_pieces_giving_check = []
        path_of_threatening_pieces_to_king = []
        for row_index in range(8):
            for column_index in range(8):
                if(self.new_board[row_index][column_index] in self.chess_piece_dict):
                    color = self.chess_piece_dict[self.new_board[row_index][column_index]][-1]
                    first_letter = self.chess_piece_dict[self.new_board[row_index][column_index]][0]
                    reachable_fields, protected_fields = self.reachable_field_function_dict[first_letter](color, row_index, column_index)

                    if(color == "w"):
                        reachable_by_white.extend(reachable_fields)
                        protected_by_white.extend(protected_fields)
                        if(self.black_kings_field in reachable_fields):
                            fields_of_pieces_giving_check.append(self.get_field_string(row_index, column_index))
                            if(first_letter in ["Q","B","R"]):
                                path_to_king = get_path_to_king(row_index, column_index, color)
                                path_of_threatening_pieces_to_king = get_path_to_king(path_to_king)

                    if(color == "b"):
                        reachable_by_black.extend(reachable_fields)
                        protected_by_black.extend(protected_fields)
                        if(self.white_kings_field in reachable_fields):
                            fields_of_pieces_giving_check.append(self.get_field_string(row_index, column_index))
                            # Only queen, bishop and rook have blockable path when giving check
                            if(first_letter in ["Q","B","R"]):
                                path_to_king = get_path_to_king(row_index, column_index, color)
                                path_of_threatening_pieces_to_king.extend(path_to_king)
        
        '''reachable_by_white_king = self.reachable_by_king("w", self.white_kings_row_index, self.white_kings_column_index)
        for field in reachable_by_white_king:
            if field in reachable_by_black+protected_by_black:
                reachable_by_white.remove(field)

        reachable_by_black_king = self.reachable_by_king("b", self.black_kings_row_index, self.black_kings_column_index)
        for field in reachable_by_black_king:
            if field in reachable_by_white+protected_by_white:
                reachable_by_black.remove(field)
            '''

        return reachable_by_white, reachable_by_black, protected_by_white, protected_by_black, fields_of_pieces_giving_check, path_of_threatening_pieces_to_king

    def get_path_to_king(self, row_index, column_index, color):

        fields_in_path = []

        if(color == "w"):
            row_difference = self.black_kings_row_index - row_index
            row_step = int(copysign(1, row_difference) if row_difference != 0 else row_difference)
            row_counter = row_index + row_step

            column_difference = self.black_kings_column_index - column_index
            column_step = int(copysign(1, column_difference) if column_difference != 0 else column_difference)
            column_counter = column_index + column_step

            while not(row_counter == self.black_kings_row_index and column_counter == self.black_kings_column_index):
                field = self.get_field_string(row_counter, column_counter)
                fields_in_path.append(field)
                row_counter += row_step
                column_counter += column_step

        if(color == "b"):
            row_difference = self.white_kings_row_index - row_index
            row_step = int(copysign(1, row_difference) if row_difference != 0 else row_difference)
            row_counter = row_index + row_step

            column_difference = self.white_kings_column_index - column_index
            column_step = int(copysign(1, column_difference) if column_difference != 0 else column_difference)
            column_counter = column_index + column_step

            while not(row_counter == self.white_kings_row_index and column_counter == self.white_kings_column_index):
                field = self.get_field_string(row_counter, column_counter)
                fields_in_path.append(field)
                row_counter += row_step
                column_counter += column_step


        return fields_in_path


    def check_for_check(self, reachable_by_white, reachable_by_black, color):
        
        check_given = self.black_kings_field in reachable_by_white if color == "w" else self.white_kings_field in reachable_by_black

        return check_given

    def check_for_en_passant(self, landing_row_index, landing_column_index):

        if(landing_row_index == 5):
            if(self.new_board[4][landing_column_index] != self.board[4][landing_column_index]):
                return True
        if(landing_row_index == 2):
            if(self.new_board[3][landing_column_index] != self.board[3][landing_column_index]):
                return True

        return False

    def check_for_pawn_promotion(self, landing_row_index, landing_column_index):

        if(landing_row_index == 7):
            if(self.new_board[6][landing_column_index]==0 and self.board[6][landing_column_index]==6):
                return True

        if(landing_row_index == 2):
            if(self.new_board[3][landing_column_index]==0 and self.board[3][landing_column_index]==12):
                return True

        return False

    def check_for_castle(self, color):
        
        if(color=="w"):
            if(self.new_board[0][6]==1):
                if(self.new_board[0][5]==5 and self.board[0][5]!=5):
                    return True

            if(self.new_board[0][2]==1):
                if(self.new_board[0][3]==5 and self.board[0][3]!=5):
                    return True
        
        if(color=="b"):
            if(self.new_board[7][6]==7):
                if(self.new_board[7][5]==11 and self.board[7][5]!=11):
                    return True

            if(self.new_board[7][2]==7):
                if(self.new_board[7][3]==11 and self.board[7][3]!=11):
                    return True

        return False

    def check_for_mate(self, reachable_by_white, reachable_by_black, protected_by_opponent, fields_of_pieces_giving_check, path_to_king, color):
        
        if(color == "w"):
            kings_field = self.black_kings_field
            kings_row_index = self.black_kings_row_index
            kings_column_index = self.black_kings_column_index
            reachable_by_black_king = self.reachable_by_king("b", kings_row_index, kings_column_index)
            for field in reachable_by_black_king:
                if(field in reachable_by_white+protected_by_white):
                    reachable_by_black_king.remove(field)
                    reachable_by_black.remove(field)
            if(len(reachable_by_black_king) == 0):
                if(len(fields_of_pieces_giving_check) == 1):
                    for field in reachable_by_white:
                        if(field in fields_of_pieces_giving_check+path_to_king):
                            return False
                        else:
                            return True



        '''
        die Felder die der König erreichen kann sind reachable - protected - reachable Gegner
        und die Schach gebenden Felder > 1 oder nicht in reachable Freund
        
        Matt wenn Checking_pieces > 1 und König keine Felder zum ausweichen hat

        oder wenn Checkin_pieces = 1 und König keine Felder zum Ausweichen hat, und Checking_piece und Path nicht erreichbar sind
        '''

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

    #method to retrieve the field on which a chess piece is standing on as a list index
    def determine_board_position(self, x, y):
        width = 1640/8
        height = 1232/8
        horizontal = x//width 
        vertical = y//height
        
        return int(horizontal), int(vertical)

    def determine_origin(self, landing_row, landing_columnn):
            #Iterate over the board. 
            for k in range(8):
                for l in range(8):
                        
                #If field in new_board does not store same piece as prev_board and field 
                #is not the landing field, it should be the origin field
                    if(self.board[k][l] != self.new_board[k][l] and k!=landing_row and l!=landing_column):
                        origin_row = k
                        origin_column = l

            return origin_row, origin_column
