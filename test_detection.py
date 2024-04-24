import unittest
import numpy as np
import copy

from chessgame import Game


class TestDetection(unittest.TestCase):


    def test_get_field_string(self):
        
        game = Game()

        columns = ["a", "b", "c", "d", "e", "f", "g", "h"]
        expected_field_strings = [column+str(i+1) for i in range(8) for column in columns]
        
        res_field_strings = [game.get_field_string(i,j) for i in range(8) for j in range(8)]

        self.assertListEqual(res_field_strings, expected_field_strings)

    def test_get_path_to_king(self):

        game = Game()

        # Black bishop or queen on a4 giving check to white king on d1 

        game.white_kings_row_index = 0
        game.white_kings_column_index = 3

        row_index = 3
        column_index = 0

        color = "b"

        expected_fields = ["b3", "c2"]

        res_fields = game.get_path_to_king(row_index, column_index, color)

        self.assertListEqual(res_fields, expected_fields)

        # White rook or queen on g7 giving check to black king on b7

        game.black_kings_row_index = 6
        game.black_kings_column_index = 1

        row_index = 6
        column_index = 6

        color = "w"

        expected_fields = ["f7", "e7", "d7", "c7"]

        res_fields = game.get_path_to_king(row_index, column_index, color)

        self.assertListEqual(res_fields, expected_fields)

        # Black bishop or queen on h1 giving check to black king on a8

        game.white_kings_row_index = 7
        game.white_kings_column_index = 0

        row_index = 0
        column_index = 7

        color = "b"

        expected_fields = ["g2", "f3", "e4", "d5", "c6", "b7"]

        res_fields = game.get_path_to_king(row_index, column_index, color)

        self.assertListEqual(res_fields, expected_fields)

    def test_determine_board_position(self):
        
        game = Game()
        
        x1 = 150
        y1 = 150
        res_x1, res_y1 = game.determine_board_position(x1, y1)
        self.assertListEqual([res_x1, res_y1], [0, 0])
        
        x2 = 1200
        y2 = 740
        res_x2, res_y2 = game.determine_board_position(x2, y2)     
        self.assertListEqual([res_x2, res_y2], [5, 4])
        
        x3 = 790
        y3 = 1210
        res_x3, res_y3 = game.determine_board_position(x3, y3)
        self.assertListEqual([res_x3, res_y3], [3, 7])

    def test_get_all_reachable_fields(self):

        game = Game()
        
        game.white_kings_field = "d2"
        game.white_kings_row_index = 1
        game.white_kings_column_index = 3

        game.black_kings_field = "c7"
        game.black_kings_row_index = 6
        game.black_kings_column_index = 2

        game.new_board[0][0] = 5
        game.new_board[1][1] = 3
        game.new_board[1][3] = 1
        game.new_board[1][5] = 4
        game.new_board[2][1] = 5
        game.new_board[2][3] = 6
        game.new_board[2][4] = 6
        game.new_board[2][5] = 2
        game.new_board[2][6] = 3
        game.new_board[3][2] = 10
        game.new_board[4][1] = 12
        game.new_board[4][2] = 12
        game.new_board[4][4] = 12
        game.new_board[4][5] = 6
        game.new_board[5][0] = 9
        game.new_board[5][3] = 8
        game.new_board[5][5] = 4
        game.new_board[6][2] = 7
        game.new_board[6][5] = 10
        game.new_board[6][7] = 9
        game.new_board[7][6] = 11

        game.board = copy.deepcopy(game.new_board)
        game.board[6][4] = 12
        game.board[4][4] = 0

        expected_fields_white = ["a2","a3", "a4", "a5", "a6", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "c3", "d4", "e5", "e6", "c2", "e2", "e4", "g4", "h3", "b4", "b5", "c4", "g2", "d5", "c6", "b7", "a8", "f4", "h5", "h2", "h4", "h7", "g8", "e8", "d7", "d5"]
        expected_fields_black = ["h8", "f8", "e8", "d8", "c8", "b8", "a8", "g7", "g6", "g5", "g4", "g3", "f5", "h6", "d7", "b7", "b6", "c6", "e7", "e6", "f6", "d5", "d5", "d4", "d3", "e4", "b4", "a5", "a3", "b2", "d2", "e3"]

        expected_prot_white = ["a1", "e3", "d3", "b2", "f2", "f5", "g3"]
        expected_prot_black = ["g8", "c7", "d6", "a6", "b5", "c5", "e5", "c4"]

        expected_giving_check = ["c4"]

        res_fields_white, res_fields_black, res_prot_white, res_prot_black, res_giving_check, res_path_to_king = game.get_all_reachable_fields()


        for field in expected_fields_white:
            self.assertIn(field, res_fields_white)

        for field in expected_fields_black:
            self.assertIn(field, res_fields_black)
        
        for field in res_fields_white:
            self.assertIn(field, expected_fields_white)
        
        for field in res_fields_black:
            self.assertIn(field, expected_fields_black)

        for field in expected_prot_white:
            self.assertIn(field, res_prot_white)

        for field in res_prot_white:
            self.assertIn(field, expected_prot_white)
        
        for field in expected_prot_black:
            self.assertIn(field, res_prot_black)

        for field in res_prot_black:
            self.assertIn(field, expected_prot_black)

        self.assertEqual(res_giving_check, expected_giving_check)
    
    
    def test_document_move(self):
        
        game = Game()

        moved_piece1 = "Bishop_w"
        origin_column1 = 0
        origin_row1 = 0
        landing_column1 = 5
        landing_row1 = 5
        capture1 = 0
        
        expected_notation1 = "Ba1-f6"
        
        res_notation1 = game.document_move(moved_piece1, origin_row1, origin_column1, landing_row1, landing_column1, capture1, False)
        self.assertEqual(res_notation1, expected_notation1)


        moved_piece2 = "N_Knight_b"
        origin_column2 = 6
        origin_row2 = 4
        landing_column2 = 7
        landing_row2 = 6
        capture2 = 1

        expected_notation2 = "Ng5xh7"
        
        res_notation2 = game.document_move(moved_piece2, origin_row2, origin_column2, landing_row2, landing_column2, capture2, False)
        self.assertEqual(res_notation2, expected_notation2)

    def test_document_en_passant(self):

        game = Game()

        origin_column1 = 2
        origin_row1 = 4
        landing_column1 = 1
        landing_row1 = 5

        expected_notation1 = "c5xb6 e.p."

        res_notation1 = game.document_en_passant(origin_row1, origin_column1, landing_row1, landing_column1, False)
        self.assertEqual(res_notation1, expected_notation1)

    def test_document_pawn_promotion(self):

        game = Game()

        origin_row1 = 6
        origin_column1 = 1
        landing_row1 = 7
        landing_column1 = 1
        moved_piece1 = "Queen_w"

        expected_notation1 = "b7-b8Q"

        res_notation1 = game.document_pawn_promotion(moved_piece1, origin_row1, origin_column1, landing_row1, landing_column1, False)

        self.assertEqual(res_notation1, expected_notation1)
        
        origin_row2 = 6
        origin_column2 = 6
        landing_row2 = 7
        landing_column2 = 7
        moved_piece2 = "Rook_w"

        expected_notation2 = "g7xh8R"

        res_notation2 = game.document_pawn_promotion(moved_piece2, origin_row2, origin_column2, landing_row2, landing_column2, False)

        self.assertEqual(res_notation2, expected_notation2)

    def test_document_castle(self):

        game = Game()

        color = "w"

        game.board = np.zeros((8,8), np.int16)

        game.new_board[0][5] = 5
        game.board[0][5] = 0
        
        expected_notation1 = "0-0"

        res_notation1 = game.document_castle(color, False)

        self.assertEqual(res_notation1, expected_notation1)


        color = "w"

        game.board = np.zeros((8,8), np.int16)

        game.new_board[0][5] = 5
        game.board[0][5] = 5
        
        check_given = True

        expected_notation2 = "0-0-0#"

        res_notation2 = game.document_castle(color, check_given)

        self.assertEqual(res_notation2, expected_notation2)

    def test_reachable_by_pawn(self):

        game = Game()

        color1 = "w"
        row_index1 = 4
        column_index1 = 3
        game.board = np.zeros((8,8), np.int16)
        game.new_board[4][2] = 12
        game.new_board[5][3] = 0
        game.new_board[5][4] = 10

        expected_fields1 = ["d6", "e6", "c6"]
        expected_prot = []

        res_fields1, res_prot1 = game.reachable_by_pawn(color1, row_index1, column_index1)

        self.assertListEqual(res_fields1, expected_fields1)
        self.assertListEqual(res_prot1, expected_prot)
        
        color2 = "w"
        row_index2 = 6
        column_index2 = 7
        game.board = np.zeros((8,8), np.int16)
        game.new_board = np.zeros((8,8), np.int16)
        game.new_board[7][7] = 11
        game.new_board[7][6] = 9 

        expected_fields2 = ["g8"]
        expected_prot2 = []

        res_fields2, res_prot2 = game.reachable_by_pawn(color2, row_index2, column_index2)

        self.assertListEqual(res_fields2, expected_fields2)
        self.assertListEqual(res_prot2, expected_prot2)
        
        color3 = "b"
        row_index3 = 5
        column_index3 = 0
        game.board = np.zeros((8,8), np.int16)
        game.new_board = np.zeros((8,8), np.int16)
        game.new_board[4][0] = 0
        game.new_board[4][1] = 5 
        

        expected_fields3 = ["a5", "b5"]
        expected_prot3 = []

        res_fields3, res_prot3 = game.reachable_by_pawn(color3, row_index3, column_index3)

        self.assertListEqual(res_fields3, expected_fields3)
        self.assertListEqual(res_prot3, expected_prot3)
        
        color4 = "b"
        row_index4 = 3
        column_index4 = 5
        game.board = np.zeros((8,8), np.int16)
        game.board[3][6] = 0
        game.new_board = np.zeros((8,8), np.int16)
        game.new_board[3][6] = 6
        game.new_board[2][4] = 9
        game.new_board[2][5] = 9 

        expected_fields4 = ["g3"]
        expected_prot4 = ["e3"]

        res_fields4, res_prot4 = game.reachable_by_pawn(color4, row_index4, column_index4)

        self.assertListEqual(res_fields4, expected_fields4)
        self.assertListEqual(res_prot4, expected_prot4)
    
    
    def test_reachable_by_king(self):
        

        game = Game()

        # White king on field c3 
        color1 = "w"
        row_index1 = 2
        column_index1 = 2

        game.new_board[row_index1-1][column_index1-1] = 4
        game.new_board[row_index1-1][column_index1] = 11
        game.new_board[row_index1-1][column_index1+1] = 0
        game.new_board[row_index1][column_index1-1] = 9
        game.new_board[row_index1][column_index1+1] = 10
        game.new_board[row_index1+1][column_index1-1] = 3
        game.new_board[row_index1+1][column_index1] = 0
        game.new_board[row_index1+1][column_index1+1] = 8

        expected_fields1 = ["c2", "d2", "b3", "d3", "c4", "d4"]
        expected_prot1 = ["b2", "b4"]

        res_fields1, res_prot1 = game.reachable_by_king(color1, row_index1, column_index1)

        self.assertListEqual(res_fields1, expected_fields1)
        self.assertListEqual(res_prot1, expected_prot1)
        
        # Black king on field a6 
        color2 = "b"
        row_index2 = 5
        column_index2 = 0
        game.new_board = np.zeros((8,8), np.int16)
        game.new_board[row_index2-1][column_index2-1] = 4
        game.new_board[row_index2-1][column_index2] = 6
        game.new_board[row_index2-1][column_index2+1] = 3
        game.new_board[row_index2][column_index2-1] = 2
        game.new_board[row_index2][column_index2+1] = 10
        game.new_board[row_index2+1][column_index2-1] = 6
        game.new_board[row_index2+1][column_index2] = 0
        game.new_board[row_index2+1][column_index2+1] = 5

        expected_fields2 = ["a5", "b5", "a7", "b7"]
        expected_prot2 = ["b6"]

        res_fields2, res_prot2 = game.reachable_by_king(color2, row_index2, column_index2)

        self.assertListEqual(res_fields2, expected_fields2)
        self.assertListEqual(res_prot2, expected_prot2)
    
    
    def test_reachable_by_bishop_or_diagonal(self):

        game = Game()

        # White bishop on c4
        color1 = "w"
        row_index1 = 3
        column_index1 = 2
        game.new_board[1][4] = 5
        game.new_board[7][6] = 11
        game.new_board[1][0] = 2

        expected_fields1 = ["b3", "d3", "b5", "a6", "d5", "e6", "f7", "g8"]
        expected_prot1 = ["a2", "e2"]

        res_fields1, res_prot1 = game.reachable_by_bishop_or_diagonal(color1, row_index1, column_index1)

        self.assertListEqual(res_fields1, expected_fields1)
        self.assertListEqual(res_prot1, expected_prot1)

        # Black bishop on g7
        color2 = "b"
        row_index2 = 6
        column_index2 = 6
        game.new_board = np.zeros((8,8), np.int16)
        game.new_board[1][1] = 5
        game.new_board[7][5] = 11
        game.new_board[5][7] = 10

        expected_fields2 = ["f6", "e5", "d4", "c3", "b2", "h8"]
        expected_prot2 = ["h6", "f8"]

        res_fields2, res_prot2 = game.reachable_by_bishop_or_diagonal(color2, row_index2, column_index2)

        self.assertListEqual(res_fields2, expected_fields2)
        self.assertListEqual(res_prot2, expected_prot2)

    
    def test_reachable_by_rook_or_straight(self):

        game = Game()

        # White rook on a1

        color = "w"
        row_index = 0
        column_index = 0
        game.new_board[6][0] = 12
        game.new_board[0][5] = 3

        expected_fields = ["a2", "a3", "a4", "a5", "a6", "a7", "b1", "c1", "d1", "e1"]
        expected_prot = ["f1"]

        res_fields, res_prot = game.reachable_by_rook_or_straight(color, row_index, column_index)

        self.assertListEqual(res_fields, expected_fields)
        self.assertListEqual(res_prot, expected_prot)
        
        # Black rook on d4

        color = "w"
        row_index = 3
        column_index = 3
        game.new_board = np.zeros((8,8), np.int16)
        game.new_board[0][3] = 5
        game.new_board[3][6] = 3

        expected_fields = ["d3", "d2", "c4", "b4", "a4", "d5", "d6", "d7", "d8", "e4", "f4"]
        expected_prot = ["d1", "g4"]

        res_fields, res_prot = game.reachable_by_rook_or_straight(color, row_index, column_index)

        self.assertListEqual(res_fields, expected_fields)
        self.assertListEqual(res_prot, expected_prot)
    

    def test_reachable_by_queen(self):

        game = Game()

        # White queen on b2

        color1 = "w"
        row_index1 = 1 
        column_index1 = 1
        game.new_board = np.zeros((8,8), np.int16)
        game.new_board[1][0] = 4
        game.new_board[2][0] = 9
        game.new_board[6][1] = 12
        game.new_board[5][5] = 5
        game.new_board[0][2] = 1

        expected_fields1 = ["a1", "a3", "c3", "d4", "e5", "b1", "b3", "b4", "b5", "b6", "b7", "c2", "d2", "e2", "f2", "g2", "h2"]
        expected_prot1 = ["c1", "f6", "a2"]

        res_fields1, res_prot1 = game.reachable_by_queen(color1, row_index1, column_index1)

        self.assertListEqual(res_fields1, expected_fields1)
        self.assertListEqual(res_prot1, expected_prot1)

    def test_reachable_by_knight(self):

        game = Game()

        # White knight on a8

        color1 = "w"
        row_index1 = 7
        column_index1 = 0
        game.new_board[5][1] = 9
        game.new_board[6][2] = 2

        expected_fields1 = ["b6"]
        expected_prot1 = ["c7"]

        res_fields1, res_prot1 = game.reachable_by_knight(color1, row_index1, column_index1)

        self.assertListEqual(res_fields1, expected_fields1)
        self.assertListEqual(res_prot1, expected_prot1)
        
        # Black knight on g2

        color2 = "b"
        row_index2 = 1
        column_index2 = 6
        game.new_board = np.zeros((8,8), np.int16)
        game.new_board[0][4] = 8
        game.new_board[2][4] = 4
        game.new_board[3][5] = 6
        game.new_board[3][7] = 3

        expected_fields2 = ["e3", "f4", "h4"]
        expected_prot2 = ["e1"]

        res_fields2, res_prot2 = game.reachable_by_knight(color2, row_index2, column_index2)

        self.assertListEqual(res_fields2, expected_fields2)
        self.assertListEqual(res_prot2, expected_prot2)

        # White knight on e5

        color3 = "w"
        row_index3 = 4
        column_index3 = 4
        game.new_board = np.zeros((8,8), np.int16)
        game.new_board[2][3] = 9
        game.new_board[3][2] = 5 
        game.new_board[2][5] = 0
        game.new_board[5][2] = 11
        game.new_board[6][3] = 0
        game.new_board[3][6] = 6
        game.new_board[6][5] = 2
        game.new_board[5][6] = 12

        expected_fields3 = ["d3", "f3", "c6", "d7", "g6"]
        expected_prot3 = ["c4", "g4", "f7"]

        res_fields3, res_prot3 = game.reachable_by_knight(color3, row_index3, column_index3)

        self.assertListEqual(res_fields3, expected_fields3)
        self.assertListEqual(res_prot3, expected_prot3)



if __name__ == '__main__':
    unittest.main()
