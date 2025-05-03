import sys
import pygame
import numpy as np


import enum


class Piece:
    def __init__(self, color, x, y, piece_type):
        self.color = color
        self.x = x
        self.y = y
        self.type = piece_type

        self.file_names = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.rank_names = ["8", "7", "6", "5", "4", "3", "2", "1"]
 
        self.chess_position = self.file_names[x] + self.rank_names[y]
        self.img = pygame.image.load(f"../resource/chess_piece_imgs/{self.color}{self.type}.png")

    def move_piece(self, _new_x, _new_y):
        #print(f'Moved piece from ({self.x},{self.y}) to ({_new_x},{_new_y})')
        self.x = _new_x
        self.y = _new_y
        self.chess_position = self.file_names[self.x] + self.rank_names[self.y]


    def draw(self, surface):
        # self.img = pygame.image.load(f"../resource/chess_piece_imgs/{self.color}{self.type}.png")
        surface.blit(self.img, (self.x*75+10, self.y*75+10))

    def print_info(self):
        print(f'\tPiece: {self.color}{self.type}\n\tPos: ({self.x},{self.y}) AKA {self.chess_position}')

#     def same_color(self, _piece1, _piece2):
#         if _piece1.color == _piece2.color:
#             return True
#         else:
#             return False


# class Player:
#     def __init__(self, _color):
#         self.color = _color
#         self.piece_positions = []


class Board:
    def __init__(self):
        # Define the names of the squares on the board:
        square_names = []
        self.file_names = enum.Enum('file_names', ["a", "b", "c", "d", "e", "f", "g", "h"])
        self.rank_names = enum.Enum('rank_names', ["1", "2", "3", "4", "5", "6", "7", "8"])
        for rank_name in self.rank_names:
            for file_name in self.file_names:
                square_names.append( file_name.name + rank_name.name)

        #self.board_square_names = np.array(square_names).reshape((8,8))
        
        self.board_square_names = square_names
        print(self.board_square_names)

        default_start_state_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

        # # Change this to FEN or PGN notation instead:
        # default_start_state = ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR',
        #                        'bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP', 'bP',
        #                        None, None, None, None, None, None, None, None,
        #                        None, None, None, None, None, None, None, None,
        #                        None, None, None, None, None, None, None, None,
        #                        None, None, None, None, None, None, None, None,
        #                        'wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP', 'wP',
        #                        'wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR'
        #                        ]

        # TODO: Add reading whose turn it is, castling rights, and ???
        self.game_state = {}
        self.ith_square = 0

        for ch in default_start_state_FEN:
            if ch == ' ':
                break   # done_reading_piece_positions_on_board = True
            elif ch == '/':
                continue
            elif ch.isdigit():
                for file_in_rank in range(int(ch)):
                    self.game_state[self.board_square_names[self.ith_square]] = {'piece': None, 'square_idx': self.ith_square}
                    self.ith_square = self.ith_square + 1
            else:
                if ch.islower():
                    piece_color = 'b'
                else:
                    piece_color = 'w'
                piece_type = ch.upper()
                # # print(type(self.board_square_names[self.ith_square]))
                # _game_state_key = 'a1' #self.board_square_names[self.ith_square]
                # piece_pos_square_name = self.square_name2xy_pos(_game_state_key)
                # self.game_state[self.board_square_names[self.ith_square]] = {'piece': Piece(piece_color, self.ith_square % 8, int(self.ith_square / 8), piece_type), 'square_idx': self.ith_square}
                self.game_state[self.board_square_names[self.ith_square]] = {'piece': ch, 'square_idx': self.ith_square} # ch
                self.ith_square = self.ith_square + 1
                # print(f'rank_idx: {rank_names[6]}')

    def square_name2xy_pos(self, _key):
        print(self.game_state[_key])
        square_idx_temp = self.game_state[_key]['square_idx']
        print(f'rank={int(square_idx_temp / 8)}, file={square_idx_temp % 8}')
        # print(f'rank={self.game_state[_key]['square_idx']}')
        return (int(square_idx_temp / 8), square_idx_temp % 8)

    def xy_pos2square_name(self, _x, _y):
        print(f'({_x},{_y}) square is named: {self.board_square_names[_x*8 + _y]}')
        # return self.file_names[_x].name + self.file_names[_y].name
        return self.board_square_names[_x*8 + _y]
    
    def get_game_state(self):
        return self.game_state
    
    # def move_piece(self):

    
#     def move_piece(self,old_x,old_y, new_x, new_y):
#         self.game_state[self.xy_pos2square_name(new_x,new_y)]['piece'] = self.game_state[self.xy_pos2square_name(old_x,old_y)]['piece']
#         self.game_state[self.xy_pos2square_name(old_x,old_y)]['piece'] = None




# class Game:
#     def __init__(self):
#         # player_white = Player('w')
#         # player_black = Player('b')

#         board_game = Board()

#         # # Test conversion between squares in algebraic notation and x,y coordinates:
#         # print(board.square_name2xy_pos('a1'))
#         # board.xy_pos2square_name(7,7)

#         pygame.init()

#         # set up the window
#         size = (640, 640)
#         screen = pygame.display.set_mode(size)
#         pygame.display.set_caption("Chess Game")

#         # set up the board
#         board = pygame.Surface((600, 600))
#         board.fill((255, 206, 158))

#         # draw the board
#         for x in range(0, 8, 2):
#             for y in range(0, 8, 2):
#                 pygame.draw.rect(board, (210, 180, 140), (x*75, y*75, 75, 75))
#                 pygame.draw.rect(board, (210, 180, 140), ((x+1)*75, (y+1)*75, 75, 75))
    
#         game_state = board_game.get_game_state()

#         # print(game_state)
#         for square_state in game_state.values():
#             print(square_state)
#             if square_state['piece'] is not None:
#                 print(square_state['piece'].type)
#                 square_state['piece'].draw(board)

#         # # set up the pieces
#         # piece_types_excl_pawns = ["R", "N", "B", "Q", "K", "B", "N", "R"]
#         # pieces = []
#         # for i in range(8):
#         #     pieces.append(Piece("b", i, 1, "p"))
#         #     pieces.append(Piece("w", i, 6, "p"))
#         #     pieces.append(Piece("b", i, 0, piece_types_excl_pawns[i]))
#         #     pieces.append(Piece("w", i, 7, piece_types_excl_pawns[i]))


#         # # draw the pieces
#         # for piece in pieces:
#         #     piece.draw(board)

#         # add the board to the screen
#         screen.blit(board, (20, 20))

#         pygame.display.flip()

#         move_piece_req_detected = False

#         # main loop
#         while True:
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     pygame.quit()
#                     sys.exit()

#                 if event.type == pygame.MOUSEBUTTONDOWN:
#                     # get the position of the click
#                     pos = pygame.mouse.get_pos()

#                     # convert the position to board coordinates
#                     x = (pos[0] - 20) // 75
#                     y = (pos[1] - 20) // 75

#                     print(f'Mouseclick detected at ({x},{y})')

#                     for square_state in game_state.values():
#                         if square_state['piece'] is not None:
#                             if square_state['piece'].x == x and square_state['piece'].y == y:
#                                 # move the piece
#                                 pos = pygame.mouse.get_pos()
#                                 x = (pos[0] - 20) // 75
#                                 y = (pos[1] - 20) // 75
#                                 # piece.x = x
#                                 # piece.y = y
#                                 # idx_piece_to_move = i
#                                 square_state['piece'].print_info()

#                                 self.color_of_moved_piece = square_state['piece'].color
#                                 self.piece_x_old = x
#                                 self.piece_y_old = y
#                                 #print(f'Move {pieces[i].color}{pieces[i].type} from ({pieces[i].x},{pieces[i].y})?')
#                                 move_piece_req_detected = True
#                                 break

#                         if move_piece_req_detected:
#                             if square_state['piece'] is None:
#                                 # move the piece
#                                 pos = pygame.mouse.get_pos()
#                                 x = (pos[0] - 20) // 75
#                                 y = (pos[1] - 20) // 75
#                                 square_state['piece'].move_piece(x,y)
#                                 # pieces[idx_piece_to_move].x = x
#                                 # pieces[idx_piece_to_move].y = y
#                                 # print(f'Moved {pieces[idx_piece_to_move].color}{pieces[idx_piece_to_move].type} to ({pieces[idx_piece_to_move].x},{pieces[idx_piece_to_move].y})')
#                             else:
#                                 square_state['piece'].print_info()
#                                 self.color_of_attacked_piece = square_state['piece'].color
#                                 if self.color_of_attacked_piece == self.color_of_moved_piece:
#                                     print('Illegal move!')
#                                 else:
#                                     square_state['piece'].move_piece(x,y)
#                             move_piece_req_detected = False
#                             break

                        

#             # redraw the board and pieces
#             board.fill((255, 206, 158))
#             for x in range(0, 8, 2):
#                 for y in range(0, 8, 2):
#                     pygame.draw.rect(board, (210, 180, 140), (x*75, y*75, 75, 75))
#                     pygame.draw.rect(board, (210, 180, 140), ((x+1)*75, (y+1)*75, 75, 75))

#             # for piece in pieces:
#             #     # piece.draw(board)
#             #     piece.draw(board)
#             game_state = board_game.get_game_state()
#             for square_state in game_state.values():
#                 if square_state['piece'] is not None:
#                     square_state['piece'].draw(board)

#             # # # add the board to the screen
#             screen.blit(board, (20, 20))

#             # update the display
#             pygame.display.update()


#             #     # if square['piece'] is not None:
#             #     #     print(square)
#             #     #     # piece = {square['piece']}


#             # self.board_square_names = np.ndarray(square_names)
#             # self.board_square_names.reshape(8,8)
#             # print(self.board_square_names)


def main():

    # game = Game()

    board_game = Board()

    pygame.init()

    # set up the window
    size = (640, 640)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Chess Game")

    # set up the board
    board = pygame.Surface((600, 600))
    board.fill((255, 206, 158))

    light_square_color = (210, 180, 140)

    # draw the board
    for x in range(0, 8, 2):
        for y in range(0, 8, 2):
            pygame.draw.rect(board, light_square_color, (x*75, y*75, 75, 75))
            pygame.draw.rect(board, (210, 180, 140), ((x+1)*75, (y+1)*75, 75, 75))
    

    # set up the pieces
    piece_types_excl_pawns = ["R", "N", "B", "Q", "K", "B", "N", "R"]
    pieces = []
    idx_pieces_in_play = []
    piece_positions_on_the_board = []
    for i in range(8):
        pieces.append(Piece("b", i, 1, "P"))
        pieces.append(Piece("w", i, 6, "P"))
        pieces.append(Piece("b", i, 0, piece_types_excl_pawns[i]))
        pieces.append(Piece("w", i, 7, piece_types_excl_pawns[i]))
    
    for i in range(32):
        idx_pieces_in_play.append(i)


    # draw the pieces
    for piece in pieces:
        # piece.print_info()
        piece.draw(board)

    # add the board to the screen
    screen.blit(board, (20, 20))

    pygame.display.flip()

    idx_piece_to_move = None
    move_is_illegal = False

    chess_move_string = ""

    # player = enum.Enum('player_to_move', ['WHITE','BLACK'])
    # player_to_move = player.WHITE
    player_to_move = "w"

    print('\nPlayer to move is: ' + player_to_move)

    # main loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                # get the position of the click
                pos = pygame.mouse.get_pos()

                # convert the position to board coordinates
                x = (pos[0] - 20) // 75
                y = (pos[1] - 20) // 75

                print(f'Mouseclick detected at ({x},{y})')

                # find the piece at the clicked position
                for i in idx_pieces_in_play: #range(len(pieces)):
                    if pieces[i].x == x and pieces[i].y == y and pieces[i].color == player_to_move:
                        # pos = pygame.mouse.get_pos()
                        # x = (pos[0] - 20) // 75
                        # y = (pos[1] - 20) // 75

                        # piece.x = x
                        # piece.y = y
                        idx_piece_to_move = i
                        # print(f'Move {pieces[i].color}{pieces[i].type} from ({pieces[i].x},{pieces[i].y})?')
                        pieces[i].print_info()
                        chess_move_string = pieces[idx_piece_to_move].chess_position
                        break

                    # Check if the requested move is legal
                    if idx_piece_to_move is not None: #and not move_is_illegal: # and pieces[i].color != player_to_move:
                        pos = pygame.mouse.get_pos()
                        x = (pos[0] - 20) // 75
                        y = (pos[1] - 20) // 75

                        # Can't move the piece to the position it is already in:
                        if pieces[idx_piece_to_move].x == x and pieces[idx_piece_to_move].y == y:
                            print('Illegal move requested! Make a different move!')
                            idx_piece_to_move = None
                            move_is_illegal = True
                            break
                        if pieces[i].x == x and pieces[i].y == y:
                            if pieces[i].color == player_to_move:
                                print('TRYING TO CAPTURE A PIECE OF THE SAME COLOR!!!')
                                idx_piece_to_move = None
                                move_is_illegal = True
                            else:
                                chess_move_string += "x"
                                idx_pieces_in_play.remove(i)
                                print(f'#Pieces on the board: {len(idx_pieces_in_play)}')
                                move_is_illegal = False
                            break
                        # else:
                        #     move_is_illegal = False

                    
                        
                        # if move_is_illegal:
                        #     move_is_illegal = False
                        #     idx_piece_to_move = None
                        #     break

                    if (idx_piece_to_move is not None) and (not move_is_illegal):
                        pieces[idx_piece_to_move].move_piece(x,y)
                        chess_move_string += pieces[idx_piece_to_move].chess_position
                        print(f'\n\tChess Move: {chess_move_string}\n')
                        chess_move_string = ""
                        # print(f'Moved {pieces[idx_piece_to_move].color}{pieces[idx_piece_to_move].type} to ({pieces[idx_piece_to_move].x},{pieces[idx_piece_to_move].y})')
                        idx_piece_to_move = None
                        move_is_illegal = False
                        
                        if player_to_move == "w":
                            player_to_move = "b"
                        else:
                            player_to_move = "w"

                        print('\nPlayer to move is: ' + player_to_move)

                        


                        # # pieces[idx_piece_to_move].x = x
                        # # pieces[idx_piece_to_move].y = y
                        # pieces[i].print_info()
                        # # print(pieces[i].x)
                        # # print(pieces[i].y)

                        # if pieces[i].x == x and pieces[i].y == y and not (pieces[i].color == player_to_move):
                        #     print('MOVING PIECE TO OCUPIED SQUARE!!!')
                        #     chess_move_string += "x"
                
                        # if (idx_piece_to_move is not None) and (not move_is_illegal):
                        #     pieces[idx_piece_to_move].move_piece(x,y)
                        #     chess_move_string += pieces[idx_piece_to_move].chess_position
                        #     print(f'\n\tChess Move: {chess_move_string}\n')
                        #     chess_move_string = ""
                        #     # print(f'Moved {pieces[idx_piece_to_move].color}{pieces[idx_piece_to_move].type} to ({pieces[idx_piece_to_move].x},{pieces[idx_piece_to_move].y})')
                        #     idx_piece_to_move = None
                        #     move_is_illegal = True
                            
                        #     if player_to_move == "w":
                        #         player_to_move = "b"
                        #     else:
                        #         player_to_move = "w"

                        #     print('\nPlayer to move is: ' + player_to_move)

                        #break


        # redraw the board and pieces
        board.fill((255, 206, 158))
        for x in range(0, 8, 2):
            for y in range(0, 8, 2):
                pygame.draw.rect(board, (210, 180, 140), (x*75, y*75, 75, 75))
                pygame.draw.rect(board, (210, 180, 140), ((x+1)*75, (y+1)*75, 75, 75))

        for i in idx_pieces_in_play:
            # piece.draw(board)
            pieces[i].draw(board)

        # # # add the board to the screen
        screen.blit(board, (20, 20))

        # update the display
        pygame.display.update()


if __name__ == '__main__':
    main()