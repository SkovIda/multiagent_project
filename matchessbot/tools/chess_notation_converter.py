#!/usr/bin/env python3

import argparse
import re
import chess


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Convert chess move notation from SAN to UCI')
    
    parser.add_argument('--from_san', type=bool, default=True,
                        help='convert the input string from SAN to UCI')
    parser.add_argument('--san_str', type=str, default="",
                        help='the SAN formatted string to convert to a list of UCI moves instead')
    
    args = parser.parse_args()
    return args


def convert_san_to_uci(san_str: str):
    board = chess.Board()
    board.reset()

    san_move_str = san_str.strip()
    
    # san_move_str = re.sub(" +\d.", " ", san_move_str) 

    # Prepend a white-space character to the input san string:
    san_move_str = " " + san_move_str # Required before using the substitution pattern below.
    san_move_str = re.sub("\s\d+.\s", " ", san_move_str) # Remove all the fullmove count numbers from the string.

    san_move_str = san_move_str.strip()

    san_move_list = san_move_str.split(" ")

    # print(f'SAN moves: {san_move_list}')

    uci_move_list = []
    for san_move in san_move_list:
        chess_move = board.push_san(san_move)
        uci_move = chess_move.uci()
        uci_move_list.append(uci_move)
    
    # print(uci_move_list)
    return uci_move_list


if __name__=='__main__':
    args = _parse_args()

    if args.from_san is True:
        uci_list = convert_san_to_uci(args.san_str)
        print(uci_list)
        # board = chess.Board()
        # board.reset()

        # san_move_str = args.san_str.strip()
        # san_move_str = re.sub("\d+. ", "", san_move_str)
        # san_move_str = san_move_str.strip()

        # san_move_list = san_move_str.split(" ")

        # print(f'SAN moves: {san_move_list}')

        # uci_move_list = []
        # for san_move in san_move_list:
        #     chess_move = board.push_san(san_move)
        #     uci_move = chess_move.uci()
        #     uci_move_list.append(uci_move)
        
        # print(uci_move_list)