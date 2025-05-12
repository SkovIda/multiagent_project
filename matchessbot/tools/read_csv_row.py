#!/usr/bin/env python3

import csv

import argparse

from chess_notation_converter import convert_san_to_uci

import re

COLUMN_IDX_PUZZLE_ID = 0
COLUMN_IDX_PUZZLE_ELO_RATING = 1    # TODO: Verify this!
COLUMN_IDX_PUZZLE_HIST_SAN = 2
COLUMN_IDX_PUZZLE_SOLUTION_SAN = 3
COLUMN_IDX_FEN = 4
COLUMN_IDX_PUZZLE_SOLUTION_UCI = 5


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Print a single row of a csv file from row number')
    
    parser.add_argument('--csv_file', type=str, default="",
                        help='absolute path to the csv file')
    parser.add_argument('--row_number', type=int, default=0,
                        help='the row number of the csv file to return')
    parser.add_argument('--result', type=str, choices=['move_hist', 'puzzle_color', 'player_color'], required=True)
    
    args = parser.parse_args()
    return args



def get_solution_length(solution_string: str):
    return len(solution_string.split(" "))

def get_solution(solution_string: str):
    return solution_string.split(" ")


# def puzzle_solver_color(uci_move_hist, solution_length):
#     if (len(uci_move_hist) - solution_length) % 2 == 0:
#         return 'white'
#     else:
#         return 'black'

# def puzzle_plays_as_color(uci_move_hist, solution_length):
#     if (len(uci_move_hist) - solution_length) % 2 != 0:
#         return 'white'
#     else:
#         return 'black'
    
def puzzle_solver_color(uci_move_setup):
    if len(uci_move_setup) % 2 == 0:
        return 'white'
    else:
        return 'black'

def puzzle_plays_as_color(uci_move_setup):
    if len(uci_move_setup) % 2 != 0:
        return 'white'
    else:
        return 'black'


if __name__=='__main__':
    args = _parse_args()

    filename = str(args.csv_file)
    # print(filename)

    with open(filename) as f:
        csv_reader = csv.reader(f)
        rows = list(csv_reader)
    
    if args.row_number <= len(rows):
        # print(rows[int(args.row_number)])
        row = rows[args.row_number]
        san_str = row[COLUMN_IDX_PUZZLE_HIST_SAN]
        # print(san_str)
        uci_move_hist = convert_san_to_uci(san_str)
        
        solution_str = row[COLUMN_IDX_PUZZLE_SOLUTION_UCI]
        solution_uci = get_solution(solution_str)

        uci_move_hist.append(solution_uci[0])

        solution_length = len(solution_str) - 1 #get_solution_length(solution_str)

        if args.result == 'move_hist':
            print(uci_move_hist)
        elif args.result == 'puzzle_color':
            print(puzzle_plays_as_color(uci_move_hist))
        elif args.result == 'player_color':
            print(puzzle_solver_color(uci_move_hist))

        # print(f"{puzzle_plays_as_color(uci_move_hist, solution_str)} {puzzle_solver_color(uci_move_hist, solution_str)} {uci_move_hist}")
        
    # try:
    #     print(rows[int(args.row_number)])
    # except Exception as err:
    #     print(f"{err.args}")
