#!/usr/bin/env python3

import argparse
import csv

import re

from tqdm import tqdm

from chess_notation_converter import convert_san_to_uci
from read_csv_row import get_solution, get_solution_length, COLUMN_IDX_PUZZLE_ID, COLUMN_IDX_PUZZLE_ELO_RATING, COLUMN_IDX_PUZZLE_HIST_SAN, COLUMN_IDX_PUZZLE_SOLUTION_SAN, COLUMN_IDX_FEN, COLUMN_IDX_PUZZLE_SOLUTION_UCI

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Create a puzzle dataset that only consists of puzzles that are categorized as: checkmate in 1')
    
    parser.add_argument('--csv_file', type=str, default="",
                        help='absolute path to the csv file')
    # parser.add_argument('--row_number', type=int, default=0,
    #                     help='the row number of the csv file to return')
    # parser.add_argument('--result', type=str, choices=['move_hist', 'puzzle_color', 'player_color', 'puzzle_id'], required=True)
    parser.add_argument('--new_puzzle_dataset_path', type=str, default="",help='absolute path and filename for saving the the filtered puzzle dataset')
    parser.add_argument('--checkmate', type=int, default=1, help='how many moves before the puzzle is solved with checkmate')
    parser.add_argument('--max_elo', type=int, default=1000, help='if the puzzles should be filtered such that only games with an ELO rating lower or equal to <max_elo> is included in the filtered dataset')
    parser.add_argument('--min_elo', type=int, default=0, help='if the puzzles should be filtered such that only games with an ELO rating higher than <max_elo> is included in the filtered dataset')
    parser.add_argument('--only_sacrifices', type=bool, default=False, help='Set to true if the newly generated puzzle dataset should only contain puzzles where one piece sacrifices itself for the team to win')
    # parser.add_argument('-puzzle_length', type=int, default=10, help='The desired length of the soluiton to the puzzle (i.e. number of player moves in solution)')
    
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = _parse_args()

    filename = str(args.csv_file)
    # print(filename)
    fieldnames = []

    with open(filename) as f:
        csv_reader = csv.reader(f)
        rows = list(csv_reader)

        fieldnames = rows[0]
        print(f'\nrows[0]={rows[0]}')

    # new_dataset_filename = "puzzles_checkmate_in_" + str(args.checkmate) + "_max_elo_" + str(args.max_elo) + ".csv"
    new_dataset_filename = "puzzles_checkmate_in_" + str(args.checkmate) + "_elo_range_" + str(args.min_elo) + "_" + str(args.max_elo) + ".csv"


    with open(new_dataset_filename, 'w') as out_file:
        csv_writer = csv.writer(out_file)
        csv_writer.writerow(fieldnames)

    puzzle_count = 0
    filtered_puzzle_count = 0

    
    # for idx, puzzle in tqdm(enumerate(rows),desc=f'filtering {len(rows) - 1} puzzles'):
    for idx, puzzle in tqdm(enumerate(rows)):
        if idx == 0:
            continue

        puzzle_count += 1
        
        puzzle_solution_is_checkmate = False
        san_str_solution = puzzle[COLUMN_IDX_PUZZLE_SOLUTION_SAN]

        if re.search('#', san_str_solution) is not None:
            puzzle_solution_is_checkmate = True
            # print(f'PUZZLE IS CHECKMATE: {san_str_solution}')

        if puzzle_solution_is_checkmate:
            solution_str = puzzle[COLUMN_IDX_PUZZLE_SOLUTION_UCI]
            solution_uci = get_solution(solution_str)

            # print(f'len(solution)={(len(solution_uci)) // 2}')

            if args.checkmate == ((len(solution_uci)) // 2):
                
                if int(puzzle[COLUMN_IDX_PUZZLE_ELO_RATING]) <= args.max_elo and int(puzzle[COLUMN_IDX_PUZZLE_ELO_RATING]) > args.min_elo:
                    # print(f'\tELO rating:\t{int(puzzle[COLUMN_IDX_PUZZLE_ELO_RATING])}')

                    filtered_puzzle_count += 1

                    with open(new_dataset_filename, 'a') as out_file:
                        csv_writer = csv.writer(out_file)
                        csv_writer.writerow(puzzle)

    print(f'generated new puzzle dataset: {new_dataset_filename}\nPuzzle dataset contains {filtered_puzzle_count} puzzles (out of {puzzle_count} total) with {args.checkmate} moves to checkmate and a max ELO rating of {args.max_elo} (min ELO rating of {args.min_elo})')