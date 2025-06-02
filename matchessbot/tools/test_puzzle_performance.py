#!/usr/bin/env python3

import csv
import json

import time

from tqdm import tqdm

import subprocess


from chess_notation_converter import convert_san_to_uci
from read_csv_row import puzzle_solver_color, get_solution, COLUMN_IDX_PUZZLE_ID, COLUMN_IDX_PUZZLE_ELO_RATING, COLUMN_IDX_PUZZLE_HIST_SAN, COLUMN_IDX_PUZZLE_SOLUTION_SAN, COLUMN_IDX_FEN, COLUMN_IDX_PUZZLE_SOLUTION_UCI




if __name__ == '__main__':
    
    puzzle_dataset_dir = "./src/matchessbot/tools/"
    # puzzle_dataset_filenames = ["puzzles_checkmate_in_1_elo_range_0_400", "puzzles_checkmate_in_2_elo_range_0_400", "puzzles_checkmate_in_1_elo_range_401_500", "puzzles_checkmate_in_2_elo_range_401_500"] #["puzzles_checkmate_in_1_elo_range_0_500", "puzzles_checkmate_in_2_elo_range_0_500", "puzzles_checkmate_in_1_elo_range_501_1000", "puzzles_checkmate_in_2_elo_range_501_1000"]
    # puzzle_dataset_checkmate_attribute = [1, 2, 1, 2]

    # puzzle_dataset_filenames = ["puzzles_checkmate_in_2_elo_range_501_600", "puzzles_checkmate_in_2_elo_range_601_700"]
    # puzzle_dataset_checkmate_attribute = [2, 2]


    # puzzle_dataset_filenames = ["puzzles_checkmate_in_2_elo_range_501_600", "puzzles_checkmate_in_2_elo_range_601_700"]
    # puzzle_dataset_checkmate_attribute = [2, 2]

    # puzzle_dataset_filenames = ["puzzles_checkmate_in_2_elo_range_601_700"]
    # puzzle_dataset_checkmate_attribute = [2]
    
    # # matchess_puzzle_performance_results = {
    # #     'puzzle_elo': [],
    # #     'puzzle_solution_length': [],
    # # }
    # results_filename = "puzzle_test_results.json"
    min_elo_ranges = [1501, 1551, 1601, 1651, 1701, 1801, 1901, 2001]
    max_elo_ranges = [1550, 1600, 1650, 1700, 1800, 1900, 2000, 3000]

    puzzle_dataset_filenames = []

    for min_elo, max_elo in zip(min_elo_ranges, max_elo_ranges):
        puzzle_dataset_filenames.append("puzzles_checkmate_in_2_elo_range_" + str(min_elo) + "_" + str(max_elo))

    print(puzzle_dataset_filenames)

    last_completed_test_dataset_idx = 5

    model_checkpoint_filename = "checkpoint_epoch_75_MATChessFormer-ImitativeRL-20.pt"
    
    model_name =  "IRL"
    print(f"\nModel:\t{model_name}\ncheckpoint:\t{model_checkpoint_filename}")

    unix_timestamps = []

    # Run "puzzle games" tests:
    for dataset_idx, puzzle_dataset_filename in enumerate(puzzle_dataset_filenames):
        if last_completed_test_dataset_idx is not None:
            if dataset_idx <= last_completed_test_dataset_idx:
                continue

        with open(puzzle_dataset_dir + puzzle_dataset_filename + ".csv") as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        unix_timestamp = int(time.time())
        unix_timestamps.append(str(unix_timestamp))
        test_result_dir_name =  "result_" + model_name + "_" + puzzle_dataset_filename + "_" + str(unix_timestamp)

        for puzzle_idx, puzzle_data in tqdm(enumerate(rows)):
            if puzzle_idx == 0:
                continue


            completed_process = subprocess.run(["./src/matchessbot/scripts/run_matchess_puzzles.sh" , str(puzzle_idx), puzzle_dataset_filename + ".csv", test_result_dir_name])

            print("\n\n")
            print(completed_process)
            print("\n\n")
        
    print("\n\nDONE TESTING:")
    print(f'puzle dataset filenames list:\t{puzzle_dataset_filenames}')
    print(f'puzzle result timestamps list:\t{unix_timestamps}')