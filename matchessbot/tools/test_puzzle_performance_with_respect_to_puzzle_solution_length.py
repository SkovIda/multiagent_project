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

    # puzzle_dataset_filenames = ["puzzles_checkmate_in_1_elo_range_1001_3000", "puzzles_checkmate_in_2_elo_range_1001_3000", "puzzles_checkmate_in_3_elo_range_1001_3000", "puzzles_checkmate_in_4_elo_range_1001_3000"]
    puzzle_dataset_filenames = ["puzzles_checkmate_in_2_elo_range_1001_1050"] #["puzzles_checkmate_in_4_elo_range_1001_3000"]

    last_completed_test_dataset_idx = None

    model_checkpoint_filename = "checkpoint_epoch_75_MATChessFormer-ImitativeRL-20.pt"
    
    model_name_IRL =  "IRL"
    print(f"\nModel:\t{model_name_IRL}\ncheckpoint:\t{model_checkpoint_filename}")

    model_name_SIL =  "SIL"
    print(f"\nModel:\t{model_name_SIL}\ncheckpoint:\t{model_checkpoint_filename}")

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
        test_result_dir_name_IRL =  "result_" + model_name_IRL + "_" + puzzle_dataset_filename + "_" + str(unix_timestamp)
        test_result_dir_name_SIL =  "result_" + model_name_SIL + "_" + puzzle_dataset_filename + "_" + str(unix_timestamp)

        for puzzle_idx, puzzle_data in tqdm(enumerate(rows)):
            if puzzle_idx == 0:
                continue
            if puzzle_idx < 30:
                continue

            completed_process = subprocess.run(["./src/matchessbot/scripts/run_matchess_puzzles.sh" , str(puzzle_idx), puzzle_dataset_filename + ".csv", test_result_dir_name_IRL])

            print("\n\n")
            print(completed_process)
            print("\n\n")

            time.sleep(1)


            completed_process = subprocess.run(["./src/matchessbot/scripts/run_matchess_puzzles_sil_model.sh" , str(puzzle_idx), puzzle_dataset_filename + ".csv", test_result_dir_name_SIL])

            print("\n\n")
            print(completed_process)
            print("\n\n")
        
    print("\n\nDONE TESTING:")
    print(f'puzle dataset filenames list:\t{puzzle_dataset_filenames}')
    print(f'puzzle result timestamps list:\t{unix_timestamps}')