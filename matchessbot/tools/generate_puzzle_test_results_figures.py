#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt

import numpy as np

import copy

import csv
import json

import re

import time

from tqdm import tqdm

# import subprocess

from chess_notation_converter import convert_san_to_uci
from read_csv_row import puzzle_solver_color, get_solution, COLUMN_IDX_PUZZLE_ID, COLUMN_IDX_PUZZLE_ELO_RATING, COLUMN_IDX_PUZZLE_HIST_SAN, COLUMN_IDX_PUZZLE_SOLUTION_SAN, COLUMN_IDX_FEN, COLUMN_IDX_PUZZLE_SOLUTION_UCI



def get_puzzle_results_from_csv(results_dir, filename):
    filepath = os.path.join(results_dir, filename)
    
    row_count = 0
    elo = []
    checkmate_in = []
    data_SIL = []
    data_IRL = []
    correct_vote_percentage_sil = []
    correct_vote_percentage_irl = []
    same_firstmove_vote_percentage = []
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
        for row in csvreader:
            if row_count == 0:
                headers_file = row
                # print(headers_file)
            else:
                elo.append(int(row[0]))
                checkmate_in.append(int(row[1]))
                data_SIL.append(int(row[2]))
                data_IRL.append(int(row[3]))
                correct_vote_percentage_sil.append(float(row[4]))
                correct_vote_percentage_irl.append(float(row[5]))
                same_firstmove_vote_percentage.append(float(row[6]))
            row_count += 1
    return elo, checkmate_in, data_SIL, data_IRL, correct_vote_percentage_sil, correct_vote_percentage_irl, same_firstmove_vote_percentage


if __name__ == '__main__':
    
    puzzle_dataset_dir = "./src/matchessbot/tools/"
    # puzzle_dataset_filenames = ["puzzles_checkmate_in_2_elo_range_0_400", "puzzles_checkmate_in_2_elo_range_401_500", "puzzles_checkmate_in_2_elo_range_501_600", "puzzles_checkmate_in_2_elo_range_601_700"] #, "puzzles_checkmate_in_2_elo_range_401_500"] # ["puzzles_checkmate_in_1_elo_range_0_400", "puzzles_checkmate_in_2_elo_range_0_400", "puzzles_checkmate_in_1_elo_range_401_500", "puzzles_checkmate_in_2_elo_range_401_500"]
    # puzzle_dataset_checkmate_attribute = [2, 2, 2, 2]
    # unix_timestamps = ['1748428627', '1748428689', '1748433689', '1748437159']

    # puzzle_dataset_filenames = ["puzzles_checkmate_in_2_elo_range_0_400", "puzzles_checkmate_in_2_elo_range_401_500", "puzzles_checkmate_in_2_elo_range_501_600", "puzzles_checkmate_in_2_elo_range_601_700", "puzzles_checkmate_in_2_elo_range_2001_3000"] 
    # puzzle_dataset_checkmate_attribute = [2, 2, 2, 2, 2]
    # unix_timestamps = ['1748428627', '1748428689', '1748433689', '1748437159', '1748684413']

    # puzzle_dataset_filenames = ["puzzles_checkmate_in_2_elo_range_2001_3000"] 
    # puzzle_dataset_checkmate_attribute = [2]
    # unix_timestamps = ['1748684413']

    # puzzle_dataset_filenames = ['puzzles_checkmate_in_1_elo_range_1501_2000', 'puzzles_checkmate_in_2_elo_range_1501_2000', 'puzzles_checkmate_in_3_elo_range_1501_2000', 'puzzles_checkmate_in_4_elo_range_1501_2000']
    # puzzle_dataset_checkmate_attribute = [1, 2, 3, 4]
    # unix_timestamps = ['1748716044', '1748716664', '1748717246', '1748717875']
    
    # elo_range_min = 1501
    # elo_range_max = 2000


    puzzle_dataset_filenames = ["puzzles_checkmate_in_1_elo_range_1001_3000", "puzzles_checkmate_in_2_elo_range_1001_3000", "puzzles_checkmate_in_3_elo_range_1001_3000", "puzzles_checkmate_in_4_elo_range_1001_3000"]
    puzzle_dataset_checkmate_attribute = [1, 2, 3, 4]
    unix_timestamps = ['1748763535', '1748767254', '1748768488', '1748769669']

    unix_timestamps_irl = unix_timestamps
    unix_timestamps_sil = unix_timestamps
    
    elo_range_min = 1001
    elo_range_max = 3000
    test_result_filename_prefix = "solution_length_test/compare_models_checkmate_1234_elo_" + str(elo_range_min) + "_" + str(elo_range_max)

    solution_lentgh_dataset_tests = True
    

    # puzzle_dataset_filenames = [ "puzzles_checkmate_in_2_elo_range_1501_1550", "puzzles_checkmate_in_2_elo_range_1551_1600", "puzzles_checkmate_in_2_elo_range_1601_1650", "puzzles_checkmate_in_2_elo_range_1651_1700", "puzzles_checkmate_in_2_elo_range_1701_1800", "puzzles_checkmate_in_2_elo_range_1801_1900", "puzzles_checkmate_in_2_elo_range_1901_2000", "puzzles_checkmate_in_2_elo_range_2001_3000"] #, "puzzles_checkmate_in_1_elo_range_1001_3000", "puzzles_checkmate_in_2_elo_range_1001_3000", "puzzles_checkmate_in_3_elo_range_1001_3000", "puzzles_checkmate_in_4_elo_range_1001_3000"]
    # puzzle_dataset_checkmate_attribute = [2, 2, 2, 2, 2, 2, 2, 2] #, 1, 2, 3, 4]
    # unix_timestamps_irl = ['1748696980', '1748697457', '1748698000', '1748698451' ,'1748698871' , '1748699757', '1748701146', '1748701660'] #, '1748763535', '1748767254', '1748768488', '1748769669']
    # unix_timestamps_sil = ['1748687884', '1748688350', '1748689498', '1748689965', '1748692907', '1748693792', '1748694231', '1748707567'] #, '1748763535', '1748767254', '1748768488', '1748769669']

    # elo_range_min = 1501
    # elo_range_max = 3000
    # test_result_filename_prefix = "difficulty_rating_test/compare_models_checkmate_2_elo_" + str(elo_range_min) + "_" + str(elo_range_max)

    # solution_lentgh_dataset_tests = False


    n_puzzle_per_length = 30

    model_checkpoint_filename_SIL = "checkpoint_epoch_60_MATChessFormer-Homogeneous-20.pt"
    model_checkpoint_filename_IRL = "checkpoint_epoch_75_MATChessFormer-ImitativeRL-20.pt"

    model_name_SIL =  "SIL"
    print(f"\nModel:\t{model_name_SIL}\ncheckpoint:\t{model_checkpoint_filename_SIL}")

    model_name_IRL =  "IRL"
    print(f"\nModel:\t{model_name_IRL}\ncheckpoint:\t{model_checkpoint_filename_IRL}")

    puzzle_test_data_dir = "test_data/puzzles/"
    # results_filename = "puzzle_test_results_" + model_name + "_checkmate_in_2.json"
    results_filename_SIL = test_result_filename_prefix + "_partial_or_complete_solve_SIL.json"
    results_filename_IRL = test_result_filename_prefix + "_partial_or_complete_solve_IRL.json"


    # results_csv_elo_vs_correct_agent_votes = "puzzle_test_results_" + model_name + "_checkmate_in_2_elo_0_600_wrt_correct_agent_vote_percentage.csv"
    results_csv_elo_vs_correct_agent_votes = test_result_filename_prefix + "_elo_wrt_same_agent_vote_percentage.csv"

    # with open(puzzle_test_data_dir + results_filename_SIL, 'w') as results_file:
    #     results_file.close()

    # with open(puzzle_test_data_dir + results_filename_IRL, 'w') as results_file:
    #     results_file.close()

    with open(puzzle_test_data_dir + results_csv_elo_vs_correct_agent_votes, 'w') as f:
        csv_writer = csv.writer(f,delimiter=',')
        csv_writer.writerow(['elo', 'checkmate_in', 'puzzle_solved_SIL', 'puzzle_solved_IRL','puzzle_firstmove_accuracy_SIL', 'puzzle_firstmove_accuracy_IRL', 'same_firstmove_agent_votes_percentage'])
        f.close()

    model_results_for_partially_or_completely_solved_puzzles_SIL = {}
    model_results_for_partially_or_completely_solved_puzzles_IRL = {}

    # Run "puzzle games" tests:
    for dataset_idx, puzzle_dataset_filename in enumerate(puzzle_dataset_filenames):
        with open(puzzle_dataset_dir + puzzle_dataset_filename + ".csv") as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        # unix_timestamp = unix_timestamps[dataset_idx] #int(time.time())
        unix_timestamp_sil = unix_timestamps_sil[dataset_idx]
        unix_timestamp_irl = unix_timestamps_irl[dataset_idx]
        
        test_result_dir_name_SIL =  "test_result_" + model_name_SIL + "_" + puzzle_dataset_filename + "_" + str(unix_timestamp_sil)
        test_result_dir_name_IRL =  "test_result_" + model_name_IRL + "_" + puzzle_dataset_filename + "_" + str(unix_timestamp_irl)

        for puzzle_idx, puzzle_data in tqdm(enumerate(rows)):
            if puzzle_idx == 0:
                continue

            if solution_lentgh_dataset_tests:
                if puzzle_idx > n_puzzle_per_length:
                    continue
                

            #### Determine the color of the pieces that the model is controlling:
            san_str = puzzle_data[COLUMN_IDX_PUZZLE_HIST_SAN]
            uci_move_hist = convert_san_to_uci(san_str)
            
            # Extract the puzzle solution in UCI notation:
            solution_str = puzzle_data[COLUMN_IDX_PUZZLE_SOLUTION_UCI]
            solution_uci = get_solution(solution_str)
            
            uci_move_hist.append(solution_uci[0])
            model_color_str = puzzle_solver_color(uci_move_hist)

            # Check wether or not the puzzle was solved:
            checkmate_in_n_fullmoves = puzzle_dataset_checkmate_attribute[dataset_idx]            

            puzzle_result_SIL = {
                'model_checkpoint_filename': model_checkpoint_filename_SIL,
                'model_type': model_name_SIL,
                'puzzle_id': puzzle_data[COLUMN_IDX_PUZZLE_ID],
                'puzzle_elo': puzzle_data[COLUMN_IDX_PUZZLE_ELO_RATING],
                'checkmate_in_n_fullmoves': checkmate_in_n_fullmoves,
                'model_color': model_color_str,
                'puzzle_solution_uci': solution_uci[1:],
                'ml_model_move_hist_vs_stockfish': [],
                'ml_models_turn': [],
                'model_solved_puzzle': False,
                'n_agents_voted_correctly': [],
                'agent_votes': [],
                'n_agents': [],
                'chess_piece_agent_votes': [],
                'piece_pos_before_move': [],
            }

            puzzle_result_IRL = {
                'model_checkpoint_filename': model_checkpoint_filename_IRL,
                'model_type': model_name_IRL,
                'puzzle_id': puzzle_data[COLUMN_IDX_PUZZLE_ID],
                'puzzle_elo': puzzle_data[COLUMN_IDX_PUZZLE_ELO_RATING],
                'checkmate_in_n_fullmoves': checkmate_in_n_fullmoves,
                'model_color': model_color_str,
                'puzzle_solution_uci': solution_uci[1:],
                'ml_model_move_hist_vs_stockfish': [],
                'ml_models_turn': [],
                'model_solved_puzzle': False,
                'n_agents_voted_correctly': [],
                'agent_votes': [],
                'n_agents': [],
                'chess_piece_agent_votes': [],
                'piece_pos_before_move': [],
            }

            # Open the test data folder for the puzzle test and extract the model's solution attempt:
            test_data_game_log_filename = puzzle_test_data_dir + test_result_dir_name_SIL + '/' + puzzle_data[COLUMN_IDX_PUZZLE_ID] + "/" + "log_game_vote_hist.json"
            with open(test_data_game_log_filename, 'r') as testdata_file:
                test_data = json.load(testdata_file)
                setup_hist_len = len(uci_move_hist) + 1
                done_processing = False
                puzzle_state_idx = setup_hist_len

                while not done_processing:
                    try:
                        model_solution_uci_str = test_data[str(puzzle_state_idx)]['Hist']
                    except:
                        print(f"Tried extracting move hist from test_data (puzzle id {puzzle_data[COLUMN_IDX_PUZZLE_ID]}), but caught an error...")
                        break

                    model_joint_action_hist = get_solution(model_solution_uci_str.strip())
                    puzzle_result_SIL['ml_model_move_hist_vs_stockfish'].append(model_joint_action_hist[-1])

                    puzzlue_solution_move_idx = puzzle_state_idx - setup_hist_len
                    if puzzlue_solution_move_idx % 2 == 0:
                        puzzle_result_SIL['ml_models_turn'].append(True)
                    else:
                        puzzle_result_SIL['ml_models_turn'].append(False)
                        # puzzle_state_idx += 1
                        # continue

                    Votes_hist_dict = test_data[str(puzzle_state_idx)]['Votes']
                    puzzle_result_SIL['agent_votes'].append(Votes_hist_dict)

                    # print(f'puzzlue_solution_move_idx={puzzlue_solution_move_idx}')
                    correct_move = puzzle_result_SIL['puzzle_solution_uci'][puzzlue_solution_move_idx]
                    if correct_move in Votes_hist_dict.keys():
                        puzzle_result_SIL['n_agents_voted_correctly'].append(Votes_hist_dict[correct_move])
                    else:
                        puzzle_result_SIL['n_agents_voted_correctly'].append(0)

                    # Compute the total number of agents:
                    vote_counts = 0
                    for key, item in Votes_hist_dict.items():
                        vote_counts += int(item)
                    puzzle_result_SIL['n_agents'].append(vote_counts)

                    # Check if puzzle was solved completely:
                    if puzzle_result_SIL['puzzle_solution_uci'] == puzzle_result_SIL['ml_model_move_hist_vs_stockfish']:
                        puzzle_result_SIL['model_solved_puzzle'] = True

                    # Save the position of the agents before the move:
                    chess_piece_agent_pos_dict = test_data[str(puzzle_state_idx)]['piece_pos_before_move']
                    puzzle_result_SIL['piece_pos_before_move'].append(chess_piece_agent_pos_dict)

                    # Save which agent voted for which move:
                    chess_piece_agents_votes_dict = test_data[str(puzzle_state_idx)]['agent Votes']
                    puzzle_result_SIL['chess_piece_agent_votes'].append(chess_piece_agents_votes_dict)
                    
                    # Check if model solution is too long:
                    if len(puzzle_result_SIL['ml_model_move_hist_vs_stockfish']) >= len(puzzle_result_SIL['puzzle_solution_uci']):
                        done_processing = True
                        break

                    puzzle_state_idx += 1

                    if (puzzle_state_idx - setup_hist_len) > len(puzzle_result_SIL['puzzle_solution_uci']) - 1:
                        done_processing = True
                        break

            
            # with open(puzzle_test_data_dir + results_filename_SIL, 'a') as puzzle_test_result_file:
            #     json.dump(puzzle_result_SIL, puzzle_test_result_file)
            #     puzzle_test_result_file.write("\n")
            #     puzzle_test_result_file.close()

            # Open the test data folder for the puzzle test and extract the model's solution attempt:
            test_data_game_log_filename = puzzle_test_data_dir + test_result_dir_name_IRL + '/' + puzzle_data[COLUMN_IDX_PUZZLE_ID] + "/" + "log_game_vote_hist.json"
            with open(test_data_game_log_filename, 'r') as testdata_file:
                test_data = json.load(testdata_file)
                setup_hist_len = len(uci_move_hist) + 1
                done_processing = False
                puzzle_state_idx = setup_hist_len

                while not done_processing:
                    try:
                        model_solution_uci_str = test_data[str(puzzle_state_idx)]['Hist']
                    except:
                        print(f"Tried extracting move hist from test_data (puzzle id {puzzle_data[COLUMN_IDX_PUZZLE_ID]}), but caught an error...")
                        break

                    model_joint_action_hist = get_solution(model_solution_uci_str.strip())
                    puzzle_result_IRL['ml_model_move_hist_vs_stockfish'].append(model_joint_action_hist[-1])

                    puzzlue_solution_move_idx = puzzle_state_idx - setup_hist_len
                    if puzzlue_solution_move_idx % 2 == 0:
                        puzzle_result_IRL['ml_models_turn'].append(True)
                    else:
                        puzzle_result_IRL['ml_models_turn'].append(False)
                        # puzzle_state_idx += 1
                        # continue

                    Votes_hist_dict = test_data[str(puzzle_state_idx)]['Votes']
                    puzzle_result_IRL['agent_votes'].append(Votes_hist_dict)

                    # print(f'puzzlue_solution_move_idx={puzzlue_solution_move_idx}')
                    correct_move = puzzle_result_IRL['puzzle_solution_uci'][puzzlue_solution_move_idx]
                    if correct_move in Votes_hist_dict.keys():
                        puzzle_result_IRL['n_agents_voted_correctly'].append(Votes_hist_dict[correct_move])
                    else:
                        puzzle_result_IRL['n_agents_voted_correctly'].append(0)

                    # Compute the total number of agents:
                    vote_counts = 0
                    for key, item in Votes_hist_dict.items():
                        vote_counts += int(item)
                    puzzle_result_IRL['n_agents'].append(vote_counts)

                    # Check if puzzle was solved completely:
                    if puzzle_result_IRL['puzzle_solution_uci'] == puzzle_result_IRL['ml_model_move_hist_vs_stockfish']:
                        puzzle_result_IRL['model_solved_puzzle'] = True

                    # Save the position of the agents before the move:
                    chess_piece_agent_pos_dict = test_data[str(puzzle_state_idx)]['piece_pos_before_move']
                    puzzle_result_IRL['piece_pos_before_move'].append(chess_piece_agent_pos_dict)

                    # Save which agent voted for which move:
                    chess_piece_agents_votes_dict = test_data[str(puzzle_state_idx)]['agent Votes']
                    puzzle_result_IRL['chess_piece_agent_votes'].append(chess_piece_agents_votes_dict)
                    
                    # Check if model solution is too long:
                    if len(puzzle_result_IRL['ml_model_move_hist_vs_stockfish']) >= len(puzzle_result_IRL['puzzle_solution_uci']):
                        done_processing = True
                        break

                    puzzle_state_idx += 1

                    if (puzzle_state_idx - setup_hist_len) > len(puzzle_result_IRL['puzzle_solution_uci']) - 1:
                        done_processing = True
                        break

            # with open(puzzle_test_data_dir + results_filename_IRL, 'a') as puzzle_test_result_file:
            #     json.dump(puzzle_result_IRL, puzzle_test_result_file)
            #     puzzle_test_result_file.write("\n")
            #     puzzle_test_result_file.close()
            
            with open(puzzle_test_data_dir + results_csv_elo_vs_correct_agent_votes, 'a') as f:
                csv_writer = csv.writer(f,delimiter=',')
                first_move_correct_votes_percentage_SIL = (puzzle_result_SIL['n_agents_voted_correctly'][0] / puzzle_result_SIL['n_agents'][0])  * 100.0

                first_move_correct_votes_percentage_IRL = (puzzle_result_IRL['n_agents_voted_correctly'][0] / puzzle_result_IRL['n_agents'][0])  * 100.0

                if first_move_correct_votes_percentage_SIL > 0:
                    # print("\n\nSOME AGENTS ON THE SIL TEAM VOTED FOR THE CORRECT MOVE TO SOLVE THE PUZZLE.\n\tPrinting the puzzle_result...\n")
                    model_results_for_partially_or_completely_solved_puzzles_SIL[puzzle_result_SIL['puzzle_id']] = copy.deepcopy(puzzle_result_SIL)
                    # for key, item in puzzle_result_SIL.items():
                    #     print(f'{key}:\t{item}')

                if first_move_correct_votes_percentage_IRL > 0:
                    # print("\n\nSOME AGENTS ON THE IRL TEAM VOTED FOR THE CORRECT MOVE TO SOLVE THE PUZZLE.\n\tPrinting the puzzle_result...\n")
                    model_results_for_partially_or_completely_solved_puzzles_IRL[puzzle_result_IRL['puzzle_id']] = copy.deepcopy(puzzle_result_IRL)
                    # for key, item in puzzle_result_SIL.items():
                    #     print(f'{key}:\t{item}')

                # csv_writer.writerow([puzzle_result_SIL['puzzle_elo'], puzzle_result_SIL['checkmate_in_n_fullmoves'], first_move_correct_votes_percentage_SIL, first_move_correct_votes_percentage_IRL])
                # ['elo', 'checkmate_in', 'puzzle_solved_SIL' 'puzzle_solved_IRL','puzzle_firstmove_accuracy_SIL' 'puzzle_firstmove_accuracy_IRL', 'same_firstmove_agent_votes_percentage']
                same_votes_count = 0
                # print(puzzle_result_IRL['chess_piece_agent_votes'][0])
                for agent_name, agent_vote in puzzle_result_IRL['chess_piece_agent_votes'][0].items():
                    if agent_vote == puzzle_result_SIL['chess_piece_agent_votes'][0][agent_name]:
                        same_votes_count += 1
                same_firstmove_votes_percentage = (same_votes_count / puzzle_result_IRL['n_agents'][0]) * 100.0 

                csv_writer.writerow([puzzle_result_SIL['puzzle_elo'], puzzle_result_SIL['checkmate_in_n_fullmoves'], int(puzzle_result_SIL['model_solved_puzzle']), int(puzzle_result_IRL['model_solved_puzzle']), first_move_correct_votes_percentage_SIL, first_move_correct_votes_percentage_IRL, same_firstmove_votes_percentage])
                f.close()

            # print("\nPUZZLE RESULT:")
            # for key, item in puzzle_result.items():
            #     print(f'{key}:\t{item}')
        if dataset_idx == 3 and puzzle_idx >= 23:
            break


    # Save all the puzzles that were partially solved by the two teams to seperate files:
    with open(puzzle_test_data_dir + results_filename_SIL, 'w') as results_file_agents_partial_solve:
        json.dump(model_results_for_partially_or_completely_solved_puzzles_SIL, results_file_agents_partial_solve)

    
    with open(puzzle_test_data_dir + results_filename_IRL, 'w') as results_file_agents_partial_solve:
        json.dump(model_results_for_partially_or_completely_solved_puzzles_IRL, results_file_agents_partial_solve)
    

    elo, checkmate_in, solved_sil, solved_irl, correct_vote_percentage_sil, correct_vote_percentage_irl, same_firstmove_vote_percentage = get_puzzle_results_from_csv(puzzle_test_data_dir, results_csv_elo_vs_correct_agent_votes)

    print(checkmate_in)

    if 1 in solved_sil:
        print("SIL model solved at least one puzzle")

    sil_solved_puzzle_count = 0
    for is_solved in solved_sil:
        if is_solved == 1:
            sil_solved_puzzle_count += 1
    print(f"SIL model solved {sil_solved_puzzle_count} puzzle(s)")
    
    if 1 in solved_irl:
        print("IRL model solved at least one puzzle")

    irl_solved_puzzle_count = 0
    for is_solved in solved_irl:
        if is_solved == 1:
            irl_solved_puzzle_count += 1
    print(f"IRL model solved {irl_solved_puzzle_count} puzzle(s)")


    # print(checkmate_in)

    # boxplot_data_accuracy_sil = [[], [], [], []]
    # for idx, checkmate_in in enumerate(checkmate_in):
    #     boxplot_data_accuracy_sil[int(checkmate_in)-1].append(int(correct_vote_percentage_sil[idx]))

    # boxplot_data_accuracy_irl = [[], [], [], []]
    # for idx, checkmate_in in enumerate(checkmate_in):
    #     boxplot_data_accuracy_irl[int(checkmate_in)-1].append(int(correct_vote_percentage_irl[idx]))
    
    # print(boxplot_data_accuracy_sil)
    # print(boxplot_data_accuracy_irl)
    # print([boxplot_data_accuracy_sil, boxplot_data_accuracy_irl])

    n_bins = 20

    elo_data, elo_rating_bins = np.histogram([elo, correct_vote_percentage_sil], bins=n_bins, range=(elo_range_min - 1, elo_range_max))
    print(elo_rating_bins)
    print(elo_data)

    print(f'Total number of puzzles in this test: {np.sum(np.array(elo_data))}')

    accuracy_barplot_data = [[], []]
    accuracy_vals = [[], []]
    accuracy_weights = [[], []]

    weights_sil = []
    weights_irl = []
    
    for elo_idx, elo_rating in enumerate(elo):
        sil_accuracy_weight = 0
        irl_accuracy_weight = 0
        
        if correct_vote_percentage_sil[elo_idx] > 50:
            accuracy_barplot_data[0].append(int(elo_rating))
            accuracy_vals[0].append(int(correct_vote_percentage_sil[elo_idx]))
            sil_accuracy_weight = 1
        if correct_vote_percentage_irl[elo_idx] > 50:
            accuracy_barplot_data[1].append(int(elo_rating))
            accuracy_vals[1].append(int(correct_vote_percentage_irl[elo_idx]))
            irl_accuracy_weight = 1

        for idx_bin, edge_val in enumerate(elo_rating_bins[1:]):
            if elo_rating <= edge_val:
                weights_sil.append((sil_accuracy_weight / elo_data[idx_bin])* 100.0)
                weights_irl.append((irl_accuracy_weight / elo_data[idx_bin]) * 100.0)

                # print(f'SIL model Accuracy weight: {sil_accuracy_weight / elo_data[idx_bin]}, weight={sil_accuracy_weight} bin_count={elo_data[idx_bin]}')
                break

        if sil_accuracy_weight > 0:
            accuracy_weights[0].append(weights_sil[-1])
        if irl_accuracy_weight > 0:
            accuracy_weights[1].append(weights_irl[-1])


    

    plt.figure(1)
    plt.grid()
    plt.scatter(elo, correct_vote_percentage_sil, alpha=0.8) #, label='Standard IL')
    # plt.hist(elo, weights=weights_sil, bins=n_bins, histtype='bar', label=['Standard IL'])
    plt.title('First Move Accuracy for the\nStandard Imitation Learning Model',fontsize="x-large", fontweight='bold')
    plt.ylabel('MoveVote Accuracy of Agents [%]',fontsize="large", fontweight='bold')
    plt.xlabel('Puzzle Rating (ELO)',fontsize="large", fontweight='bold')
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_scatter_firstmove_accuracy_wrt_elo_SIL.png')
    plt.savefig(save_file)

    plt.figure(2)
    plt.grid()
    plt.scatter(elo, correct_vote_percentage_irl, alpha=0.8) #, label=['Imitative RL'])
    # plt.hist(elo, weights=weights_irl, bins=n_bins, histtype='bar', label=['Imitative RL'])
    plt.title('First Move Accuracy for the\nImitative Reinforcement Learning Model',fontsize="x-large", fontweight='bold')
    plt.ylabel('MoveVote Accuracy of Agents [%]',fontsize="large", fontweight='bold')
    plt.xlabel('Puzzle Rating (ELO)',fontsize="large", fontweight='bold')
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_scatter_firstmove_accuracy_wrt_elo_IRL.png')
    plt.savefig(save_file)


    # boxplot_data = [[], [], [], []]
    boxplot_data = [[], [], [], []]
    for idx, c in enumerate(checkmate_in):
        boxplot_data[int(c)-1].append(int(elo[idx]))

    plt.figure(3)
    plt.boxplot(boxplot_data)
    plt.title('Elo Rating wrt. Solution Length',fontsize="x-large", fontweight='bold')
    plt.ylabel('Solution Length [fullmove count]',fontsize="large", fontweight='bold')
    plt.xlabel('Puzzle Rating [ELO]',fontsize="large", fontweight='bold')    
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_boxplot_elo_wrt_solution_length.png')
    plt.savefig(save_file)
    

    # print(accuracy_barplot_data)
    # print(accuracy_vals)
    # print(accuracy_weights)

    plt.figure(4)
    # plt.hist(accuracy_barplot_data, weights=accuracy_vals, histtype='bar', label=['Standard IL', 'Imitative RL'])
    plt.hist(accuracy_barplot_data, weights=accuracy_weights, bins=n_bins, histtype='bar', label=['Standard IL', 'Imitative RL'])
    plt.legend(['Standard IL', 'Imitative RL'])
    plt.title(' First Move Accuracy of the Teams\' Chosen Move',fontsize="x-large", fontweight='bold')
    plt.ylabel('First Move Accuracy of Model [%]',fontsize="large", fontweight='bold')
    plt.xlabel('Puzzle Rating [ELO]',fontsize="large", fontweight='bold')    
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_bar_firstmove_accuracy_both_models_wrt_elo.png')
    plt.savefig(save_file)


    plt.figure(5)
    plt.hist(elo, bins=n_bins, histtype='bar')
    # bar_input = np.array(elo_data) / np.sum(elo_data)
    # print(f'bar input:\nbar_input:\t{bar_input}\nelo data:\t{elo_data}')
    # plt.bar(bar_input, elo_data, align='edge')
    plt.title('Histogram of Puzzle Rating for Puzzles\nwith a Checkmate in 2 Solution',fontsize="x-large", fontweight='bold')
    plt.ylabel('Frequency [%]',fontsize="large", fontweight='bold')
    plt.xlabel('Puzzle Rating [ELO]',fontsize="large", fontweight='bold')    
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_hist_elo_rating_of_puzzles.png')
    plt.savefig(save_file)


    # all_agents_voted_the_same = []
    # for vote_percent in same_firstmove_vote_percentage:
    #     if vote_percent == 100:
    #         all_agents_voted_the_same.append(1)
    #     else:
    #         all_agents_voted_the_same.append(0)


    plt.figure(6)
    # plt.grid()
    # plt.scatter(elo, checkmate_in, alpha=0.5, label='checkmate in N fullmoves')
    # plt.scatter(checkmate_in, correct_vote_percentage_irl, alpha=0.5, label='Imitative RL')
    # plt.bar(checkmate_in, correct_vote_percentage_sil, correct_vote_percentage_irl)
    #plt.hist(elo_rating_histogram_wrt_solution_length, histtype='bar')
    # plt.hist(same_firstmove_vote_percentage, histtype='bar')    
    
    # plt.bar(elo, same_firstmove_vote_percentage, width=2.0)   # Works somewhat

    # data, bins = np.histogram(same_firstmove_vote_percentage)
    # plt.hist(data, bins=bins, histtype='bar')
    # same_vote_hist, same_vote_bins = np.histogram(same_firstmove_vote_percentage, bins=n_bins, range=(0,100))
    plt.hist(same_firstmove_vote_percentage, density=True)

    # plt.hist(same_firstmove_vote_percentage, histtype='bar')  # Works somewhat better?
    
    # plt.bar(model_accuracy_wrt_elo_ratings_bin_edges[1:], heights)
    # plt.hist(model_accuracy_wrt_elo_ratings, bins=model_accuracy_wrt_elo_ratings_bin_edges, histtype='bar')
    
    # plt.hist(model_accuracy_wrt_elo_ratings)
    #plt.xticks(np.arange(6), ['1500','1600', '1700',  '1800', '1900', '2000'])
    plt.title('Accuracy of the Two models wrt. Each Other for\nFirst Move in Chess Puzzle Solution',fontsize="x-large", fontweight='bold')
    plt.ylabel('Number of Puzzles',fontsize="large", fontweight='bold')
    plt.xlabel('Same agent votes [%]',fontsize="large", fontweight='bold')    
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_hist_same_agent_votes.png')
    plt.savefig(save_file)





    accuracy_barplot_data = [[], []]
    accuracy_vals = [[], []]
    accuracy_weights = [[], []]

    weights_sil = []
    weights_irl = []
    
    for elo_idx, elo_rating in enumerate(elo):
        sil_accuracy_weight = 0
        irl_accuracy_weight = 0
        
        if correct_vote_percentage_sil[elo_idx] > 0:
            accuracy_barplot_data[0].append(int(elo_rating))
            accuracy_vals[0].append(int(correct_vote_percentage_sil[elo_idx]))
            sil_accuracy_weight = 1
        if correct_vote_percentage_irl[elo_idx] > 0:
            accuracy_barplot_data[1].append(int(elo_rating))
            accuracy_vals[1].append(int(correct_vote_percentage_irl[elo_idx]))
            irl_accuracy_weight = 1

        for idx_bin, edge_val in enumerate(elo_rating_bins[1:]):
            if elo_rating <= edge_val:
                weights_sil.append((sil_accuracy_weight / elo_data[idx_bin])* 100.0)
                weights_irl.append((irl_accuracy_weight / elo_data[idx_bin]) * 100.0)

                # print(f'SIL model Accuracy weight: {sil_accuracy_weight / elo_data[idx_bin]}, weight={sil_accuracy_weight} bin_count={elo_data[idx_bin]}')
                break

        if sil_accuracy_weight > 0:
            accuracy_weights[0].append(weights_sil[-1])
        if irl_accuracy_weight > 0:
            accuracy_weights[1].append(weights_irl[-1])

    plt.figure(7)
    # plt.hist(accuracy_barplot_data, weights=accuracy_vals, histtype='bar', label=['Standard IL', 'Imitative RL'])
    plt.hist(accuracy_barplot_data, weights=accuracy_weights, bins=n_bins, histtype='bar', label=['Standard IL', 'Imitative RL'])
    plt.legend(['Standard IL', 'Imitative RL'])
    plt.title('First Move Accuracy of Agents on the Team ',fontsize="x-large", fontweight='bold')
    plt.ylabel('MoveVote Accuracy of Agents [%]',fontsize="large", fontweight='bold')
    plt.xlabel('Puzzle Rating [ELO]',fontsize="large", fontweight='bold')    
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_bar_first_move_agent_vote_accuracy_votes_wrt_elo_bar.png')
    plt.savefig(save_file)

    ##############################################################################
    if solution_lentgh_dataset_tests:

        solution_length_correct_votes_sil = [[], [], [], []]
        elo_per_solution_length = [[], [], [], []]
        for vote_idx, correct_agent_votes in enumerate(correct_vote_percentage_sil):
            solution_length_correct_votes_sil[int(checkmate_in[vote_idx])-1].append(correct_agent_votes)
            elo_per_solution_length[int(checkmate_in[vote_idx])-1].append(elo[vote_idx])

        solution_length_correct_votes_irl = [[], [], [], []]
        for vote_idx, correct_agent_votes in enumerate(correct_vote_percentage_irl):
            solution_length_correct_votes_irl[int(checkmate_in[vote_idx])-1].append(correct_agent_votes)
        
        solution_lengths = ['checkmate in 1', 'checkmate in 2', 'checkmate in 3', 'checkmate in 4']
        plt.figure(8)
        # plt.scatter(elo, correct_vote_percentage_sil, alpha=0.8) #, label='Standard IL')
        # plt.hist(elo, weights=weights_sil, bins=n_bins, histtype='bar', label=['Standard IL'])
        plt.hist(solution_length_correct_votes_sil)
        plt.legend(solution_lengths)
        plt.title('First Move Accuracy for the\nStandard Imitation Learning Model',fontsize="x-large", fontweight='bold')
        plt.ylabel('MoveVote Accuracy of Agents [%]',fontsize="large", fontweight='bold')
        plt.xlabel('Puzzle Rating (ELO)',fontsize="large", fontweight='bold')
        save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_bar_first_move_accuracy_wrt_elo_per_length_SIL.png')
        plt.savefig(save_file)


        plt.figure(9)
        # plt.scatter(elo, correct_vote_percentage_irl, alpha=0.8) #, label=['Imitative RL'])
        # plt.hist(elo, weights=weights_irl, bins=n_bins, histtype='bar', label=['Imitative RL'])
        plt.hist(solution_length_correct_votes_irl)
        plt.legend(solution_lengths)
        plt.title('First Move Accuracy for the\nImitative Reinforcement Learning Model',fontsize="x-large", fontweight='bold')
        plt.ylabel('MoveVote Accuracy of Agents [%]',fontsize="large", fontweight='bold')
        plt.xlabel('Puzzle Rating (ELO)',fontsize="large", fontweight='bold')
        
        save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_bar_first_move_accuracy_wrt_elo_per_length_IRL.png')
        plt.savefig(save_file)


        plt.figure(10)
        plt.grid()
        for length in range(4):
            plt.scatter(elo_per_solution_length[length], solution_length_correct_votes_sil[length], alpha=0.8) #, label='Standard IL')
        plt.legend(solution_lengths)
        # plt.hist(elo, weights=weights_sil, bins=n_bins, histtype='bar', label=['Standard IL'])
        plt.title('First Move Accuracy for the\nStandard Imitation Learning Model',fontsize="x-large", fontweight='bold')
        plt.ylabel('MoveVote Accuracy of Agents [%]',fontsize="large", fontweight='bold')
        plt.xlabel('Puzzle Rating (ELO)',fontsize="large", fontweight='bold')
        save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_scatter_firstmove_accuracy_wrt_elo_per_length_SIL.png')
        plt.savefig(save_file)

        plt.figure(11)
        plt.grid()
        for length in range(4):
            plt.scatter(elo_per_solution_length[length], solution_length_correct_votes_irl[length], alpha=0.8) #, label=['Imitative RL'])
        plt.legend(solution_lengths)
        # plt.hist(elo, weights=weights_irl, bins=n_bins, histtype='bar', label=['Imitative RL'])
        plt.title('First Move Accuracy for the\nImitative Reinforcement Learning Model',fontsize="x-large", fontweight='bold')
        plt.ylabel('MoveVote Accuracy of Agents [%]',fontsize="large", fontweight='bold')
        plt.xlabel('Puzzle Rating (ELO)',fontsize="large", fontweight='bold')
        
        save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_scatter_firstmove_accuracy_wrt_elo_per_length_IRL.png')
        plt.savefig(save_file)


        # first_move_accuracy_histogram_wrt_solution_length = [[], []]
        # for idx, c_in in enumerate(checkmate_in):
        #     if correct_vote_percentage_sil[idx] >= 50:
        #         first_move_accuracy_histogram_wrt_solution_length[0].append(c_in)
        #     if correct_vote_percentage_irl[idx] >= 50:
        #         first_move_accuracy_histogram_wrt_solution_length[1].append(c_in)

        # print(first_move_accuracy_histogram_wrt_solution_length)

        first_move_accuracy_histogram_wrt_solution_length = [[2, 3, 3, 4], [1, 3, 3, 3, 4]]

        # solution_lengths = ['1', '2', '3', '4']
        # height_sil = [0,1,2,1]
        # height_irl = [1, 0, 3, 1]

        plt.figure(12)
        # plt.scatter(elo, correct_vote_percentage_irl, alpha=0.8) #, label=['Imitative RL'])
        # plt.hist(elo, weights=weights_irl, bins=n_bins, histtype='bar', label=['Imitative RL'])
        # for model_data in first_move_accuracy_histogram_wrt_solution_length:
        # plt.hist(first_move_accuracy_histogram_wrt_solution_length, bins=[0.5, 1.5, 2.5, 3.5, 4.5], label=['1', '2', '3', '4'], align='mid')
        plt.hist(first_move_accuracy_histogram_wrt_solution_length, bins=np.arange(1,6)-0.5)
        # plt.bar(solution_lengths, height=height_irl)
        plt.xticks(range(1,5))
        plt.legend(['Standard IL', 'Imitative RL'])
        plt.title('First Move Accuracy of Models wrt. Solution Length',fontsize="x-large", fontweight='bold')
        plt.ylabel('MoveVote Accuracy of Team [count]',fontsize="large", fontweight='bold')
        plt.xlabel('Solution Length [#Moves before solved]',fontsize="large", fontweight='bold')
        
        save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_bar_first_move_accuracy_hist_wrt_solution_length.png')
        plt.savefig(save_file)
        
