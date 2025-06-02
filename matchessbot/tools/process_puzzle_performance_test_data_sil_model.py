#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt

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
    data = []
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')#, quotechar='|')
        for row in csvreader:
            if row_count == 0:
                headers_file = row
                # print(headers_file)
            else:
                elo.append(int(row[0]))
                data.append(float(row[1]))
            row_count += 1
    return elo, data


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

    puzzle_dataset_filenames = [ "puzzles_checkmate_in_2_elo_range_1501_1550", "puzzles_checkmate_in_2_elo_range_1551_1600", "puzzles_checkmate_in_2_elo_range_1601_1650", "puzzles_checkmate_in_2_elo_range_1651_1700", "puzzles_checkmate_in_2_elo_range_1701_1800", "puzzles_checkmate_in_2_elo_range_1801_1900", "puzzles_checkmate_in_2_elo_range_1901_2000", "puzzles_checkmate_in_2_elo_range_2001_3000"]
    puzzle_dataset_checkmate_attribute = [2, 2, 2, 2, 2, 2, 2, 2]
    unix_timestamps = ['1748687884', '1748688350', '1748689498', '1748689965', '1748692907', '1748693792', '1748694231', '1748707567']
    

    elo_range_min = 1501
    elo_range_max = 2500
    test_result_filename_prefix = "puzzle_test_results_elo_" + str(elo_range_min) + " _" + str(elo_range_max) + "_"

    model_checkpoint_filename = "checkpoint_epoch_60_MATChessFormer-Homogeneous-20.pt"
    model_name =  "SIL"

    puzzle_test_data_dir = "test_data/puzzles/"
    # results_filename = "puzzle_test_results_" + model_name + "_checkmate_in_2.json"
    results_filename = test_result_filename_prefix + model_name + "_checkmate_in_2.json"

    # results_csv_elo_vs_correct_agent_votes = "puzzle_test_results_" + model_name + "_checkmate_in_2_elo_0_600_wrt_correct_agent_vote_percentage.csv"
    results_csv_elo_vs_correct_agent_votes = test_result_filename_prefix + model_name + "_checkmate_in_2_elo_wrt_correct_agent_vote_percentage.csv"

    with open(puzzle_test_data_dir + results_filename, 'w') as results_file:
        results_file.close()

    with open(puzzle_test_data_dir + results_csv_elo_vs_correct_agent_votes, 'w') as f:
        csv_writer = csv.writer(f,delimiter=',')
        csv_writer.writerow(['puzzle_elo', 'first_move_correct_vote_percentage'])
        f.close()

    print(f"\nModel:\t{model_name}\ncheckpoint:\t{model_checkpoint_filename}")

    model_results_for_partially_or_completely_solved_puzzles = {}

    

    # Run "puzzle games" tests:
    for dataset_idx, puzzle_dataset_filename in enumerate(puzzle_dataset_filenames):
        with open(puzzle_dataset_dir + puzzle_dataset_filename + ".csv") as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        unix_timestamp = unix_timestamps[dataset_idx] #int(time.time())
        test_result_dir_name =  "test_result_" + model_name + "_" + puzzle_dataset_filename + "_" + str(unix_timestamp)

        for puzzle_idx, puzzle_data in tqdm(enumerate(rows)):
            if puzzle_idx == 0:
                continue

            # if dataset_idx == 3 and puzzle_idx >= 23:
            #     break
            
            if dataset_idx == 0 and puzzle_idx >= 27:
                break

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

            # Run puzzle simulation:
            # completed_process = subprocess.run(["./src/matchessbot/scripts/run_matchess_puzzles.sh" , str(puzzle_idx), puzzle_dataset_filename + ".csv", test_result_dir_name])
            # print("\n\n")
            # print(completed_process)
            # print("\n\n")
            

            puzzle_result = {
                'model_checkpoint_filename': model_checkpoint_filename,
                'model_type': model_name,
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

            # print("\nPUZZLE RESULT BEFORE DATA PROCESSING:")
            # for key, item in puzzle_result.items():
            #     print(f'{key}:\t{item}')

            # Open the test data folder for the puzzle test and extract the model's soluiton attempt:
            test_data_game_log_filename = puzzle_test_data_dir + test_result_dir_name + '/' + puzzle_data[COLUMN_IDX_PUZZLE_ID] + "/" + "log_game_vote_hist.json"
            with open(test_data_game_log_filename, 'r') as testdata_file:
                test_data = json.load(testdata_file)

                # print(test_data)
                
                setup_hist_len = len(uci_move_hist) + 1
                
                done_processing = False
                puzzle_state_idx = setup_hist_len
                # print(f"puzzle_state_idx={puzzle_state_idx}")
                # print(f'\nPUZZLE ID: {puzzle_data[COLUMN_IDX_PUZZLE_ID]}')
                # print(puzzle_result['puzzle_solution_uci'])
                while not done_processing:
                    # print(puzzle_state_idx- setup_hist_len)

                    
                    try:
                        model_solution_uci_str = test_data[str(puzzle_state_idx)]['Hist']
                    except:
                        print(f"Tried extracting move hist from test_data (puzzle id {puzzle_data[COLUMN_IDX_PUZZLE_ID]}), but caught an error...")
                        break

                    model_joint_action_hist = get_solution(model_solution_uci_str.strip())
                    puzzle_result['ml_model_move_hist_vs_stockfish'].append(model_joint_action_hist[-1])

                    puzzlue_solution_move_idx = puzzle_state_idx - setup_hist_len
                    if puzzlue_solution_move_idx % 2 == 0:
                        puzzle_result['ml_models_turn'].append(True)
                    else:
                        puzzle_result['ml_models_turn'].append(False)
                        # puzzle_state_idx += 1
                        # continue

                    Votes_hist_dict = test_data[str(puzzle_state_idx)]['Votes']
                    puzzle_result['agent_votes'].append(Votes_hist_dict)

                    # print(f'puzzlue_solution_move_idx={puzzlue_solution_move_idx}')
                    correct_move = puzzle_result['puzzle_solution_uci'][puzzlue_solution_move_idx]
                    if correct_move in Votes_hist_dict.keys():
                        puzzle_result['n_agents_voted_correctly'].append(Votes_hist_dict[correct_move])
                    else:
                        puzzle_result['n_agents_voted_correctly'].append(0)

                    # Compute the total number of agents:
                    vote_counts = 0
                    for key, item in Votes_hist_dict.items():
                        vote_counts += int(item)
                    puzzle_result['n_agents'].append(vote_counts)

                    # Check if puzzle was solved completely:
                    if puzzle_result['puzzle_solution_uci'] == puzzle_result['ml_model_move_hist_vs_stockfish']:
                        puzzle_result['model_solved_puzzle'] = True

                    # Save the position of the agents before the move:
                    chess_piece_agent_pos_dict = test_data[str(puzzle_state_idx)]['piece_pos_before_move']
                    puzzle_result['piece_pos_before_move'].append(chess_piece_agent_pos_dict)

                    # Save which agent voted for which move:
                    chess_piece_agents_votes_dict = test_data[str(puzzle_state_idx)]['agent Votes']
                    puzzle_result['chess_piece_agent_votes'].append(chess_piece_agents_votes_dict)
                    
                    # Check if model solution is too long:
                    if len(puzzle_result['ml_model_move_hist_vs_stockfish']) >= len(puzzle_result['puzzle_solution_uci']):
                        done_processing = True
                        break

                    puzzle_state_idx += 1

                    if (puzzle_state_idx - setup_hist_len) > len(puzzle_result['puzzle_solution_uci']) - 1:
                        done_processing = True
                        break

            
            with open(puzzle_test_data_dir + results_filename, 'a') as puzzle_test_result_file:
                json.dump(puzzle_result, puzzle_test_result_file)
                puzzle_test_result_file.write("\n")
                puzzle_test_result_file.close()

            with open(puzzle_test_data_dir + results_csv_elo_vs_correct_agent_votes, 'a') as f:
                csv_writer = csv.writer(f,delimiter=',')
                first_move_correct_votes_percentage = (puzzle_result['n_agents_voted_correctly'][0] / puzzle_result['n_agents'][0])  * 100.0

                if first_move_correct_votes_percentage > 0:
                    print("\n\nSOME AGENTS ON THE TEAM VOTED FOR THE CORRECT MOVE TO SOLVE THE PUZZLE.\n\tPrinting the puzzle_result...\n")
                    model_results_for_partially_or_completely_solved_puzzles[puzzle_result['puzzle_id']] = copy.deepcopy(puzzle_result)
                    for key, item in puzzle_result.items():
                        print(f'{key}:\t{item}')

                csv_writer.writerow([puzzle_result['puzzle_elo'], first_move_correct_votes_percentage])
                f.close()

            # print("\nPUZZLE RESULT:")
            # for key, item in puzzle_result.items():
            #     print(f'{key}:\t{item}')
        if dataset_idx == 3 and puzzle_idx >= 23:
            break

    # Plot the percentage of agents who voted correctly wrt. ELO rating of the puzzle:
    elo, correct_vote_percentage = get_puzzle_results_from_csv(puzzle_test_data_dir, results_csv_elo_vs_correct_agent_votes)
    plt.figure(1)
    plt.grid()
    # plt.axis([0, n_epochs + 1, 0, 8.1])
    plt.scatter(elo, correct_vote_percentage, alpha=0.8)
    # plt.plot(epochs_train[:n_epochs], avg_train_loss[:n_epochs], epochs_val[:n_epochs],avg_val_loss[:n_epochs], linewidth=2)
    # plt.legend(['ELO Rating', 'Percentage of agents that voted correctly [%]'])
    # plt.title('Percentage of agents that voted for the correct\nfirst move when solving the puzzle',fontsize="x-large", fontweight='bold')
    plt.title('Accuracy of the Agent\'s MoveVotes for the\nFirst Move in the Solution to the Chess Puzzle',fontsize="x-large", fontweight='bold')
    plt.ylabel('MoveVote Accuracy of Agents on Team [%]',fontsize="large", fontweight='bold')
    plt.xlabel('Puzzle Rating (ELO)',fontsize="large", fontweight='bold')
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + model_name + '_plot_elo_vs_first_move_correct_vote_percentage.png')
    plt.savefig(save_file)

    
    # Save dict with all the puzzles that were partially/completely solved by some agents on the team:
    print("\n\nPUZZLE PARTIALLY/COMPLETELY SOLVED BY SOME/ALL AGENTS ON THE TEAM:")
    for key, item in model_results_for_partially_or_completely_solved_puzzles.items():
        print(f'{key}:\t{item}')

    partially_solved_puzzle_results_filename = test_result_filename_prefix + model_name + "_checkmate_in_2_AGENTS_PARTIAL_SOLVE.json"
    with open(puzzle_test_data_dir + partially_solved_puzzle_results_filename, 'w') as results_file_agents_partial_solve:
        json.dump(model_results_for_partially_or_completely_solved_puzzles, results_file_agents_partial_solve)