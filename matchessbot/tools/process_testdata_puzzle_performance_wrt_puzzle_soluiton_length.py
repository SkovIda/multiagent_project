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
    
    elo_range_min = 1001
    elo_range_max = 3000
    test_result_filename_prefix = "compare_SIL_IRL_models_elo_" + str(elo_range_min) + " _" + str(elo_range_max) + "_"

    # model_checkpoint_filename = "checkpoint_epoch_60_MATChessFormer-Homogeneous-20.pt"
    # model_name =  "SIL"

    model_checkpoint_filename_SIL = "checkpoint_epoch_60_MATChessFormer-Homogeneous-20.pt"
    model_checkpoint_filename_IRL = "checkpoint_epoch_75_MATChessFormer-ImitativeRL-20.pt"

    model_name_SIL =  "SIL"
    print(f"\nModel:\t{model_name_SIL}\ncheckpoint:\t{model_checkpoint_filename_SIL}")

    model_name_IRL =  "IRL"
    print(f"\nModel:\t{model_name_IRL}\ncheckpoint:\t{model_checkpoint_filename_IRL}")

    puzzle_test_data_dir = "test_data/puzzles/"
    # results_filename = "puzzle_test_results_" + model_name + "_checkmate_in_2.json"
    # results_filename = test_result_filename_prefix + "_checkmate_in_2.json"

    # results_csv_elo_vs_correct_agent_votes = "puzzle_test_results_" + model_name + "_checkmate_in_2_elo_0_600_wrt_correct_agent_vote_percentage.csv"
    results_csv_elo_vs_correct_agent_votes = test_result_filename_prefix + "_elo_wrt_same_agent_vote_percentage.csv"

    # with open(puzzle_test_data_dir + results_filename, 'w') as results_file:
    #     results_file.close()

    with open(puzzle_test_data_dir + results_csv_elo_vs_correct_agent_votes, 'w') as f:
        csv_writer = csv.writer(f,delimiter=',')
        csv_writer.writerow(['elo', 'checkmate_in', 'puzzle_solved_SIL' 'puzzle_solved_IRL','puzzle_firstmove_accuracy_SIL' 'puzzle_firstmove_accuracy_IRL', 'same_firstmove_agent_votes_percentage'])
        f.close()

    model_results_for_partially_or_completely_solved_puzzles_SIL = {}
    model_results_for_partially_or_completely_solved_puzzles_IRL = {}

    # Run "puzzle games" tests:
    for dataset_idx, puzzle_dataset_filename in enumerate(puzzle_dataset_filenames):
        with open(puzzle_dataset_dir + puzzle_dataset_filename + ".csv") as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)

        unix_timestamp = unix_timestamps[dataset_idx] #int(time.time())
        test_result_dir_name_SIL =  "test_result_" + model_name_SIL + "_" + puzzle_dataset_filename + "_" + str(unix_timestamp)
        test_result_dir_name_IRL =  "test_result_" + model_name_IRL + "_" + puzzle_dataset_filename + "_" + str(unix_timestamp)

        for puzzle_idx, puzzle_data in tqdm(enumerate(rows)):
            if puzzle_idx == 0:
                continue
            elif puzzle_idx > 15:
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

            
            # with open(puzzle_test_data_dir + results_filename, 'a') as puzzle_test_result_file:
            #     json.dump(puzzle_result, puzzle_test_result_file)
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

            with open(puzzle_test_data_dir + results_csv_elo_vs_correct_agent_votes, 'a') as f:
                csv_writer = csv.writer(f,delimiter=',')
                first_move_correct_votes_percentage_SIL = (puzzle_result_SIL['n_agents_voted_correctly'][0] / puzzle_result_SIL['n_agents'][0])  * 100.0

                first_move_correct_votes_percentage_IRL = (puzzle_result_IRL['n_agents_voted_correctly'][0] / puzzle_result_IRL['n_agents'][0])  * 100.0

                # if first_move_correct_votes_percentage > 0:
                #     print("\n\nSOME AGENTS ON THE TEAM VOTED FOR THE CORRECT MOVE TO SOLVE THE PUZZLE.\n\tPrinting the puzzle_result...\n")
                #     model_results_for_partially_or_completely_solved_puzzles_SIL[puzzle_result_SIL['puzzle_id']] = copy.deepcopy(puzzle_result_SIL)
                #     for key, item in puzzle_result_SIL.items():
                #         print(f'{key}:\t{item}')

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

    elo, checkmate_in, solved_sil, solved_irl, correct_vote_percentage_sil, correct_vote_percentage_irl, same_firstmove_vote_percentage = get_puzzle_results_from_csv(puzzle_test_data_dir, results_csv_elo_vs_correct_agent_votes)

    print(checkmate_in)

    # boxplot_data_accuracy_sil = [[], [], [], []]
    # for idx, checkmate_in in enumerate(checkmate_in):
    #     boxplot_data_accuracy_sil[int(checkmate_in)-1].append(int(correct_vote_percentage_sil[idx]))

    # boxplot_data_accuracy_irl = [[], [], [], []]
    # for idx, checkmate_in in enumerate(checkmate_in):
    #     boxplot_data_accuracy_irl[int(checkmate_in)-1].append(int(correct_vote_percentage_irl[idx]))
    
    # print(boxplot_data_accuracy_sil)
    # print(boxplot_data_accuracy_irl)
    # print([boxplot_data_accuracy_sil, boxplot_data_accuracy_irl])

    plt.figure(1)
    plt.grid()
    # plt.scatter(elo, correct_vote_percentage_sil, c=checkmate_in, alpha=0.5, label='Standard IL')
    plt.scatter(elo, correct_vote_percentage_sil, alpha=0.8) #, label='Standard IL')
    # plt.scatter(elo, same_firstmove_vote_percentage, c=checkmate_in, alpha=0.5, label=['1', '2', '3', '4'])
    # plt.legend('Standard IL')
    # # plt.boxplot(boxplot_data_accuracy_sil)
    # # plt.boxplot(boxplot_data_accuracy_irl)
    # # plt.boxplot(boxplot_data_accuracy_sil)
    plt.title('First Move Accuracy for the\nStandard Imitation Learning Model',fontsize="x-large", fontweight='bold')
    plt.ylabel('MoveVote Accuracy of Agents [%]',fontsize="large", fontweight='bold')
    plt.xlabel('Puzzle Rating (ELO)',fontsize="large", fontweight='bold')
    # plt.xlabel('Solution Length [fullmove count]',fontsize="large", fontweight='bold')

    # plt.plot(checkmate_in, correct_vote_percentage_sil, correct_vote_percentage_irl)
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_plot_elo_vs_first_move_correct_vote_percentage_SIL.png')
    plt.savefig(save_file)



    plt.figure(2)
    plt.grid()
    plt.scatter(elo, correct_vote_percentage_irl, alpha=0.8) #, label=['Imitative RL'])
    # plt.legend('Imitative RL')
    plt.title('First Move Accuracy for the\nImitative Reinforcement Learning Model',fontsize="x-large", fontweight='bold')
    plt.ylabel('MoveVote Accuracy of Agents [%]',fontsize="large", fontweight='bold')
    plt.xlabel('Puzzle Rating (ELO)',fontsize="large", fontweight='bold')
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_plot_elo_vs_first_move_correct_vote_percentage_IRL.png')
    plt.savefig(save_file)

    boxplot_data = [[], [], [], []]
    for idx, c in enumerate(checkmate_in):
        boxplot_data[int(c)-1].append(int(elo[idx]))
    plt.figure(3)
    plt.grid()
    # plt.scatter(elo, checkmate_in, alpha=0.5, label='checkmate in N fullmoves')
    # plt.scatter(checkmate_in, correct_vote_percentage_irl, alpha=0.5, label='Imitative RL')
    # plt.bar(checkmate_in, correct_vote_percentage_sil, correct_vote_percentage_irl)
    plt.boxplot(boxplot_data)
    plt.title('Elo Rating wrt. Solution Length',fontsize="x-large", fontweight='bold')
    plt.ylabel('Solution Length [fullmove count]',fontsize="large", fontweight='bold')
    plt.xlabel('Puzzle Rating [ELO]',fontsize="large", fontweight='bold')    
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_plot_elo_vs_length_boxplot.png')
    plt.savefig(save_file)



    accuracy_barplot_data = [[], []]
    accuracy_vals = [[], []]
    for elo_idx, elo_rating in enumerate(elo):
        if correct_vote_percentage_sil[elo_idx] > 50:
            accuracy_barplot_data[0].append(int(elo_rating))
            accuracy_vals[0].append(int(correct_vote_percentage_sil[elo_idx]))
        if correct_vote_percentage_irl[elo_idx] > 50:
            accuracy_barplot_data[1].append(int(elo_rating))
            accuracy_vals[1].append(int(correct_vote_percentage_irl[elo_idx]))

    print("Accuracy bar plot data (elo, accuracy):")
    print(accuracy_barplot_data)
    print(accuracy_vals)


    plt.figure(4)
    # plt.grid()
    # plt.scatter(elo, checkmate_in, alpha=0.5, label='checkmate in N fullmoves')
    # plt.scatter(checkmate_in, correct_vote_percentage_irl, alpha=0.5, label='Imitative RL')
    # plt.bar(checkmate_in, correct_vote_percentage_sil, correct_vote_percentage_irl)
    # plt.boxplot(accuracy_barplot_data)

    # plt.hist(accuracy_barplot_data, histtype='bar', label=['Standard IL', 'Imitative RL'])
    plt.hist(accuracy_barplot_data, weights=accuracy_vals, histtype='bar', label=['Standard IL', 'Imitative RL'])
    plt.legend(['Standard IL', 'Imitative RL'])
    plt.title('First Move Accuracy wrt. Elo Rating',fontsize="x-large", fontweight='bold')
    plt.ylabel('First Mode accuracy [%]',fontsize="large", fontweight='bold')
    plt.xlabel('Puzzle Rating [ELO]',fontsize="large", fontweight='bold')    
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_plot_first_move_accuracy_wrt_elo.png')
    plt.savefig(save_file)



    elo_rating_histogram_wrt_solution_length = [[], [], [], []]
    for idx, c_in in enumerate(checkmate_in):
        elo_rating_histogram_wrt_solution_length[int(c_in)-1].append(int(elo[idx]))

    
    plt.figure(5)
    plt.grid()
    # plt.scatter(elo, checkmate_in, alpha=0.5, label='checkmate in N fullmoves')
    # plt.scatter(checkmate_in, correct_vote_percentage_irl, alpha=0.5, label='Imitative RL')
    # plt.bar(checkmate_in, correct_vote_percentage_sil, correct_vote_percentage_irl)
    plt.hist(elo_rating_histogram_wrt_solution_length, histtype='bar')
    plt.title('Elo Rating wrt. Solution Length',fontsize="x-large", fontweight='bold')
    plt.ylabel('Puzzle Rating [ELO]',fontsize="large", fontweight='bold')
    plt.xlabel('Solution Length [fullmove count]',fontsize="large", fontweight='bold')    
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_plot_elo_dist_vs_length.png')
    plt.savefig(save_file)


    model_accuracy_wrt_elo_ratings = [[], [], [], [], []]
    model_accuracy_wrt_elo_ratings_bin_edges = [1500, 1600, 1700, 1800, 1900, 2000]

    for idx_elo, elo_rating in enumerate(elo):
        for idx_bin, edge_val in enumerate(model_accuracy_wrt_elo_ratings_bin_edges[1:]):
            if elo_rating <= edge_val:
                model_accuracy_wrt_elo_ratings[idx_bin].append(same_firstmove_vote_percentage[idx_elo])
                break

    #model_accuracy_wrt_elo_ratings = [correct_vote_percentage_sil, correct_vote_percentage_irl]

    print(model_accuracy_wrt_elo_ratings)
    heights = []
    for list_i in model_accuracy_wrt_elo_ratings:
        heights.append(np.mean(np.array(list_i)))

    print(np.array(heights))



    plt.figure(6)
    plt.grid()
    # plt.scatter(elo, checkmate_in, alpha=0.5, label='checkmate in N fullmoves')
    # plt.scatter(checkmate_in, correct_vote_percentage_irl, alpha=0.5, label='Imitative RL')
    # plt.bar(checkmate_in, correct_vote_percentage_sil, correct_vote_percentage_irl)
    #plt.hist(elo_rating_histogram_wrt_solution_length, histtype='bar')
    # plt.hist(same_firstmove_vote_percentage, histtype='bar')    
    
    # plt.bar(elo, same_firstmove_vote_percentage, width=2.0)   # Works somewhat

    # data, bins = np.histogram(same_firstmove_vote_percentage)
    # plt.hist(data, bins=bins, histtype='bar')

    plt.hist(same_firstmove_vote_percentage, histtype='bar')  # Works somewhat better?
    
    
    # plt.bar(model_accuracy_wrt_elo_ratings_bin_edges[1:], heights)
    # plt.hist(model_accuracy_wrt_elo_ratings, bins=model_accuracy_wrt_elo_ratings_bin_edges, histtype='bar')
    
    # plt.hist(model_accuracy_wrt_elo_ratings)
    #plt.xticks(np.arange(6), ['1500','1600', '1700',  '1800', '1900', '2000'])
    plt.title('Accuracy of the two models wrt. Each Other for\nFirst Move in Chess Puzzle Solution',fontsize="x-large", fontweight='bold')
    plt.ylabel('Frequency [%]',fontsize="large", fontweight='bold')
    plt.xlabel('Same agent votes [%]',fontsize="large", fontweight='bold')    
    
    save_file = os.path.join(puzzle_test_data_dir, test_result_filename_prefix + '_plot_same_agent_votes.png')
    plt.savefig(save_file)

    
    # # Save dict with all the puzzles that were partially/completely solved by some agents on the team:
    # print("\n\nPUZZLE PARTIALLY/COMPLETELY SOLVED BY SOME/ALL AGENTS ON THE TEAM:")
    # for key, item in model_results_for_partially_or_completely_solved_puzzles_SIL.items():
    #     print(f'{key}:\t{item}')

    # partially_solved_puzzle_results_filename = test_result_filename_prefix + "_checkmate_in_2_AGENTS_PARTIAL_SOLVE.json"
    # with open(puzzle_test_data_dir + partially_solved_puzzle_results_filename, 'w') as results_file_agents_partial_solve:
    #     json.dump(model_results_for_partially_or_completely_solved_puzzles_SIL, results_file_agents_partial_solve)