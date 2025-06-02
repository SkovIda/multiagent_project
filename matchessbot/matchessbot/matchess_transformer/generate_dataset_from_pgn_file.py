import os
import sys

import argparse
import pathlib

from tqdm import tqdm

import copy

import json

import chess
import chess.pgn

from data.process_pgn_data import GamePGN, Reward

import numpy as np

def _parse_args():
    parser = argparse.ArgumentParser(
        prog='GenerateMATChessDataset',
        description='Generate training data for Multi-Agent Team Chess (MATChess) Transformer.')
    
    parser.add_argument('--dataset_dir', type=str, default="dataset",
                        help='directory to save the generated dataset')
    parser.add_argument('--dataset_filename', type=str, help='the filename of the generated dataset')
    parser.add_argument('--pgn_dir', type=str, default="data/lichess_elite_database",
                        help='directory of the PGN files')
    parser.add_argument('--pgn_file', type=str, default="lichess_elite_2025-02.pgn",
                        help='name of the PGN file')
    parser.add_argument('--max_n_games', type=int, default=None, help='the maximum number of games to be extracted from the PGN file')
        # parser.add_argument('--filter_games', type=str, default="checkmate_only", help="the dataset is filtered such that it only contains games that ended in a checkmate")
    parser.add_argument('--compute_returns', type=bool, default=False,
                        help='Set to True if the returns should be included in the dataset in addition to the computed state-action rewards')
    args = parser.parse_args()
    return args

# def create_dataset_from_pgn_file(pgn_file_dir, pgn_file, max_dataset_entries=np.inf):
#     pgn_filename = os.path.join(pgn_file_dir, pgn_file)

#     with open(pgn_filename) as pgn:
#         done_reading_pgn_file = False
#         game_offsets = []

#         all_PGN_games = []

#         # max_dataset_entries = 1    # set max_dataset_entries to inf to process all the games in the PGN file.
#         game_count = 0
#         while not done_reading_pgn_file:

#             if game_count >= max_dataset_entries:
#                 break

#             game_offset = pgn.tell()
            
#             pgn_game = chess.pgn.read_game(pgn)
#             if pgn_game is None:
#                 done_reading_pgn_file = True
#                 break

#             pgn_game_datapoint = GamePGN(pgn_game)
#             # pgn_game_datapoint.print_pgn_info()

#             try:
#                 pgn_game_state_action_reward_state_attr = pgn_game_datapoint.all_reward_attributes_for_game()
#             except AttributeError as err:
#                 print(f"Skipping Game #{game_count} due to AttributeError: {err.args}")
#                 continue

#             game_offsets.append(game_offset)

#             # print("\n\n========== PGN GAME DATA ==========")
#             # pgn_game_datapoint.print_pgn_info()

#             # last_input_state_idx = len(pgn_game_state_action_reward_state_attr) - 1
#             # print(f"Game # {game_count}\nHalfmove = {last_input_state_idx}")
#             # print(f"Game # {game_count}

#             # # # for key, item in pgn_game_state_action_reward_state_attr[last_input_state_idx].items():
#             # # #     print(f'\n{key}:\t{item}')

#             # reward_function = Reward()
#             # pgn_game_rewards = reward_function.get_rewards_from_reward_state_attributes(pgn_game_state_action_reward_state_attr[last_input_state_idx]['reward_state_attributes'])
#             # # print("\nRewards:")
#             # # for key, item in pgn_game_rewards.items():
#             # #     print(f'Reward:\t{key.name}:\t{item}')
#             # # ##############
#             # # # TODO: Convert this to a dataset and save it in a json file:
#             # # # Each entry is a dictonary of pgn_game_info (incl. info to find the state in the orinigal pgn game dataset file), 
#             # # # state-action-rewards dictionary, plus all state-attributes for the input state and the reward state
#             # # ##############
            
#             game_count += 1

#             game_i = {
#                 'pgn_filename': str(pgn_file),
#                 'game_id': str(pgn_game_datapoint.get_game_id()),
#                 'pgn_header_info': pgn_game_datapoint.pgn_header_info(),
#                 'states_action_next_state_attr': pgn_game_state_action_reward_state_attr,
#                 #'actions': pgn_game_state_action_reward_state_attr[:][:]['action']['uci_move'],
#                 # 'rewards': pgn_game_rewards
#             }

#             all_PGN_games.append(copy.deepcopy(game_i))



        
#         print(f'\n\n#Games extracted from PGN file:\t{len(game_offsets)}')

#         return all_PGN_games

def create_dataset_from_pgn_file(pgn_file_dir, pgn_file, max_dataset_entries=None):
    pgn_filename = os.path.join(pgn_file_dir, pgn_file)

    print(pgn_filename)

    with open(pgn_filename) as pgn:
        # done_reading_pgn_file = False
        game_offsets = []

        all_PGN_games = []

        # # max_dataset_entries = 1    # set max_dataset_entries to inf to process all the games in the PGN file.
        game_count = 0
        skip_bot_game_count = 0
        skip_draw_game_count = 0
        skip_black_win_game_count = 0

        print(f"Skimming through game headers in PGN file...")

        while True:
            if max_dataset_entries is not None:
                if game_count >= int(max_dataset_entries):
                    break

            offset = pgn.tell()
            headers = chess.pgn.read_headers(pgn)
            if headers is None:
                break

            if "BOT" in headers.get("WhiteTitle", "?") or "BOT" in headers.get("BlackTitle", "?"):
                # print(f'Skip game #{len(game_offsets) + 1} since at least one of the players are bots')
                skip_bot_game_count +=1
                continue

            if "1/2-1/2" in headers.get("Result", "?"):
                # print(f'Skip game #{len(game_offsets) + 1} since it ends in a draw')
                skip_draw_game_count +=1
                continue
            if "0-1" in headers.get("Result", "?"):
                # print(f'Skip game #{len(game_offsets) + 1} since it ends in a draw')
                skip_black_win_game_count +=1
                continue

            game_offsets.append(offset)
            game_count += 1

        print(f'\nRead {game_count + skip_bot_game_count + skip_draw_game_count + skip_black_win_game_count} chess game headers in PGN file to extract {len(game_offsets)} games for further processing')
        # print(f'\t{len(game_offsets)} of those games were extracted.')
        print(f'\tskipped {skip_bot_game_count} BOT games\n\tskipped {skip_draw_game_count} games that ended in a draw\n\tSkipped {skip_black_win_game_count} games that was won by black')

        game_count = 0
        skip_game_count = 0

        # while not done_reading_pgn_file:
        print("Generating MATChess dataset...")
        for game_offset in tqdm(game_offsets, total=len(game_offsets)):
            # game_offset = pgn.tell()
            pgn.seek(game_offset)
            pgn_game = chess.pgn.read_game(pgn)
            if pgn_game is None:
                # done_reading_pgn_file = True
                break

            if len([move for move in pgn_game.mainline_moves()]) < 2:
                # print(f"Skipping Game #{game_count} due to #moves < 2")
                skip_game_count += 1
                continue

            pgn_game_datapoint = GamePGN(pgn_game)
            # pgn_game_datapoint.print_pgn_info()

            try:
                pgn_game_state_action_reward_state_attr = pgn_game_datapoint.all_reward_attributes_for_game()

            except AttributeError as err:
                print(f"Skipping Game #{game_count} due to AttributeError: {err.args}")
                continue

            # game_offsets.append(game_offset)

            # print("\n\n========== PGN GAME DATA ==========")
            # pgn_game_datapoint.print_pgn_info()

            # last_input_state_idx = len(pgn_game_state_action_reward_state_attr) - 1
            # print(f"Game # {game_count}\nHalfmove = {last_input_state_idx}")
            # print(f"Game # {game_count}

            # # # for key, item in pgn_game_state_action_reward_state_attr[last_input_state_idx].items():
            # # #     print(f'\n{key}:\t{item}')

            # reward_function = Reward()
            # pgn_game_rewards = reward_function.get_rewards_from_reward_state_attributes(pgn_game_state_action_reward_state_attr[last_input_state_idx]['reward_state_attributes'])
            # # print("\nRewards:")
            # # for key, item in pgn_game_rewards.items():
            # #     print(f'Reward:\t{key.name}:\t{item}')
            # # ##############
            # # # TODO: Convert this to a dataset and save it in a json file:
            # # # Each entry is a dictonary of pgn_game_info (incl. info to find the state in the orinigal pgn game dataset file), 
            # # # state-action-rewards dictionary, plus all state-attributes for the input state and the reward state
            # # ##############
            
            game_count += 1

            # game_i = {
            #     'pgn_filename': str(pgn_file),
            #     'game_id': str(pgn_game_datapoint.get_game_id()),
            #     'pgn_header_info': pgn_game_datapoint.pgn_header_info(),
            #     'states_action_next_state_attr': pgn_game_state_action_reward_state_attr,
            #     #'actions': pgn_game_state_action_reward_state_attr[:][:]['action']['uci_move'],
            #     # 'rewards': pgn_game_rewards
            # }
            game_i = {
                'pgn_filename': str(pgn_file),
                'game_id': str(pgn_game_datapoint.get_game_id()),
                'game_offset_in_pgn': game_offset,
                'pgn_header_info': pgn_game_datapoint.pgn_header_info(),
                'states_action_next_state_attr': pgn_game_state_action_reward_state_attr,
                #'actions': pgn_game_state_action_reward_state_attr[:][:]['action']['uci_move'],
                # 'rewards': pgn_game_rewards
            }

            all_PGN_games.append(copy.deepcopy(game_i))

        print(f"#Games included in the final dataset:\tgame_count={game_count}\tlen(all_PGN_games)={len(all_PGN_games)}")
        print(f'\tSkipped {skip_game_count} games due to error: len(pgn_game.moves) < 2')

        return all_PGN_games



def generate_matchess_transformer_dataset():
    args = _parse_args()
    
    # os.makedirs(args.dataset_dir, exist_ok=True)
    dataset_filepath = os.path.join(args.dataset_dir, args.dataset_filename)
    
    ############### Generate dataset with state-action pairs and next-state rewards ######################
    if pathlib.Path(dataset_filepath).exists():
        # dataset_filepath = os.path.join(args.dataset_dir, args.dataset_filename)
        user_input = input(f'{dataset_filepath} already exists...\nDo you want to overwrite it? [y/n]')
        if user_input == 'y':
            
            with open(dataset_filepath, 'w') as outfile:
                print(f'Successfully deleted the content of dataset file: {dataset_filepath}')
                outfile.close()

            pgn_games = create_dataset_from_pgn_file(pgn_file_dir=args.pgn_dir, pgn_file=args.pgn_file, max_dataset_entries=args.max_n_games)
            dataset_entry_count = 0
            dataset_game_count = 0
            for idx, game in tqdm(enumerate(pgn_games)):
                dataset_game_count += 1
                # print(f'\n\nGame #{idx}')
                entry = {
                        'pgn_filename': str(game['pgn_filename']),
                        'game_id': str(game['game_id']),
                        'game_offset_in_pgn': game['game_offset_in_pgn'],
                        'pgn_header_info': game['pgn_header_info'],
                        'state': {},
                        'action': "",
                        'rewards': {} #game['rewards']
                    }
                for state_idx, state_in_game in enumerate(game['states_action_next_state_attr']):
                    # print("\nGame State:")
                    entry['state'] = state_in_game['state']
                    entry['action'] = state_in_game['action']["uci_move"]
                    entry['rewards'] = state_in_game['rewards']
                    with open(dataset_filepath, 'a') as outfile:
                        json.dump(entry, outfile)
                        outfile.write('\n')
                        dataset_entry_count +=1
            print(f'Re-generated the MATChess dataset and saved it as: {dataset_filepath}')
            print(f'\tThe generated dataset contains {dataset_entry_count} states (for the winning player) from {dataset_game_count} games')
    else:
        os.makedirs(args.dataset_dir, exist_ok=True)
        dataset_filepath = os.path.join(args.dataset_dir, args.dataset_filename)
        
        with open(dataset_filepath, 'w') as outfile:
            pgn_games = create_dataset_from_pgn_file(pgn_file_dir=args.pgn_dir, pgn_file=args.pgn_file, max_dataset_entries=args.max_n_games)
            dataset_entry_count = 0
            dataset_game_count = 0
            for idx, game in tqdm(enumerate(pgn_games)):
                dataset_game_count += 1
                # print(f'\n\nGame #{idx}')
                entry = {
                        'pgn_filename': str(game['pgn_filename']),
                        'game_id': str(game['game_id']),
                        'game_offset_in_pgn': game['game_offset_in_pgn'],
                        'pgn_header_info': game['pgn_header_info'],
                        'state': {},
                        'action': "",
                        'rewards': {} #game['rewards']
                    }
                for state_idx, state_in_game in enumerate(game['states_action_next_state_attr']):
                    # print("\nGame State:")
                    entry['state'] = state_in_game['state']
                    entry['action'] = state_in_game['action']["uci_move"]
                    entry['rewards'] = state_in_game['rewards']
                    with open(dataset_filepath, 'a') as outfile:
                        json.dump(entry, outfile)
                        outfile.write('\n')
                        dataset_entry_count +=1
            print(f'Generated the MATChess dataset and saved it as: {dataset_filepath}')
            print(f'\tThe generated dataset contains {dataset_entry_count} states from {len(pgn_games)} games')

        # if args.compute_returns:

        #     if pathlib.Path(dataset_filepath).exists():
        #         # dataset_filepath = os.path.join(args.dataset_dir, args.dataset_filename)
        #         user_input = input(f'{dataset_filepath} already exists...\nDo you want to overwrite it? [y/n]')
        #         if user_input == 'y':
        #             with open(dataset_filepath, 'w') as outfile:
        #                 print(f'Successfully deleted the content of dataset file: {dataset_filepath}')
        #                 outfile.close()
        #         else:
        #             print(f'Exiting program without generating a new dataset...')
        #             return
        #     else:
        #         os.makedirs(args.dataset_dir, exist_ok=True)

        #     with open(dataset_filepath, 'w') as outfile:
        #         print(f'Generating new dataset: {dataset_filepath} ...')

        #         pgn_games = create_dataset_from_pgn_file(pgn_file_dir=args.pgn_dir, pgn_file=args.pgn_file, max_dataset_entries=args.max_n_games)
        #         dataset_entry_count = 0
        #         dataset_game_count = 0
        #         for idx, game in tqdm(enumerate(pgn_games)):
        #             dataset_game_count += 1
        #             # print(f'\n\nGame #{idx}')
        #             game_length = len(game['states_action_next_state_attr'])
        #             print(f'game_length={game_length}')
        #             rewards_to_go = {}
                    
        #             entry = {
        #                     'pgn_filename': str(game['pgn_filename']),
        #                     'game_id': str(game['game_id']),
        #                     'game_offset_in_pgn': game['game_offset_in_pgn'],
        #                     'pgn_header_info': game['pgn_header_info'],
        #                     'state': {},
        #                     'action': "",
        #                     'rewards': {} #game['rewards']
        #                 }
        #             for state_idx, state_in_game in enumerate(game['states_action_next_state_attr']):
        #                 # print("\nGame State:")
        #                 entry['state'] = state_in_game['state']
        #                 entry['action'] = state_in_game['action']["uci_move"]
        #                 entry['rewards'] = state_in_game['rewards']
        #                 with open(dataset_filepath, 'a') as outfile:
        #                     json.dump(entry, outfile)
        #                     outfile.write('\n')
        #                     dataset_entry_count +=1
        #         print(f'Generated the MATChess dataset and saved it as: {dataset_filepath}')
        #         print(f'\tThe generated dataset contains {dataset_entry_count} states from {len(pgn_games)} games')

    #     with open(dataset_filepath, 'w') as outfile:
    #         print(f'Creating MATChess dataset:\t{dataset_filepath}\n from PGN file:\t{args.pgn_file}')
    #         pgn_games = create_dataset_from_pgn_file(pgn_file_dir=args.pgn_dir, pgn_file=args.pgn_file)
    #         dataset_entry_count = 0
    #         for idx, game in tqdm(enumerate(pgn_games)):
    #             # print(f'\n\nGame #{idx}')
    #             entry = {
    #                     'pgn_filename': str(game['pgn_filename']),
    #                     'game_id': str(game['game_id']),
    #                     'game_idx': idx, #game['states_action_next_state_attr'],
    #                     'pgn_header_info': game['pgn_header_info'],
    #                     'state': {},
    #                     'action': "",
    #                     'rewards': {} #game['rewards']
    #                 }
    #             for state_idx, state_in_game in enumerate(game['states_action_next_state_attr']):
    #                 # print("\nGame State:")
    #                 entry['state'] = state_in_game['state']
    #                 entry['action'] = state_in_game['action']["uci_move"]
    #                 entry['rewards'] = state_in_game['rewards']
    #                 json.dump(entry, outfile)
    #                 outfile.write('\n')
    #                 dataset_entry_count +=1
    #         print(f'Generated the MATChess dataset and saved it as: {dataset_filepath}')
    #         print(f'\tThe generated dataset contains {dataset_entry_count} states from {len(pgn_games)} games')
    # ###########################################


    
    # try:
    #     with open(dataset_filepath, 'x') as outfile:
    #         # TODO: Write dataset entries to file
    #         os.makedirs(args.dataset_dir, exist_ok=True)
    #         pgn_games = create_dataset_from_pgn_file(pgn_file_dir=args.pgn_dir, pgn_file=args.pgn_file)
    #         print(type(pgn_games))
    #         # for game in pgn_games:
    #         #     for key, state_action_reward in game.items():
    #         #         entry = state_action_reward
    #         #         json.dump(entry, outfile)
    #         #         outfile.write('\n')
    #         # print(f'Generated the MATChess Transformer dataset: {dataset_filepath}')
    # except:
    #     user_input = input(f'{dataset_filepath} already exists...\nDo you want to overwrite it? [y/n]')
    #     if user_input == 'y':
    #         with open(dataset_filepath, 'w') as outfile:
    #             print(f'Deleted the content of dataset file: {dataset_filepath}')
    #             pgn_games = create_dataset_from_pgn_file(pgn_file_dir=args.pgn_dir, pgn_file=args.pgn_file)
    #             # print(type(pgn_games))
    #             # for game in pgn_games:
    #             #     for key, state_action_reward in game.items():
    #             #         entry = state_action_reward
    #             #         json.dump(entry, outfile)
    #             #         outfile.write('\n')
    #             # print(f'Re-generated the dataset and saved it as: {dataset_filepath}')



def create_selfplay_dataset_from_pgn_file(pgn_file_dir, pgn_file, max_dataset_entries=None):
    pgn_filename = os.path.join(pgn_file_dir, pgn_file)

    print(pgn_filename)

    with open(pgn_filename) as pgn:
        # done_reading_pgn_file = False
        game_offsets = []

        all_PGN_games = []

        # # max_dataset_entries = 1    # set max_dataset_entries to inf to process all the games in the PGN file.
        game_count = 0
        skip_bot_game_count = 0
        skip_draw_game_count = 0
        skip_black_win_game_count = 0

        print(f"Skimming through game headers in PGN file...")

        while True:
            if max_dataset_entries is not None:
                if game_count >= int(max_dataset_entries):
                    break

            offset = pgn.tell()
            headers = chess.pgn.read_headers(pgn)
            if headers is None:
                break

            # if "BOT" in headers.get("WhiteTitle", "?") or "BOT" in headers.get("BlackTitle", "?"):
            #     # print(f'Skip game #{len(game_offsets) + 1} since at least one of the players are bots')
            #     skip_bot_game_count +=1
            #     continue

            # if "1/2-1/2" in headers.get("Result", "?"):
            #     # print(f'Skip game #{len(game_offsets) + 1} since it ends in a draw')
            #     skip_draw_game_count +=1
            #     continue
            # if "0-1" in headers.get("Result", "?"):
            #     # print(f'Skip game #{len(game_offsets) + 1} since it ends in a draw')
            #     skip_black_win_game_count +=1
            #     continue

            game_offsets.append(offset)
            game_count += 1

        print(f'\nRead {game_count + skip_bot_game_count + skip_draw_game_count + skip_black_win_game_count} chess game headers in PGN file to extract {len(game_offsets)} games for further processing')
        # print(f'\t{len(game_offsets)} of those games were extracted.')
        print(f'\tskipped {skip_bot_game_count} BOT games\n\tskipped {skip_draw_game_count} games that ended in a draw\n\tSkipped {skip_black_win_game_count} games that was won by black')

        game_count = 0
        skip_game_count = 0

        # while not done_reading_pgn_file:
        print("Generating MATChess dataset...")
        print(game_offsets)
        #for idx, game_offset in tqdm(game_offsets, total=len(game_offsets)):
        for idx, game_offset in tqdm(enumerate(game_offsets)):
            # game_offset = pgn.tell()
            pgn.seek(game_offset)
            pgn_game = chess.pgn.read_game(pgn)
            if pgn_game is None:
                # done_reading_pgn_file = True
                break

            if len([move for move in pgn_game.mainline_moves()]) < 2:
                # print(f"Skipping Game #{game_count} due to #moves < 2")
                skip_game_count += 1
                continue

            pgn_game_datapoint = GamePGN(pgn_game)
            # pgn_game_datapoint.print_pgn_info()

            try:
                pgn_game_state_action_reward_state_attr = pgn_game_datapoint.all_reward_attributes_for_game(only_winning_players_moves=False)
                # print(pgn_game_state_action_reward_state_attr)

            except AttributeError as err:
                print(f"Skipping Game #{game_count} due to AttributeError: {err.args}")
                continue
            
            game_count += 1

            game_i = {
                'pgn_filename': str(pgn_file),
                'game_id': str(pgn_game_datapoint.get_game_id()),
                'game_offset_in_pgn': game_offset,
                'pgn_header_info': pgn_game_datapoint.pgn_header_info(),
                'states_action_next_state_attr': pgn_game_state_action_reward_state_attr,
                #'actions': pgn_game_state_action_reward_state_attr[:][:]['action']['uci_move'],
                # 'rewards': pgn_game_rewards
            }

            all_PGN_games.append(copy.deepcopy(game_i))

        print(f"#Games included in the final dataset:\tgame_count={game_count}\tlen(all_PGN_games)={len(all_PGN_games)}")
        print(f'\tSkipped {skip_game_count} games due to error: len(pgn_game.moves) < 2')

        return all_PGN_games





if __name__ == '__main__':        
    generate_matchess_transformer_dataset()