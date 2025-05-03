import os
import torch
from pathlib import Path
from torch.utils.data import Dataset
from .tokenizer import Tokenizer

import json

from tqdm import tqdm

import copy


class MATChessDataset(Dataset):
    chess_piece_agent_ids = ['king', 'queen', 'rook0', 'rook7', 'knight1', 'knight6', 'bishop2', 'bishop5', 'pawn0', 'pawn1', 'pawn2', 'pawn3', 'pawn4', 'pawn5', 'pawn6', 'pawn7']
    reward_types_len = 11
    chess_piece_agent_reward_weights = [[1.0] * reward_types_len] * len(chess_piece_agent_ids)
    
    def __init__(self, tokenizer: Tokenizer, dataset_path: str, n_datapoints=None, **unused):
        self.tokenizer = tokenizer
        self.games = []

        print(f'Loading dataset from file: {dataset_path}...')

        n_datapoint_counter = 0

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                dict_entry = {
                        'pgn_filename': None,
                        'game_id': None,
                        'game_offset_in_pgn': None,
                        'pgn_header_info': None,
                        'state': {},
                        'action': "",
                        'rewards': {}
                    }
                dict_entry = json.loads(line)
                self.games.append(copy.deepcopy(self.tokenizer.encode_pgn_dataset_entry(dict_entry, chess_piece_agent_ids=self.chess_piece_agent_ids, chess_piece_agent_reward_weights=self.chess_piece_agent_reward_weights)))
                
                n_datapoint_counter += 1
                if n_datapoints is not None:
                    if n_datapoint_counter >= n_datapoints:
                        print(f"Reached maximum number of entries in dataset: n_datapoints={n_datapoints}.")
                        break

        print("Done loading dataset.")

    def __len__(self):
        return len(self.games)

    def __getitem__(self, i):
        # game = self.games[i]
        
        # encoded_input = self.tokenizer.encode_model_input(game['state']['fen'], self.chess_piece_agent_ids, game['state']['agent_pos_square_idx'], self.chess_piece_agent_reward_weights)

        # # # TODO: add target_move and target_rewards to the output of __getitem__()
        # print(game['action'])
        # encoded_output = self.tokenizer.encode_model_targets(game['action'], game['rewards'], self.chess_piece_agent_reward_weights)

        # return encoded_input | encoded_output
        return self.games[i]


if __name__ == '__main__':
    tokenizer = Tokenizer()

    dataset = MATChessDataset(tokenizer=tokenizer, dataset_path="dataset/gen_dataset_test.json")

    print("\n\n ===== item in dataset =====")
    first_item_in_dataset = dataset.__getitem__(0)
    print(first_item_in_dataset)

    last_item_in_dataset = dataset.__getitem__(dataset.__len__() - 1)
    print(last_item_in_dataset)