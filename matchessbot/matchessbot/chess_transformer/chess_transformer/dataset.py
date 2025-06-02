import os
import json
import torch
import argparse
import tables as tb
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, data_folder, h5_file, split, n_moves=None, n_datapoints=98304,**unused):
        """
        Init.

        Args:

            datasets (list): A list of tuples, each containing:

                (dict): A data configuration representing the
                dataset.

                (str): The data split. One of "train", "val", or
                None, which means that all datapoints will be
                included.

            n_moves (int, optional): Number of moves into the future to
            return. Defaults to None, which means that all moves in the
            H5 data column will be returned.
        """
        # Number of datapoints in the dataset to include in the training:
        self.n_train_datapoints = n_datapoints
        # n validation datapoints for a split percentage of 0.8 for training and 0.2 for validation:
        self.n_valid_datapoints = int(n_datapoints * 0.25)

        print(f'\nInit dataset...\n\ttrain dataset size:\t{self.n_train_datapoints}\n\tvalid dataset size:\t{self.n_valid_datapoints}')

        if n_moves is not None:
            assert n_moves > 0

        # Open table in H5 file
        self.h5_file = tb.open_file(os.path.join(data_folder, h5_file), mode="r")
        self.encoded_table = self.h5_file.root.encoded_data
        self.split = split

        # Create indices
        if split == "train":
            self.first_index = 0
        elif split == "val":
            self.first_index = self.encoded_table.attrs.val_split_index
        elif split is None:
            self.first_index = 0
        else:
            raise NotImplementedError

        # How many moves should be returned per row? Remember, there's a
        # "<move>" token prepended to each row's move sequence in the H5
        # data -- subtract 1 to get MAX_MOVE_SEQUENCE_LENGTH
        if n_moves is not None:
            # This is the same as min(MAX_MOVE_SEQUENCE_LENGTH, n_moves)
            self.n_moves = min(
                len(self.encoded_table[self.first_index]["moves"]) - 1, n_moves
            )
        else:
            self.n_moves = len(self.encoded_table[self.first_index]["moves"]) - 1

    def __getitem__(self, i):
        turns = torch.IntTensor([self.encoded_table[self.first_index + i]["turn"]])
        white_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["white_kingside_castling_rights"]]
        )  # (1)
        white_queenside_castling_rights = torch.IntTensor(
            [
                self.encoded_table[self.first_index + i][
                    "white_queenside_castling_rights"
                ]
            ]
        )  # (1)
        black_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["black_kingside_castling_rights"]]
        )  # (1)
        black_queenside_castling_rights = torch.IntTensor(
            [
                self.encoded_table[self.first_index + i][
                    "black_queenside_castling_rights"
                ]
            ]
        )  # (1)
        board_position = torch.IntTensor(
            self.encoded_table[self.first_index + i]["board_position"]
        )  # (64)
        moves = torch.LongTensor(
            self.encoded_table[self.first_index + i]["moves"][: self.n_moves + 1]
        )  # (n_moves + 1)
        length = torch.LongTensor(
            [self.encoded_table[self.first_index + i]["length"]]
        ).clamp(
            max=self.n_moves
        )  # (1), value <= n_moves

        return {
            "turns": turns,
            "white_kingside_castling_rights": white_kingside_castling_rights,
            "white_queenside_castling_rights": white_queenside_castling_rights,
            "black_kingside_castling_rights": black_kingside_castling_rights,
            "black_queenside_castling_rights": black_queenside_castling_rights,
            "board_positions": board_position,
            "moves": moves,
            "lengths": length,
        }

    def __len__(self):
        if self.split == "train":
            if self.n_train_datapoints < self.encoded_table.attrs.val_split_index:
                return self.n_train_datapoints
            else:
                return self.encoded_table.attrs.val_split_index
        elif self.split == "val":
            if self.n_valid_datapoints < (self.encoded_table.nrows - self.encoded_table.attrs.val_split_index):
                return self.n_valid_datapoints
            else:
                return self.encoded_table.nrows - self.encoded_table.attrs.val_split_index
        elif self.split is None:
            return self.encoded_table.nrows
        else:
            raise NotImplementedError
