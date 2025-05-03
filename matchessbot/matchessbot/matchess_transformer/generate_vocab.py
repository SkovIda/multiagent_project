import os
import sys

import argparse
import pathlib

def _parse_args():
    parser = argparse.ArgumentParser(
        prog='GenerateVocab',
        description='Generate output vocabulary (UCI Moves) for Multi-Agent Team Chess Transformer')
    
    parser.add_argument('--vocab_dir', type=str, default="vocabs",
                        help='directory of the generated vocabulary file')
    parser.add_argument('--vocab_filename', type=str, default='uci_move_vocab.txt')
    args = parser.parse_args()
    return args

'''
The following code was taken from: https://github.com/stone50/all-chess-moves/blob/main/generator.py
Minor adaptations have been made to save the vocabulary in a format that is suitable to use as a vocabulary for training a Transformer model model.
'''

FILES = [
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h'
]

PROMOTION_PIECES = {
    'n',
    'b',
    'r',
    'q'
}

moves = []


def constructMoveString(startFile, startRank, endFile, endRank):
    return FILES[startFile] + str(startRank + 1) + FILES[endFile] + str(endRank + 1)


def addSlidingMoves(file, rank):
    upOffsetMax = 8 - rank
    downOffsetMax = rank + 1
    rightOffsetMax = 8 - file
    leftOffsetMax = file + 1

    offsetMaxGrid = [
        [min(downOffsetMax, leftOffsetMax), leftOffsetMax, min(upOffsetMax, leftOffsetMax)],
        [downOffsetMax, 0, upOffsetMax],
        [min(downOffsetMax, rightOffsetMax), rightOffsetMax, min(upOffsetMax, rightOffsetMax)]
    ]

    for fileDirection in range(-1, 2):
        for rankDirection in range(-1, 2):
            for offset in range(1, offsetMaxGrid[fileDirection + 1][rankDirection + 1]):
                moves.append(constructMoveString(file, rank, file + (offset * fileDirection), rank + (offset * rankDirection)))


def addKnightMoves(file, rank):
    rank0 = (rank == 0)
    rank1 = (rank == 1)
    rank6 = (rank == 6)
    rank7 = (rank == 7)
    file0 = (file == 0)
    file1 = (file == 1)
    file6 = (file == 6)
    file7 = (file == 7)

    # up 2 left 1
    if not (rank7 or rank6 or file0):
        moves.append(constructMoveString(file, rank, file - 1, rank + 2))

    # up 2 right 1
    if not (rank7 or rank6 or file7):
        moves.append(constructMoveString(file, rank, file + 1, rank + 2))

    # up 1 right 2
    if not (rank7 or file7 or file6):
        moves.append(constructMoveString(file, rank, file + 2, rank + 1))

    # down 1 right 2
    if not (rank0 or file6 or file7):
        moves.append(constructMoveString(file, rank, file + 2, rank - 1))

    # down 2 right 1
    if not (rank0 or rank1 or file7):
        moves.append(constructMoveString(file, rank, file + 1, rank - 2))

    # down 2 left 1
    if not (rank0 or rank1 or file0):
        moves.append(constructMoveString(file, rank, file - 1, rank - 2))

    # down 1 left 2
    if not (rank0 or file0 or file1):
        moves.append(constructMoveString(file, rank, file - 2, rank - 1))

    # up 1 left 2
    if not (rank7 or file0 or file1):
        moves.append(constructMoveString(file, rank, file - 2, rank + 1))


def addPieceMoves():
    for rank in range(8):
        for file in range(8):
            addSlidingMoves(file, rank)
            addKnightMoves(file, rank)


def addCastleMoves():
    moves.append("O-O")
    moves.append("O-O-O")


def addPromotionMoves():
    for promotionPiece in PROMOTION_PIECES:
        for file in range(8):
            for fileDelta in range(-1, 2):
                newFile = file + fileDelta
                if 0 <= newFile <= 7:
                    moves.append(constructMoveString(file, 6, newFile, 7) + promotionPiece)
                    moves.append(constructMoveString(file, 1, newFile, 0) + promotionPiece)


def generate_uci_move_vocab():
    addPieceMoves()
    # addCastleMoves()
    addPromotionMoves()

    # print(moves)
    args = _parse_args()
    # os.makedirs(args.vocab_dir, exist_ok=True)
    
    os.makedirs(args.vocab_dir, exist_ok=True)
    # pathlib.Path(args.vocab_dir).mkdir(parents=True, exist_ok=True)
    vocab_filepath = os.path.join(args.vocab_dir, args.vocab_filename) 
    # vocab_filepath = pathlib.Path.joinpath(args.vocab_dir, args.vocab_filename) #'vocabs/uci_vocab.txt'

    # with open(vocab_filepath, 'w') as outfile:
    #     print(f'deleted the content of vocab file: {vocab_filepath}')
    try:
        with open(vocab_filepath, 'x') as outfile:
            for move in moves:
                entry = move
                outfile.write(entry)
                outfile.write('\n')
            print(f'Generated the uci vocab file: {vocab_filepath}')
    except:
        user_input = input(f'{vocab_filepath} already exists...\nDo you want to overwrite it? [y/n]')
        if user_input == 'y':
            with open(vocab_filepath, 'w') as outfile:
                print(f'Deleted the content of vocab file: {vocab_filepath}')
                for move in moves:
                    entry = move
                    outfile.write(entry)
                    outfile.write('\n')
                print(f'Re-generated the uci vocab file: {vocab_filepath}')    


if __name__ == '__main__':
    generate_uci_move_vocab()