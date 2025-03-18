
import typing
import chess


def piece_color_from_str(color_name: str) -> chess.Color:
    if color_name == "white":
        return chess.WHITE
    elif color_name == "black":
        return chess.BLACK
    else:
        return None


def piece_type_from_str(type_name: str) -> chess.PieceType:
    piece_type_enums = {'king': chess.KING, 'queen': chess.QUEEN, 'rook': chess.ROOK, 'bishop': chess.BISHOP, 'knight': chess.KNIGHT, 'pawn': chess.PAWN}
    if type_name in piece_type_enums:
        return piece_type_enums[type_name]
    else:
        return None


def chess_piece(color: str, type: str) -> chess.Piece:
    return chess.Piece(piece_type=piece_type_from_str(type), color=piece_type_from_str(color))


def color_name(color: chess.Color) -> str:
    return typing.cast(str, chess.COLOR_NAMES[color])