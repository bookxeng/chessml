"""Chess rules engine and gym-style environment for training ML agents."""
from .board import Board
from .constants import WHITE, BLACK, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, STARTING_FEN
from .encoding import NUM_ACTIONS, OBS_SHAPE, encode_board, move_to_action, action_to_move, legal_action_mask
from .env import ChessEnv
from .move import Move
from .movegen import legal_moves, perft
from .rules import Outcome, outcome
from .san import move_to_san, parse_san

__all__ = [
    "Board", "Move", "ChessEnv", "Outcome",
    "WHITE", "BLACK", "PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING", "STARTING_FEN",
    "NUM_ACTIONS", "OBS_SHAPE", "encode_board", "move_to_action", "action_to_move", "legal_action_mask",
    "legal_moves", "perft", "outcome", "move_to_san", "parse_san",
]
