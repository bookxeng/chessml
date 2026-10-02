import random

import pytest

from chessml import Board, Move, move_to_san, parse_san
from chessml.players import parse_move


def san(fen, uci):
    return move_to_san(Board(fen), Move.from_uci(uci))


def test_basic():
    assert san(Board().fen(), "e2e4") == "e4"
    assert san(Board().fen(), "g1f3") == "Nf3"


def test_capture_check_mate_promotion():
    assert san("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "e4d5") == "exd5"
    assert san("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", "a1a8") == "Ra8+"
    assert san("7k/8/6K1/8/8/8/8/R7 w - - 0 1", "a1a8") == "Ra8#"
    assert san("8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7a8q") == "a8=Q"
    assert san("8/P6k/8/8/8/8/8/K7 w - - 0 1", "a7a8n") == "a8=N"


def test_castling():
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    assert san(fen, "e1g1") == "O-O"
    assert san(fen, "e1c1") == "O-O-O"


def test_disambiguation():
    # Knights on b1 and f1 can both reach d2: file disambiguates.
    assert san("4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1", "b1d2") == "Nbd2"
    # Rooks on a1 and a5 can both reach a3: rank disambiguates.
    assert san("4k3/8/8/R7/8/8/8/R3K3 w - - 0 1", "a1a3") == "R1a3"
    # Queens on a1, c1 and a3 can all reach b2: needs full square.
    assert san("4k3/8/8/8/8/Q7/8/Q1Q1K3 w - - 0 1", "a1b2") == "Qa1b2"


def test_parse_lenient_forms():
    board = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert parse_san(board, "0-0").uci() == "e1g1"
    assert parse_san(board, "O-O-O").uci() == "e1c1"
    board = Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    assert parse_san(board, "a8Q").uci() == "a7a8q"
    assert parse_san(board, "a8=N").uci() == "a7a8n"


def test_parse_errors():
    with pytest.raises(ValueError):
        parse_san(Board(), "e5")
    with pytest.raises(ValueError):
        parse_san(Board(), "Nd2")  # blocked


def test_parse_move_accepts_uci_and_san():
    board = Board()
    assert parse_move(board, "e2e4") == parse_move(board, "e4")
    board = Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    assert parse_move(board, "a7a8").promotion  # bare UCI promotion defaults to queen


def test_san_round_trip_random_games():
    rng = random.Random(3)
    for _ in range(6):
        board = Board()
        for _ in range(120):
            moves = board.legal_moves()
            if not moves:
                break
            for m in moves:
                assert parse_san(board, move_to_san(board, m, moves)) == m
            board.push(rng.choice(moves))
