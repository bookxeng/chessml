from chessml import Board, WHITE, BLACK
from chessml.players import parse_move


def play(board, *moves):
    for text in moves:
        board.push(parse_move(board, text))
    return board


def test_fools_mate():
    board = play(Board(), "f3", "e5", "g4", "Qh4")
    out = board.outcome()
    assert out.reason == "checkmate"
    assert out.result == "0-1"
    assert out.winner == BLACK


def test_scholars_mate():
    board = play(Board(), "e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7")
    out = board.outcome()
    assert (out.reason, out.winner) == ("checkmate", WHITE)


def test_stalemate():
    board = Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    out = board.outcome()
    assert out.reason == "stalemate"
    assert out.result == "1/2-1/2"


def test_game_in_progress():
    assert Board().outcome() is None


def test_fifty_move_rule():
    board = Board("8/8/8/4k3/8/8/4K3/4R3 w - - 99 80")
    assert board.outcome() is None
    play(board, "Ra1")
    assert board.outcome().reason == "fifty_moves"


def test_checkmate_beats_fifty_move_rule():
    board = Board("7k/8/6K1/8/8/8/8/R7 w - - 99 80")
    play(board, "Ra8")
    assert board.outcome().reason == "checkmate"


def test_threefold_repetition():
    board = Board()
    shuffle = ["Nf3", "Nf6", "Ng1", "Ng8"]
    play(board, *shuffle)
    assert board.outcome() is None       # start position seen twice
    play(board, *shuffle)
    assert board.outcome().reason == "threefold_repetition"


def test_repetition_needs_same_castling_rights():
    # Rook moves out and back: same placement, but castling rights were lost.
    board = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    play(board, "Rb1", "Rb8", "Ra1", "Ra8", "Rb1", "Rb8", "Ra1", "Ra8")
    # Initial position (with rights) seen once; the rightless one seen twice.
    assert board.outcome() is None


def test_insufficient_material():
    assert Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1").outcome().reason == "insufficient_material"
    assert Board("8/8/8/4k3/8/8/4K3/5N2 w - - 0 1").outcome().reason == "insufficient_material"
    assert Board("8/8/8/4k3/8/8/4K3/5B2 w - - 0 1").outcome().reason == "insufficient_material"
    # Bishops on the same color (f1 and c8 are both light squares).
    assert Board("2b5/8/8/4k3/8/8/4K3/5B2 w - - 0 1").outcome().reason == "insufficient_material"


def test_sufficient_material():
    for fen in [
        "8/8/8/4k3/8/8/4K3/4R3 w - - 0 1",   # rook
        "8/8/8/4k3/8/8/3PK3/8 w - - 0 1",    # pawn
        "8/8/8/4k3/8/8/4K3/4NN2 w - - 0 1",  # two knights (mate possible with help)
        "1b6/8/8/4k3/8/8/4K3/5B2 w - - 0 1", # opposite-colored bishops
        "8/8/8/4k3/8/8/4K3/4BN2 w - - 0 1",  # bishop + knight
    ]:
        assert Board(fen).outcome() is None, fen


def test_castling_through_check_is_illegal():
    # Black rook on f8 attacks f1: white may castle queenside but not kingside.
    board = Board("5r1k/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    ucis = {m.uci() for m in board.legal_moves()}
    assert "e1c1" in ucis
    assert "e1g1" not in ucis


def test_castling_out_of_check_is_illegal():
    board = Board("4r2k/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    ucis = {m.uci() for m in board.legal_moves()}
    assert "e1c1" not in ucis and "e1g1" not in ucis


def test_en_passant():
    board = play(Board(), "e4", "a6", "e5", "d5")
    assert board.ep_square is not None
    move = parse_move(board, "exd6")
    board.push(move)
    assert board.fen().startswith("rnbqkbnr/1pp1pppp/p2P4/8/8/8/PPPP1PPP/RNBQKBNR b")


def test_en_passant_expires():
    board = play(Board(), "e4", "a6", "e5", "d5", "h3", "h6")
    assert "e5d6" not in {m.uci() for m in board.legal_moves()}


def test_promotion_choices():
    board = Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    ucis = {m.uci() for m in board.legal_moves()}
    assert {"a7a8q", "a7a8r", "a7a8b", "a7a8n"} <= ucis
