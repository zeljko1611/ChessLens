from chessvision.position import (
    STARTING_BOARD_FEN,
    board_to_fen,
    fen_to_board,
    full_fen,
    in_check,
    infer_castling,
    suggest_side_to_move,
    validate,
)


def test_fen_roundtrip():
    for fen in [STARTING_BOARD_FEN, "8/8/8/4k3/8/8/8/4K3", "r3k2r/pp1n1ppp/2p5/8/3P4/2N5/PP3PPP/R3K2R"]:
        assert board_to_fen(fen_to_board(fen)) == fen


def test_full_fen_start_position():
    assert full_fen(fen_to_board(STARTING_BOARD_FEN)) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_castling_only_when_king_and_rook_home():
    board = fen_to_board("r3k3/8/8/8/8/8/8/4K2R")
    assert infer_castling(board) == "Kq"


def test_validate_detects_errors():
    assert validate(fen_to_board(STARTING_BOARD_FEN)).ok
    assert not validate(fen_to_board("8/8/8/8/8/8/8/4K3")).ok  # no black king
    assert not validate(fen_to_board("4k3/8/8/8/8/8/8/P3K3")).ok  # pawn on the first rank
    assert not validate(fen_to_board("8/8/8/3kK3/8/8/8/8")).ok  # adjacent kings


def test_check_and_side_to_move():
    board = fen_to_board("4k3/8/8/8/8/8/8/4R1K1")  # the rook on e1 checks the black king
    assert in_check(board, white=False)
    assert not in_check(board, white=True)
    assert suggest_side_to_move(board) == "b"
    assert not validate(board, side_to_move="w").ok  # it cannot be White's move


def test_pawn_attacks_direction():
    assert in_check(fen_to_board("8/8/8/3k4/4P3/8/8/4K3"), white=False)  # the white pawn on e4 attacks d5
    assert not in_check(fen_to_board("8/8/8/4P3/3k4/8/8/4K3"), white=False)  # pawns do not attack backwards

