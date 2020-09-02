import chess.pgn
import chess
import numpy as np
import re


def fen_to_matrix(fen_str):
    _ranks = fen_str.split(" ")[0].split("/")
    _piece_codes = {
        "P": chess.PAWN,
        "N": chess.KNIGHT,
        "B": chess.BISHOP,
        "R": chess.ROOK,
        "Q": chess.QUEEN,
        "K": chess.KING,
        "p": chess.PAWN + chess.KING,
        "n": chess.KNIGHT + chess.KING,
        "b": chess.BISHOP + chess.KING,
        "r": chess.ROOK + chess.KING,
        "q": chess.QUEEN + chess.KING,
        "k": chess.KING + chess.KING,
    }

    def _fen_to_piece(_fen_rank):
        _pieces_in_rank = list(_fen_rank)
        _explicit_rank = []
        for char in list(_pieces_in_rank):
            if re.match(r"\d", char):
                for count in range(int(char)):
                    _explicit_rank.append(char)
            else:
                _explicit_rank.append(char)

        return [
            _piece_codes[_piece] if _piece in _piece_codes else 0
            for _piece in _explicit_rank
        ]

    return np.array([_fen_to_piece(_rank) for _rank in _ranks])


def extract_data_from_game(game_model):
    positions = []
    moves = []
    for position in game_model.mainline():
        positions.append(fen_to_matrix(position.board().fen()))
        moves.append(position.move.from_square)

    return (np.array(positions), np.array(moves))


pgn = open("./data/lichess_db_standard_rated_2014-09.pgn")

game_positions = np.empty((1, 8, 8))
move_positions = np.empty((64))

for count in range(1000):
    game_model = chess.pgn.read_game(pgn)
    game_data = extract_data_from_game(game_model)
    if count == 0:
        game_positions = game_data[0]
        move_positions = game_data[1]
    else:
        game_positions = np.append(game_positions, game_data[0], axis=0)
        move_positions = np.append(move_positions, game_data[1], axis=0)


np.save("./data/game_positions.npy", game_positions)
np.save("./data/move_positions.npy", move_positions)

