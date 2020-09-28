import unittest
import numpy as np

import extract_game as eg


class TestExtractionFunctions(unittest.TestCase):
    def test_fen_to_matrix_white_turn(self):
        np.testing.assert_equal(
            eg.fen_to_matrix(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            ),
            np.array(
                [
                    [10, 8, 9, 11, 12, 9, 8, 10],
                    [7, 7, 7, 7, 7, 7, 7, 7],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [4, 2, 3, 5, 6, 3, 2, 4],
                ]
            ),
        )

    def test_fen_to_matrix_black_turn(self):
        np.testing.assert_equal(
            eg.fen_to_matrix(
                "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
            ),
            np.array(
                [
                    [4, 2, 3, 5, 6, 3, 2, 4],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 7, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [7, 7, 7, 7, 0, 7, 7, 7],
                    [10, 8, 9, 11, 12, 9, 8, 10],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
