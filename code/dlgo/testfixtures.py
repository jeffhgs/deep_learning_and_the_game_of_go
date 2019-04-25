import numpy as np
from dlgo import goboard_fast as goboard
from dlgo.utils import coords_from_point
from dlgo.utils import point_from_coords
from dlgo.gotypes import Player, Point

def get_test_game_states():
    board_size = 19
    row=3
    col=3
    game_state0 = goboard.GameState.new_game(board_size)

    move1 = goboard.Move.play(Point(row, col))
    game_state1 = game_state0.apply_move(move1)
    return [game_state0, game_state1]
