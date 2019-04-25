# tag::e2e_imports[]
import h5py

from keras.models import Sequential
from keras.layers import Dense

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent
from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.httpfrontend import get_web_app
from dlgo.networks import large
# end::e2e_imports[]

import os
adirCode=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rdirOut='../../log/agent'
os.makedirs(os.path.join(rdirOut,'test_dir'), exist_ok=True)

# tag::e2e_processor[]
go_board_rows, go_board_cols = 19, 19
nb_classes = go_board_rows * go_board_cols
encoder = SevenPlaneEncoder((go_board_rows, go_board_cols))
processor = GoDataProcessor(encoder=encoder.name(),
                            data_directory=os.path.join(adirCode,"data"))

X, y = processor.load_go_data(num_samples=128)
# end::e2e_processor[]

# tag::e2e_model[]
input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
model = Sequential()
network_layers = large.layers(input_shape)
for layer in network_layers:
    model.add(layer)
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(X, y, batch_size=128, epochs=1, verbose=1)
# end::e2e_model[]

# tag::e2e_agent[]
deep_learning_bot = DeepLearningAgent(model, encoder)

#how would you like to play?

import numpy as np
from dlgo import goboard_fast as goboard
from dlgo.utils import coords_from_point
from dlgo.utils import point_from_coords
from dlgo.gotypes import Player, Point

board_size = 19
row=3
col=3
game_state0 = goboard.GameState.new_game(board_size)
print("prediction 0: {}".format(model.predict(np.array([encoder.encode(game_state0)]))))

move1 = goboard.Move.play(Point(row, col))
game_state1 = game_state0.apply_move(move1)

print("prediction 1: {}".format(model.predict(np.array([encoder.encode(game_state1)]))))


with h5py.File(os.path.join(rdirOut,"deep_bot.h5"), 'w') as file_agent_out:
    deep_learning_bot.serialize(file_agent_out)
# end::e2e_agent[]

# tag::e2e_load_agent[]
model_file = h5py.File(os.path.join(rdirOut,"deep_bot.h5"), "r")
bot_from_file = load_prediction_agent(model_file)

print("successfully round tripped {}".format(os.path.join(rdirOut,"deep_bot.h5")))
#web_app = get_web_app({'predict': bot_from_file})
#web_app.run()
# end::e2e_load_agent[]
