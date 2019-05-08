# tag::load_opponents[]
from dlgo.agent.pg import PolicyAgent
from dlgo.agent.predict import load_prediction_agent
#from dlgo.encoders.alphago import AlphaGoEncoder
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.rl.simulate import experience_simulation
import h5py

import sys
import os
def usage():
    print("usage:")
    print("  alphago_policy_rl.py <policy file>")
    sys.exit(1)

rfileSl=sys.argv[1]
if(not os.path.exists(rfileSl)):
    usage()

sl_agent = load_prediction_agent(h5py.File(rfileSl))
sl_opponent = load_prediction_agent(h5py.File(rfileSl))

encoder = sl_agent.encoder

alphago_rl_agent = PolicyAgent(sl_agent.model, encoder)
opponent = PolicyAgent(sl_opponent.model, encoder)
# end::load_opponents[]

# tag::run_simulation[]
num_games = 1000
experience = experience_simulation(num_games, alphago_rl_agent, opponent)

alphago_rl_agent.train(experience)

with h5py.File('alphago_rl_policy.h5', 'w') as rl_agent_out:
    alphago_rl_agent.serialize(rl_agent_out)

with h5py.File('alphago_rl_experience.h5', 'w') as exp_out:
    experience.serialize(exp_out)
# end::run_simulation[]
