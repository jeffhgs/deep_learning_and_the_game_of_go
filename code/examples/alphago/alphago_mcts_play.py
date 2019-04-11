# tag::run_alphago[]
from dlgo.agent import load_prediction_agent, load_policy_agent, AlphaGoMCTS
from dlgo.rl import load_value_agent
import h5py

import os

adirCode=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fast_policy = load_prediction_agent(h5py.File(os.path.join(adirCode, 'alphago_sl_policy.h5'), 'r'))
strong_policy = load_policy_agent(h5py.File(os.path.join(adirCode, 'alphago_rl_policy.h5'), 'r'))
value = load_value_agent(h5py.File(os.path.join(adirCode, 'alphago_value.h5'), 'r'))

alphago = AlphaGoMCTS(strong_policy, fast_policy, value)
# end::run_alphago[]

# TODO: register in frontend
