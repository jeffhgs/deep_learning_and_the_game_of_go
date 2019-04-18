import argparse

import h5py

from dlgo import agent
from dlgo import httpfrontend
from dlgo import mcts
from dlgo import rl
from dlgo.agent import load_prediction_agent, load_policy_agent, AlphaGoMCTS
from dlgo.rl import load_value_agent

import os
adirCode=os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind-address', default='127.0.0.1')
    parser.add_argument('--port', '-p', type=int, default=5000)
    parser.add_argument('--pg-agent')
    parser.add_argument('--predict-agent')
    parser.add_argument('--q-agent')
    parser.add_argument('--ac-agent')

    args = parser.parse_args()

    bots = {'mcts': mcts.MCTSAgent(800, temperature=0.7)}
    if args.pg_agent:
        bots['pg'] = agent.load_policy_agent(h5py.File(args.pg_agent))
    if args.predict_agent:
        bots['predict'] = agent.load_prediction_agent(
            h5py.File(args.predict_agent))
    if args.q_agent:
        q_bot = rl.load_q_agent(h5py.File(args.q_agent))
        q_bot.set_temperature(0.01)
        bots['q'] = q_bot
    if args.ac_agent:
        ac_bot = rl.load_ac_agent(h5py.File(args.ac_agent))
        ac_bot.set_temperature(0.05)
        bots['ac'] = ac_bot
    if True:
        # see code/dlgo/agent/alphago_tst.py
        fast_policy = load_prediction_agent(h5py.File(os.path.join(adirCode,'test_alphago_sl_policy.h5'), 'r'))
        strong_policy = load_policy_agent(h5py.File(os.path.join(adirCode,'test_alphago_rl_policy.h5'), 'r'))
        value = load_value_agent(h5py.File(os.path.join(adirCode,'test_alphago_value.h5'), 'r'))

        alphago = AlphaGoMCTS(strong_policy, fast_policy, value,
                              num_simulations=20, depth=5, rollout_limit=10)
        bots['alphago'] = alphago

    web_app = httpfrontend.get_web_app(bots)
    web_app.run(host=args.bind_address, port=args.port)


if __name__ == '__main__':
    main()
