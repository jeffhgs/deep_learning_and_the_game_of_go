import unittest

import os
import sys
adirCode=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,adirCode)

import random
import time
import numpy as np
import json
from dlgo.nn import load_mnist
from dlgo.nn import network
from dlgo.nn.layers import DenseLayer, ActivationLayer

class RunSampler():
    def __init__(self):
        self.training_data, self.test_data = load_mnist.load_data()

    def limit_data(self, data, n):
        return data[:n]

    def run_net_one_layer_once(self, samples_per_dataset, num_epochs, learning_rate):
        random.shuffle(self.training_data)
        random.shuffle(self.test_data)
        training_data = self.limit_data(self.training_data, samples_per_dataset)
        test_data = self.limit_data(self.test_data, samples_per_dataset)
        net = network.SequentialNetwork()  # <2>

        net.add(DenseLayer(784, 392))  # <3>
        net.add(ActivationLayer(392))
        net.add(DenseLayer(392, 196))
        net.add(ActivationLayer(196))
        net.add(DenseLayer(196, 10))
        net.add(ActivationLayer(10))  # <4>
        m = net.train(training_data, epochs=num_epochs, mini_batch_size=10,
                  learning_rate=learning_rate, test_data=test_data)  # <1>
        return m

    def sample_runs(self, num_runs, samples_per_dataset, num_epochs, learning_rate):
        runs = []
        for i in range(0,num_runs):
            run = self.run_net_one_layer_once(samples_per_dataset, num_epochs, learning_rate)
            runs.append(run)
        return runs

    def scores(self, runs):
        return [run[-1]['correct']/run[-1]['possible'] for run in runs]

    def distribution(self,scores):
        return {
            'mean': np.mean(np.array(scores)),
            'stdev': np.std(np.array(scores))}


    def sample_runs_and_report(self, num_runs, samples_per_dataset, num_epochs, learning_rate):
        print("about to sample_runs_and_report num_runs={}, samples_per_dataset={}, num_epochs={} learning_rate={}".format(num_runs, samples_per_dataset, num_epochs, learning_rate))
        t0 = time.time()
        runs = self.sample_runs(num_runs, samples_per_dataset, num_epochs, learning_rate)
        scores = sorted(self.scores(runs), key=(lambda x: -x))
        distribution = self.distribution(scores)
        dtElapsed = time.time() - t0
        print(json.dumps({
            'parameters': {
                'num_runs' : num_runs,
                'samples_per_dataset': samples_per_dataset,
                'num_epochs': num_epochs,
                'learning_rate': learning_rate
            },
            'dtElapsed' : dtElapsed,
            'distribution':distribution,
            'scores':scores,
            'runs':runs},
            sort_keys=True))


class BoardTest(unittest.TestCase):
    def setUp(self):
        random.seed(time.time())
        self.runSampler = RunSampler()

    def test_load_data(self):
        pass

    def test_01_learning_rate(self):
        self.runSampler.sample_runs_and_report(
            num_runs=10,
            samples_per_dataset=500,
            num_epochs=10,
            learning_rate=3.0
        )

    def test_02_learning_rate(self):
        self.runSampler.sample_runs_and_report(
            num_runs=10,
            samples_per_dataset=500,
            num_epochs=10,
            learning_rate=1.5
        )

    def test_03_learning_rate(self):
        self.runSampler.sample_runs_and_report(
            num_runs=10,
            samples_per_dataset=500,
            num_epochs=10,
            learning_rate=0.1
        )

    def test_04_fit1(self):
        self.runSampler.sample_runs_and_report(
            num_runs=1,
            samples_per_dataset=50000,
            num_epochs=10,
            learning_rate=3.0
        )

    def test_04_fit2(self):
        self.runSampler.sample_runs_and_report(
            num_runs=2,
            samples_per_dataset=50000,
            num_epochs=10,
            learning_rate=3.0
        )

    def test_05_fit3(self):
        self.runSampler.sample_runs_and_report(
            num_runs=10,
            samples_per_dataset=50000,
            num_epochs=10,
            learning_rate=0.1
        )

if __name__ == '__main__':
    unittest.main()