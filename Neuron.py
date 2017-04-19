import random
import math

class Neuron:
    def __init__(self, num_input):
        self.num_input = num_input

        self.bias = 0.0  # bias
        self.weights = []  # weights
        self.gradients = []  # gradients (error * derivative)
        self.deltas = []  # deltas (error * derivative * x)

        self.output = 0.0  # store the output for each activation

        # reset neuron
        self.reset_deltas()
        self.reset_gradients()
        self.init_weights_and_bias()

    def reset_deltas(self):
        self.deltas = [0.0] * self.num_input

    def reset_gradients(self):
        self.gradients = [0.0] * self.num_input

    def init_weights_and_bias(self):
        for i in range(self.num_input):
            self.weights.append(random.random())

    def af(self, u):
        return 1 / (1 + math.exp(-u))

    def d_af(self, y):
        return y * (1.0 - y)

    def sum(self, input):
        potential = self.bias
        for i, weight in enumerate(self.weights):
            potential += input[i] * weight
        return potential

    def activate(self, input):
        self.output = self.af(self.sum(input))   # save the output for later
        return self.output
