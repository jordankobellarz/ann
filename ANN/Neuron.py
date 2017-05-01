import random
import math

class Neuron:
    def __init__(self, num_input):
        self.num_input = num_input

        self.bias = 0.0  # bias

        self.weights = []  # weights

        # gradients (error * derivative)
        self.gradients = []  # one gradient for each weight
        self.bias_gradient = 0.0

        # deltas (error * derivative * x)
        self.deltas = []   # one delta for each weight
        self.bias_delta = 0.0

        # last deltas (to use with momentum)
        self.last_deltas = [1.0] * num_input
        self.last_bias_delta = 1.0

        self.output = 0.0  # store the output for each activation

        # reset neuron
        self.reset_deltas()
        self.reset_gradients()
        self.init_weights_and_bias()

    def reset_deltas(self):
        self.deltas = [0.0] * self.num_input
        self.bias_delta = 0.0

    def reset_gradients(self):
        self.gradients = [0.0] * self.num_input
        self.bias_gradient = 0.0

    def init_weights_and_bias(self):
        self.bias = random.randint(-10, 10) / 10  # between 0.1 and 0.9
        self.weights = []
        for i in range(self.num_input):
            self.weights.append(random.randint(-10, 10) / 10)

    def af(self, u):
        return 1 / (1 + math.exp(-u))

    def d_af(self, y):
        return y * (1.0 - y)

    def sum(self, input):
        potential = self.bias * 1
        for i, weight in enumerate(self.weights):
            potential += input[i] * weight
        return potential

    def activate(self, input):
        self.output = self.af(self.sum(input))   # save the output for later
        return self.output

    def update_weights(self, learning_rate, momentum = 0.5):
        self.bias += learning_rate * self.bias_delta + momentum * self.last_bias_delta
        for i, weight in enumerate(self.weights):
            self.weights[i] += learning_rate * self.deltas[i] + momentum * self.last_deltas[i]

        # save current deltas to use with momentum on the next iteration
        self.last_deltas = self.deltas[:]
        self.last_bias_delta = self.bias_delta

    def dump(self):
        return {
            'bias': self.bias,
            'weights': self.weights,
            'gradients': self.gradients,
            'bias_gradient': self.bias_gradient,
            'deltas': self.deltas,
            'bias_delta': self.bias_delta,
            'output': self.output
        }