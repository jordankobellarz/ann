import random
import math


class Model:
    def __init__(self, num_input):
        self.num_input = num_input

        self.iteration = 0  # iterations counter
        self.error = 0.0  # error for each iteration

        self.layers = []  # model layers

    def batch_train(self, patterns, learning_rate=0.1,  momentum=0.85,
                    min_error=0.001, max_iterations=-1, log_each_iterations=100):

        print("\n>>> Training: ")
        while self.iteration < max_iterations or max_iterations == -1:

            self.start_epoch()

            for pattern in patterns:
                output = self.feed_forward(pattern['input'])
                self.back_propagate(pattern['desired'])
                self.error += Model.mean_squared_error(pattern['desired'], output)

            self.update_weights(learning_rate, momentum)

            if self.end_epoch(len(patterns), min_error, log_each_iterations):
                break

    def online_train(self, patterns, learning_rate=0.1,  momentum=0.85,
                     min_error=0.001, max_iterations=-1, log_each_iterations=100):

        print("\n>>> Training: ")
        while self.iteration < max_iterations or max_iterations == -1:

            self.start_epoch()

            random.shuffle(patterns)
            for pattern in patterns:
                output = self.feed_forward(pattern['input'])
                self.back_propagate(pattern['desired'])
                self.error += Model.mean_squared_error(pattern['desired'], output)
                self.update_weights(learning_rate, momentum)

            if self.end_epoch(len(patterns), min_error, log_each_iterations):
                break

    def start_epoch(self):
        self.iteration += 1
        self.error = 0.0

    def end_epoch(self, num_patterns, min_error, log_each_iterations):
        self.error /= num_patterns
        if self.error <= min_error:
            self.log_iteration(log_each_iterations, True)
            return True
        self.log_iteration(log_each_iterations)
        return False

    def log_iteration(self, log_each_iterations, force=False):
        if not self.iteration % log_each_iterations and self.iteration != 0 or force:
            print("iteration: " + str(self.iteration) + " error: " + str(self.error))

    def feed_forward(self, input):
        last_output = input
        for layer in self.layers:
            last_output = layer.activate(last_output)
        return last_output

    def back_propagate(self, desired):

        # calculate output layer gradients and deltas
        output_layer = self.layers[-1]
        for k, neuron_k in enumerate(output_layer.neurons):
            derivative = neuron_k.d_af(neuron_k.output)  # derivative for neuron k
            error = (desired[k] - neuron_k.output)  # error for neuron k

            # calculate gradients and deltas for each weight
            for i, weight_i in enumerate(neuron_k.weights):
                neuron_k.gradients[i] = error * derivative
                neuron_k.deltas[i] += error * derivative * output_layer.input[i]

            # calculate gradient and delta for bias
            neuron_k.bias_gradient = error * derivative
            neuron_k.bias_delta += error * derivative * 1

        # calculate hidden layer gradients and deltas
        for l, layer in enumerate(self.layers[0:-1]):
            for j, neuron_j in enumerate(layer.neurons):
                derivative = neuron_j.d_af(neuron_j.output)  # derivative for neuron j
                error = 0.0  # error for neuron j

                # calculate the retropropagated error (from last layer backwards)
                for k, neuron_k in enumerate(self.layers[l+1].neurons):
                    error += neuron_k.gradients[j] * neuron_k.weights[k]

                # calculate gradients and deltas for each weight
                for i, weight_i in enumerate(neuron_j.weights):
                    neuron_j.gradients[i] = error * derivative
                    neuron_j.deltas[i] += error * derivative * layer.input[i]

                # calculate gradient and delta for bias
                neuron_j.bias_gradient = error * derivative
                neuron_j.bias_delta += error * derivative * 1

    def add_layer(self, num_neurons):
        if len(self.layers) == 0:
            # if is the first hidden layer
            self.layers.append(Layer(num_input=self.num_input, num_neurons=num_neurons))
        else:
            # if is the nth hidden layer
            last_layer_num_outputs = self.layers[-1].num_neuron
            self.layers.append(Layer(num_input=last_layer_num_outputs, num_neurons=num_neurons))

    def update_weights(self, learning_rate, momentum):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update_weights(learning_rate, momentum)

    def test(self, patterns, min_error):
        correct_count = 0
        print("\n>>> Testing: ")
        for i, pattern in enumerate(patterns):
            desired = pattern['desired'][0]
            output = self.feed_forward(pattern['input'])[0]
            error = abs(desired - output)
            if error <= min_error:
                correct_count += 1
            print("#" + str(i+1) + "\tdesired: " + str(desired) + "\tgot: " + str(round(output, 3)) + "\terror: " + str(round(error, 5)))
        accuracy = round(float(correct_count)/float(len(patterns))*100, 2)
        print("\n>>> Accuracy: " + str(accuracy) + "%")

    @staticmethod
    def mean_squared_error(desired, output):
        sum = 0.0
        for i in range(len(desired)):
            sum += ((desired[i] - output[i]) ** 2) / 2.0
        return sum

    def dump(self):
        return {
            'hidden_layer': self.hidden_layer.dump(),
            'output_layer': self.output_layer.dump(),
        }

    def dump_json(self):
        import json
        print(json.dumps(self.dump(), indent=2, sort_keys=True))


class Layer:
    def __init__(self, num_input=1, num_neurons=1):
        self.num_input = num_input
        self.num_neuron = num_neurons

        self.neurons = []
        self.input = []  # store this layer input for each feedforward to facilitate backpropagation

        for i in range(num_neurons):
            self.add_neuron()

    def add_neuron(self):
        self.neurons.append(Neuron(self.num_input))

    def get_outputs(self):  # get the stored outputs in this layer
        output = []
        for neuron in self.neurons:
            output.append(neuron.output)
        return output

    def activate(self, input):  # activate all neurons on this layer
        self.input = input  # store the input to use in backpropagation

        output = []
        for neuron in self.neurons:
            output.append(neuron.activate(self.input))

        return output

    def update_weights(self):
        for neuron in self.neurons:
            neuron.update_weights()

    def dump(self):
        obj = {
            'input': self.input,
            'output': self.get_outputs(),
            'neurons': []
        }

        for neuron in self.neurons:
            obj['neurons'].append(neuron.dump())

        return obj


class Neuron:
    def __init__(self, num_input):
        self.num_input = num_input

        self.bias = 0.0  # bias

        self.weights = []  # weights

        # gradients (error * derivative)
        self.gradients = [0.0] * num_input  # one gradient for each weight
        self.bias_gradient = 0.0

        # accumulate deltas for batch train (error * derivative * x)
        self.deltas = [0.0] * num_input   # one delta for each weight
        self.bias_delta = 0.0

        # last deltas (to use with momentum)
        self.last_deltas = [0.0] * num_input
        self.last_bias_delta = 0.0

        self.output = 0.0  # store the output for each activation

        # reset neuron
        self.init_weights_and_bias()

    def init_weights_and_bias(self):
        self.bias = random.randint(-9, 9) / 10.0  # between -0.9 and 0.9
        self.weights = []
        for i in range(self.num_input):
            self.weights.append(random.randint(-9, 9) / 10.0)

    def af(self, u):
        return 1.0 / (1.0 + math.exp(-u))

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

    def update_weights(self, learning_rate, momentum=0):
        self.bias += learning_rate * self.bias_delta + momentum * self.last_bias_delta
        for i, weight in enumerate(self.weights):
            self.weights[i] += learning_rate * self.deltas[i] + momentum * self.last_deltas[i]

        # save current deltas to use with momentum on the next iteration
        self.last_deltas = self.deltas[:]
        self.last_bias_delta = self.bias_delta

        # reset gradients and deltas
        self.gradients = [0.0] * self.num_input
        self.bias_gradient = 0.0
        self.deltas = [0.0] * self.num_input
        self.bias_delta = 0.0

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

