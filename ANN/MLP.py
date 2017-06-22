import random
import math

class Net:

    def __init__(self, num_input, num_hidden, num_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.iteration = 0  # iterations counter
        self.error = 0.0  # error for each iteration

        self.hidden_layer = Layer(num_input, num_hidden)
        self.output_layer = Layer(num_hidden, num_output)

    def batch_train(self, patterns, max_iterations = -1, learning_rate = .001, momentum = .85, min_error = .001, log_each_iterations = 1000):

        while self.iteration < max_iterations or max_iterations == -1:

            self.error = 0.0
            self.reset_deltas()

            # batch forward and backward propagation
            for pattern in patterns:
                output = self.feed_forward(pattern['input'])
                self.back_propagate(pattern['desired'])

                # calculate the error for this pattern
                self.error += self.mean_square_error(pattern['desired'], output)

            # update weights for hidden and output layer
            for neuron in self.hidden_layer.neurons:
                neuron.update_weights(learning_rate, momentum)
            for neuron in self.output_layer.neurons:
                neuron.update_weights(learning_rate, momentum)

            #  stop training if the total error is below the minimum error
            self.error /= len(patterns)
            if self.error < min_error:
                break

            if not self.iteration % log_each_iterations and self.iteration != 0:
                print("iteration: " + str(self.iteration) + " error: " + str(self.error))

            self.iteration += 1


    def online_train(self, patterns, max_iterations = -1, learning_rate = .001, momentum = .85, min_error = .001, log_each_iterations = 1000):

        while self.iteration < max_iterations or max_iterations == -1:

            self.error = 0.0

            random.shuffle(patterns)

            # batch forward and backward propagation
            for pattern in patterns:
                self.reset_deltas()
                output = self.feed_forward(pattern['input'])
                self.back_propagate(pattern['desired'])

                # calculate the error for this pattern
                self.error += self.mean_square_error(pattern['desired'], output)

                # update weights for hidden and output layer
                for neuron in self.hidden_layer.neurons:
                    neuron.update_weights(learning_rate, momentum)
                for neuron in self.output_layer.neurons:
                    neuron.update_weights(learning_rate, momentum)

            #  stop training if the total error is below the minimum error
            self.error /= len(patterns)
            if self.error < min_error:
                break

            if not self.iteration % log_each_iterations and self.iteration != 0:
                print("iteration: " + str(self.iteration) + " error: " + str(self.error))

            self.iteration += 1

    def feed_forward(self, input):
        output = self.hidden_layer.activate(input)
        output = self.output_layer.activate(output)
        return output

    def back_propagate(self, desired):

        # calculate output layer gradients and deltas
        for k, neuron_k in enumerate(self.output_layer.neurons):
            derivative = neuron_k.d_af(neuron_k.output)  # derivative for neuron k
            error = (desired[k] - neuron_k.output)  # error for neuron k

            # calculate gradients and deltas for each weight
            for i, weight_i in enumerate(neuron_k.weights):
                neuron_k.gradients[i] = error * derivative
                neuron_k.deltas[i] += error * derivative * self.output_layer.input[i]

            # calculate gradient and delta for bias
            neuron_k.bias_gradient = error * derivative
            neuron_k.bias_delta += error * derivative * 1

        # calculate hidden layer gradients and deltas
        for j, neuron_j in enumerate(self.hidden_layer.neurons):
            derivative = neuron_j.d_af(neuron_j.output)  # derivative for neuron j
            error = 0.0  # error for neuron j

            # calculate the retropropagated error (from output layer)
            for k, neuron_k in enumerate(self.output_layer.neurons):
                error += neuron_k.gradients[j] * neuron_k.weights[k]

            # calculate gradients and deltas for each weight
            for i, weight_i in enumerate(neuron_j.weights):
                neuron_j.gradients[i] = error * derivative
                neuron_j.deltas[i] += error * derivative * self.hidden_layer.input[i]

            # calculate gradient and delta for bias
            neuron_j.bias_gradient = error * derivative
            neuron_j.bias_delta += error * derivative * 1

    def reset_deltas(self):
        for neuron in self.hidden_layer.neurons:
            neuron.reset_deltas()
        for neuron in self.output_layer.neurons:
            neuron.reset_deltas()

    def reset_weights_and_bias(self):
        self.iteration = 0
        self.error = 0
        for neuron in self.hidden_layer.neurons:
            neuron.init_weights_and_bias()
        for neuron in self.output_layer.neurons:
            neuron.init_weights_and_bias()

    def test(self, patterns, log_to_file = False):
        for i, pattern in enumerate(patterns):
            desired = pattern['desired'][0]
            output = self.feed_forward(pattern['input'])[0]
            error = desired - output
            print("test " + str(i+1) + " = desired: " + str(desired) + ", got: " + str(output) + ", error: " + str(error))

    def mean_square_error(self, desired, output):
        sum = 0.0
        for i in range(len(desired)):
            sum += 1.0/2.0 * ((desired[i] - output[i]) ** 2)
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
    def __init__(self, num_input, num_neurons):
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

