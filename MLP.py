from Layer import Layer

class MLP:

    def __init__(self, num_input, num_hidden, num_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.iteration = 0  # iterations counter
        self.error = 0.0  # error for each iteration

        self.hidden_layer = Layer(num_input, num_hidden)
        self.output_layer = Layer(num_hidden, num_output)

    def train(self, patterns, learning_rate = .001, max_iterations = -1, min_error = 0.001):

        while (self.iteration < max_iterations or max_iterations == -1) and (self.error > min_error or self.error == 0.0):

            self.error = 0
            self.reset_deltas()

            # batch forward and backward propagation
            for pattern in patterns:
                output = self.feed_forward(pattern['input'])
                self.back_propagate(pattern['desired'])

                # calculate the error for this pattern
                self.error += self.mean_square_error(pattern['desired'], output)

            # update weights for hidden and output layer
            for neuron in self.hidden_layer.neurons:
                neuron.update_weights(learning_rate)
            for neuron in self.output_layer.neurons:
                neuron.update_weights(learning_rate)

            if not self.iteration % 1000 and self.iteration != 0:
                print("iteration: " + str(self.iteration) + " error: " + str(self.error / len(patterns)))

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
                error += neuron_k.gradients[j] * neuron_j.weights[k]

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

    def reset(self):
        self.iteration = 0
        self.error = 0
        for neuron in self.hidden_layer.neurons:
            neuron.init_weights_and_bias()
        for neuron in self.output_layer.neurons:
            neuron.init_weights_and_bias()

    def mean_square_error(self, desired, output):
        sum = 0.0
        for i in range(len(desired)):
            sum += (desired[i] - output[i]) ** 2
        return sum

    def dump(self):
        return {
            'hidden_layer': self.hidden_layer.dump(),
            'output_layer': self.output_layer.dump(),
        }

    def dump_json(self):
        import json
        print(json.dumps(self.dump(), indent=2, sort_keys=True))