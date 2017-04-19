from Layer import Layer

class MLP:

    def __init__(self, num_input, num_hidden, num_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        self.hidden_layer = Layer(num_input, num_hidden)
        self.output_layer = Layer(num_hidden, num_output)

    def train(self, patterns, max_iterations = -1, min_error = 0.001):
        return

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

    def dump(self):
        return {
            'hidden_layer': self.hidden_layer.dump(),
            'output_layer': self.output_layer.dump(),
        }

    def dump_json(self):
        import json
        print(json.dumps(self.dump(), indent=2, sort_keys=True))