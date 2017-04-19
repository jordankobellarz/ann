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

    def back_propagate(self):
        return

    def calculate_deltas(self):
        return

