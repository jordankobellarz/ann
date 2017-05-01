from Neuron import Neuron

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