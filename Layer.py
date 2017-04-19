from Neuron import Neuron

class Layer:
    def __init__(self, num_input, num_neurons):
        self.num_input = num_input
        self.num_neuron = num_neurons

        self.neurons = []

        for i in range(num_neurons):
            self.add_neuron()

    def add_neuron(self):
        self.neurons.append(Neuron(self.num_input))

    def activate(self, input):  # activate all neurons on this layer
        output = []
        for neuron in self.neurons:
            output.append(neuron.activate(input))
        return output