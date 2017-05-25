import Utils
import math
import MLP
from ANN.DataSet import DataSet


class Net:
    def __init__(self, num_inputs, num_hidden_neurons):
        self.num_hidden_neurons = num_hidden_neurons
        self.num_inputs = num_inputs

        self.iteration = 0  # iterations counter
        self.error = 0.0  # error for each iteration

        self.hidden_layer = Layer(num_inputs, num_hidden_neurons)
        self.output_layer = MLP.Layer(num_hidden_neurons, 1)

    def activate(self, input):
        output = self.hidden_layer.activate(input)
        output = self.output_layer.neurons[0].sum(output)
        return output

    def train(self, patterns, max_iterations=-1, learning_rate=.001, min_error=.001, log_each_iterations=1000):

        inputs = DataSet.getValues(patterns, 'input')
        desired = DataSet.getValues(patterns, 'desired')

        # phase 1
        centers = Utils.k_means(inputs, self.num_hidden_neurons)
        radius = Utils.k_nearest_neighbors(inputs, centers)
        for i, neuron in enumerate(self.hidden_layer.neurons):
            neuron.updateCluster(centers[i], radius[i])

        # phase 2 (adaline)
        output_neuron = self.output_layer.neurons[0]
        while self.iteration < max_iterations or max_iterations == -1:

            self.error = 0.0

            for i, input in enumerate(inputs):
                hiddenLayerOutput = self.hidden_layer.activate(input)
                e = output_neuron.sum(hiddenLayerOutput) - desired[i][0]
                for j, weight in enumerate(output_neuron.weights):
                    output_neuron.weights[j] -= learning_rate * e * hiddenLayerOutput[j]
                self.error += math.pow(e, 2)

            #  stop training if the total error is below the minimum error
            self.error /= len(patterns)
            if self.error < min_error:
                break

            if not self.iteration % log_each_iterations and self.iteration != 0:
                print("iteration: " + str(self.iteration) + " error: " + str(self.error))

            self.iteration += 1

class Layer:
    def __init__(self, num_input, num_neurons):
        self.num_input = num_input
        self.num_neuron = num_neurons
        self.neurons = []

        for i in range(num_neurons):
            self.neurons.append(Neuron(self.num_input))

    def activate(self, input):  # activate all neurons on this layer
        output = []
        for neuron in self.neurons:
            output.append(neuron.activate(input))
        return output


class Neuron:
    def __init__(self, num_input):
        self.num_input = num_input

        self.center = [0.0] * num_input
        self.radius = [1.0] * num_input

    # Radial Basis Function
    def RBF(self, input):
        result = 0.0
        for i in range(len(input)):
            if self.radius[i] > 0:
                result += - math.pow(input[i] - self.center[i], 2) / math.pow(self.radius[i], 2)
            else:
                result += - math.pow(input[i] - self.center[i], 2)
        return math.exp(result)

    def activate(self, input):
        return self.RBF(input)

    def updateCluster(self, center, radius):
        self.center = center
        self.radius = radius
