import Utils
import math
import MLP
import SOM
from ANN.DataSet import DataSet


class Config:
    # center function constants
    CENTER_FN_KMEANS = 'kmeans'
    CENTER_FN_KOHONEN = 'kohonen'

    def __init__(self, center_function=CENTER_FN_KMEANS, max_iterations=-1, learning_rate=.001, min_error=.001,
                 log_each_iterations=1000):
        self.center_function = center_function
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.min_error = min_error
        self.log_each_iterations = log_each_iterations


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

    def train(self, patterns, config=Config(), center_fn_config=SOM.Config(), centers=[], radius=[]):

        inputs = DataSet.getValues(patterns, 'input')
        desired = DataSet.getValues(patterns, 'desired')

        # ---------------------------------------------------------
        # Phase 1 (calculating center and radius for each neuron)
        # ---------------------------------------------------------

        # calculate centers
        if config.center_function == Config.CENTER_FN_KMEANS:
            centers = Utils.k_means(inputs, self.num_hidden_neurons)
            radius = Utils.k_nearest_neighbors(inputs, centers)
        elif config.center_function == Config.CENTER_FN_KOHONEN:
            kohonen = SOM.Net(self.num_inputs, self.num_hidden_neurons)
            kohonen.train(inputs, center_fn_config)
            centers = kohonen.get_weights()
            radius = Utils.k_nearest_neighbors(inputs, centers)

        # update the centers and radius for each neuron
        for i, neuron in enumerate(self.hidden_layer.neurons):
            neuron.update_cluster(centers[i], radius[i])

        # ---------------------------------------------------------
        # Phase 2 (adaline)
        # ---------------------------------------------------------
        output_neuron = self.output_layer.neurons[0]
        while self.iteration < config.max_iterations or config.max_iterations == -1:

            self.error = 0.0

            for i, input in enumerate(inputs):
                hiddenLayerOutput = self.hidden_layer.activate(input)
                e = output_neuron.sum(hiddenLayerOutput) - desired[i][0]
                for j, weight in enumerate(output_neuron.weights):
                    output_neuron.weights[j] -= config.learning_rate * e * hiddenLayerOutput[j]
                self.error += math.pow(e, 2)

            #  stop training if the total error is below the minimum error
            self.error /= len(patterns)
            if self.error < config.min_error:
                break

            if not self.iteration % config.log_each_iterations and self.iteration != 0:
                print("iteration: " + str(self.iteration) + " error: " + str(self.error))

            self.iteration += 1

    def test(self, patterns):
        for i, pattern in enumerate(patterns):
            desired = pattern['desired'][0]
            output = self.activate(pattern['input'])[0]
            diff = desired - output
            print("test " + str(i) + " = desired: " + str(desired) + ", got: " + str(output) + ", diff: " + str(diff))


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

    def update_cluster(self, center, radius):
        self.center = center
        self.radius = radius
