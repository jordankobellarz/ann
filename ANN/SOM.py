from __builtin__ import staticmethod

import Utils
import math
import random
from ANN.MLP import Neuron


class Net:
    def __init__(self, num_inputs, num_neurons):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs

        self.iteration = 0  # iterations counter

        self.neurons = []  # SOM neurons

        self.add_neurons(num_neurons)

    def add_neurons(self, num_neurons):
        """Add num_neurons to the SOM"""
        for i in range(num_neurons):
            neuron = Neuron(self.num_inputs)
            for j, weight in enumerate(neuron.weights):
                neuron.weights[j] = random.randint(10, 90) / 100
            self.neurons.append(Neuron(self.num_inputs))

    def train(self, inputs, max_iterations=-1, initial_learning_rate=0.1, initial_radius=1.0, log_each_iterations=100):
        """
        Train the Self Organizing Map (SOM).
        Inspired from: https://page.mi.fu-berlin.de/rojas/neural/chapter/K15.pdf
        Inspired from: http://www.shy.am/wp-content/uploads/2009/01/kohonen-self-organizing-maps-shyam-guthikonda.pdf
        Inspired from: https://www.youtube.com/watch?v=H9H6s-x-0YE
        :param inputs: 
        :param max_iterations: 
        :param initial_learning_rate: 
        :param initial_radius: 
        :param log_each_iterations: 
        :return: 
        """

        learning_rate = initial_learning_rate
        radius = initial_radius

        while self.iteration < max_iterations or max_iterations == -1:

            # select one random input
            random.shuffle(inputs)
            input = inputs[0]

            # find the best matching unit (BMU)
            bmu = self.best_matching_unit(input)

            # Update the weights of all neurons on the BMU neighborhood, and of the BMU itself
            for j, neuron in enumerate(self.neurons):
                euclidean_distance = Utils.euclidean_distance(neuron.weights, bmu.weights)

                # relative_distance will be a number between 0 (distant from BMU) and 1 (near to BMU)
                # The relative_distance between the BMU and itself always be 1, and because that
                # it will have the the biggest increase on weights.
                relative_distance = math.exp(-math.pow(euclidean_distance, 2) / (2 * math.pow(radius, 2)))

                # if this neuron is near the BMU by the radius, update his weights
                if relative_distance <= radius:
                    for i, weight in enumerate(neuron.weights):
                        neuron.weights[i] += learning_rate * relative_distance * (input[i] - weight)

            # update learning rate and radius (they will decay over time)
            learning_rate = initial_learning_rate * math.exp(-self.iteration / float(max_iterations))
            radius = initial_radius * math.exp(-self.iteration / float(max_iterations))

            if not self.iteration % log_each_iterations and self.iteration != 0:
                print("iteration: " + str(self.iteration))

            self.iteration += 1

    def best_matching_unit(self, input):
        """
        Find the neuron with the minimum euclidean distance between his weights and the input.
        This Neuron is called 'best matching unit' or BMU
        """
        best_matching_unit = None
        min_distance = 0
        for j, neuron in enumerate(self.neurons):
            euclidean_distance = Utils.euclidean_distance(input, neuron.weights)
            if euclidean_distance < min_distance or best_matching_unit is None:
                best_matching_unit = neuron
                min_distance = euclidean_distance
        return best_matching_unit

    def classify(self, inputs):
        centers = []  # each center will be the weights of one SOM neuron
        for neuron in self.neurons:
            centers.append(neuron.weights)
        return Utils.clusterize(inputs, centers)

    def u_matrix(self, width, height):
        """Create the U matrix to visualize the map"""
        # cada ponto na matrix contem a distancia euclidiana de celulas vizinhas
        u_matrix = [[0 for x in range(width)] for y in range(height)]
        for j, neuron in enumerate(self.neurons):
            for k, neuron in enumerate(self.neurons):
                print("Neuron #" + j + " weights:" + neuron.weights)

    def dump(self):
        for j, neuron in enumerate(self.neurons):
            print("Neuron #" + str(j) + " weights:", neuron.weights)