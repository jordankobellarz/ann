from ANN.MLP import Model
from ANN.MLP import Neuron
import Datasets

# create the model
mlp = Model(num_input=18)
mlp.add_layer(num_neurons=20, activation=Neuron.FN_SIGMOID)  # hidden layer
mlp.add_layer(num_neurons=1, activation=Neuron.FN_SIGMOID)  # output layer

ds = Datasets.facebook()
mlp.online_train(ds.training_patterns, learning_rate=0.06, momentum=0.85,
                min_error=.0001, max_iterations=-1, log_each_iterations=1)
mlp.test(ds.testing_patterns, 0.01)