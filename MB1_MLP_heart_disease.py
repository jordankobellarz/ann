from ANN.MLP import Model
from ANN.MLP import Neuron
import Datasets

# create the model
mlp = Model(num_input=22)
mlp.add_layer(num_neurons=20, activation=Neuron.FN_SIGMOID)  # hidden layer
mlp.add_layer(num_neurons=1, activation=Neuron.FN_LINEAR)  # output layer

ds = Datasets.heart_disease()
mlp.online_train(ds.training_patterns, learning_rate=0.01, momentum=0.01,
                 min_error=2.78711e-2, max_iterations=-1, log_each_iterations=10)
mlp.test(ds.testing_patterns, min_error=0.4999)