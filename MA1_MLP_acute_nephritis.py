from ANN.MLP import Model
from ANN.MLP import Neuron
import Datasets

# create the model
mlp = Model(num_input=6)
mlp.add_layer(num_neurons=20, activation=Neuron.FN_SIGMOID)  # hidden layer
mlp.add_layer(num_neurons=1, activation=Neuron.FN_LINEAR)  # output layer

ds = Datasets.acute_nephritis()
mlp.online_train(ds.training_patterns, learning_rate=0.02, momentum=0.2,
                 min_error=3.96199e-5, max_iterations=10000, log_each_iterations=1)
mlp.test(ds.testing_patterns, min_error=0.49)