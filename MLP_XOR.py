from ANN.MLP import Model
from ANN.MLP import Neuron
import Datasets

# create the model
mlp = Model(num_input=2)
mlp.add_layer(num_neurons=10, activation=Neuron.FN_SIGMOID)  # hidden layer
mlp.add_layer(num_neurons=1, activation=Neuron.FN_SIGMOID)  # output layer

ds = Datasets.XOR()
mlp.batch_train(ds.training_patterns, learning_rate=0.1, momentum=0.85,
                min_error=0.0001, max_iterations=-1, log_each_iterations=1000)
mlp.test(ds.training_patterns, min_error=0.1)


