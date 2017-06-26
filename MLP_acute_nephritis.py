from ANN.MLP import Model
import Datasets

num_input = 6
num_hidden = 1
num_output = 1

# create the model
mlp = Model(num_input=num_input)
mlp.add_layer(num_neurons=num_hidden)  # hidden layer
mlp.add_layer(num_neurons=num_output)  # output layer

ds = Datasets.acute_nephritis()
mlp.online_train(ds.training_patterns, learning_rate=0.9, momentum=0.9,
                 min_error=0.001, max_iterations=10000, log_each_iterations=1)
mlp.test(ds.testing_patterns, min_error=0.1)