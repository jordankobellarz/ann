from ANN.DataSet import DataSet

from ANN.MLP import MLP

num_input = 2
num_hidden = 2
num_output = 1

# create the network
mlp = MLP(num_input, num_hidden, num_output)

# create the data set
ds = DataSet(num_input, num_output, [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
], 1)

learning_rate = .99
momentum = 0.9
max_iterations = -1
min_error = .0001
log_each_iterations = 100

mlp.batch_train(ds.training_patterns, max_iterations, learning_rate, momentum, min_error, log_each_iterations)

mlp.test(ds.training_patterns)

# mlp.dump_json()


