from MLP import MLP
from DataSet import DataSet

num_input = 2
num_hidden = 2
num_output = 1

# create the network
mlp = MLP(num_input, num_hidden, num_output)

# create the data set
ds = DataSet(num_input, num_output, [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
], 1)

learning_rate = .01
momentum = .85
max_iterations = 10000
min_error = .001
log_each_iterations = 100

mlp.online_train(ds.training_patterns, max_iterations, learning_rate, momentum, min_error, log_each_iterations)

print(mlp.feed_forward([0, 0]))
print(mlp.feed_forward([0, 1]))
print(mlp.feed_forward([1, 0]))
print(mlp.feed_forward([1, 1]))

# mlp.dump_json()


