from MLP import MLP
from DataSet import DataSet

max_iterations = -1
min_error = .001

num_input = 2
num_hidden = 2
num_output = 1

# create the network
mlp = MLP(num_input, num_hidden, num_output)

# create the data set
training_data = DataSet(num_input, num_output, [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
])

print(mlp.feed_forward([1, 2]))
mlp.back_propagate([1])