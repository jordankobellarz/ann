from ANN.DataSet import DataSet

from ANN import MLP

num_input = 2
num_hidden = 2
num_output = 1

# create the network
mlp = MLP.Net(num_input, num_hidden, num_output)

# create the data set
ds = DataSet(num_input, num_output, [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
], 1)

mlp.batch_train(ds.training_patterns, learning_rate=0.01, momentum=0.85,
                min_error=0.0001, max_iterations=-1, log_each_iterations=1000)

mlp.test(ds.training_patterns, min_error=0.0001)

# mlp.dump_json()


