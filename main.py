from MLP import MLP
from DataSet import DataSet
import correlation

num_input = 18
num_hidden = 7
num_output = 1

# create the network
mlp = MLP(num_input, num_hidden, num_output)

# create the data set
ds = DataSet(num_input, num_output, correlation.ds_facebook_normalized, 0.8)

# training parameters
learning_rate = .00001
momentum = 0.05
max_iterations = -1
min_error = .001
log_each_iterations = 10

mlp.batch_train(ds.training_patterns, max_iterations, learning_rate, momentum, min_error, log_each_iterations)

mlp.test(ds.testing_patterns)

# mlp.dump_json()
