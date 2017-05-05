from ANN.DataSet import DataSet
from ANN.MLP import MLP
import FB_correlation

num_input = 18
num_hidden = 4
num_output = 1

# create the network
mlp = MLP(num_input, num_hidden, num_output)

# create the data set
ds = DataSet(num_input, num_output, FB_correlation.ds_facebook_normalized, 0.8)

# training parameters
learning_rate = .005
momentum = .08
max_iterations = -1
min_error = .00001
log_each_iterations = 10

mlp.batch_train(ds.training_patterns, max_iterations, learning_rate, momentum, min_error, log_each_iterations)

mlp.test(ds.testing_patterns)

# mlp.dump_json()
