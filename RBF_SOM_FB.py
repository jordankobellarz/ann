from ANN import RBFNet
from ANN.DataSet import DataSet
import FB_correlation

num_input = 18
num_hidden = 10
num_output = 1

# create the network
rbf_net = RBFNet.Net(num_input, num_hidden)

# create the data set
ds = DataSet(num_input, num_output, FB_correlation.ds_facebook_normalized, 0.8)

# training parameters
learning_rate = .005
momentum = .08
max_iterations = -1
min_error = .00001
log_each_iterations = 10

rbf_net.train(ds.training_patterns, RBFNet.Net.CENTER_FN_KOHONEN, max_iterations, learning_rate, min_error, log_each_iterations)

rbf_net.test(ds.testing_patterns)