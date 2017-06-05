from ANN import RBFNet
from ANN import SOM
from ANN.DataSet import DataSet
import FB_correlation

num_input = 18
num_hidden = 20
num_output = 1

# create the network
rbf_net = RBFNet.Net(num_input, num_hidden)

# create the data set
ds = DataSet(num_input, num_output, FB_correlation.ds_facebook_normalized, 0.8)

# kohonen parameters
max_iterations = 100
initial_learning_rate = 0.1
initial_radius = 1.1
log_each_iterations = 10
kohonen_net_config = SOM.Config(max_iterations, initial_learning_rate, initial_radius, log_each_iterations)

# rbf parameters
max_iterations = -1
learning_rate = .01
min_error = .0001
log_each_iterations = 10
rbf_net_config = RBFNet.Config(RBFNet.Config.CENTER_FN_KOHONEN, max_iterations, learning_rate, min_error, log_each_iterations)

# train
rbf_net.train(ds.training_patterns, rbf_net_config, kohonen_net_config)

rbf_net.test(ds.testing_patterns)