from ANN import RBFNet
from ANN.DataSet import DataSet

num_input = 2
num_hidden = 4
num_output = 1

# create the network
rbf_net = RBFNet.Net(num_input, num_hidden)

# create the data set
ds = DataSet(num_input, num_output, [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
], 1)

# rbf parameters
max_iterations = 1000000
learning_rate = .1
min_error = .001
log_each_iterations = 10
rbf_net_config = RBFNet.Config(RBFNet.Config.CENTER_FN_KMEANS, max_iterations, learning_rate, min_error, log_each_iterations)

# train
rbf_net.train(ds.training_patterns, rbf_net_config)

print(rbf_net.activate([0, 0]))
print(rbf_net.activate([0, 1]))
print(rbf_net.activate([1, 0]))
print(rbf_net.activate([1, 1]))