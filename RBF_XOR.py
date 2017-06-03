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

rbf_net.train(ds.training_patterns, RBFNet.Net.CENTER_FN_KMEANS, 1000000, .1, 0.001, 10)

print(rbf_net.activate([0, 0]))
print(rbf_net.activate([0, 1]))
print(rbf_net.activate([1, 0]))
print(rbf_net.activate([1, 1]))