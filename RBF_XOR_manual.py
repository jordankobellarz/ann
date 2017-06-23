from ANN import RBFNet
from ANN.DataSet import DataSet

num_input = 2
num_hidden = 3
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
config = RBFNet.Config(center_function=None, max_iterations=1000000,
                       learning_rate=0.1, min_error=0.001, log_each_iterations=10)

# manually setting centers and radius
centers = [
    [0, 0],
    [1, 1],
    [0.5, 0.5]
]

radius = [
    [1, 1],
    [1, 1],
    [2, 2]
]

# train
rbf_net.train(ds.training_patterns, config=config, centers=centers, radius=radius)

print(rbf_net.activate([0, 0]))
print(rbf_net.activate([0, 1]))
print(rbf_net.activate([1, 0]))
print(rbf_net.activate([1, 1]))