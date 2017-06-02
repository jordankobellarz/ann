from ANN import SOM

inputs = [
    [10, 15],
    [10, 10],
    [12, 13],
    [112, 120],
    [115, 115],
    [110, 112],
    [1120, 1200],
    [1150, 1150],
    [1100, 1120],
]

tests = [
    [12, 13],
    [11, 10],
    [111, 115],
    [115, 113],
    [1000, 1100],
    [1050, 1180],
]

# create the network
num_inputs = 2
num_neurons = 3
SOM_net = SOM.Net(num_inputs, num_neurons)

# train
max_iterations = 1000
initial_learning_rate = 0.01
initial_radius = 1.1
SOM_net.train(inputs, max_iterations, initial_learning_rate, initial_radius)

# show outputs
print("Classes", SOM_net.classify(tests))
SOM_net.dump()
