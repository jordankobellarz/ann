from ANN import SOM

inputs = [
    [10, 15],
    [10, 10],
    [12, 13],
    [112, 120],
    [115, 115],
    [110, 112]
]

tests = [
    [12, 13],
    [11, 10],
    [111, 115],
    [115, 113]
]

# create the network
num_inputs = 2
num_neurons = 2
SOM_net = SOM.Net(num_inputs, num_neurons)

# manually setting initial weights
# TODO: remove in production
# SOM_net.neurons[0].weights = [0, 0]
# SOM_net.neurons[1].weights = [90, 90]

# train
max_iterations = 100000
initial_learning_rate = 0.01
initial_radius = 1.1
SOM_net.train(inputs, max_iterations, initial_learning_rate, initial_radius)

# show outputs
print("Classes", SOM_net.classify(tests))
SOM_net.dump()
