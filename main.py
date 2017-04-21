from MLP import MLP
from DataSet import DataSet
import correlation

num_input = 18
num_hidden = 5
num_output = 1

# create the network
mlp = MLP(num_input, num_hidden, num_output)

# create the data set
ds = DataSet(num_input, num_output, correlation.ds_facebook_normalized)

# training parameters
learning_rate = .0005
momentum = 0.01
max_iterations = -1
min_error = .0001
log_each_iterations = 20

mlp.batch_train(ds.training_patterns, max_iterations, learning_rate, momentum, .001, log_each_iterations)
print("learning rate reduced to .0005")
mlp.batch_train(ds.training_patterns, max_iterations, .005, momentum, .0005, log_each_iterations)
print("learning rate reduced to .0001")
mlp.batch_train(ds.training_patterns, max_iterations, .003, momentum, .0001, log_each_iterations)

# testing patterns
for i, pattern in enumerate(ds.testing_patterns):
    desired = pattern['desired'][0]
    output = mlp.feed_forward(pattern['input'])[0]
    print("test " + str(i) + " = desired: " + str(desired) + ", got: " + str(output) + ", error: " + str(desired - output))

# mlp.dump_json()
