import time
from ANN import combinator
from ANN.DataSet import DataSet

# training parameters
max_iterations = 1000000
min_error = .0001
log_each_iterations = 10000
max_tests = 7  # number of tests for each parameters combination
file_name = 'results/XOR_' + str(time.time()) + '.csv'  # name of the output file

# base parameters
num_input = 2
num_output = 1

# parameter datasets
num_hidden_ds = [2]
learning_rate_ds = [0.01, 0.05, 0.09, 0.1, 0.5, 0.9, 0.99]
momentum_ds = [0, 0.8, 0.85, 0.9]

# create the data set
ds = DataSet(num_input, num_output, [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
], 1)

# test all possible combinations
combinator.test_combinations(
    num_input,
    num_output,
    ds,
    num_hidden_ds,
    learning_rate_ds,
    momentum_ds,
    min_error,
    max_iterations,
    max_tests,
    file_name,
    log_each_iterations)