import time
import MLP_FB_correlation
from ANN import combinator
from ANN.DataSet import DataSet

# training parameters
max_iterations = 3000
min_error = .0001
log_each_iterations = 10
max_tests = 2  # number of tests for each parameters combination
file_name = 'results/FB_' + str(time.time()) + '.csv'  # name of the output file

# base parameters
num_input = 18
num_output = 1

# parameter datasets
num_hidden_ds = [5, 6, 7, 8, 9, 10]
learning_rate_ds = [0.0001, 0.0005, 0.0009, 0.001, 0.005, 0.009]
momentum_ds = [0.03, 0.04, 0.05]

# create the data set
ds = DataSet(num_input, num_output, MLP_FB_correlation.ds_facebook_normalized, 0.8)

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

