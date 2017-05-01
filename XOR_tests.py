import time
from ANN import combinator
from ANN.DataSet import DataSet

# training parameters
max_iterations = 100000
min_error = .0001
log_each_iterations = 10000
max_tests = 6  # number of tests for each parameters combination
file_name = 'results/XOR_' + str(time.time()) + '.csv'  # name of the output file

# base parameters
num_input = 2
num_output = 1

# parameter datasets
num_hidden_ds = [2]
learning_rate_ds = [0.99, 0.9, 0.01, 0.02, 0.001, 0.002]
momentum_ds = [0, 0.8, 0.85, 0.9]

# create the data set
ds = DataSet(num_input, num_output, [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
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



num_input = 2
num_output = 1

# create the data set
ds = DataSet(num_input, num_output, [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
], 1)

# training parameters
max_iterations = 20000
min_error = .0001
log_each_iterations = 10000000000

# parameter datasets
num_hidden_ds = [2]
learning_rate_ds = [0.01, 0.02, 0.1, 0.2, 0.9, 0.99]
momentum_ds = [0, 0.8, 0.85, 0.9]

total_combinations = len(num_hidden_ds) * len(learning_rate_ds) * len(momentum_ds)

with open('results/XOR_' + str(datetime.datetime.now()) + '.csv', 'wb') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=';')
    wr.writerow(['ID', 'Type', 'Hidden', 'Learning rate', 'Momentum', 'Final error', 'Epochs'])

    # test all possible parameter combinations
    combination_number = 1
    for num_hidden in num_hidden_ds:
        for learning_rate in learning_rate_ds:
            for momentum in momentum_ds:
                print('Combinacao ' + str(combination_number) + ' de ' + str(total_combinations))
                for i in range(5):
                    mlp = MLP(num_input, num_hidden, num_output)
                    mlp.batch_train(ds.training_patterns, max_iterations, learning_rate, momentum, min_error, log_each_iterations)
                    wr.writerow(['C' + str(combination_number), 'batch', num_hidden, str(learning_rate).replace('.', ','), str(momentum).replace('.', ','), str(round(mlp.error, 6)).replace('.', ','), mlp.iteration])
                combination_number += 1


from ANN.DataSet import DataSet
import FB_correlation
from ANN import combinator
import time

# training parameters
max_iterations = 1000
min_error = .0001
log_each_iterations = 100
max_tests = 5  # number of tests for each parameters combination
file_name = 'FB_' + str(time.time())

# base parameters
num_input = 18
num_output = 1

# parameter datasets
num_hidden_ds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
learning_rate_ds = [0.0001, 0.001, 0.1]
momentum_ds = [0, 0.8, 0.85, 0.9]

# create the data set
ds = DataSet(num_input, num_output, FB_correlation.ds_facebook_normalized, 0.8)

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
    100)
