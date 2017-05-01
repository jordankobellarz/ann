import csv
import time
from ANN.MLP import MLP

def test_combinations(num_input, num_output, dataset, num_hidden_ds, learning_rate_ds, momentum_ds, min_error=0.0001, max_iterations=-1, max_tests=3, file_name='results/DS.csv', log_each_iterations = 10000000):
    with open(file_name, 'wb') as file:

        # create the csv writer
        wr = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=';')
        wr.writerow(['ID', 'Type', 'Hidden', 'Learning rate', 'Momentum', 'Final error', 'Epochs', 'Total time'])

        # store the config ID
        config_id = 1

        total_tests = len(num_hidden_ds) + len(learning_rate_ds) + len(momentum_ds)

        # test all possible parameter combinations
        for num_hidden in num_hidden_ds:  # num_hidden
            for learning_rate in learning_rate_ds:  # learning_rate
                for momentum in momentum_ds:  # momentum
                    for i in range(max_tests):  # number of test for this configuration

                        percentage = (config_id * i + 1) / total_tests
                        print(str(percentage) + '%')

                        time_start = time.time()

                        # create the MLP and train it
                        mlp = MLP(num_input, num_hidden, num_output)
                        mlp.batch_train(dataset.training_patterns, max_iterations, learning_rate, momentum, min_error, log_each_iterations)

                        time_stop = time.time()

                        # log this configuration results
                        wr.writerow([
                            'C' + str(config_id), 'batch',  # config ID
                            num_hidden, str(learning_rate).replace('.', ','),  # number of hidden neurons
                            str(momentum).replace('.', ','),  # momentum
                            str(round(mlp.error, 6)).replace('.', ','),
                            mlp.iteration,
                            round((time_start - time_stop)/1000, 0)
                        ])

                    # compute the next config ID
                    config_id += 1