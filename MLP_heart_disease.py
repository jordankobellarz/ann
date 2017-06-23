from ANN.DataSet import DataSet
from ANN import MLP
import csv

# http://archive.ics.uci.edu/ml/datasets/SPECT+Heart
ds = []
with open('datasets/heart_disease.csv') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        for i, value in enumerate(row):
            row[i] = float(value)
        ds.append(row)

num_input = 22
num_hidden = 20
num_output = 1

# create the network
mlp = MLP.Net(num_input, num_hidden, num_output)

# create the data set (80 samples to train and 187 to test)
ds = DataSet(num_input, num_output, ds, 0.3)

mlp.batch_train(ds.training_patterns, learning_rate=0.1, momentum=0.05,
                min_error=0.025, max_iterations=10000, log_each_iterations=10)

mlp.test(ds.testing_patterns, min_error=0.3)