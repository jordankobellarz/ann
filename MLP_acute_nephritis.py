from ANN.DataSet import DataSet
from ANN import MLP
import csv
import codecs
import numpy as np
from ANN.Utils import Normalizer

# gera o CSV a partir do arquivo .data
# http://archive.ics.uci.edu/ml/datasets/Acute+Inflammations
ds = []
with codecs.open('datasets/acute_nephritis.data', 'rU', 'utf-16') as csv_file:
    reader = csv.reader(csv_file, delimiter='\t')
    for row in reader:
        ds.append(row)

# remove the 7th column (Inflammation of urinary bladder)
np_ds = np.array(ds)
np_ds = np_ds[:, [0, 1, 2, 3, 4, 5, 7]]

# normalize the dataset
Normalizer.numbers(np_ds[:, 0])
Normalizer.yes_no(np_ds[:, 1])
Normalizer.yes_no(np_ds[:, 2])
Normalizer.yes_no(np_ds[:, 3])
Normalizer.yes_no(np_ds[:, 4])
Normalizer.yes_no(np_ds[:, 5])
Normalizer.yes_no(np_ds[:, 6])

# get the normalized dataset
normalized_ds = np_ds.astype(np.float).tolist()

num_input = 6
num_hidden = 1
num_output = 1

# create the network
mlp = MLP.Net(num_input, num_hidden, num_output)

# create the data set
ds = DataSet(num_input, num_output, normalized_ds, 0.8)

# training parameters
learning_rate = .09
momentum = .1
max_iterations = -1
min_error = .001
log_each_iterations = 10

mlp.batch_train(ds.training_patterns, max_iterations, learning_rate, momentum, min_error, log_each_iterations)

mlp.test(ds.testing_patterns)