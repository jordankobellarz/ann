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

# save the dataset as csv
np.savetxt('datasets/acute_nephritis.csv', np_ds, delimiter=";", fmt='%s')

# get the normalized dataset
normalized_ds = np_ds.astype(np.float).tolist()

num_input = 6
num_hidden = 1
num_output = 1

# create the network
mlp = MLP.Net(num_input, num_hidden, num_output)

# create the data set (90 samples to train and 30 to test)
ds = DataSet(num_input, num_output, normalized_ds, 0.75)

mlp.batch_train(ds.training_patterns, max_iterations=-1, learning_rate=0.1,
                 momentum=0.1, min_error=0.0001, log_each_iterations=10)

mlp.test(ds.testing_patterns)