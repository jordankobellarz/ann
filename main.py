from MLP import MLP
from DataSet import DataSet

num_input = 2
num_hidden = 2
num_output = 1

# create the network
mlp = MLP(num_input, num_hidden, num_output)

# create the data set
ds = DataSet(num_input, num_output, [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
])

mlp.train(ds.patterns, 0.01, 1)

'''
print("\n\nBatch training 1000 epochs")
mlp.train(ds.patterns, 0.01, 500)
print(mlp.feed_forward([0, 0]))
print(mlp.feed_forward([1, 0]))
print(mlp.feed_forward([0, 1]))
print(mlp.feed_forward([1, 1]))

print("\n\nBatch training 10000 epochs")
mlp.train(ds.patterns, 0.01, 10000)
print(mlp.feed_forward([0, 0]))
print(mlp.feed_forward([1, 0]))
print(mlp.feed_forward([0, 1]))
print(mlp.feed_forward([1, 1]))

print("\n\nBatch training 100000 epochs")
mlp.train(ds.patterns, 0.01, 100000)
print(mlp.feed_forward([0, 0]))
print(mlp.feed_forward([1, 0]))
print(mlp.feed_forward([0, 1]))
print(mlp.feed_forward([1, 1]))
'''


