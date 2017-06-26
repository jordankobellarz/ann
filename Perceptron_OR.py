from ANN import Perceptron
import Datasets

# create the network
perceptron = Perceptron.Model(num_inputs=2)

ds = Datasets.OR()
perceptron.online_train(ds.training_patterns, learning_rate=0.5,
                       min_error=0.001, max_iterations=-1, log_each_iterations=10)
perceptron.test(ds.training_patterns, min_error=0.1)


