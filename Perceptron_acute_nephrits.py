from ANN import Perceptron
import Datasets

perceptron = Perceptron.Model(num_inputs=6)

ds = Datasets.acute_nephritis()
perceptron.online_train(ds.training_patterns, learning_rate=0.9,
                       min_error=0.001, max_iterations=-1, log_each_iterations=1)
perceptron.test(ds.testing_patterns, min_error=0.2)


