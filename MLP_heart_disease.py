from ANN.MLP import Model
import Datasets

# create the model
mlp = Model(num_input=22)
mlp.add_layer(num_neurons=20)  # hidden layer
mlp.add_layer(num_neurons=1)  # output layer

ds = Datasets.heart_disease()
mlp.online_train(ds.training_patterns, learning_rate=0.3, momentum=0.9,
                min_error=0.00001, max_iterations=-1, log_each_iterations=10)
mlp.test(ds.testing_patterns, min_error=0.4999)