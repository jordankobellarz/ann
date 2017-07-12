from MLP import Neuron

class Model():
    def __init__(self, num_inputs):
        self.iteration = 0  # iterations counter
        self.error = 0.0  # error for each iteration
        self.neuron = Neuron(num_inputs)

    def online_train(self, patterns, learning_rate=0.1, min_error=0.001, max_iterations=-1, log_each_iterations=100):

        print("\n>>> Training: ")
        while self.iteration < max_iterations or max_iterations == -1:

            self.error = 0.0

            for pattern in patterns:
                input = pattern['input']
                output = self.neuron.activate(input)

                error = pattern['desired'][0] - output
                self.error += error ** 2

                self.neuron.bias_delta = error
                for i in range(len(self.neuron.deltas)):
                    self.neuron.deltas[i] += error * input[i]
                self.neuron.update_weights(learning_rate)

            self.iteration += 1
            self.error /= len(patterns)
            if self.error <= min_error:
                self.log_iteration(log_each_iterations, True)
                break
            self.log_iteration(log_each_iterations)

    def log_iteration(self, log_each_iterations, force=False):
        if not self.iteration % log_each_iterations and self.iteration != 0 or force:
            print("iteration: " + str(self.iteration) + " error: " + str(self.error))

    def test(self, patterns, min_error):
        correct_count = 0
        print("\n>>> Testing: ")
        for i, pattern in enumerate(patterns):
            desired = pattern['desired'][0]
            output = self.neuron.activate(pattern['input'])
            error = abs(desired - output)
            if error <= min_error:
                correct_count += 1
            print("#" + str(i + 1) + "\tdesired: " + str(desired) + "\tgot: "
                  + str(round(output, 3)) + "\terror: " + str(round(error, 5)))
        accuracy = round(float(correct_count) / float(len(patterns)) * 100, 2)
        print("\n>>> Accuracy: " + str(accuracy) + "%")