class DataSet:

    def __init__(self, num_inputs, num_outputs, flat_patterns, percentage_training_patterns=0.8):
        """
        Transform the flat patterns matrix in two lists: input and expected
        :param num_inputs: number of inputs 
        :param num_outputs: number of outputs
        :param flat_patterns: a matrix with all patterns 
        :param percentage_training_patterns: percentage of training patterns
        """

        self.training_patterns = []
        self.testing_patterns = []

        self.num_training_patterns = int(len(flat_patterns) * percentage_training_patterns)
        self.num_testing_patterns = len(flat_patterns) - self.num_training_patterns

        # verify if all patterns match the correct number of inputs and outputs
        entries_per_pattern = num_inputs + num_outputs
        for pattern in flat_patterns:
            if len(pattern) < entries_per_pattern:
                raise Exception("Each pattern needs to have " + str(entries_per_pattern) + " values: "
                                + str(num_inputs) + " for input and " + str(num_outputs) + " for the expected output.")

        # create patterns list for inputs and expected outputs
        for i, pattern in enumerate(flat_patterns):
            input = pattern[0:num_inputs]
            desired = pattern[num_inputs:num_inputs + num_outputs]
            if i < self.num_training_patterns:
                self.training_patterns.append({
                    'input': input,
                    'desired': desired
                })
            else:
                self.testing_patterns.append({
                    'input': input,
                    'desired': desired
                })

    @staticmethod
    def getValues(patterns, type='input'):
        values = []
        for pattern in patterns:
            values.append(pattern[type])
        return values