class DataSet:

    def __init__(self, num_inputs, num_outputs, flat_patterns):
        """
        Transform the flat patterns matrix in two lists: input and expected
        :param num_inputs: number of inputs 
        :param num_outputs: number of outputs
        :param flat_patterns: a matrix with all patterns 
        """

        self.patterns = []

        # verify if all patterns match the correct number of inputs and outputs
        entries_per_pattern = num_inputs + num_outputs
        for pattern in flat_patterns:
            if len(pattern) < entries_per_pattern:
                raise Exception("Each pattern needs to have " + str(entries_per_pattern) + " values: "
                                + str(num_inputs) + " for input and " + str(num_outputs) + " for the expected output.")

        # create patterns list for inputs and expected outputs
        for pattern in flat_patterns:
            self.patterns.append({
                'input': pattern[0:num_inputs],
                'desired': pattern[num_inputs:num_inputs + num_outputs]
            })
