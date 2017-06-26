class DataSet:

    def __init__(self, num_inputs, num_outputs, ds, percentage_training_patterns=0.8):
        """
        Transform the dataset in two lists: input and expected
        :param num_inputs: number of inputs 
        :param num_outputs: number of outputs
        :param ds: a matrix with all patterns 
        :param percentage_training_patterns: percentage of training patterns
        """

        self.training_patterns = []
        self.testing_patterns = []

        self.num_training_patterns = int(len(ds) * percentage_training_patterns)
        self.num_testing_patterns = len(ds) - self.num_training_patterns

        # verify if there is conflicting patterns
        self.verify_conflicts(ds, num_inputs)

        # verify if all patterns match the correct number of inputs and outputs
        entries_per_pattern = num_inputs + num_outputs
        for pattern in ds:
            if len(pattern) < entries_per_pattern:
                raise Exception("Each pattern needs to have " + str(entries_per_pattern) + " values: "
                                + str(num_inputs) + " for input and " + str(num_outputs) + " for the expected output.")

        # create patterns list for inputs and expected outputs
        for i, pattern in enumerate(ds):
            input = self.get_input(pattern, num_inputs)
            desired = self.get_desired(pattern, num_inputs)
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
    def getValues(ds, type='input'):
        values = []
        for pattern in ds:
            values.append(pattern[type])
        return values

    def verify_conflicts(self, ds, num_inputs):
        conflict_indexes = []
        conflict_count = 0
        for i, pattern in enumerate(ds):
            input = self.get_input(pattern, num_inputs)
            desired = self.get_desired(pattern, num_inputs)
            for j, pattern_aux in enumerate(ds[i:len(ds)]):
                input_aux = self.get_input(pattern_aux, num_inputs)
                desired_aux = self.get_desired(pattern_aux, num_inputs)
                if input == input_aux and desired != desired_aux:  # a conflict!!
                    conflict_count += 1
                    print "Pattern " + str(i+1) + " and " + str(i+j+1) + " are conflicting!!"
                    if not (i+j) in conflict_indexes:
                        conflict_indexes.append(i+j)

        # remove conflicting indexes
        conflict_indexes.sort()
        ini_len = len(ds)
        for i in conflict_indexes:
            ds.pop(i - (ini_len - len(ds)))

        if conflict_count:
            print str(conflict_count) + " conflicting patterns found, " + str(len(conflict_indexes)) + " removed!!"

    def get_input(self, pattern, num_inputs):
        return pattern[0:num_inputs]

    def get_desired(self, pattern, num_inputs):
        return pattern[num_inputs:len(pattern)]