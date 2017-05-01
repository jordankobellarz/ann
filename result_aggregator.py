import numpy as np
import csv

file_name = 'results/XOR_1493650844.09'

results = []
with open(file_name + '.csv', 'rb') as file:
    reader = csv.reader(file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    next(reader, None) # skip the header
    results = list(reader)

with open(file_name + '_aggregated.csv', 'wb') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC, delimiter=';')

    wr.writerow(['ID', 'Type', 'Hidden', 'Learning rate', 'Momentum', 'Final error', 'Epochs', 'Total time'])

    last_row = []
    config_rows = []

    for row in results:

        # if is a new configuration
        if len(last_row) == 0 or last_row[0] != row[0]:

            # aggregate to find the median value of the last config rows
            if len(config_rows) > 0:
                np_config_rows = np.array(config_rows).astype(np.float)
                last_row[2] = int(last_row[2])  # minimum error
                last_row[5] = str(np.min(np_config_rows[:, 0])).replace('.', ',')  # minimum error
                last_row[6] = str(np.min(np_config_rows[:, 1])).replace('.', ',')  # minimum number of iterations
                last_row[7] = str(round(last_row[7], 3)).replace('.', ',') + 's'

                wr.writerow(last_row)

            # set the new config ID and reset the rows for this new config
            last_row = row
            config_rows = []

        config_rows.append([
            row[5].replace(',', '.'),  # final error
            row[6]  # iterations
        ])




