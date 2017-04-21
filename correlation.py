import numpy as np
import csv

ds_facebook = []

#  get the facebook dataset in memory
with open('dataset_Facebook.csv', 'rb') as csv_file:
    reader = csv.reader(csv_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        aux = []
        for i, col in enumerate(row):
            if type(col) is float:
                aux.append(float(col))
            else:
                aux.append(0)
        ds_facebook.append(aux)

t = 400
avg = [0.0] * 19
sum_corr = [0.0] * 19

# create one array for each columns on the facebook dataset to calculate the correlations between columns
columns = []
for i in range(19):
    col = []
    for j in range(t):
        col.append(ds_facebook[j][i])
    columns.append(np.array(col))

for i in range(19):

    avg_i = np.sum(columns[i]) / t  # feature i average
    std_i = np.std(columns[i])  # feature i standard deviation

    for j in range(19):
        if j != i:
            avg_j = np.sum(columns[j]) / t  # feature j average
            std_j = np.sum(columns[j])  # feature j standard deviation

            cov_X_Y = 0
            for l in range(t):
                cov_X_Y += (columns[i][l] - avg_i) * (columns[j][l] - avg_j)

            r = cov_X_Y / (std_i * std_j)  # correlation between feature i and j

            sum_corr[i] += r # sum of all correlations for column i and other columns

# return the column index of the column with the maximum correlation with other columns
def max_corr_var(sum_corr):
    max = 0
    max_i = -1
    for i, corr in enumerate(sum_corr):
        if sum_corr[i] > max:
            max = sum_corr[i]
            max_i = i
    return max_i

#  max_corr_var that's the index of the column on the facebook dataset that will be the output of the network
max_corr_var = max_corr_var(sum_corr)

max_diff_cols = []
for col in columns:
    max = np.max(col)
    min = np.min(col)
    max_diff_cols.append({
        'max': max,
        'min': min,
        'diff': max - min
    })

# create the patterns dataset for the neural network,
# >> putting the max_corr_var column at the end of the dataset
# >> we will normalize all attributes, dividing 1 by then
ds_facebook_normalized = []
for row in ds_facebook:
    row_aux = []

    # add all columns that aren't the max_corr_var column
    for i, col in enumerate(row):
        if i != max_corr_var:
            if col != 0:
                row_aux.append((col - max_diff_cols[i]['min']) / max_diff_cols[i]['diff'])  # X[i] - min(X) / (max(X) - min(X)
            else:
                row_aux.append(0)

    # finally, add the max_corr_var column at the end of the row (will be the output of the network)
    if row[max_corr_var] != 0:
        row_aux.append((row[max_corr_var] - max_diff_cols[i]['min']) / max_diff_cols[i]['diff'])
    else:
        row_aux.append(0)

    ds_facebook_normalized.append(row_aux)