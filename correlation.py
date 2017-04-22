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

t = 400  # amostral number
avg = [0.0] * 19
sum_corr = [0.0] * 19

# create one array for each columns on the facebook dataset to calculate the correlations between columns
columns = []
for i in range(19):
    col = []
    for j in range(t):  # calculate the correlation only for the first "t" rows
        col.append(ds_facebook[j][i])
    columns.append(np.array(col))

# calculate the linear correlation for all columns
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
#  max_corr_var will be the index of the column on the facebook dataset that will be the output of the network
max = 0
max_corr_var = -1
for i, corr in enumerate(sum_corr):
    if sum_corr[i] > max:
        max = sum_corr[i]
        max_corr_var = i

# get the minimum and maximum value for each column of the dataset
# we will use these values to normalize the dataset
max_diff_cols = []
for col in columns:
    max = np.max(col)
    min = np.min(col)
    max_diff_cols.append({
        'max': max,
        'min': min,
        'diff': max - min
    })

# create the normalized patterns dataset for the neural network,
# >> putting the max_corr_var column at the end of the dataset
# >> we will normalize all attributes
ds_facebook_normalized = []
for row in ds_facebook:
    row_aux = []

    # add all columns that aren't the max_corr_var column
    for i, col in enumerate(row):
        if i != max_corr_var:
            row_aux.append((col - max_diff_cols[i]['min']) / max_diff_cols[i]['diff'])  # X[i] - min(X) / (max(X) - min(X)

    # finally, add the max_corr_var column at the end of the row (will be the output of the network)
    row_aux.append((row[max_corr_var] - max_diff_cols[i]['min']) / max_diff_cols[i]['diff'])

    ds_facebook_normalized.append(row_aux)