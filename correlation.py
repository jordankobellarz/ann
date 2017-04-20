import numpy as np
import csv

ds_facebook = []

#  get the facebook dataset in memory
with open('dataset_Facebook.csv', 'rb') as csv_file:
    reader = csv.reader(csv_file, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
    for j, row in enumerate(reader):
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

columns = []
for i in range(19):
    col = []
    for j in range(t):
        col.append(ds_facebook[j][i])
    columns.append(np.array(col))

for i in range(19):

    # average
    avg_i = np.sum(columns[i]) / t
    std_i = np.std(columns[i])

    for j in range(19):
        if j != i:
            avg_j = np.sum(columns[j]) / t
            std_j = np.sum(columns[j])

            cov_X_Y = 0
            for l in range(t):
                cov_X_Y += (columns[i][l] - avg_i) * (columns[j][l] - avg_j)
            r = cov_X_Y / (std_i * std_j)

            sum_corr[i] += r

def max_sum_cor(sum_corr):
    max = 0
    max_i = -1
    for i, corr in enumerate(sum_corr):
        if sum_corr[i] > max:
            max = sum_corr[i]
            max_i = i
    return max_i, max

print(max_sum_cor(sum_corr)[0])
