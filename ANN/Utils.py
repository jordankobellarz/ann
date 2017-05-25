import math
import random

# Euclidean distance between two scalars or lists
# https://en.wikipedia.org/wiki/Euclidean_distance
def euclidean_distance(p, q):
    result = 0

    if isinstance(p, list) and isinstance(q, list) and len(p) != len(q):
        raise Exception("Both p and q needs to have the same length.")

    # if listA and listB are scalar values, transform to list before
    if not isinstance(p, list) and not isinstance(q, list):
        p = [p]
        q = [q]

    # sum all distances
    for i in range(len(p)):
        result += math.pow(p[i] - q[i], 2)

    # return the square root
    return math.sqrt(result)


# set a label (center index) for each input
def clusterize(inputs, centers):

    closest_center_indexes = [-1] * len(inputs)

    for i, input in enumerate(inputs):

        closest_center_index = -1
        min_euclidean_distance = -1.0

        for j, center in enumerate(centers):
            distance = euclidean_distance(input, center)
            if distance < min_euclidean_distance or closest_center_index == -1:
                closest_center_index = j
                min_euclidean_distance = distance

        closest_center_indexes[i] = closest_center_index

    clusters = [None] * len(centers)
    for i in range(len(centers)):
        cluster = []
        for j in range(len(inputs)):
            if closest_center_indexes[j] == i:
                cluster.append(j)
        clusters[i] = cluster

    return clusters


# K-means algorithm
# it partitionate the 'inputs' in 'k' clusters
# from https://pt.coursera.org/learn/machine-learning/lecture/93VPG/k-means-algorithm
# return the centers for each of the k clusters
def k_means(inputs, k):

    # select k random inputs to initialize the centers for each cluster
    centers = []
    inputsAux = inputs[:]
    random.shuffle(inputsAux)
    for i in range(k):
        centers.append(inputsAux[i])

    # TODO: trocar a condicao de parada por alguma heuristica
    iteration = 0
    while iteration < 10000:

        # get the closest center index for each input
        clusters = clusterize(inputs, centers)

        # move the centers
        for i in range(len(centers)):

            # get all inputs near to this center
            cluster_input_indexes = clusters[i]

            # calculate the mean for all inputs near to this center
            count = len(cluster_input_indexes)
            if count > 0:
                new_center = [0.0] * len(centers[i])
                for j in range(len(new_center)):
                    for input_index in cluster_input_indexes:
                        new_center[j] += inputs[input_index][j]
                    new_center[j] /= count
                centers[i] = new_center

        iteration += 1

    return centers

# https://www.youtube.com/watch?v=OUtTI99uRf4
def k_nearest_neighbors(inputs, centers):

    # clusterize the inputs
    clusters = clusterize(inputs, centers)

    # calculate the mean radius for all values
    radius = [[]] * len(centers)
    for i in range(len(centers)):

        # get the ce
        center = centers[i]
        sums = [0.0] * len(center)
        cluster_input_indexes = clusters[i]

        # sum the distances for each input on this cluster
        for input_index in cluster_input_indexes:
            for j in range(len(center)):
                sums[j] += math.pow(center[j] - inputs[input_index][j], 2)

        # calculate the medium euclidean distance between all neighbors
        for j in range(len(center)):
            number_of_neighbors = len(cluster_input_indexes)
            sums[j] = math.sqrt(sums[j] / number_of_neighbors)

        radius[i] = sums

    return radius


# show informations about a cluster
def dump_clusters(inputs, centers, radius):

    clusters = clusterize(inputs, centers)

    # show information about clusters
    for i, cluster in enumerate(clusters):
        print("\nCluster #" + str(i+1) + " >>> center:  " + str(centers[i]) + ", radius: " + str(radius[i]) + ", inputs:")
        for input_index in cluster:
            print(inputs[input_index])