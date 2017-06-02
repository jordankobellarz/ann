import math
import random


def euclidean_distance(p, q):
    """
    Calculate the euclidean distance between the points on the n-dimensional space p and q.
    Inspired from https://en.wikipedia.org/wiki/Euclidean_distance.
    :param p: an scalar or an list
    :param q: an scalar or an list (with the same size as p)
    :return: the euclidean distance between 
    """
    result = 0

    # verify if p and q are the same type and have the same size
    if isinstance(p, list) and isinstance(q, list) and len(p) != len(q):
        raise Exception("Both p and q needs to have the same length.")

    # if p and q are scalar values, transform to list before doing calculations
    if not isinstance(p, list) and not isinstance(q, list):
        p = [p]
        q = [q]

    # sum all distances
    for i in range(len(p)):
        result += math.pow(p[i] - q[i], 2)

    # return the square root
    return math.sqrt(result)


def clusterize(inputs, centers):
    """
    Set a label (center index) for each input.
    Classify each input according to the minimum euclidean distance from an center. 
    :param inputs: list of inputs
    :param centers: list of clusters centers
    :return: 
    """
    nearest_center_indexes = [-1] * len(inputs)

    for i, input in enumerate(inputs):

        nearest_center_index = -1
        min_euclidean_distance = -1.0

        # get the nearest center from this input
        for j, center in enumerate(centers):
            distance = euclidean_distance(input, center)
            if distance < min_euclidean_distance or nearest_center_index == -1:
                nearest_center_index = j
                min_euclidean_distance = distance

        nearest_center_indexes[i] = nearest_center_index

    clusters = [None] * len(centers)
    for i in range(len(centers)):
        cluster = []
        for j in range(len(inputs)):
            if nearest_center_indexes[j] == i:
                cluster.append(j)
        clusters[i] = cluster

    return clusters


def k_means(inputs, k):
    """
    K-means algorithm. It partitionate the 'inputs' in 'K' clusters
    Inspired from: https://pt.coursera.org/learn/machine-learning/lecture/93VPG/k-means-algorithm
    :param inputs: inputs to classify
    :param k: number of clusters to create
    :return: the center for each of the K clusters
    """

    # select k random inputs to initialize the centers for each cluster
    centers = []
    inputsAux = inputs[:]
    random.shuffle(inputsAux)
    for i in range(k):
        centers.append(inputsAux[i])

    # TODO: change this stop condition for an heuristic
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


def k_nearest_neighbors(inputs, centers):
    """
    Inspired from: https://www.youtube.com/watch?v=OUtTI99uRf4
    :param inputs: 
    :param centers: 
    :return: 
    """

    # clusterize the inputs
    clusters = clusterize(inputs, centers)

    # calculate the mean radius for all values
    radius = [[]] * len(centers)
    for i in range(len(centers)):

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
        print("\n Cluster #" + str(i+1) + " >>> center:  " + str(centers[i]) + ", radius: " + str(radius[i]) + ":")
        for input_index in cluster:
            print inputs[input_index]
    print("\n")