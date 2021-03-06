import math
import random
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot
from numpy import unique, where
from scipy import stats
from sklearn import mixture

from print import printPartition, printCentroids


def distance(data1, data2):
    """
    Calculates euclidean distance between two data points.
    :param data1: Data point as array (x,y)
    :param data2: Data point as array (x,y)
    :return: Euclidean distance between two data points
    """
    data1 = list(map(int, data1))
    data2 = list(map(int, data2))
    return math.sqrt((data1[0] - data2[0]) ** 2 + (data1[1] - data2[1]) ** 2)


def SSE(data, centroids):
    """
    Sum-of-squared errors
    :param data: Data = [[x, t], [x, y]]
    :param centroids: Centroids = [[x, y], [x, y]]
    :return: SSE
    """

    distances = []
    for i in centroids:
        for j in data:
            distances.append(distance(i, j) ** 2)

    sse = 0
    for i in distances:
        sse += i

    return sse


def NNS(data, dataset):
    """
    Nearest neighbour search
    :param data: data point [x, y]
    :param dataset: set of data points [[x, y], [x, y]...]
    :return: nearest neighbour [x, y]
    """

    nearest = []
    dist = math.inf

    for i in dataset:
        tmpDist = distance(data, i)
        if tmpDist < dist:
            dist = tmpDist
            nearest = i

    return nearest


def random_swap(centroids, data):
    """
    Swaps one cluster to random point in data space
    :param centroids: list of centroids [[x, y], [x, y]]
    :param data: list of dataset [[x, y], [x, y], [x, y]...]
    :return: new centroid list [[x, y], [x, y]]
    """

    newCentroids = []
    swap = random.randrange(0, len(centroids))

    Max_x = 0
    Min_x = math.inf
    Max_y = 0
    Min_y = math.inf

    # find max and min in both values in list
    for i in data:
        if Max_x < i[0]:
            Max_x = i[0]

        if Min_x > i[0]:
            Min_x = i[0]

        if Max_y < i[1]:
            Max_y = i[1]

        if Min_y > i[1]:
            Min_y = i[1]

    for i in range(len(centroids)):
        if i == swap:
            newCentroids.append([random.randrange(Min_x, Max_x),
                                 random.randrange(Min_y, Max_y)])
        else:
            newCentroids.append(centroids[i])

    return newCentroids


def optimalPartition(dataset, centroid_set):
    """

    :param dataset: dataset [[x, y], [x, y], [x, y]...]
    :param centroid_set: centroid set [[x, y], [x, y]]
    :return: partitions [[[x, y], cluster], [[x, y], cluster]...]
    """

    partition = []
    nearest = []

    for i in dataset:
        nearest.append(NNS(i, centroid_set))

    for i in range(len(dataset)):
        calc = 1
        for j in centroid_set:
            if nearest[i] == j:
                partition.append(calc)
            else:
                calc += 1

    return partition


def centroid_point(data):
    avg_x = 0
    avg_y = 0
    n = len(data)
    if n > 0:
        for i in data:
            avg_x += float(i[0])
            avg_y += float(i[1])

    return [round(avg_x / n), round(avg_y / n)]


def centroid_step(partition, data, k):
    centroid = []
    tmp_data = []
    clusters = unique(partition)

    for i in clusters:
        for j in range(len(partition)):
            if i == partition[j]:
                tmp_data.append(data[j])

        centroid.append(centroid_point(tmp_data))
        tmp_data.clear()

    data_min = np.amin(data)
    data_max = np.amax(data)
    while len(centroid) < k:
        centroid.append([random.randrange(data_min, data_max),
                         random.randrange(data_min, data_max)])

    return centroid


def cluster(data, k):
    """
    Calculates k amount of clusters to data

    :param data: numpy array of shape {1 2}
    :param k: number of clusters
    """

    Max_x = 0
    Min_x = math.inf
    Max_y = 0
    Min_y = math.inf

    # find max and min in both values in list
    for i in data:
        if Max_x < i[0]:
            Max_x = i[0]

        if Min_x > i[0]:
            Min_x = i[0]

        if Max_y < i[1]:
            Max_y = i[1]

        if Min_y > i[1]:
            Min_y = i[1]

    # pick two random numbers between min and max values
    i = 0
    centroids = []
    while i < k:
        tmp = [random.randrange(Min_x, Max_x + 1), random.randrange(Min_x, Max_y + 1)]
        centroids.append(tmp)
        i += 1

    sse = 0
    j = 1
    tmp_sse = 1
    clusters = []
    tmp_centroids = centroids
    partitions = optimalPartition(data, centroids)

    f = open("Output/stats.csv", 'w')
    f.write("SSE-value;Number of active centroids(n);Number of active centroids(%)\n")

    # Main loop
    while tmp_sse != sse:

        """
        10 ajoa datalla S1

        Parempien rando swappien m????r??:
        Ajo 1: 36
        Ajo 2: 22
        Ajo 3: 42
        Ajo 4: 49
        Ajo 5: 11
        Ajo 6: 16
        Ajo 7: 2
        Ajo 8: 25
        Ajo 9: 31
        Ajo 10: 18
        """
        #Loop for exercise 2.6
        if j == 5:
            before_sse = SSE(data, centroids)
            print("SSE-value before random swap:", before_sse)
            times = 0
            for i in range(100):
                centroids = random_swap(centroids, data)
                swap_sse = SSE(data, centroids)

                if before_sse > swap_sse:
                    times += 1
                    print(times, ": Better SSE:", swap_sse)
        sse = tmp_sse

        centroids = centroid_step(partitions, data, k)
        partitions = optimalPartition(data, centroids)
        clusters = unique(partitions)
        tmp_sse = SSE(data, centroids)

        active_centroids = 0

        for i in range(len(centroids)):
            if centroids[i] != tmp_centroids[i]:
                active_centroids += 1

        f.write(str(round(tmp_sse)) + ';' + str(active_centroids)
                + ";" + str((active_centroids / len(centroids)) * 100) + '\n')

        tmp_centroids = centroids

        j += 1

    f.close()

    # create scatter plot for samples from each cluster
    for i in clusters:
        row_ix = where(partitions == i)
        pyplot.scatter(data[row_ix, 0], data[row_ix, 1])

    # add centroids to the plot
    for i in centroids:
        pyplot.scatter(i[0], i[1], s=20, c="black")

    d = pd.read_csv("Output/stats.csv", header=0, delimiter=';')
    print(d)

    # print to both .txt files
    printPartition(data, partitions)
    printCentroids(centroids)

    pyplot.show()


def gaussianMixtureClustering(data, k):
    """
    Clusters the data and visualizes it and writes partition and centroids to files

    :param data: numpy array of shape {1 2}
    :param k: number of clusters
    """

    # define the model
    model = mixture.GaussianMixture(n_components=k)

    # fit the model
    model.fit(data)

    # assign a cluster to each example
    yhat = model.predict(data)

    # retrieve unique clusters
    clusters = unique(yhat)

    # create scatter plot for samples from each cluster
    for i in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == i)
        # create scatter of these samples
        pyplot.scatter(data[row_ix, 0], data[row_ix, 1])

    # calculate centroids
    centers = np.empty(shape=(model.n_components, data.shape[1]))
    for i in range(model.n_components):
        density = scipy.stats.multivariate_normal(cov=model.covariances_[i], mean=model.means_[i]).logpdf(data)
        centers[i, :] = data[np.argmax(density)]
    pyplot.scatter(centers[:, 0], centers[:, 1], s=20, c="black")

    # Shows the plot
    pyplot.show()

    # print partition and centroid files
    printPartition(data, yhat)
    printCentroids(centers)
