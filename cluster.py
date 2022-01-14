import math
import numpy as np
import scipy
import random
from matplotlib import pyplot
from numpy import unique, where
from scipy import stats
from sklearn import mixture
from print import printPartition, printCentroids


def cluster(data, k):
    """
    Calculates k amount of clusters to data

    :param data: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :return: numpy array of shape (2, k)
    """
    Max_x = 0
    Min_x = math.inf
    Max_y = 0
    Min_y = math.inf

    for i in data:
        if Max_x < i[0]:
            Max_x = i[0]

        if Min_x > i[0]:
            Min_x = i[0]

        if Max_y < i[1]:
            Max_y = i[1]

        if Min_y > i[1]:
            Min_y = i[1]

    i = 0
    centers = []
    while i < k:
        tmp = [random.randrange(Min_x, Max_x+1), random.randrange(Min_x, Max_y+1)]
        centers.append(tmp)
        i += 1

    partitions = []
    for i in data:
        partitions.append(random.randrange(1, k+1))

    printPartition(data, partitions)
    printCentroids(centers)


def gaussianMixtureClustering(data, k):
    """
    Clusters the data and visualizes it and writes partition and centroids to files

    :param data: numpy array of shape (r, 2), where r is the number of rows
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

    # show the plot
    pyplot.show()

    # print partition and centroid files
    printPartition(data, yhat)
    printCentroids(centers)
