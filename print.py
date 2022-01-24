def printPartition(data, partition):
    """
    data in format {    1234    1234} and partition in format {1}
    :param data numpy array of shape
    :param partition array of shape {p} where p is partition value
    """

    f = open("Output/partition.txt", 'w')
    j = 0
    for i in data:
        f.write(str(i)[1:-1].replace(',', '') + " " + str(partition[j]) + "\n")
        j += 1
    f.close()


def printCentroids(centroids):
    """
    centroids in format {    1234    1234}

    :param centroids in numpy array of shape
    """

    f = open("Output/centroid.txt", 'w')
    for i in centroids:
        f.write(str(i).replace('.', '').replace(',', '')[1:-1] + " " + "\n")
    f.close()
