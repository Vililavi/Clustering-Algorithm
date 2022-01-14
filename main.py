import numpy as np
from cluster import gaussianMixtureClustering, cluster


def main():
    print("Enter the location of the input file (n/PATH):")
    path = input()

    k = 0
    while k < 1:
        print("Enter the k value:")
        k = int(input())
    data = []

    if path != "n":
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
                f.close()
        except FileExistsError:
            print("Error")

        for i in lines:
            i.replace("\n", "")
            data.append(i.split())

        data = np.array([list(map(int, i)) for i in data])

    else:
        print("Shutting down")

    print("By which algorithm do you want to use?:\n"
          "1. Gaussian Mixture Clustering\n"
          "2. Dummy Clustering (by Vili Lavikainen)")

    chose = input()
    if chose == '1':
        gaussianMixtureClustering(data, k)
    elif chose == '2':
        cluster(data, k)
    else:
        print("It seem you can't follow instructions")


main()
