import numpy as np
from cluster import gaussianMixtureClustering, cluster


def main():
    print("Enter the location of the input file (n/PATH):")
    print("(input files data must be in format:")
    print(" 1234 1234")
    print(" 4321 4321)")
    path = input()

    k = 0
    while k < 1:
        print("Enter the k value:")
        try:
            k = int(input())
        except ValueError:
            print("Value is not a number")
    data1 = []
    data2 = []

    if path != "n":
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
                f.close()
        except FileExistsError:
            print("Error")

        for i in lines:
            i.replace("\n", "")
            data1.append(list(map(int, i.split())))

        data2 = np.array([list(map(int, i)) for i in data1])

    else:
        print("Shutting down")

    print("By which algorithm do you want to use?:\n"
          "1. Gaussian Mixture Clustering\n"
          "2. Dummy Clustering (by Vili Lavikainen)")

    chose = input()
    if chose == '1':
        gaussianMixtureClustering(data2, k)
    elif chose == '2':
        cluster(data2, k)
    else:
        print("It seem you can't follow instructions")


main()
