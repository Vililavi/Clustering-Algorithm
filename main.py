def main():
    print("Enter the location of the input file:")
    path = input()

    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            f.close()
    except FileExistsError:
        print("An error occurred with reading file")

    data = []
    for i in lines:
        i.replace("\n", "")
        data.append(i.split())

    cluster(data)


def cluster(data):
    print(data)


main()
