import os

f = open("changed_file_names.txt", "a")

def main():
    path = '../input/20news-18828/'
    count = 10000

    for root, dirs, files in os.walk(path):
        for i in files:
            os.rename(os.path.join(root, i), os.path.join(root, str(count) + ".txt"))
            f.write(root+"/"+i + "\t" + str(count) + ".txt" + "\n")
            count += 1


if __name__ == '__main__':
    main()
f.close()