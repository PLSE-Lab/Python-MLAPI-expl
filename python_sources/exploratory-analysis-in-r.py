import csv

with open('../input/train.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row)


#train = read.csv("../input/train.csv")
#test  = read.csv("../input/test.csv")

