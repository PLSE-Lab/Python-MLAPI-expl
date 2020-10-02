import pandas as pd
import random as rand

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as output

numClusters = 10

digits = []
for row in test.values:
    digits.append(row)

centroids = []
for count in range(0, numClusters):
    index = int(rand.random() * len(test.values))
    centroids.append(digits[index])

finished = 0
while finished == 0:
    for centroid in centroids:
        for row in digits:
            print("Hello")
    finished = 1;
