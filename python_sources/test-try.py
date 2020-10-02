import csv
import pandas as np
import numpy as nump
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
# The competition datafiles are in the directory ../input
# Read competition data files:
train = np.read_csv("../input/train.csv")
test  = np.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as out

PCA_COMPONENTS = 100


def doWork(train, labels, test):
    print ("Converting training set to matrix")
    X_train = nump.mat(train)

    print ("Fitting PCA. Components: %d" % PCA_COMPONENTS)
    pca = decomposition.PCA(n_components=PCA_COMPONENTS).fit(X_train)

    print ("Reducing training to %d components" % PCA_COMPONENTS)
    X_train_reduced = pca.transform(X_train)

    print ("Fitting kNN with k=10, kd_tree")
    knn = KNeighborsClassifier(n_neighbors=100, algorithm="kd_tree")
    print (knn.fit(X_train_reduced, labels))

    print ("Reducing test to %d components" % PCA_COMPONENTS)
    X_test_reduced = pca.transform(test)

    print ("Preddicting numbers")
    predictions = knn.predict(X_test_reduced)

    print ("Writing to file")
    write_to_file(predictions)

    return predictions


def write_to_file(predictions):
    f = open("output-pca-knn-skilearn-v3.csv", "w")
    for p in predictions:
        f.write(str(p))
        f.write("\n")
    f.close()

def read_data(f, header=True, test=False):
    data = []
    labels = []

    csv_reader = csv.reader(open(f, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index = index + 1
        if header and index == 1:
            continue

        if not test:
            labels.append(int(row[0]))
            row = row[1:]

        data.append(nump.array(nump.int64(row)))
    return (data, labels)


if __name__ == '__main__':
    train, labels = read_data("../input/train.csv")
    test, tmpl = read_data("../input/test.csv", test=True)
print (doWork(train, labels, test))