import csv
import numpy
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures


def do(C=1.0, degree=2):
    print ('reading training data...')
    trainingData = []
    trainingLabels = []
    with open('../input/train.csv', 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if reader.line_num == 1: continue
            trainingData.append([1 if int(i) > 0 else 0 for i in line[1:]])
            trainingLabels.append(int(line[0]))

    print ('pca...')
    trainingData = numpy.array(trainingData)
    pca = PCA(50, whiten=True)
    pca.fit(trainingData)
    trainingData = pca.transform(trainingData)
    trainingData = [list(i) for i in trainingData]

    poly = PolynomialFeatures(degree)
    trainingData = poly.fit_transform(trainingData)

    model = LogisticRegression(solver='newton-cg', C=C)
    print ('training model...')
    model.fit(trainingData, trainingLabels)

    print ('validating...')
    validatingLabels = model.predict(trainingData[-3000:])
    print (validatingLabels)
    cnt = 0
    tmp = trainingLabels[-3000:]
    for idx in range(len(validatingLabels)):
        if(validatingLabels[idx] != tmp[idx]): cnt += 1
    print ((3000 - cnt) / 3000.0 * 100, '%')

    print ('reading testing data...')
    testingData = []
    with open('../input/test.csv', 'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if reader.line_num == 1: continue
            testingData.append([1 if int(i) > 0 else 0 for i in line])

    print ('pca...')
    testingData = numpy.array(testingData)
    testingData = pca.transform(testingData)
    testingData = [list(i) for i in testingData]

    print ('poly...')
    testingData = poly.fit_transform(testingData)

    print ('predicting...')
    labels = model.predict(testingData)

    print ('writing csv...')
    with open('res.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageId', 'Label'])
        writer.writerows([(i + 1, int(labels[i])) for i in range(len(labels))])


do(1.0)