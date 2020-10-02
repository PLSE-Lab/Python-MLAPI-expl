import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


def InformationFirstOrder(arr, AveragedOverAllFrequencies):
    if arr == 0:
        return 0
    else:
        Pk = arr/AveragedOverAllFrequencies
    return Pk*math.log(Pk)

informationFirstOrder = np.vectorize(InformationFirstOrder)


def InformationSecondOrder(arr, AveragedOverAllFrequenciesSquared):
    if arr == 0:
        return 0
    else:
        Pk = arr**2/AveragedOverAllFrequenciesSquared
    return Pk*math.log(Pk)

informationSecondOrder = np.vectorize(InformationSecondOrder)


def MungeData(im,
              imageheight=28,
              imagewidth=28,
              cellheight=4,
              cellwidth=4,
              stride=2):
    newimage = np.zeros((int(imagewidth/stride),
                         int(imageheight/stride)))
    for row in range(int(imagewidth/stride)-1):
        for col in range(int(imageheight/stride)-1):
            cell = (im[stride * row:stride*(row) + cellwidth,
                       stride * col:stride * col + cellheight]
                    ).reshape(1, cellheight * cellwidth)
            newimage[row, col] = (-informationFirstOrder(cell,
                                                         cell.sum()).sum())
    return newimage.ravel()


if __name__ == "__main__":
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    # == Remove - script takes 20 minutes== #
    train = train[:1000]
    test = test[:1000]
    # == Remove - script takes 20 minutes== #
    trainlabels = train.label.values
    traindata = None
    for i in range(train.shape[0]):
        im = train.iloc[i][1:].reshape(28, 28)
        im = im.reshape(28, 28)
        imentropy = MungeData(im)
        if(traindata is None):
            traindata = np.zeros((train.shape[0], len(imentropy)))
        traindata[i, :] = imentropy
    print('Finished Building Train')

    testdata = None
    for i in range(test.shape[0]):
        im = test.iloc[i].reshape(28, 28)
        im = im.reshape(28, 28)
        imentropy = MungeData(im)
        if(testdata is None):
            testdata = np.zeros((test.shape[0], len(imentropy)))
        testdata[i, :] = imentropy

    print('Finished Building Test')
    knn = KNeighborsClassifier(n_neighbors=5)
    mm = MinMaxScaler()
    knn.fit(mm.fit_transform(traindata), trainlabels)
    p = knn.predict(mm.transform(testdata))
    testsubmission = pd.DataFrame({'ImageId': range(1,
                                                    p.shape[0]+1),
                                   'Label': p})
    testsubmission.to_csv('knnsubmission1.csv', index=False)