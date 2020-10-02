import pandas as pd
import numpy as np
#import sklearn.cross_validation as cv
from skimage.filters import threshold_otsu
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from skimage import img_as_ubyte
from skimage import data
from skimage.filters.rank import median
from skimage.morphology import disk

def get_data():
    dataset = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    target = dataset[[0]].values.ravel()
    train = dataset.iloc[:,1:].values
    target = target.astype(np.uint8)
    train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
    test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)
    return train,target,test

def preprocess(X,num):
    flat = []
    for i in range(0,num):
        thresh = threshold_otsu(X[i][0])
        X[i][0] = X[i][0] > thresh
        noisy_image = img_as_ubyte(X[i][0])
        noise = np.random.random(noisy_image.shape)
        noisy_image[noise > 0.99] = 255
        noisy_image[noise < 0.01] = 0
        X[i][0] = median(noisy_image, disk(1))
        flat.append(X[i][0].flatten())
    X_preprocessed = np.array(flat, 'float64')
    return X_preprocessed

train,target,test = get_data()
X_preprocessed = preprocess(train,42000)
test_preprocessed = preprocess(test,28000)

model = SVC()
model.fit(X_preprocessed[:15000], target[:15000])

#print(np.mean(cv.cross_val_score(model, X_preprocessed[:15000], target[:15000])))

pred = model.predict(test_preprocessed)

np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')