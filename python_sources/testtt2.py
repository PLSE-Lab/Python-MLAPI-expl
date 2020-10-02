import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog

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

def hog_fe(df,num): # df is a [48k,1] shape containing 28x28 matrices
    images = np.array(df).reshape((-1, 1, 28, 28)).astype(np.uint8)
    hog_features_list = []
    for i in range(0,num):
        image = images[i][0]
        fd = hog(image, orientations=9, pixels_per_cell=(2, 2), cells_per_block=(1, 1), visualise=False)
        hog_features_list.append(fd)
    hog_features = np.array(hog_features_list, 'float64')
    return hog_features

train,target,test = get_data()
X_preprocessed = preprocess(train,42000)
test_preprocessed = preprocess(test,28000)

hog_features_train = hog_fe(X_preprocessed,42000)
hog_features_test = hog_fe(test_preprocessed,28000)

model = RandomForestClassifier(n_estimators=150)
model.fit(hog_features_train,target)

pred = model.predict(hog_features_test)

np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')