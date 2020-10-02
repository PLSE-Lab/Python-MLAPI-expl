import pandas as pd
import numpy as np
from skimage.filters import threshold_otsu
from sklearn.ensemble import RandomForestClassifier

from skimage import img_as_ubyte
from skimage import data
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.filters import sobel

def get_data():
    dataset = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    target = dataset[[0]].values.ravel()
    train = dataset.iloc[:,1:].values
    target = target.astype(np.uint8)
    train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
    test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)
    return train,target,test

def preprocess_and_sobel(X,num):
    
    sobel_images = []

    for i in range(0,num):

        thresh = threshold_otsu(X[i][0])
        X[i][0] = X[i][0] > thresh

        noisy_image = img_as_ubyte(X[i][0])
        noise = np.random.random(noisy_image.shape)
        noisy_image[noise > 0.99] = 255
        noisy_image[noise < 0.01] = 0
        X[i][0] = median(noisy_image, disk(1))
        
        sobel_image = sobel(X[i][0])
        sobel_images.append(sobel_image.flatten())

    return np.array(sobel_images, 'float64')

train,target,test = get_data()
sobel_images_train = preprocess_and_sobel(train,42000)
sobel_images_test = preprocess_and_sobel(test,28000)

model = RandomForestClassifier(n_estimators=150)
model.fit(sobel_images_train,target)

pred = model.predict(sobel_images_test)

np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')