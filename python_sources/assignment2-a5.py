import pandas as pd
import numpy as np
import sklearn.cross_validation as cv
from skimage.filters import threshold_otsu
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from skimage.feature import daisy
import skimage.transform as transform

import skimage
from skimage import img_as_ubyte
from skimage import data
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.filters import sobel

def get_data():
    dataset = pd.read_csv("../input/train.csv")
    #test = pd.read_csv("../input/test.csv")
    target = dataset[[0]].values.ravel()
    train = dataset.iloc[:,1:].values
    target = target.astype(np.uint8)
    train = np.array(train).reshape((-1, 1, 28, 28)).astype(np.uint8)
    #test = np.array(test).reshape((-1, 1, 28, 28)).astype(np.uint8)
    return train,target
    #,test

def preprocess_and_corners(X,num):
    
    images_p = []
    images_n = []
    original = []

    for i in range(0,num):

        thresh = threshold_otsu(X[i][0])
        X[i][0] = X[i][0] > thresh

        noisy_image = img_as_ubyte(X[i][0])
        noise = np.random.random(noisy_image.shape)
        noisy_image[noise > 0.99] = 255
        noisy_image[noise < 0.01] = 0
        X[i][0] = median(noisy_image, disk(1))
        
        #corners_image = sobel(X[i][0])
        images_p.append(transform.rotate(X[i][0],45).flatten())
        images_n.append(transform.rotate(X[i][0],315).flatten())
        original.append(X[i][0].flatten())

    im1 = np.array(images_p, 'float64')
    im2 = np.array(images_n, 'float64')
    ori = np.array(original, 'float64')
    return im1, im2, ori

train,target = get_data()
#,test 

im1,im2,ori = preprocess_and_corners(train,42000)
#hog_features_test = preprocess_and_hog(test,28000)

model1 = RandomForestClassifier(n_estimators=100)
model2 = RandomForestClassifier(n_estimators=100)
model3 = RandomForestClassifier(n_estimators=100)
model1.fit(im1,target)
model2.fit(im2,target)
model3.fit(ori,target)

ensemble = VotingClassifier(estimators=[('45', model1), ('315', model2), ('0', model3)], voting='hard', weights=[1,1,2])

print(np.mean(cv.cross_val_score(ensemble, ori, target)))

#pred = model.predict(hog_features_test)
#np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')