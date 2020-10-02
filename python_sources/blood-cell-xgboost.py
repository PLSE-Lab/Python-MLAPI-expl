# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from tqdm import tqdm
import cv2

import mahotas

# Any results you write to the current directory are saved as output.

TRAIN_PATH = '../input/dataset2-master/dataset2-master/images/TRAIN/'
TEST_PATH = '../input/dataset2-master/dataset2-master/images/TEST/'

dict_characters = {0:'EOSINOPHIL', 1:'LYMPHOCYTE', 2:'MONOCYTE', 3:'NEUTROPHIL'}


# LOAD DATA
def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []

    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            if wbc_type in ['NEUTROPHIL']:
                label = 3
            elif wbc_type in ['EOSINOPHIL']:
                label = 0
            elif wbc_type in ['MONOCYTE']:
                label = 2  
            elif wbc_type in ['LYMPHOCYTE']:
                label = 1
                
            for image_filename in os.listdir(folder + wbc_type):
                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img_file is not None:
                    img_file =  cv2.resize(img_file, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X,y
    
# METHODS TO EXTRACT FEATS

# Hu Moments
def ftd_hu_moments(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ft = cv2.HuMoments(cv2.moments(im)).flatten()
    return ft

# Haralick Texture 
def ftd_haralick(im):
    # convert the image to grayscale
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    hl = mahotas.features.haralick(g).mean(axis=0)
    # return the result
    return hl
# Texture
def lbp(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 8
    n_points = radius * 3
    feats = mahotas.features.lbp(g,radius, n_points)
    return feats

# Color Histogram
def ftd_histogram(im, mask=None):
    # convert the image to HSV color-space
    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([im], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


# EXTRACT IMAGE FEATURES FROM ARRAY
def feat_extract(X):
    X_feat = []
   
    for im in X:
       

        f_h_moments = ftd_hu_moments(im)
        f_haralick   = ftd_haralick(im)

        f_hist  = ftd_histogram(im)
        f_tas  = mahotas.features.pftas(im)

        im_features = np.hstack([f_hist, f_haralick ,f_h_moments,f_tas])
        
        X_feat.append(im_features)
        

    return np.asarray(X_feat)


print("# LOAD DATA #")
X_train, y_train = get_data(TRAIN_PATH)
X_test, y_test = get_data(TEST_PATH)

print("# EXTRACT FEATURES ")
X_feat_train = feat_extract(X_train)
X_feat_test  = feat_extract(X_test)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


import xgboost as xgb

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("\n[STATUS] Xboost init\n")

model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,
                             colsample_bytree = 0.7072947899294361,
                             gamma = 0.3227361479535839,
                             learning_rate = 0.08313320382211467,
                             max_depth = 5,
                             n_estimators = 129,
                             subsample = 0.9815714308010349, verbose=1)


print("\n[STATUS] Xboost train\n")
model.fit(X_feat_train, y_train)

print("\n[STATUS] Xboost predict\n")
pred = model.predict(X_feat_test)

cm=accuracy_score(y_test, pred)

print('Accuracy: ',cm)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))