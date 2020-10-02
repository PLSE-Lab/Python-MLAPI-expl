# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from numpy import linalg as LA
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data_path = "../input/images"
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
images = os.listdir(data_path)
ids = test["id"]
#print(ids)
print(images)
# Any results you write to the current directory are saved as output.


def preprocess(imgs):
    imgs_p = np.ndarray(( imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

img_rows = 160
img_cols = 160
i = 0
features = []
test_feats = []
imgs = np.ndarray((1584, 1, img_rows, img_cols), dtype=np.uint8)
for image_name in images:
       
            #print(i)
            img = cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_GRAYSCALE)
            img= cv2.resize(img,(img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
            img = np.array([img])
            #print(np.shape(img))
            
            w, v = LA.eig(img)
          #  print(w)
            eig = np.sort(w)
            if i in ids.as_matrix():
                 test_feats.append(eig)
            else:
                 features.append(eig)
                # test_feats.append(eig)
            imgs[i] = img
            i+=1
            #img =  preprocess(img)
           # print(np.shape( cv2.resize(img,(img_cols, img_rows), interpolation=cv2.INTER_CUBIC)) )

print(len(features))
#print(features[1])
print(np.shape(features))

features = np.asarray(features)
features = np.reshape(features,(990,160,))
print(np.shape(features))
test_feats = np.asarray(test_feats)
test_feats = np.reshape(test_feats ,(594,160,))
print(np.shape(test_feats))
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species'])

#scaler = StandardScaler().fit(features)
x_train = features#scaler.transform(features)
#test = scaler.transform(test_feat)
print("hello")
print(np.shape(test_feats))
print(np.shape(features))
#TODO CV 
from sklearn import datasets, neighbors, linear_model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
logistic = linear_model.LogisticRegression()
clf.fit(x_train, y_train)

y_test = clf.predict_proba(test_feats)

submission = pd.DataFrame(y_test, index=ids, columns=le.classes_)
submission.to_csv('submission_log_reg.csv')



























