# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


features_train = train_df[['Sex','Pclass','Age','Fare',]].values
labels_train = train_df['Survived'].values

import math
for feature in features_train:
    if feature[0] == 'male':
        feature[0] = 0
    else:
        feature[0] = 1
    
    if math.isnan(feature[2]):
        feature[2] = 4
    if math.isnan(feature[3]):
        feature[3] = 2
    feature[2] = int(feature[2]//10)
    feature[3] = int(feature[3]//10)
features_train = features_train.tolist()
labels_train = labels_train.tolist()


features_test = test_df[['Sex','Pclass','Age','Fare',]].values
for feature in features_test:
    if feature[0] == 'male':
        feature[0] = 0
    else:
        feature[0] = 1
    
    if math.isnan(feature[2]):
        feature[2] = 4
    if math.isnan(feature[3]):
        feature[3] = 2
    feature[2] = int(feature[2]//10)
    feature[3] = int(feature[3]//10)
features_test = features_test.tolist()


from time import time
from sklearn.svm import SVC
cls = SVC()
t0 = time()
cls.fit(features_train, labels_train)
print("Training time: ", time()-t0)

t0 = time()
labels_pred = cls.predict(features_test)
print("Prediction time: ", time()-t0)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": labels_pred
    })
#print(submission)
submission.to_csv('submission.csv', index=False)

# Any results you write to the current directory are saved as output.