import pandas as pd

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 18:50:32 2016

@author: mayank
"""

import numpy as np 


from sklearn.decomposition import PCA
from sklearn.svm import SVC

COMPONENT_NUM = 50

print('Read training data...')

with open('../input/train.csv', 'r') as reader:
    reader.readline()
    train_label = []
    train_data = []
    for line in reader.readlines():
        data = list(map(int, line.rstrip().split(',')))
        train_label.append(data[0])
        train_data.append(data[1:])

        
print('Loaded ' + str(len(train_label)))

print('Reduction...')

train_label = np.array(train_label)
train_data = np.array(train_data)

pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train_data)
train_data = pca.transform(train_data)

print('Train SVM...')
svc = SVC()
svc.fit(train_data, train_label)

print('Read testing data...')



with open('../input/test.csv', 'r') as reader:
    reader.readline()
    test_data = []
    for line in reader.readlines():
        pixels = list(map(int, line.rstrip().split(',')))
        test_data.append(pixels)

print('Loaded ' + str(len(test_data)))

print('Predicting...')
test_data = np.array(test_data)

test_data = pca.transform(test_data)
predict = svc.predict(test_data)

print('Saving...')


with open('predict02.csv', 'w') as writer:
    writer.write('"ImageId","Label"\n')
    count = 0
    for p in predict:
        count += 1
        writer.write(str(count) + ',"' + str(p) + '"\n')
        