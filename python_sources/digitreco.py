#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:29:51 2017

@author: pegasus
"""

import pandas as pd
import numpy as np

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

print(train.shape)
print(test.shape)
train.head()

import matplotlib.pyplot as plt
"""
plt.hist(train["label"])
plt.title("Frequency histogram of numbers in training data")
plt.xlabel("Number Value")
plt.ylabel("Frequency")
plt.show()
plt.savefig("output1.png")
"""
import math

f, ax=plt.subplots(5,5)
for i in range(1,26):
    #print(train["label"][i-1])
    data=train.iloc[i,1:785].values
    nrows,ncols =28,28
    grid=data.reshape((nrows,ncols))
    n=math.ceil(i/5)-1
    m=[0,1,2,3,4]*5
    image=ax[m[i-1],n].imshow(grid)
    plt.show()
    plt.savefig("output2.png")
    
label_train=train['label']
train=train.drop('label',axis=1)

train=train/255
test=test/255

train['label']=label_train

from sklearn import decomposition
from sklearn import datasets
pca=decomposition.PCA(n_components=200)

pca.fit(train.drop('label',axis=1))
plt.plot(pca.explained_variance_ratio_)
plt.ylabel('% of variance explained')

pca=decomposition.PCA(n_components=50)
pca.fit(train.drop('label',axis=1))
PCtrain=pd.DataFrame(pca.transform(train.drop('label',axis=1)))

PCtrain['label']=train['label']

PCtest=pd.DataFrame(pca.transform(test))

#print(PCtrain)
#print(PCtest)

from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')
x=PCtrain[0]
y=PCtrain[1]
z=PCtrain[2]
#print(PCtrain.shape())
colors=[int(i%9) for i in PCtrain['label']] 
ax.scatter(x,y,z, c=colors,marker='o', label=colors)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()
#plt.savefig("output3.png")

from sklearn.neural_network import MLPClassifier
y=PCtrain['label'][0:20000]
X=PCtrain.drop('label',axis=1)[0:20000]
alg=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2500,), random_state=1)
alg.fit(X,y)

from sklearn import metrics

predicted=alg.predict(PCtrain.drop('label',axis=1)[20001:42000])
expected=PCtrain['label'][20001:42000]

print("Classification report for classifier %s:\n%s\n" % (alg,metrics.classification_report(expected,predicted)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected,predicted))

output=pd.DataFrame(alg.predict(PCtest), columns=['Label'])
output.reset_index(inplace=True)
output.rename(columns={'index':'ImageId'},inplace=True)
output['ImageId']=output['ImageId']+1
output.to_csv('digitreco.csv',index=False)