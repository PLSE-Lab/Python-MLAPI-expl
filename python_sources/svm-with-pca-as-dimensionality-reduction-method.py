# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from pylab import *
import random
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as ss
import sqlite3
print("=============Reading training set=============")
train=pd.read_csv("train.csv",header=0,nrows=42000)
print(train.describe())
labels=train.iloc[:,0]
pixels=train.iloc[:,1:]
training,cv,training_label,cv_label=tts(pixels,labels,test_size=0.2,random_state=0)
print("==============loading images=================")
def loadimage(training,training_label,i):
    img=training.iloc[i].as_matrix()
    img=img.reshape(28,28)
    plt.imshow(img,cmap='gray')
    plt.title(training_label.iloc[i])
a=list(range(1,33600))
for i in range(1,10):
	subplot(330+i)
	id=random.sample(a,1)
	loadimage(training,training_label,id)
plt.show()
print("==============Dimensionality reduction using PCA==================")
pca=PCA(n_components=90,whiten=True)
pca.fit(training)
training_r=pca.transform(training)
print("%f of the variance has been retrieved for n=90 in PCA"%(pca.explained_variance_ratio_.sum()))
print("==============train================")
C=3
gamma=0.009
classifier=svm.SVC(C=C,gamma=gamma)
classifier.fit(training_r,training_label)
train_accuracy=classifier.score(training_r,training_label)
cv_accuracy=classifier.score(pca.transform(cv),cv_label)
print("Training accuracy for C=%f gamma=%f is:" %(C,gamma),train_accuracy)
print("CV accuracy for C=%f gamma=%f is:" %(C,gamma),cv_accuracy)
test=pd.read_csv("test.csv",header=0,nrows=28200)
test_label=classifier.predict(pca.transform(test))
test_file=pd.DataFrame(test_label,range(1,1+len(test_label)))
test_file.reset_index
test_file.columns=["Label"]
test_file.index.name="ImageID"
test_file.to_csv("Test_label.csv")
test_label=pd.Series(test_label)
loadimage(test,test_label,89)