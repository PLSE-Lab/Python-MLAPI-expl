#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Loading and displaying 1 .pgm file

# In[3]:


a=cv2.imread('../input/cambridge-orl-dataset/att_faces/s1/1.pgm',0)
print(a.shape)
plt.imshow(a,cmap='gray')
plt.show()


# # Loading and displaying 10 .pgm files of same person

# In[4]:


fig,ax=plt.subplots(2,5,figsize=(15,7))
for i in range(0,10):
    a=cv2.imread('../input/cambridge-orl-dataset/att_faces/s1/{}.pgm'.format(i+1),0)
    print(a.shape)
    ax[i%2,i//2].imshow(a,cmap='gray')
plt.show()


# # Detecting rectangle on 10 images of s1 folder

# In[5]:


# face_cascade = cv2.CascadeClassifier('../input/trained-model-haarcascade/repository/opencv-opencv-8c25a8e/data/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('../input/trained-model-haarcascade/repository/opencv-opencv-8c25a8e/data/lbpcascades/lbpcascade_frontalface_improved.xml')
fig, ax=plt.subplots(10,2,figsize=(10,50))
for i in range(10):
    img=cv2.imread('../input/cambridge-orl-dataset/att_faces/s1/{}.pgm'.format(i+1))
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax[i,0].imshow(gray)
    ax[i,0].axis('off')
    ax[i,0].set_title('Original Image')
    faces = face_cascade.detectMultiScale(img)
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        print(i,w*h)
    ax[i,1].imshow(gray)
    ax[i,1].axis('off')
    ax[i,1].set_title('Face detected Image')
plt.show()


# # Preprocessing and accuracy of face detection

# In[48]:


face_cascade = cv2.CascadeClassifier('../input/trained-model-haarcascade/repository/opencv-opencv-8c25a8e/data/lbpcascades/lbpcascade_frontalface_improved.xml')
tp,fp,fn=[],[],[]
exception=[]
count=0
length=48
dataset=np.zeros((375,length*length+1),dtype=np.uint8)
sh,sw=0,0
for test_case in range(1,41):
    if test_case==34:
        continue
    for i in range(10):
        img=cv2.imread('../input/cambridge-orl-dataset/att_faces/s{}/{}.pgm'.format(test_case,i+1),0)
        assert img.shape==(112,92)
        equ = cv2.equalizeHist(img)
        blur=cv2.blur(equ,(3,3))
        faces = face_cascade.detectMultiScale(blur)
        if len(faces)==0:
            fn.append((test_case,i+1))
        elif len(faces)==1 and faces[0][2]*faces[0][3]>2300:
            x,y,w,h=faces[0]
#             print(faces[0])
            tp.append((test_case,i+1))
            reduced_img=img[x:x+w,y:y+h]
            reduced_img=cv2.resize(reduced_img,(length,length))
            reduced_img.shape=(1,length*length)
            dataset[count,:length*length]=reduced_img
            dataset[count,length*length]=test_case
            count+=1
            sw+=faces[0][2]
            sh+=faces[0][3]
            print(test_case,i,faces[0],sw,sh)
        elif len(faces)==1 and faces[0][2]*faces[0][3]<2300:
            fp.append((test_case,i+1,faces[0][2]*faces[0][3]))
        else:
            print('Multiple detection for {}/{}'.format(test_case,i+1))
            count=0
            for (x,y,w,h) in faces:
                if w*h>2300:
                    if count==0:
                        tp.append((test_case,i+1))
#                         sw+=w
#                         sh+=h
                    else:
                        fp.append((test_case,i+1,w*h))
                    count+=1
                else:
                    fp.append((test_case,i+1,w*h))
                


# In[49]:


print(len(tp),len(fp),len(fn))


# In[50]:


print(fn)


# In[51]:


df=pd.DataFrame(dataset)
df.to_csv('dataset.csv')


# In[52]:


from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
#import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

from sklearn.ensemble import RandomForestClassifier # A combine model of many decision t

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn.externals import joblib


# In[53]:


df.head()


# In[54]:


print('phase1')
X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:,:length**2],df.iloc[:,length**2] , test_size = 0.25)
print('phase2')
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
print(type(X_train))
print(X_train.describe())


# In[55]:


# Ramdom Forest
clf2 = RandomForestClassifier(n_estimators=100 ,random_state=5)
clf2.fit(X_train, Y_train)
pre = clf2.predict(X_test)

#Saving model
filename = 'random_forest2.sav'
joblib.dump(clf2, filename)

#Using Current Classfier
print('phase4')
pre=clf2.predict(X_test)

#Printing the accuracy score
print('phase5')
print(accuracy_score(Y_test,pre))
print(confusion_matrix(Y_test, pre))


# In[35]:


#decison tree classifier
clf3 = DecisionTreeClassifier()
clf3=clf3.fit(X_train,Y_train)
pre=clf3.predict(X_test)

#Saving model
filename = 'Decison_Tree2.sav'
joblib.dump(clf3, filename)

#Using Current Classfier
print('phase4')
pre=clf3.predict(X_test)

#Printing the accuracy score
print('phase5')
print(accuracy_score(Y_test,pre))
print(confusion_matrix(Y_test, pre))

