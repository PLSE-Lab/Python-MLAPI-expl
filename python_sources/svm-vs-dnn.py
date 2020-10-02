#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Importing data:

# In[ ]:


train=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
submission= pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')


# # Preliminary work :

# In[ ]:


# Let's take preview on train datas 
train.head()


# In[ ]:


# Now, we will give preview on test datas
test.head()


# In[ ]:


# Let's give some statstics on the train datas 
train.describe()


# In[ ]:


# Hereunder, we will give general informations about train datas.
train.info()


# In[ ]:


# We will do the same work for test datas.
# We start by showing some statstics about the test datas.
test.describe()


# In[ ]:


# Then , we show some general informations about the test datas .
test.info()


# **The above brief analyse, show that our datas don't encompasse missing values.**

# # Preprocessing datas:

# In[ ]:


xtr=train.iloc[:,1:].to_numpy() # extract train pixels datas .
xts=test.to_numpy()             # extract test pixels datas.
Ytr=train.iloc[:,0].to_numpy()  # extract train label datas.


# In[ ]:


Xtr=xtr.reshape(xtr.shape[0],28,28) # reshape the train pixels datas accordingly to the origin image size.
Xts=xts.reshape(xts.shape[0],28,28) # reshape the test pixels datas accordingly to the origin image size.


# # Outliers:

# To check if there is outliers in our dataset , we will create the boxeplot of the pixels values which should have values between 0 and 255 . Moreover , we will check the values of training labels, which should have values between 0 and 9. 

# In[ ]:


import matplotlib.pyplot as plt 
meanprops={"marker":"o","markeredgecolor":"black","markeredgecolor":"firebrick"}
medianprops={'color':'black'}


# In[ ]:


plt.subplot(211)
plt.boxplot([xtr,xts],labels=['training pixels','test pixels'],meanprops=meanprops,            medianprops=medianprops,showfliers=True,showmeans=True,patch_artist=True,vert=False)
plt.subplot(212)
plt.boxplot([Ytr.flatten()],labels=['labels training datas'],showfliers=True,showmeans=True,meanprops=meanprops,           medianprops=medianprops,patch_artist=True,vert=False)
plt.show()


# **The box plot above show that neither the pixel datas nor the labels trainig data encompasse outliers.**

# # Visualization :

# In[ ]:


# we choice randomly 5 observations from our training dataset to compare the handwritten digit to his 
# correspond label .
import random 
samples=random.sample(range(xtr.shape[0]+1),5)
j=0
for i in samples :
    j=j+1
    plt.subplot(150+j)
    plt.imshow(Xtr[i],cmap=plt.get_cmap('gray'))
    plt.title(Ytr[i])
plt.show()


# # SVM Classifier:

# ### Contents :
# 1. Import required librairies .
# 2. Split datas & Implement SVM.
# 3. Compute the estimated score of the svm estimator .
# 4. Predict the test handwritten digits labels.

# ### 1. Import required librairies :

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# ### 2.Split datas and implement SVM:

# In[ ]:


# prepare datas
xxtr,xxts,ytr,yts=train_test_split(xtr,Ytr,test_size=0.1)


# In[ ]:


# Implement svm estimator .
estimator= SVC()


# In[ ]:


# Training the SVM estimator .
estimator.fit(xxtr,ytr)


# ### 3. Compute the estimated score of the svm estimator:

# In[ ]:


sc=estimator.score(xxts,yts)
print("The estimated score of the SVM method is : {}".format(sc))


# ### 4. Predict the test handwritten digits labels:

# In[ ]:


submission['Label']=estimator.predict(xts)


# In[ ]:


submission.to_csv('svm.csv',index='False')


# # Neural network Classifier:

# ### Contents:
# 1. Import required librairies.
# 2. Preprocessing datas.
# 3. Implement Deep Neural Network.
# 4. Train & test the DNN.
# 5. Results analyse
# 6. Predict the test handwritten digits labels.

# ### 1. Import required librairies :

# In[ ]:


from keras.layers import Dense , Flatten 
from keras.models import Sequential
from keras.callbacks import EarlyStopping 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


# ### 2. Preprocessing datas:

# In[ ]:


Xtr=Xtr.reshape(Xtr.shape[0],Xtr.shape[1],Xtr.shape[2],1)
Xts=Xts.reshape(Xts.shape[0],Xts.shape[1],Xts.shape[2],1)


# In[ ]:


Ytr=to_categorical(Ytr)


# In[ ]:


dtgen=ImageDataGenerator()


# In[ ]:


X_train,X_val,Y_train,Y_val=train_test_split(Xtr,Ytr,test_size=0.1)


# In[ ]:


training=dtgen.flow(X_train,Y_train,batch_size=32)
validation=dtgen.flow(X_val,Y_val,batch_size=32)


# ### 3.Implement Deep Neural Network (DNN):

# In[ ]:


NN=Sequential()
# Add input layer
NN.add(Dense(128,input_shape=(28,28,1),activation='relu'))
NN.add(Flatten())

# Add Hidden layers 
NN.add(Dense(256,activation='relu'))
NN.add(Dense(256,activation='relu'))

# output layer.

NN.add(Dense(10,activation='softmax'))


# ### 4. Train & test the DNN :

# In[ ]:


# Compile the neural network 
NN.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# In[ ]:


# Training the DNN.
history=NN.fit_generator(generator=training,steps_per_epoch=training.n,epochs=3,validation_data=validation,                validation_steps=validation.n)


# ### 5. Results analyse :

# In[ ]:


ht=history.history
ht.keys()


# In[ ]:


epochs=range(1,len(ht['loss'])+1)
plt.plot(epochs,ht['loss'],'bo')
plt.plot(epochs,ht['val_loss'],'b+')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[ ]:


plt.plot(epochs,ht['accuracy'],'bo')
plt.plot(epochs,ht['val_accuracy'],'b+')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()


# ### 6.Predict the test handwritten digits labels

# In[ ]:


submission1=pd.DataFrame({'ImageId':submission['ImageId']})


# In[ ]:


submission1.insert(1,'Label',NN.predict_classes(Xts),True)


# In[ ]:


submission1.to_csv('dnn.csv',index=False)

