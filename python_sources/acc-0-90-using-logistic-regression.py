#!/usr/bin/env python
# coding: utf-8

# Hello, this is my first Machine Learning Program in Kaggle. And i have to learn Python these days for the Competitions. if there are something wrong, welcome to take it out. As everyone know, Digital Recognizer is the HelloWorld question in the field of ML. OK let's Go!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# load the data and change the data from pd frame to np
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# Data Preprocessing:  in this question, we do not need spent more time on the data. And we had better write several lines to see what the image is(visualization). It is an easy task.

# In[ ]:


x_train = train_df.drop(['label'], axis=1).values.astype('float32')
y_train = train_df['label'].values
x_test = test_df.values.astype('float32')


# In[ ]:


# visualization: just to plot the number
def plot_number(x):
    xx=x.reshape((28,28))
    plt.figure(1, figsize=(3, 3))
    plt.imshow(xx, cmap=plt.cm.gray_r, interpolation='nearest')
plot_number(x_train[1,:])


# Sometimes, we can use a small set data instead of the Train Data to speed up during hyperparameter optimization. 

# In[ ]:


N=3000
x1_train,x1_cross,y1_train,y1_cross = train_test_split(x_train[:N], y_train[:N], 
                                                       train_size=0.7, random_state=12)


# Train Model : LogisticRegression, it is a linear classifier and more details in the sk-learn help document.
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# 
# If we don`t do anything , we can get a accuracy of 0.83 .  It seems not bad !  (Be careful, we just use the small Data of 3000)

# In[ ]:


lr = LogisticRegression()
lr.fit(x1_train, y1_train)
print("Accuracy = {:.2f}".format(lr.score(x1_cross, y1_cross)))


# Hyperparameter Optimization: we can choose some parameters to increase our model, for example the regularization weight.  We change "C" from 1e-10 to 1e4 to see what happens. Plot the acc_train & acc_cross Vs C. We know that larger C will cause overfitting.
# From the result, may be set C 2e-6 is best, and we increase the acc_cross from 0.83 to 0.89. If we change the solver from 'liblinear'(default) to 'newton-cg', we can also increase the accuracy a little (not much, less than 0.01).

# In[ ]:


Acc_train,Acc_cross,params = [],[],[]
for c in np.arange(-30,12):
    CC=10**(c/3)
    lr = LogisticRegression(C=CC,max_iter=100)
    lr.fit(x1_train,y1_train)
    print("CC = %3f , Accuracy = %3f "% (CC,(lr.score(x1_cross, y1_cross)) )  )
    Acc_train = np.append(Acc_train,lr.score(x1_train, y1_train))
    Acc_cross = np.append(Acc_cross,lr.score(x1_cross, y1_cross))
    params = np.append(params,CC)

plt.plot(params,Acc_train,label='Acc_train')    
plt.plot(params,Acc_cross,label='Acc_cross')    
plt.xlabel('C')
plt.ylabel('Acc')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()


# You can also choose another model: LogisticRegressionCV, it can choose the best C automatically. More details in http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

# Now we get hyperparameters.  let us turn back to the big data. Get the acc of 0.917

# In[ ]:


N=42000
x1_train,x1_cross,y1_train,y1_cross = train_test_split(x_train[:N], y_train[:N], 
                                                       train_size=0.7, random_state=12)
lr = LogisticRegression(C=2e-6,max_iter=1000)
lr.fit(x_train,y_train)
print("Accuracy = %3f "% ((lr.score(x1_cross, y1_cross)) )  )


# Save the result and submit~~ 

# In[ ]:


pred = lr.predict(x_test)
np.savetxt('submission.csv', np.c_[range(1,len(x_test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')


# Conclusions:
# My score is 0.90971 ,  Maybe i can increase the score by choosing hyper-parameters more carefully but i think change another model such as CNN, SVM or RF is more efficiency. I will try and write another notebook if possible.
# 
# First time for the Kaggle Competition. Far from perfect but enjoy it. 
# 

# In[ ]:




