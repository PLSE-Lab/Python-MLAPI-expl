#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[ ]:


iris = pd.read_csv("../input/Iris.csv")


# In[ ]:


iris


# In[ ]:



#separating the independent variables - pick all rows and all columns except
# the  last one
x=iris.iloc[:,1:5].values # independent variables should always be a matrix 
#the dependent variables
y=iris.iloc[:,5].values


# In[ ]:


x


# In[ ]:


y


# In[ ]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# In[ ]:



from sklearn.cross_validation import train_test_split
#to match the same data in the sets, set random_state to the same number as the trainer
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=1/3,random_state = 0)


# In[ ]:



from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)


# In[ ]:



# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state=0)
classifier.fit(xtrain,ytrain)


# In[ ]:


# Predicting the Test set results
ypred = classifier.predict(xtest)


# In[ ]:


ypred


# In[ ]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)


# In[ ]:


cm

