#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


dataset=pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
dataset.head()
import seaborn as sn
import matplotlib.pyplot as plt
plt.figure(figsize=(18,10))
sn.lineplot(data=dataset.iloc[:,:-1])


# In[ ]:


plt.figure(figsize=(18,10))
sn.scatterplot(x=dataset['petal_width'],y=dataset['petal_length'],hue=dataset['petal_length']>1.5)


# In[ ]:


plt.figure(figsize=(18,10))
sn.scatterplot(x=dataset['sepal_width'],y=dataset['sepal_length'],hue=dataset['sepal_length']>5.0)


# In[ ]:


dataset=pd.DataFrame(dataset)
dataset.head()


# In[ ]:


X=dataset.iloc[:,[0,1,2,3]]
Y=dataset.iloc[:,[4]]
Y.head()

#X.head()


# In[ ]:


dataset.species.value_counts()


# In[ ]:


dataset.replace(to_replace=['Iris-setosa','Iris-virginica','Iris-versicolor'],value=['1','2','3'],inplace=True)
dataset.head()


# In[ ]:



from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[ ]:


#applying naive bayes model for classification
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(xtrain,ytrain)
ypred=classifier.predict(xtest)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print(confusion_matrix(ytest,ypred))


# In[ ]:


print(accuracy_score(ytest,ypred)*100)
#96 percent accuracy


# In[ ]:


from sklearn.gaussian_process import GaussianProcessClassifier
classet=GaussianProcessClassifier()
classet.fit(xtrain,ytrain)
classet.predict(xtest)
print(accuracy_score(ytest,ypred)*100)

