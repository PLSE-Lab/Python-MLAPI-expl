#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This kernel we will use sklearn for KNN algorithm
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/column_2C_weka.csv")


# In[ ]:


# Do it first df.info() because we dont know is there any NaN value or length of data
df.info()


# In[ ]:


# to know about features an target variable
df.head()


# ##  K-NEAREST NEIGHBORS (KNN)
# * x = features 
# * y = target variables(normal,abnormal)
# 

# In[ ]:


x = df.loc[:,df.columns != 'class']
#x  = df.drop(['class'],axis =1 )

y = df.loc[:,df.columns == 'class']

#x = pd.DataFrame(df.iloc[:,:-1].values)
#y = pd.DataFrame(df.iloc[:,6].values)

Normal = df.loc[df['class'] == 'Normal']
Abnormal = df.loc[df['class'] == 'Abnormal']


# In[ ]:


# Scatter Plot
plt.scatter(Normal.pelvic_radius,Normal.pelvic_incidence,color='r',label="Normal",alpha=0.3)
plt.scatter(Abnormal.pelvic_radius,Abnormal.pelvic_incidence,color='g',label="Abnormal",alpha=0.3)
plt.xlabel("pelvic_radius")
plt.ylabel("pelvic_incidence")
plt.legend()
plt.show()


# ### Categorical Data
# 

# In[ ]:


#from sklearn.preprocessing import LabelEncoder
#labelencoder_y=LabelEncoder()
#y=pd.DataFrame(labelencoder_y.fit_transform(y).reshape(-1,1))

# Categorical Data without sklearn liblary

df['class'] = [ 1 if each == "Normal" else 0 for each in df['class']]
y = df['class'].values


# ###  Normalization
# 
# 

# In[ ]:


x = ((x-np.min(x)) / ( np.max(x)-np.min(x)))


# ### Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state=1)


# ### KNN Model

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn= KNeighborsClassifier()
knn.fit(x,y)
prediction = knn.predict(x_test)

print("{} nn score : {}".format(3,knn.score(x_test,y_test)))


# ### Model complexity
# 

# In[ ]:


neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('Value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy)))) 

