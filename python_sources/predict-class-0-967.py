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


# # 1-Import the dataset
# 
# The first step is to import the dataset into Python and to have a quick look at it. As we can see, there is no missing value, and all the features seem categorical, so we will keep the dataset this way to build the model later on.
# 

# In[ ]:


#Import the libraries

import numpy as np # linear algebra 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

#Import and visualize the dataset

df=pd.read_csv("../input/mushroom-classification/mushrooms.csv")
df.head()


# In[ ]:


df.tail()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df = df.apply(label_encoder.fit_transform)
df


# # 2- Selecting variables for the model
# 
# We will try to keep all the features, except veil class, veil type and veil colour since they have missing values. I will also scale the values to do Machine Learning thereafter.

# In[ ]:


#Set the class as the response variable

y=df[['class']]

#Set the remaining variables in the dataframe as features

x=df.drop(columns=['class','veil-type','veil-color'])

X = (x - np.min(x))/(np.max(x)-np.min(x)).values
X.info


# # 4-Build the logistic regression model
# 
# I will now run the logistic regression model with the cross validation method (which is, by definition, splitting the testing and the training dataset multiple times). The accuracy is 88.23%, which is not too bad. I will use the KNN model to see if we get the same result.

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

mushroom_model = LogisticRegression()
scores=cross_val_score(mushroom_model, X, y.values.ravel(),cv=10)
scores.mean()


# # 5-Use the KNN model
# 
# In this step, we will use the KNN model. Before running this mode, we will first determine the best number of neighbors by repeating the process 31 times and choose the model with the best accuracy. In that case, 1 is the best number with an accuracy of 96.7%.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y.values.ravel(), cv=10, scoring='accuracy')
k_range = list(range(1, 31))
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y.values.ravel(), cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
    
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


# We choose 1 for the number of neighbors since it seems to give a better accuracy from the graph above

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=1)
print(cross_val_score(knn, X, y.values.ravel(), cv=10, scoring='accuracy').mean())


# # 6-PCA and Logistic Regression
# 
# Since in a logistic regression model, we do not want to have a lot of interactions between variables, we will try to reduce the number of components in the correlation matrix in order to keep as much information as possible with less features in the model. As we can notice on the graph, we can represent 90% of the data by only keeping ten (out of 20) of the most important components, which is very good. We will then try to calculate the accuracy with a cumulative variance of 70%,80%, 90% and 95% to see if the accuracy will increase (88.23% if we keep all the components). The best choice is to keep all the components in the model. Indeed, by checking the correlation matrix, most of them have a very low correlation with each other, which means that they are almost all independant with each other, which is very good for a logistic regression model.

# In[ ]:


from sklearn.decomposition import PCA

pca=PCA()  
pca.n_components=20  
pca_data=pca.fit_transform(X)
percentage_var_explained = pca.explained_variance_ratio_;  
cum_var_explained=np.cumsum(percentage_var_explained)
 
plt.figure(1,figsize=(6,4))
plt.clf()  
plt.plot(cum_var_explained,linewidth=2)  
plt.axis('tight')  
plt.grid() 
plt.xlabel('n_components') 
plt.ylabel('Cumulative_Variance_explained')  
plt.show()


# In[ ]:


i_range=[0.7,0.8,0.9,0.95]
scores1=[]

for i in i_range:
  pca=PCA(i) 
  pca.fit(X) 
  X1=pca.transform(X) 
  mushroom_model1 = LogisticRegression()
  scores1=cross_val_score(mushroom_model1, X1, y.values.ravel(),cv=10)
  print(scores1.mean())


# In[ ]:


df=df.drop(columns=['veil-type','veil-color'])
plt.figure(figsize=(15, 10))
sn.heatmap(df.corr(), annot=True)
plt.show()


# # 7-Conclusion
# 
# Even by using Principal Component Analysis to reduce the number of components in the model,  KNN seems to be a better choice since it gives a better accuracy than logistic regression.
