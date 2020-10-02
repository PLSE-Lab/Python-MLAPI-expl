#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings                          # to hide error messages(if any)
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# ### Loading required libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading the data set

# In[ ]:


df = pd.read_csv('../input/Iris.csv')
df.head()


# In[ ]:


#Removing unnecessary column i.e. Id
df = df.drop(['Id'], axis = 1)


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# ### Checking if any column has null values

# In[ ]:


df.isnull().sum()


# In[ ]:


df['Species'].unique()


# ### Different type of Species present are:
# 1.Iris-setosa<br/>
# 2.Iris-versicolor<br/>
# 3.Iris-virginica

# In[ ]:


df['Species'].value_counts()


# #### There are 50 Species of each type

# In[ ]:


df.describe()


# The features described in the above data set are:
# 
# **1. count** tells us the number of NoN-empty rows in a feature.
# 
# **2. mean** tells us the mean value of that feature.
# 
# **3. std** tells us the Standard Deviation Value of that feature.
# 
# **4. min** tells us the minimum value of that feature.
# 
# **5. 25%**, **50%**, and **75%** are the percentile/quartile of each features.
# 
# **6. max** tells us the maximum value of that feature.

# ### Visualizing the given data

# In[ ]:


sns.countplot(x = 'Species', data = df)
plt.show()


# In[ ]:


corr = df.corr()
plt.figure(figsize = (10,6))

#Drawing a heatmap to show how various features are correlated

sns.heatmap(corr,annot = True)
plt.yticks(rotation = 45)
plt.show()


# In[ ]:


sns.pairplot(df, hue = 'Species')
plt.show()


# In[ ]:


#Scatter plot between petal length and petal witdth
plt.figure(figsize = (10,6))
sns.lmplot(x = 'PetalLengthCm', y = 'PetalWidthCm',data = df, hue = 'Species')
plt.show()


# In[ ]:


plt.figure(figsize =(10,6) )
sns.lmplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = df, hue = 'Species')


# In[ ]:


#swarmplot
plt.figure(figsize = (10,6))
sns.swarmplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data = df, hue = 'Species')
plt.show()


# In[ ]:


#Box Plot
plt.figure(figsize =(10,7) )
sns.boxplot(x = 'Species', y = 'SepalWidthCm', data = df)
plt.show()


# ### Loading Machine Learning Libraries

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


X = np.array(df.iloc[:,0:4])
y = np.array(df['Species'])


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33, random_state = 42)


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train,y_train)


# In[ ]:


knn.score(X_test,y_test)


# ### Performing Cross Validation

# In[ ]:


neighbors = []     #empty list to store the number of neighbors
cv_scores = []     #empty list to score cross validation scores

from sklearn.model_selection import cross_val_score
for i in range(1,51,2):
    neighbors.append(i)
    knn = KNeighborsClassifier(n_neighbors = i)
    
    #Performing 10 fold cross-validation
    
    scores = cross_val_score(knn,X_train,y_train, cv = 10, scoring = 'accuracy')
    cv_scores.append(scores.mean())


# In[ ]:


#Misclassification error rates
MSE = [1-x for x in cv_scores]

#determining the best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is %d ' %optimal_k)


# In[ ]:


#Plotting misclassification versus k(number of nearest neighbors)

sns.set()
plt.figure(figsize = (10,6))
plt.plot(neighbors,MSE, 'c')
plt.xlabel('Neighbors')
plt.ylabel('Misclassification Error Rate')
plt.title('Misclassification Error Rate vs. Nearest Neighbors')
plt.show()

