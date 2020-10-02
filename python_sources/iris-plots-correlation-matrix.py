#!/usr/bin/env python
# coding: utf-8

# **
# 
# **Iris dataset with simple visualisations and correlation matrix**
# --------------------------------------------------------------
# 
# **

# In[ ]:


import pandas as pd
pd.options.mode.chained_assignment = None 
dataframe = pd.read_csv("../input/Iris.csv")
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB #Classifier 
from sklearn import svm 
import seaborn as sns 
import matplotlib.pyplot as plt #Visualize 
from sklearn.metrics import mean_squared_error 
from math import sqrt 


# In[ ]:


# Lets se how our dataframe looks like
dataframe.head()


# In[ ]:


#Let see how the classes are separated

sns.FacetGrid(dataframe, hue="Species", size=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()


# In[ ]:


#Let see how the classes are separated
sns.FacetGrid(dataframe, hue= "Species", size = 5).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()
sns.plt.show()


# In[ ]:


# import correlation matrix to see parametrs which best correlate each other
# According to the correlation matrix results PetalLengthCm and
#PetalWidthCm have possitive correlation which is proved by the plot above

import seaborn as sns
corr = dataframe.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.show()


# In[ ]:


#In order to rus Naive_Bayes classifier we have to replace the "Species" values
dataframe['Species'].replace("Iris-setosa",1,inplace= True)
dataframe['Species'].replace("Iris-virginica",2,inplace = True)
dataframe['Species'].replace("Iris-versicolor",3,inplace=True)


# In[ ]:


#Now check if everything was changed properly
dataframe['Species'].unique()


# In[ ]:


X = dataframe.iloc[:, 0:4]  
Y = dataframe['Species']


# In[ ]:


# I prefer to use train_test_split for cross-validation
# This peace will prove us if we have overfitting 
X_train, X_test, y_train, y_test = train_test_split(
  X, Y, test_size=0.4, random_state=0)
print(" X_train",X_train)
print("x_test",X_test)
print("y_train",y_train)
print("y_test",y_test)


# In[ ]:


#Train and test model
clf = GaussianNB()
clf = clf.fit(X_train ,y_train)
clf.score(X_test, y_test) 

