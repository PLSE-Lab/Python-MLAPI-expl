#!/usr/bin/env python
# coding: utf-8

# **KNN Algorithms**
# 
# ----Content
# 
# 1-Import Dataset
# 
# 2-Investigation Dataset
# 
# 3-Visualizaiton Dataset
# 
# 4-What is KNN algoritms?
# 
# 5-KNN with Sklearn
# 
# 6-Conclusion

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


#read data
data = pd.read_csv("../input/data.csv")


# In[3]:


data.tail() #tail is opposide head


# In[4]:


#We can drop some columns
data.drop(["id","Unnamed: 32"],axis = 1,inplace = True)


# In[5]:


data.info()


# In[6]:


#Split Data as M&B
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]


# In[9]:


#Visualization, Scatter Plot

plt.scatter(M.radius_mean,M.area_mean,color = "Black",label="Malignant",alpha=0.2)
plt.scatter(B.radius_mean,B.area_mean,color = "Orange",label="Benign",alpha=0.3)
plt.xlabel("Radius Mean")
plt.ylabel("Area Mean")
plt.legend()
plt.show()

#We appear that it is clear segregation.


# In[10]:


#Visualization, Scatter Plot

plt.scatter(M.radius_mean,M.texture_mean,color = "Black",label="Malignant",alpha=0.2)
plt.scatter(B.radius_mean,B.texture_mean,color = "Lime",label="Benign",alpha=0.3)
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.legend()
plt.show()


# In[11]:


#change M & B 
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
#seperate data as x (features) & y (labels)
y= data.diagnosis.values
x1= data.drop(["diagnosis"],axis= 1) #we remowe diagnosis for predict


# In[12]:


#normalization
x = (x1-np.min(x1))/(np.max(x1)-np.min(x1))


# In[13]:


#Train-Test-Split 
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest =  train_test_split(x,y,test_size=0.3,random_state=42)


# In[14]:


#Create-KNN-model
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 4) #n_neighbors = K value
KNN.fit(xtrain,ytrain) #learning model
prediction = KNN.predict(xtest)


# In[15]:


print("{}-NN Score: {}".format(4,KNN.score(xtest,ytest)))


# In[16]:


#Find Optimum K value
scores = []
for each in range(1,15):
    KNNfind = KNeighborsClassifier(n_neighbors = each)
    KNNfind.fit(xtrain,ytrain)
    scores.append(KNNfind.score(xtest,ytest))
    
plt.plot(range(1,15),scores,color="black")
plt.xlabel("K Values")
plt.ylabel("Score(Accuracy)")
plt.show()


# # Conclusion
# 
# 1-Thank you for investigation my kernel.
# 
# 2-I tried K value as 3,4,5 and I find 4 optimum value.
# 
# 3-Finally, I found optimum value by aid of for loop.
# 
# # If you like this kernel, Please Upvote :) Thanks
