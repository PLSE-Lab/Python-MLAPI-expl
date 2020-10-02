#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import Perceptron 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# # Introduction 
# My goal for this project is to see whether volatile acidity and alcohol content are good predictors of wine quality. <br>
# I am utilizing a perceptron model to train my data and achieved ~82% accuracy on my first try. 

# In[ ]:


df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')


# In[ ]:


df.tail()


# # Data Visualization 
# For my first plot, I'm finding the distribution of the wines based on quality. </br> 
# I then try and visualize the additional two variables (volatile acidity and alcohol content) that I will be using for my prediction. </br> 

# In[ ]:


#The mean is at a quality rating of 5 with a count of 681 wines 
plt.figure(figsize=(10,6))
sns.countplot(df["quality"])


# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(x=df["quality"],y=df["alcohol"])


# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(x=df["quality"],y=df["volatile acidity"])


# # Data Analysis 
# I first need to insert an additional column to my dataframe that initializes the output class labels of each wine based on quality. </br>
# We are going to classify wines with a quality rating >= 7 as a good wine and wines with a quality rating <= 7 as a bad wine.

# In[ ]:


#Select the quality column of all the wines available in our dataset 
y = df.iloc[0:1599, 11].values

#Classify the quality of the wine based on our ranking
y = np.where(y >= 7, 1, 0)

#Set the tansformed qualities into a new column
df['quality output'] = y

#X label represent our volatile acidity and alcohol content, y label represents the qualities of the wine
y = df.iloc[:,-1].values.astype(int)
X = df.iloc[0:1599, [1, 10]].values

print('Class labels:', np.unique(y))


# In[ ]:


#We are splitting our X and y lists into 30% test data and 70% training data. This helps us evaluate how well our model works on new data. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print(np.bincount(y))
print(np.bincount(y_train))
print(np.bincount(y_test))


# In[ ]:


#Feature scaling helps optimize our algorithm and allows for faster convergence
sc = StandardScaler()
#.fit() allows us to estimate the sample mean and std. deviation of each feature dimension from the training data 
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print(X_train_std)


# In[ ]:


#Training our perceptron model
#We set a maximum of 40 iterations and learning rate of 0.1
per = Perceptron(max_iter=40, eta0=0.1, random_state=1)
per.fit(X_train_std,y_train)

y_predict = per.predict(X_test_std)
print("Accuracy: %.2f" % accuracy_score(y_test, y_predict))

