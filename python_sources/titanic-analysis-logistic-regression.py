#!/usr/bin/env python
# coding: utf-8

# My understanding on a logistics regression is that it is a classification model and it produces results in a binary format (discrete/categorical). <br> <br>
# Means the prediction will be in 0 or 1, Yes or No etc. which is very fitting in this Titanic scenario where the result we want to find out is alive or not.

# # Importing Libraries

# In[ ]:


#import libraries 

#structures
import pandas as pd
import numpy as np

#visualization
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import seaborn as sns
sns.set()
from mpl_toolkits.mplot3d import Axes3D

#get model duration
import time
from datetime import date

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Description of data

# In[ ]:


#load dataset
data = '../input/titanicdataset-traincsv/train.csv'
dataset = pd.read_csv(data)
dataset.shape


# The titanic train data consists of 891 rows and 12 columns.

# In[ ]:


dataset.dtypes


# In[ ]:


dataset.describe()


# In[ ]:


dataset.head(10)


# In[ ]:


print("Total number of passengers in the dataset: " + str(len(dataset.index)))


# **Data Definitions** <br> <br>
# 
# * PassengerId - Unique Id of each passenger on the ship
# * Survived - '0' for not survived & '1' for survived
# * Pclass - Passenger class: '1' for 1st class, '2' for 2nd class & '3' for 3rd class
# * Name - Passenger name
# * Sex - Passenger gender: 'male' or 'female'
# * Age - Passenger age
# * SibSp - No. of siblings or spouses aborded Titanic together with the passenger
# * Parch - No. of parents or children aborded Titanic together with the passenger
# * Ticket - Passenger ticket number
# * Fare - Passenger ticket fare
# * Cabin - Passenger cabin number
# * Embarked - Encoded name of city passenger embarked

# # Analyzing Data

# In[ ]:


sns.countplot(x="Survived", data=dataset)


# We can see only one third of the total passengers survived the incident.

# In[ ]:


sns.countplot(x="Survived", hue="Sex", data=dataset)


# We can see females have a higher survival rate than males in this scenario. <br>
# Based on the Titanic movie, they arranged females and children to board the lifeboats so it makes sense, I guess.

# In[ ]:


sns.countplot(x="Survived", hue="Pclass", data=dataset)


# The death rate of 3rd class passengers are much higher than the other 2 passenger classes. <br>
# Again, with a reference to the Titanic movie, 3rd class passengers are staying at the lowest level of the ship and they have longer route to get to the top when the ship was sinking.

# In[ ]:


dataset["Age"].plot.hist()


# We can see average population of passengers on the titanic are young to middle age group.

# In[ ]:


sns.boxplot(x="Survived", y="Age", data=dataset)


# We can see that younger people tend to have a slightly higher survival rate than the older counterpart.

# In[ ]:


dataset["Pclass"].plot.hist()


# In[ ]:


sns.boxplot(x="Pclass", y="Age", data=dataset)


# We can see that older population of passengers are more likely to be in Passenger Class 1 & Class 2 than Class 3.

# In[ ]:


dataset["Fare"].plot.hist(figsize=(10,10))


# So we can see most of the passengers on Titanic are 3rd class passengers.

# In[ ]:


dataset.info()


# In[ ]:


sns.countplot(x="SibSp", data=dataset)


# In[ ]:


sns.countplot(x="Parch", data=dataset)


# Based on the 2 graphs above, we can conclude most of the passengers on titanic are single passengers and the 2nd most are most likely couples. <br> <br>
# 
# This sounds plausible because as we already saw above that most of the passengers on Titanic are 3rd class passengers and normally, when you are travelling with a family or a spouse, you definitely don't want to choose a 3rd class unless you have no choice.

# # Cleaning Data

# In[ ]:


dataset.isnull()


# In[ ]:


dataset.isnull().sum()


# As we can see, there are many null values under 'Cabin' column. <br>
# 
# 687 out of 891 data points is a really high amount. Also there are quite a number of null values under age. <br>
# 
# This will surely affect the prediction results if left unhandled. <br>
# 
# Handling null values in dataset has two approaches. We can determine the null value at a given point by averaging out the surrounding values under the feature. However, this only works given that the data is an ordinal data and we know that it is in either ascending or descending order or have some sort of pattern. In our case,  it is not. And same for the 'Cabin' feature as well which actually looks like a nominal data. <br>
# 
# Therefore, I decided to remove 177 lines of data from Age along with the whole column of 'Cabin' from the dataset.

# In[ ]:


sns.heatmap(dataset.isnull(), yticklabels=False, cmap="viridis")


# In[ ]:


dataset.head()


# In[ ]:


#dropping 'Cabin' feature
dataset.drop("Cabin", axis=1, inplace=True)


# In[ ]:


#check if the 'Cabin' feature is dropped
dataset.head()


# In[ ]:


sns.heatmap(dataset.isnull(), yticklabels=False, cmap="viridis")


# In[ ]:


#dropping rows of data with 'Age' null
dataset.dropna(inplace=True)


# In[ ]:


sns.heatmap(dataset.isnull(), yticklabels=False, cmap="viridis")


# In[ ]:


#checking if any more null value in the dataset
dataset.isnull().sum()


# So now it is confirmed that the dataset is clean without any null value.

# In[ ]:


dataset.shape


# We are left with 712 data points which is plenty.

# In[ ]:


dataset.head()


# Now we need to convert features: "Pclass", "Sex" & "Embarked" into categorical binary data which is 0 & 1 or True or False or something along. We have 2 options: we can use a label encoder or we can use a pandas method as well.

# First, lets check how many unique values in each of these features.

# In[ ]:


dataset.Pclass.unique()


# We have 3 unique values. <br>
# So we can convert 'Pclass' feature into 2 columns '2' for 2nd Class & '3' for 3rd Class with 0 for No & 1 for Yes values. <br>
# 0 in both of these columns will be automatically 1st class.

# In[ ]:


dataset.Sex.unique()


# We have 2 unique values only. So we can convert 'Sex' feature values into '0' for female & '1' for male.

# In[ ]:


dataset.Embarked.unique()


# We can convert 'Embarked' feature into 3 columns 'S', 'C', 'Q' with 0 for No & 1 for Yes values.

# In[ ]:


Pcl=pd.get_dummies(dataset["Pclass"],drop_first=True)
Pcl.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X = dataset
a = dataset['Sex']

X['Sex'] = le.fit_transform(X['Sex'])

a = le.transform(a)
dataset = X


# In[ ]:


embark=pd.get_dummies(dataset["Embarked"])
embark.head()


# In[ ]:


dataset=pd.concat([dataset,embark,Pcl],axis=1)
dataset.head()


# Lets drop the redundant columns which includes 'PassengerId', 'Pclass', 'Name', 'Ticket' & 'Embarked'.

# In[ ]:


dataset.drop(['PassengerId','Pclass', 'Name','Ticket','Embarked'],axis=1, inplace=True)
dataset.head()


# # Pearson's Correlation

# In[ ]:


#get correlation map
corr_mat=dataset.corr()


# In[ ]:


#visualise data
plt.figure(figsize=(13,5))
sns_plot=sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
plt.show()


# # Logistics Regression

# ## Train & Test Data

# In[ ]:


# Train
X = dataset.drop("Survived", axis=1)
y = dataset["Survived"]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel=LogisticRegression()


# In[ ]:


logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)
print(predictions)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test, predictions)


# So we have following confusion matrix values: <br> <br>
# 
# Predicted No:Actual No: 140 <br>
# Predicted No:Actual Yes: 32 <br> <br>
# 
# Predicted Yes:Actual No: 30 <br>
# Predicted Yes:Actual Yes: 83 <br> <br>
# 
# Lets look at Sensitivity & Specificity values. <br> <br>
# 
# Sensitivity = (Predicted No:Actual No value)/(Total No) = 140/172 = 0.814 <br> <br>
# 
# Specificity = (Predicted Yes:Actual Yes value)/(Total Yes) = 83/113 = 0.735 <br> <br>
# 
# We have higher Sensitivity score over Specificity score. So we can conclude our model is better at predicting the number of people who do no survive the incident.

# In[ ]:


from sklearn.metrics import accuracy_score 


# In[ ]:


accuracy_score(y_test,predictions) #(0+1)/(0+1+1+3) = 0.2

