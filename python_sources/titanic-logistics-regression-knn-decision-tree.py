#!/usr/bin/env python
# coding: utf-8

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


#load train dataset
train_data = '../input/titanic/train.csv'
train_dataset = pd.read_csv(train_data)
train_dataset.shape


# The titanic train data consists of 891 rows and 12 columns. <br>
# Means we have a total of 891 passengers and 12 features in the train dataset.

# In[ ]:


print("Total number of passengers in the train dataset: " + str(len(train_dataset.index)))


# In[ ]:


test_data = '../input/titanic/test.csv'
test_dataset = pd.read_csv(test_data)
test_dataset.shape


# In[ ]:


print("Total number of passengers in the test dataset: " + str(len(test_dataset.index)))


# In[ ]:


train_dataset.dtypes


# In[ ]:


test_dataset.dtypes


# In[ ]:


train_dataset.describe()


# In[ ]:


test_dataset.describe()


# In[ ]:


train_dataset.head(10)


# In[ ]:


test_dataset.head(10)


# **Data Definitions**
# 
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


sns.countplot(x="Survived", data=train_dataset)


# In[ ]:


sns.countplot(x="Survived", hue="Sex", data=train_dataset)


# We can see females have a higher survival rate than males in this scenario. 

# In[ ]:


sns.countplot(x="Survived", hue="Pclass", data=train_dataset)


# The death rate of 3rd class passengers are much higher than the other 2 passenger classes.

# In[ ]:


total_dataset = pd.concat([train_dataset, test_dataset])


# In[ ]:


total_dataset.shape


# In[ ]:


total_dataset.head()


# In[ ]:


total_dataset.tail()


# In[ ]:


train_dataset["Age"].plot.hist()


# In[ ]:


test_dataset["Age"].plot.hist()


# In[ ]:


total_dataset["Age"].plot.hist()


# From 3 histograms above, we can see both train & test datasets have similar distribution of data in terms of age. <br>
# We can see average population of passengers on the titanic are young to middle age group.

# In[ ]:


sns.boxplot(x="Survived", y="Age", data=train_dataset)


# We can see that younger people tend to have a slightly higher survival rate than the older counterpart.

# In[ ]:


train_dataset["Pclass"].plot.hist()


# In[ ]:


test_dataset["Pclass"].plot.hist()


# In[ ]:


total_dataset["Pclass"].plot.hist()


# From 3 histograms above, we can see both train & test datasets have similar distribution of data in terms of Passenger Class as well.

# In[ ]:


sns.boxplot(x="Pclass", y="Age", data=train_dataset)


# In[ ]:


sns.boxplot(x="Pclass", y="Age", data=test_dataset)


# In[ ]:


sns.boxplot(x="Pclass", y="Age", data=total_dataset)


# We can see that older population of passengers are more likely to be in Passenger Class 1 & Class 2 than Class 3.

# In[ ]:


train_dataset["Fare"].plot.hist(figsize=(10,10))


# In[ ]:


test_dataset["Fare"].plot.hist(figsize=(10,10))


# In[ ]:


total_dataset["Fare"].plot.hist(figsize=(10,10))


# So we can see most of the passengers on Titanic are 3rd class passengers.

# In[ ]:


train_dataset.info()


# In[ ]:


test_dataset.info()


# In[ ]:


total_dataset.info()


# In[ ]:


sns.countplot(x="SibSp", data=train_dataset)


# In[ ]:


sns.countplot(x="SibSp", data=test_dataset)


# In[ ]:


sns.countplot(x="SibSp", data=total_dataset)


# In[ ]:


sns.countplot(x="Parch", data=train_dataset)


# In[ ]:


sns.countplot(x="Parch", data=test_dataset)


# In[ ]:


sns.countplot(x="Parch", data=total_dataset)


# # Cleaning Data

# In[ ]:


total_dataset.isnull()


# In[ ]:


train_dataset.isnull().sum()


# As we can see, there are many null values under 'Cabin' column. <br>
# 
# 687 out of 891 data points is a really high amount. Also there are quite a number of null values under age. <br>
# 
# This will surely affect the prediction results if left unhandled. <br>
# 
# Handling null values in dataset has two approaches. We can determine the null value at a given point by averaging out the surrounding values under the feature. However, this only works given that the data is an ordinal data and we know that it is in either ascending or descending order or have some sort of pattern. In our case, it is not. And same for the 'Cabin' feature as well which actually looks like a nominal data. <br>
# 
# Therefore, I decided to remove 177 lines of data from Age along with the whole column of 'Cabin' from the dataset.

# In[ ]:


test_dataset.isnull().sum()


# Same can be said for test dataset. Again features "Age" & "Cabin".

# In[ ]:


sns.heatmap(train_dataset.isnull(), yticklabels=False, cmap="viridis")


# In[ ]:


sns.heatmap(test_dataset.isnull(), yticklabels=False, cmap="viridis")


# In[ ]:


train_dataset.head()


# In[ ]:


test_dataset.head()


# In[ ]:


#dropping 'Cabin' feature
train_dataset.drop("Cabin", axis=1, inplace=True)
test_dataset.drop("Cabin", axis=1, inplace=True)


# In[ ]:


#check if the 'Cabin' feature is dropped
train_dataset.head()


# In[ ]:


test_dataset.head()


# In[ ]:


sns.heatmap(train_dataset.isnull(), yticklabels=False, cmap="viridis")


# In[ ]:


sns.heatmap(test_dataset.isnull(), yticklabels=False, cmap="viridis")


# In[ ]:


#dropping rows of data with 'Age' null
train_dataset.dropna(inplace=True)


# We can't drop null rows of "Age" under test dataset because if we do so, the number of predictions will be less. <br>
# 
# But we can't leave these null values as it is as well. That will become a problem during a prediction. <br>
# 
# So we will fill up these null values with average values. Under Description of Data section - we already have average values for the test dataset. Lets call it back.

# In[ ]:


test_dataset.describe()


# In[ ]:


#replacing null values with average values
test_dataset['Age'].fillna((test_dataset['Age'].mean()), inplace=True)
test_dataset['Fare'].fillna((test_dataset['Fare'].mean()), inplace=True)


# In[ ]:


sns.heatmap(train_dataset.isnull(), yticklabels=False, cmap="viridis")


# In[ ]:


sns.heatmap(test_dataset.isnull(), yticklabels=False, cmap="viridis")


# In[ ]:


#checking if any more null value in the dataset
train_dataset.isnull().sum()


# In[ ]:


test_dataset.isnull().sum()


# So now it is confirmed that both the train & test datasets are clean without any null value.

# In[ ]:


train_dataset.shape


# We are left with 712 data points which is plenty.

# In[ ]:


test_dataset.shape


# Test dataset still have full data points.

# In[ ]:


train_dataset.head()


# In[ ]:


test_dataset.head()


# Now we need to convert features: "Pclass", "Sex" & "Embarked" into categorical binary data which is 0 & 1 or True or False or something along. We have 2 options: we can use a label encoder or we can use a pandas method as well. <br>
# 
# First, lets check how many unique values in each of these features.

# In[ ]:


train_dataset.Pclass.unique()


# In[ ]:


test_dataset.Pclass.unique()


# We have 3 unique values.
# So we can convert 'Pclass' feature into 2 columns '2' for 2nd Class & '3' for 3rd Class with 0 for No & 1 for Yes values.
# 0 in both of these columns will be automatically 1st class.

# In[ ]:


train_dataset.Sex.unique()


# In[ ]:


test_dataset.Sex.unique()


# We have 2 unique values only. So we can convert 'Sex' feature values into '0' for female & '1' for male.

# In[ ]:


train_dataset.Embarked.unique()


# In[ ]:


test_dataset.Embarked.unique()


# We can convert 'Embarked' feature into 3 columns 'S', 'C', 'Q' with 0 for No & 1 for Yes values.

# In[ ]:


Pcl_train=pd.get_dummies(train_dataset["Pclass"],drop_first=True)


# In[ ]:


Pcl_test=pd.get_dummies(test_dataset["Pclass"],drop_first=True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

X1 = train_dataset
a = train_dataset['Sex']

X1['Sex'] = le.fit_transform(X1['Sex'])

a = le.transform(a)
train_dataset = X1


# In[ ]:


X2 = test_dataset
b = test_dataset['Sex']

X2['Sex'] = le.fit_transform(X2['Sex'])

b = le.transform(b)
test_dataset = X2


# In[ ]:


embark_train=pd.get_dummies(train_dataset["Embarked"])


# In[ ]:


embark_test=pd.get_dummies(test_dataset["Embarked"])


# In[ ]:


train_dataset=pd.concat([train_dataset,embark_train,Pcl_train],axis=1)
train_dataset.head()


# In[ ]:


test_dataset=pd.concat([test_dataset,embark_test,Pcl_test],axis=1)
test_dataset.head()


# Lets drop the redundant columns which includes 'Pclass', 'Name', 'Ticket' & 'Embarked'.

# In[ ]:


train_dataset.drop(['Pclass', 'Name','Ticket','Embarked'],axis=1, inplace=True)
train_dataset.head()


# In[ ]:


test_dataset.drop(['Pclass', 'Name','Ticket','Embarked'],axis=1, inplace=True)
test_dataset.head()


# # Pearson's Correlation

# In[ ]:


#get correlation map
corr_mat=train_dataset.corr()


# In[ ]:


#visualise data
plt.figure(figsize=(13,5))
sns_plot=sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
plt.show()


# # Scaling Dataset

# In[ ]:


#to run for model without scaling
dropped_passengerId = train_dataset.drop("PassengerId", axis=1)

X_train = dropped_passengerId.drop("Survived", axis=1)
y_train = train_dataset["Survived"]

X_test = test_dataset.drop("PassengerId", axis=1)


# In[ ]:


#to run for model with scaling
dropped_passengerId = train_dataset.drop("PassengerId", axis=1)

dropped_survived = dropped_passengerId.drop("Survived", axis=1)

dropped_survived.head()


# In[ ]:


test_dropped_passengerId = test_dataset.drop("PassengerId", axis=1)
test_dropped_passengerId.head()


# In[ ]:


X_train = dropped_survived.iloc[:,0:10]
y_train = train_dataset["Survived"]

X_test = test_dropped_passengerId.iloc[:,0:10]


# ## Standard Scaler

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


#stadardize data
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)


# In[ ]:


#get feature names
X_train_columns = train_dataset.columns[:10]
X_test_columns = test_dataset.columns[:10]


# ## Min-Max Scaler

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


#stadardize data
X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_test_scaled = MinMaxScaler().fit_transform(X_test)


# In[ ]:


#get feature names
X_train_columns = train_dataset.columns[:10]
X_test_columns = test_dataset.columns[:10]


# ## Robust Scaler

# In[ ]:


from sklearn.preprocessing import RobustScaler


# In[ ]:


#stadardize data
X_train_scaled = RobustScaler().fit_transform(X_train)
X_test_scaled = RobustScaler().fit_transform(X_test)


# In[ ]:


#get feature names
X_train_columns = train_dataset.columns[:10]
X_test_columns = test_dataset.columns[:10]


# ## Training using Train Dataset

# ## Logistics Regression

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


output = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_noscaling.csv', index=False)
print("Your submission was successfully saved!")


# ## K-NN

# In[ ]:


import math
math.sqrt(len(X_test))


# Result is 20, so we can use 19 or 21 for K. Hence I will use 21.

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knnmodel = KNeighborsClassifier(n_neighbors=21, p=2, metric='euclidean') #p is 2 cuz we are looking for survived or not: 2 results


# In[ ]:


#Fit Model
knnmodel.fit(X_train_scaled, y_train)


# In[ ]:


#predict the test set results
predictions = knnmodel.predict(X_test_scaled)
print(predictions)


# In[ ]:


output = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_standard.csv', index=False)
print("Your submission was successfully saved!")


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


decisionmodel = DecisionTreeRegressor()


# In[ ]:


#Fit Model
decisionmodel.fit(X_train_scaled, y_train)


# In[ ]:


#predict the test set results
predictions = decisionmodel.predict(X_test_scaled)
print(predictions)


# In[ ]:


output = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived': predictions})
output.to_csv('my_submission_robust.csv', index=False)
print("Your submission was successfully saved!")


# # Scores record

# **Logistics Regression Model**
# * Without scaling - accuracy score: 0.75598
# * Standard scaler - accuracy score: 0.76076
# * Min-max scaler - accuracy score: 0.74162
# * Robust scaler - accuracy score: 0.53588
# 
# **K-NN Model** <br>
# *n_neighbors = 19*
# * Without scaling - accuracy score: 0.62679
# * Standard scaler - accuracy score: 0.75119
# * Min-max scaler - accuracy score: 0.74641
# * Robust scaler - accuracy score: 0.71770
# 
# *n_neighbors = 21*
# * Without scaling - accuracy score: 0.62200
# * Standard scaler - accuracy score: 0.74641
# * Min-max scaler - accuracy score: 0.76555
# * Robust scaler - accuracy score: 0.73684
