#!/usr/bin/env python
# coding: utf-8

# ### ****Predicting Titanic Survival****
# 
#   This is a simple and elegant ML Problem for a beginner. This problem has input with 12 features and output of 0/1 like   a binary classification problem. There are mnay binary classifications algorithms available to classify this problem     at first.   But the predictions accuracy can been improved  further with the following techniques:-
#   
#   1. Data Analysis.
#   
#      a) Data Visualising to find the relevance of the 12 features, their relation to the prediction and finding the data         discripencies.
#      
#      b) Data Cleaning  to remove the undesired data, handeling the missing data and making necessary data type                   convesions before training the model.
#      
#   2. Feature Engineering .
#   
#   3. Comparing the various ML models.
#   
#   This is true for all the ML model but  as beginner following this procedure step by step allows one to have a complete   good hold of our ML model and helps us in keeping our foundations(data) strong. 

# In[ ]:


import numpy as np


# In[ ]:


import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


df = pd.read_csv(r"/kaggle/input/titanic/train.csv")


# In[ ]:


df.head()


# In[ ]:


test_data = pd.read_csv(r"/kaggle/input/titanic/test.csv")


# In[ ]:


test_data.head()


# Getting the number of null values for all the features. Such that it can be furthur used for cleaning the data using different techniques depeding upon the its number for each feature. 

# In[ ]:


total = df.isnull().sum().sort_values(ascending= False)


# In[ ]:


percent1 = df.isnull().sum()/df.isnull().count()*100


# In[ ]:


percent2 = (round(percent1,1)).sort_values(ascending= False)


# In[ ]:


missing_data = pd.concat([total, percent2], axis=1, keys= ['Total','%'])


# In[ ]:


missing_data.head(5)


# The above table illustrates the percentage of null values for the different features int the dataset. The column 'Cabin' has about 77% null values,  we can drop the column from our dataset to be used for prediction. And we can use a different approach for the second most null values feature i.e the 'Age' column. Have replaced the null age values with the mean age value. 

# In[ ]:


df= df.drop("Cabin",axis =1)


# In[ ]:


data = [df,test_data]


# In[ ]:


for dataset in data:
    mean = df["Age"].mean()
    std = test_data["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    rand_age=  np.random.randint(mean-std,mean+std, size= is_null)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = df["Age"].astype(int)
    
    


# In[ ]:


df["Age"].isnull().sum()


# For the 'Embarked' column too, have replaced the missing values with the most common values for this column.

# In[ ]:


common_value = 'S'


# In[ ]:


data = [df,test_data]


# In[ ]:


for dataset in data:
    dataset["Embarked"] = dataset["Embarked"].fillna(common_value)


# In[ ]:


df.info()


# ### **Converting value types**

# Converting the different data types into a single specific like int datatype here for efficient working of our ML models.

# In[ ]:


genders= {"male":0, "female":1}
data = [df, test_data]


# In[ ]:


for dataset in data:
    dataset["Sex"] = dataset["Sex"].map(genders)


# In[ ]:


data = [df, test_data]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)


# In[ ]:


ports = {"S":0, "C":1,"Q":2}
data = [df, test_data]


# In[ ]:


for dataset in data:
    dataset["Embarked"] = dataset["Embarked"].map(ports)


# In[ ]:


df.info()


# ### **Categorising Age**

# In[ ]:


data = [df, test_data]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6


# In[ ]:


df.head()


# In[ ]:


Xtrain = df.drop(["PassengerId","Survived","Name","Ticket"], axis =1)


# In[ ]:


Ytrain = df["Survived"]


# In[ ]:


Xtest = test_data.drop(["PassengerId","Name","Ticket"], axis =1)


# ### **Building Model**
Now we have cleaned our data, and hence can now feed to a ML alorithm. Will be using Feature Engineering in the next version of this notebook, for now we will proceed with the first step i.e data cleaning and  the model implementation. 
# In[ ]:


random_forest = RandomForestClassifier(n_estimators = 100)


# In[ ]:


Xtest = Xtest.drop("Cabin" , axis=1)


# In[ ]:


random_forest.fit(Xtrain,Ytrain)


# In[ ]:


Xtest.head()


# In[ ]:


Xtrain.head()


# In[ ]:


Y_prediction = random_forest.predict(Xtest)


# In[ ]:


random_forest.score(Xtrain, Ytrain)


# In[ ]:


acc_random_forest = round(random_forest.score(Xtrain, Ytrain) * 100, 2)


# In[ ]:


print(acc_random_forest)


# In[ ]:


output = pd.DataFrame({"PassengerId": test_data.PassengerId,"Survived": Y_prediction})


# In[ ]:


output.to_csv("my_submssion.csv", index = False)


# In[ ]:


print("Your submission was successufly saved!")


# Will  be trying feature engineering more significantly in the next version to improve our models accuracy.
# Hope it was a interesting read!!!
