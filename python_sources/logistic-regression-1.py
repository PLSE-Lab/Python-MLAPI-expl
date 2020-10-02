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


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


train=pd.read_csv("../input/titanic_train.csv")
train.head()
#SibSp represent number of siblings or spouses
#Parch represents number of parents/children on board
#Fare represents how much passengers pay for the ticket
#Embarked which places passengers embarked into abroad


# In[ ]:


train.isnull() #firstly we need check the missing values and .isnull() return True for missing values and False for non missing


# In[ ]:


# We can see better the missing value with a heatmap
plt.figure(figsize=(20,15))
sns.heatmap(train.isnull(),cmap="ocean")
# we see that there are missing values only in the Age and Cabin columns in our dataset


# In[ ]:


sns.set_style("darkgrid")#Set the aesthetic style of the plots like darkgrid, whitegrid, dark, white, ticks
plt.figure(figsize=(10,5))
sns.countplot(train["Survived"])# it seem that there 350 survivor versus 550 non survivors


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(train["Survived"],hue="Sex",data=train)
#It seems that the percentage of females among survived is much more higher than males


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(train["Survived"],hue="Pclass",data=train,palette="RdBu_r")
#It seems that the percentage of class 3 or the lowes class has the highest deaths than others


# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(x="SibSp",data=train,hue="Sex")
#It seems that most of the passenger do not have children or spouse,particularly among males


# In[ ]:


#I can get more interactive plot
import cufflinks as cf
cf.go_offline()
train["Age"].iplot(kind="hist")


# In[ ]:


train["Survived"].value_counts()


# In[ ]:


len(train[(train["Survived"] == 1) & (train["Pclass"] == 3) & (train["Age"] <15)])


# In[ ]:


len(train[(train["Age"] < 15) & (train["Pclass"] == 3)])


# In[ ]:


len(train[(train["Survived"] == 1) & (train["Pclass"] == 3)])


# In[ ]:


len(train[train["Pclass"] == 3])


# In[ ]:


print(22/54,97/448) #here we understand that %40 of children in the third class survived while it is only %22 in other ages


# In[ ]:


len(train[(train["Survived"] == 1) & (train["Pclass"] == 1) & (train["Age"] <15)])


# In[ ]:


len(train[(train["Age"] < 15) & (train["Pclass"] == 1)])


# In[ ]:


len(train[(train["Survived"] == 1) & (train["Pclass"] == 1)])


# In[ ]:


len(train[train["Pclass"] == 1])


# In[ ]:


print(4/5, 131/211)
#when it comes to the first class,
#here we understand that %80 of children in the third class survived while it is only %62 in other ages


# In[ ]:


#In order to start mechine learning algorithm, we need to transform our data into an acceptable form 
plt.figure(figsize=(15,15))
sns.heatmap(train.isnull(),cmap="viridis")


# In[ ]:


# we can fill the missing values in the Age column with the median age 
#and get ride of Cabin column because there are many missing values there
plt.figure(figsize=(15,15))
sns.boxplot(x="Pclass",y="Age",data=train)
#instead of using the mean of the age column, we can use separate means by every class by looking from the boxplot
#becasue the mean of every class is different


# In[ ]:


def age_mean(col): # here I create a function in order to assign the mean of every class to the missing values
    Age=col[0]
    Pclass=col[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


train["Age"]=train[["Age","Pclass"]].apply(age_mean,axis=1)


# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(train.isnull(),cmap="coolwarm")
#As it is seen below there is not any null value in the Age column


# In[ ]:


#Because there are alot of missing values in the Cabin column, it is better to drop it
train.drop("Cabin",axis=1,inplace=True)
train.head()


# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(train.isnull(),cmap="coolwarm")


# In[ ]:


#Now there is just one missing value in the Embark column and we can just drop it
train.dropna(inplace=True)


# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(train.isnull(),cmap="winter") #Now there is not any missing value in the data


# In[ ]:


#Before applying logistic regression algorithm, we need to convert categorical values into dummy variable as 0 or 1
#Otherwise the algorithm will not be able to directly take these features as inputs
#we use pandas.get_dummies() method in order to convert categorical variables into numeric dummy ones
Sex=pd.get_dummies(train["Sex"],drop_first=True)# we need to use drop_first=True in order to get 1 for only one gender
Sex.head()


# In[ ]:


Embark=pd.get_dummies(train["Embarked"],drop_first=True)
Embark.head()


# In[ ]:


#Now we will add these values into our dataframe by using .concat() method
train=pd.concat([train,Sex,Embark],axis=1)
train.head()


# In[ ]:


# we dont need Sex and Embarked columns anymore because we have replacement values for them for the algorithm
# We do not need also Name and Ticket column because they are not useful for our purpose and algorithm
train.drop(["Name","Sex","Ticket","Embarked"],axis=1,inplace=True)
train.head() 
#All the data is numerical ready for the algorithm


# In[ ]:


#PassegerId is just an index , so it should also be dropped
train.drop("PassengerId",axis=1,inplace=True)
train.head() #Now the data is perfectly ready for our algorithm


# In[ ]:


X=train[["Pclass","Age","SibSp","Parch","Fare","male","Q","S"]]
y=train["Survived"]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


# In[ ]:


#After splitting data we import our model
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression() # we create an instance of the model


# In[ ]:


#The next step is to train the model
logmodel.fit(X_train,y_train)


# In[ ]:


predictions=logmodel.predict(X_test)
predictions


# In[ ]:


#The next step is to evaluate our model
#Sklearn has very good classification report to use
from sklearn.metrics import classification_report # this return the model's accuracy.precision etc.
print(classification_report(y_test,predictions))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions) #It sound our model works good
#TP=147
#FN=16
#FP=30
#TN=74


# In[ ]:




