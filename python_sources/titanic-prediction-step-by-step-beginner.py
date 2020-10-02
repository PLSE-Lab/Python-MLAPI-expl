#!/usr/bin/env python
# coding: utf-8

# # Titanic Prediction Step by Step (Beginner)
# 
# As a newcomer to data science and machine learning, i did an initial level analysis and some predictions. If you liked the work and i was able to help, please don't forget to upvote :)

# ## 1st Step: Import Libraries
# 
# In the first step, we will import the necessary libraries for us.

# In[ ]:


## Some visualization tools and tools we mostly use.
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from pandas import read_csv
import seaborn as sns

## This part is optional, if you don't want to see 
## the warnings, you can turn them off in this way.
import warnings
warnings.filterwarnings("ignore")

## Machine learning tools.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


# ## 2nd Step: Upload and Reading the Data
# 
# In the second step we will load the data. Then we can take a first look at our data. 
# 
# a) Let's make some comments about our data: there are some variables in our study that are not important to us like "Name" and "Ticket". We will clean them in the next stage and continue on our way.

# In[ ]:


train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")

train.head()


# b) Now we are looking at our missing datas for our train and test data. As far as we can see, we can make the following conclusion: There is a level of missing data in the "Age" and "Cabin" variables that we can call serious. Since there are too many missing data that we cannot work with the "Cabin" variable, we will delete it. However, since the "Age" variable is both an important variable and we have the ability to correct it, we can delete or replace the missing datas in this variable.

# In[ ]:


print(pd.isnull(train).sum(),"\n------------\n",pd.isnull(test).sum())


# ## 3rd Step: Data Analysis and Some Visiuality
# 
# This part, especially the visualization part, is a part that I am still working on. The data visualization and analysis part is the most crucial and important part in such studies. I couldn't keep this part too wide, but don't do what i do, filter and analyze the data as much as possible. And try to improve yourself on this subject because if you are going to work on a data, you must first warm up with that data.

# In[ ]:


train[["Sex", "Survived"]].groupby(["Sex"]).mean()*100
## Survival rates by gender.


# In[ ]:


train[["Pclass", "Survived"]].groupby(["Pclass"]).mean()*100
## Apparently the socioeconomic situation had an impact on the chance to survive.


# In[ ]:


train[["Embarked", "Survived"]].groupby(["Embarked"]).mean()*100


# #### This image gives us the correlation analysis. We use it to explain the relationship between variables, and we can say that the relationship increases as the ratio on boxes approaches 1.

# In[ ]:


f,ax = plt.subplots(figsize = (5,5))
sns.heatmap(train.corr(), annot=True)

## this part is optional. I had to do it because the 
## plot had been disproportionate.
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()


# In[ ]:


survived_map = {1:"Survived", 0:"Death"}

sns.swarmplot(x="Sex", y="Age", 
              hue=train["Survived"].map(survived_map).copy(), 
              data=train)
plt.show()


# In[ ]:


survived_map = {1:"Survived", 0:"Death"}

sns.swarmplot(x="Sex", y="Fare", 
              hue=train["Survived"].map(survived_map).copy(), 
              data=train)
plt.show()

## Socioeconomics status was important for survive. We can see that 
## 3 person who status of passangers with the highest ticket fee.


# In[ ]:


survived_map = {1:"Survived", 0:"Death"}

sns.swarmplot(x="Embarked", y="Age", 
              hue=train["Survived"].map(survived_map).copy(), 
              data=train)
plt.show()


# ## 4th Step: Data Cleaning and Editing
# 
# In this section, we will make our data ready to predict and work on. This part is also important. We will find and clear our missing datas, remove the variables that we will not use from the data set. And group them by putting them into data types that we can operate on.
# 
# 
# As always, let's first take a look at our raw data. And let's decide what we should work on. As I said from the beginning, some variables are useless, so we will delete them. And we will fill in some of the missing data in a way that works for us. We will also need to change the dtypes of some variables.

# In[ ]:


print(train.info(),test.info())


# In[ ]:


## We will delete unimportant variables for us.
train.drop("PassengerId",axis=1,inplace=True)
train.drop(["Cabin","Name","Ticket"], axis=1, inplace=True)
test.drop(["Cabin","Name","Ticket"], axis=1, inplace=True)

## In this section we will group the sibsp and patch variables into 
## one variable and group them together and delete the rest.
## First, let's find out who the traveler is traveling with how many people.
train["Alone?"] = train["SibSp"]+train["Parch"]+1
train.drop(["SibSp","Parch"], axis=1, inplace=True)

test["Alone?"] = test["SibSp"]+test["Parch"]+1
test.drop(["SibSp","Parch"], axis=1, inplace=True)

## Now we will classify our passengers and apply mapping.
bins = [0,1,4,11]
labels = ["Alone","NotAlone","Crowd"]
train["Person"] = pd.cut(train["Alone?"],bins, labels = labels, include_lowest = True)

person_map = {"Alone":1, "NotAlone":2, "Crowd":3}
train["Person"] = train["Person"].map(person_map)
train.drop("Alone?", axis=1, inplace=True)

bins = [0,1,4,11]
labels = ["Alone","NotAlone","Crowd"]
test["Person"] = pd.cut(test["Alone?"],bins, labels = labels, include_lowest = True)

person_map = {"Alone":1, "NotAlone":2, "Crowd":3}
test["Person"] = test["Person"].map(person_map)
test.drop("Alone?", axis=1, inplace=True)

train.head()


# ### In this section, we will randomly fill the missing data in our age variable with a value between the average of the age variable and the standard error.

# In[ ]:


def age_train(x):
    train_mean = train[x].mean()
    train_std = train[x].std()
    train_null = train[x].isnull().sum()
    return np.random.randint(train_mean - train_std, train_mean + train_std, size = train_null)
def age_test(x):
    test_mean = test[x].mean()
    test_std = test[x].std()
    test_null = test[x].isnull().sum()
    return np.random.randint(test_mean - test_std, test_mean + test_std, size = test_null)

train["Age"][np.isnan(train["Age"])] = age_train("Age")
test["Age"][np.isnan(test["Age"])] = age_test("Age")


train["Age"] = train["Age"].astype("int64")
test["Age"] = test["Age"].astype("int64")

train.head()


# ### We fill in the missing data and categorize the remaining variables.

# #### Fare

# In[ ]:


test["Fare"].fillna(test["Fare"].median(), inplace = True)
train["Fare"] = train["Fare"].astype("int64")
test["Fare"] = test["Fare"].astype("int64")

train["FareGroup"] = pd.qcut(train["Fare"], 4, labels = [1,2,3,4])
train.drop("Fare", axis=1, inplace=True)

test["FareGroup"] = pd.qcut(test["Fare"], 4, labels = [1,2,3,4])
test.drop("Fare", axis=1, inplace=True)


# #### Embarked

# In[ ]:


train = train.fillna({"Embarked": "S"})

embarked_map = {"S": 0, "C": 1, "Q":2}
train["Embarked"] = train["Embarked"].map(embarked_map)

embarked_map = {"S": 0, "C": 1, "Q":2}
test["Embarked"] = test["Embarked"].map(embarked_map)

train["Embarked"] = train["Embarked"].astype("int64")
test["Embarked"] = test["Embarked"].astype("int64")


# #### Sex

# In[ ]:


sex_map = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_map)

sex_map = {"male": 0, "female": 1}
test["Sex"] = test["Sex"].map(sex_map)

train["Sex"] = train["Sex"].astype("int64")
test["Sex"] = test["Sex"].astype("int64")


# #### Age

# In[ ]:


bins = [0,18,30,50,70,120]
labels = ["0-17","18-29","30-49","50-69","70+"]
train["AgeGroup"] = pd.cut(train["Age"],bins, labels = labels, include_lowest = True)

age_mapping = {"0-17":0, "18-29":1, "30-49":2, "50-69":3, "70+":4}
train["AgeGroup"] = train["AgeGroup"].map(age_mapping)
train.drop("Age", axis=1, inplace=True)

train.head()


# In[ ]:


bins = [0,18,30,50,70,120]
labels = ["0-17","18-29","30-49","50-69","70+"]
test["AgeGroup"] = pd.cut(test["Age"],bins, labels = labels, include_lowest = True)

age_mapping = {"0-17":0, "18-29":1, "30-49":2, "50-69":3, "70+":4}
test["AgeGroup"] = test["AgeGroup"].map(age_mapping)
test.drop("Age", axis=1, inplace=True)

test.head()


# # 5th Step: Predictions
# 
# ### We can now implement our machine learning algorithms.
# 
# ##### First of all, we will separate our data set as train and test.

# In[ ]:


from sklearn.model_selection import train_test_split

independent_var = train.drop(["Survived"], axis=1)
dependent_var = train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(independent_var, dependent_var, 
                                                  test_size = 20, random_state = 0)


# ### And finally we can implement our models. I applied all of them here at the same time and sorted the accuracy scores of the models in a dataframe. You can implement it one by one.

# In[ ]:


models = []
models.append(SVC())
models.append(LinearSVC())
models.append(Perceptron())
models.append(GaussianNB())
models.append(SGDClassifier())
models.append(LogisticRegression())
models.append(KNeighborsClassifier())
models.append(RandomForestClassifier())
models.append(DecisionTreeClassifier())
models.append(GradientBoostingClassifier())

accuracy_list = []
for model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = (accuracy_score(y_pred, y_test, normalize=True)*100)
    accuracy_list.append(accuracy)


model_name_list = ["SVM","Linear SVC","Perceptron","Gaussian NB","SGD Classifier","Logistic Regression",
                   "K-Neighbors Classifier","Random Forest Classifier","Decision Tree","Gradient Boosting"]

best_model = pd.DataFrame({"Model": model_name_list, "Score": accuracy_list})
best_model.sort_values(by="Score", ascending=False)


# # Submission Part

# In[ ]:


DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)

passanger_id = test["PassengerId"]
pred = DT.predict(test.drop("PassengerId", axis=1))
predictions = pd.DataFrame({ "PassengerId" : passanger_id, "Survived": pred })

## predictions.to_csv("submission.csv", index=False)


# **References**
# 
# * [A Journey through Titanic](http://https://www.kaggle.com/omarelgabry/a-journey-through-titanic)
# * [Titanic Data Science Solutions](http://https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# * [Titanic Survival Predictions (Beginner)](http://https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner)
