#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hello, if you're reading this and you are like me, you are somewhat new to data science and are trying to expand your skillset. The goal of this kernal is to do a full on data science project and get your feet wet with a few of the typical techniques that are used in a data science projects.

# ## Importing Libraries <br/>  
# Most kernals start out like this because obviously we need some higher level python-based libraries to work with our data. I typically import numpy for any linear algebra work, pandas for data cleaning and transformation, seaborn and matplotlib.pyplot for data visualizations and sklearn for machine learning. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# ## Importing Data

# obligatory read table:

# In[ ]:


train = pd.read_csv("../input/train.csv")


# ## EDA <br/>
# I typically start with train.head() and train.info() to try to get an idea of what information the data is trying to convey, the data types and some of the data values. Luckily for us, most of this information can be found in kaggle's [data dictionary](https://www.kaggle.com/c/titanic/data). I'll leave the link for you to refer to, but below is the jist of the information.
# 
# **Survived** : nominal - indicates whether a person survived the titanic or not<br />
# **Pclass** : ordinal - indicates the ticket class of the person, ie "first", "second", or "third" class<br />
# **Sex** : nominal - indicated the sex of the passenger<br />
# **Age** : Numeric - indicates age<br />
# **SibSP** : Numeric - indicates the number of siblings or spouses related to the passanger onboard<br />
# **Parch** : Numeric - indicates the number of parents related to the passanger onboard<br />
# **Ticket** : nominal - ticket number of the passanger <br />
# **Fare** : Numeric - the amount paid for their ticket by the passanger<br />
# **Cabin** : nominal - the cabin number of the passanger<br />
# **Embarked** : nominal - indicates from what port the passanger embarked from<br />  
# 

# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:





# In[ ]:


plt.figure(figsize = (10,8))
sns.heatmap(train.corr(), cmap= sns.color_palette(palette="RdBu"),linewidths=.5,annot=True)
plt.title("Correlation Matrix of Train DF")
plt.yticks(rotation = "0")


# In[ ]:


train.corr()["Survived"] # Fare, Pclass


# In[ ]:


sns.countplot(data = train, x = "Sex", hue = "Survived")
plt.title("Count of Sexes by Survival")


# In[ ]:


sns.boxplot(data = train, y = "Fare", x = "Survived")
plt.title("Boxplot of Fare by Survival")


# In[ ]:


sns.boxplot(data = train, y = "Fare", x = "Sex", hue = "Survived")
plt.title("Boxplot of Fare by Sex, and Survival")


# In[ ]:


sns.boxplot(data = train, x = "Sex", y = "Age", hue = "Survived")
plt.title("Age vs Sex grouped by Survived")


# In[ ]:


sns.boxplot(data = train, x = "Survived", y = "Age", hue = "Pclass",)
plt.title("Age vs Sex grouped by Survived")


# In[ ]:


train["bool_cabin"] = train["Cabin"].isna() == False
train["bool_cabin"].head()


# In[ ]:


sns.countplot(data = train, x = "bool_cabin", hue = "Survived")
plt.title("Count of Known/Unknown Passenger's Cabins grouped by Survival")


# In[ ]:


x = train["bool_cabin"].values
y = train["Survived"].values
print(np.corrcoef(x,y)[1,0])


# In[ ]:


sns.countplot(data = train, x = "SibSp", hue = "Survived")


# In[ ]:


sns.countplot(data = train, x = "Parch", hue = "Survived")


# # Missing Values
# 

# In[ ]:


train.isna().sum()


# ### Embarked EDA, How to deal with the 2 NA values

# In[ ]:


train[train["Embarked"].isna()]


# In[ ]:


train["Embarked"].value_counts()


# In[ ]:


train.groupby(["Sex","Embarked"]).count()["PassengerId"].rename({"PassengerId":"Count"})


# In[ ]:


train.groupby(["Sex","Pclass","Embarked"]).count()["PassengerId"].rename({"PassengerId":"Count"})


# In[ ]:


def fill_embarked(df,fill = "S"):
    copy = df.copy()
    copy.loc[copy.isna()["Embarked"],"Embarked"] = fill
    return copy


# In[ ]:


example = fill_embarked(train)
example.isna()[["Embarked"]].sum()


# ### Age EDA, How to deal with these Missing Values

# In[ ]:


train.isna()[["Age"]].sum()


# In[ ]:


train[["Age"]].mean()


# In[ ]:


train[["Age"]].median()


# In[ ]:


temp = train.groupby(["Sex","Pclass"]).median()[["Age"]]
temp = temp.merge(train.groupby(["Sex","Pclass"]).mean()[["Age"]], left_index = True, right_index = True, suffixes= ("Median","Mean"))
age = temp.merge(train.groupby(["Sex","Pclass"]).count()[["Age"]], left_index= True, right_index = True, suffixes = ("Count","Count"))
age


# In[ ]:


def fill_age(df, method = np.median):
    global age
    copy = df.copy()
    copy.loc[copy["Age"].isna(),"Age"] = method(copy.dropna()["Age"])
    return copy    


# In[ ]:


temp = train.copy()
temp = fill_age(temp)
temp["Age"].isna().sum()


# ### Cabin EDA, How to deal with these Missing Values

# In[ ]:


train.count().max(), train.isna()["Cabin"].sum()


# So we see that a good chunk of the data is missing here. MORE THAN HALF! Since it would probably be a little erroneous to make assumptions about the Data, its probably better to not fill in for these missing values.

# ## Data Transformation

# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train = fill_age(train)
train = fill_embarked(train)
train["bool_cabin"] = train["Cabin"].isna() == False
train.drop("Cabin",axis=1,inplace = True)


# In[ ]:


train.isna().sum()


# In[ ]:


def one_hot_encoding(df):
    copy = df.copy()
    columns = copy.columns.tolist()
    
    first_column = copy[columns[0]]
    temp = pd.get_dummies(first_column, prefix = columns[0], drop_first = True)
    
    for i in columns[1:]:
        curr_column = copy[i]
        curr_df = pd.get_dummies(curr_column, prefix = i, drop_first = True)
        temp = temp.merge(curr_df,right_index = True, left_index = True)
    return temp


# In[ ]:


train = pd.read_csv("../input/train.csv")
train = fill_age(train)
train = fill_embarked(train)
train["bool_cabin"] = train["Cabin"].isna() == False
train.drop("Cabin",axis=1,inplace = True)
num = ["SibSp","Parch","Fare", "Age","Survived"]
cats = ["Pclass","Sex","Embarked","bool_cabin"]
train_cats = train[cats]
train_num = train[num]
train_cats = one_hot_encoding(train_cats)
train_x = train_num.merge(train_cats, right_index = True, left_index= True)
train_x.head()


# ## EDA Finding Relations Amongst the Data and Survived

# In[ ]:


x = train_x.copy()
x.drop("Survived", axis=1,inplace = True)
y = train_x["Survived"]


# In[ ]:


from sklearn.linear_model import LogisticRegressionCV


# In[ ]:


model = LogisticRegressionCV(cv = 5, fit_intercept= True)
model.fit(x,y)
model.score(x,y)


# In[ ]:


num = ["SibSp","Parch","Fare", "Age"]
cats = ["Pclass","Sex","Embarked","bool_cabin"]


# In[ ]:


test = pd.read_csv("../input/test.csv")
test = fill_age(test)
test = fill_embarked(test)
test["bool_cabin"] = test["Cabin"].isna() == False
test.drop("Cabin",axis=1,inplace = True)
PassengerId = test["PassengerId"]
test_cats = test[cats]
test_num = test[num]
test_cats = one_hot_encoding(test_cats)
test_x = test_num.merge(test_cats, right_index = True, left_index= True)
test_x.head()


# In[ ]:


test_x.isna().sum()


# In[ ]:


test_x.loc[test["Fare"].isna(),"Fare"] = np.mean(test["Fare"])


# In[ ]:


predictions = model.predict(test_x)


# In[ ]:


submission = pd.DataFrame({"PassengerId":PassengerId, "Survived":predictions})


# In[ ]:


submission.to_csv("submission.csv", index=False)

