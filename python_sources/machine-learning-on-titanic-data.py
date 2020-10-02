#!/usr/bin/env python
# coding: utf-8

# # Titanic in Machine Learning
# Yutao Chen  
# 04/05/2019

# In[ ]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib as plt

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_score


# # Prepara the data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_len = len(train)
test_ID = test["PassengerId"]


# In[ ]:


print(train.columns.values)  # see all the titles of the data


# In[ ]:


# see the general info about the data
train.info()
print('-'*40)
test.info()


# In[ ]:


# combine training and testing for processing
all_data = pd.concat(objs=[train, test], axis=0, sort=False).reset_index(drop=True)
all_data = all_data.fillna(np.nan) # fill the all different kinds of missing data with NaN
all_data.head()


# In[ ]:


all_data.tail()


# In[ ]:


all_data.isnull().sum()  # see if there are some missing values


# # Analysis and process the data column by colum

# ## 1. Pclass

# In[ ]:


g = sns.catplot(x="Pclass",y="Survived",data=train,kind="bar",palette = "muted")
g = g.set_ylabels("survival probability")


# We can see that the people in first class have higher survival probability. While the people in third class have the lowest.   
# There is no missing data for this column, so we don't need to fill missing data.   
# The data type is int and is 1, 2 and 3. So we don't need to do anythin with the data.  

# ## 2. Name

# In[ ]:


# Name (Title)
all_data['Name'].head()


# In[ ]:


# Only take the title in the name
title = [i.split(",")[1].split(".")[0].strip() for i in all_data["Name"]]
all_data["Title"] = pd.Series(title)
all_data["Title"].head()


# In[ ]:


# show all the different titles
all_data["Title"].unique()


# In[ ]:


# see if there are any missing data
all_data["Title"].isnull().sum()


# In[ ]:


g = sns.countplot(x="Title",data=all_data)
g = plt.pyplot.setp(g.get_xticklabels(), rotation=90)


# In[ ]:


# Encode the title as 0, 1, 2 and 3
# All the title with very few people are grouped together
all_data["Title"] = all_data["Title"].replace(["Don","Rev","Dr","Major","Mlle","Col","Mme",
                                               "Ms","Lady","Sir","Capt","the Countess","Jonkheer","Dona"], 'Rare')
all_data["Title"] = all_data["Title"].map({"Mr":0, "Mrs":1, "Miss" : 1 , "Master":2, "Rare":3})
all_data["Title"] = all_data["Title"].astype(int)


# In[ ]:


g = sns.catplot(x="Title",y="Survived",data=all_data,kind="bar")
g = g.set_xticklabels(["Mr","Miss/Mrs","Master","Rare"])
g = g.set_ylabels("survival probability")


# In[ ]:


# don't need the column "Name" 
all_data.drop(labels = ["Name"], axis = 1, inplace = True)


# In general, the title in name tell us some about this people. Maybe people with some titles have higher probability to survive.  
# 

# ## 3. Sex

# In[ ]:


# Sex
g = sns.catplot(x="Sex",y="Survived",data=all_data,kind="bar",palette = "muted")
g = g.set_ylabels("survival probability")


# In[ ]:


all_data["Sex"] = all_data["Sex"].map({"male": 0, "female":1})
all_data["Sex"] = all_data["Sex"].astype(int)
all_data["Sex"].head()


# Female have higher probability to survive in general.   
# The only thing we need to do with "Sex" is to encode it into 0 and 1.  

# ## 4. Age

# In[ ]:


# See which kind of the info is related to the age
# I choose sex, SibSp, Parch, Pclass and Title as candidates
g = sns.heatmap(all_data[["Age","Sex","SibSp","Parch","Pclass","Title"]].corr(),cmap="coolwarm",annot=True)


# In[ ]:


# Fill all the missing data
Nan_index_age = list(all_data["Age"][all_data["Age"].isnull()].index)  # index of missing data

for i in Nan_index_age:
    median = all_data["Age"].median()
    # the median of all the data with similiar background
    predict = all_data["Age"][((all_data['SibSp'] == all_data.iloc[i]["SibSp"]) & 
                              (all_data['Parch'] == all_data.iloc[i]["Parch"]) & 
                              (all_data['Pclass'] == all_data.iloc[i]["Pclass"]) &
                              (all_data['Title'] == all_data.iloc[i]["Title"]))].median()
    if np.isnan(predict):
        all_data['Age'].iloc[i] = median
    else:
        all_data['Age'].iloc[i] = predict


# In[ ]:


all_data['Age'].isnull().sum()


# In[ ]:


g = sns.catplot(x="Survived", y = "Age",data = all_data, kind="violin")


# The most important thing is to find a way to fill the missing data in this columne.  
# I used the median number of ages of people with similiar background.  
# If there is no median number, we use the overall average age.  
# The people at age 0-5 have a relatively high probability to survive.  

# ## 5. SibSp

# In[ ]:


g = sns.catplot(x="SibSp",y="Survived",data=train,kind="bar",palette = "muted")
g = g.set_ylabels("survival probability")


# People have 1 or 2 SibSp have higher survival probability

# ## 6. Parch

# In[ ]:


g  = sns.catplot(x="Parch",y="Survived",data=train,kind="bar",palette = "muted")
g = g.set_ylabels("survival probability")


# People have 1, 2 or 3 Parch have higher survival probability

# ## 7. Ticket

# In[ ]:


all_data['Ticket'].head()


# In[ ]:


# Take the prefix of the ticket to represent different kind of ticket
# If the ticket is all number, I use "Num" to represent
Ticket = []
for i in list(all_data.Ticket):
    if i.isdigit():
        Ticket.append('Num')
    else:
        Ticket.append(i.replace(".","").strip().split(" ")[0])
all_data["Ticket"] = Ticket
all_data["Ticket"].head()


# ## 8. Fare

# In[ ]:


all_data["Fare"] = all_data["Fare"].fillna(all_data["Fare"].median())


# In[ ]:


g = sns.distplot(all_data["Fare"], color="g")


# In[ ]:


all_data["Fare"] = all_data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(all_data["Fare"], color="g")


# Wide range of the data may reduce the accuracy. So, we take the log to narrow the range.

# ## 9. Cabin

# In[ ]:


all_data['Cabin'].head()


# In[ ]:


all_data['Cabin'].isnull().sum()


# In[ ]:


# 'X' to represent NaN
# Use the first character to represent the other
all_data["Cabin"] = all_data["Cabin"].map(lambda i: 'X' if pd.isnull(i) else i[0])
all_data['Cabin'].isnull().sum()


# In[ ]:


all_data['Cabin'].head()


# In[ ]:


g = sns.catplot(y="Survived",x="Cabin",data=all_data,kind="bar")
g = g.set_ylabels("Survival Probability")


# In[ ]:


all_data["Cabin"].unique()


# The only thing we need to do is to fill the missing data.  
# Since there are too many missing data, I decide not to predict, just treat it as another class.   

# ## 10. Embarked

# In[ ]:


all_data["Embarked"].head()


# In[ ]:


all_data["Embarked"].isnull().sum()


# In[ ]:


# Fille the missing data
# Only two missing data. So, just use the most common one
all_data["Embarked"] = all_data["Embarked"].fillna("S")
all_data["Embarked"].isnull().sum()


# In[ ]:


all_data["Embarked"].unique()


# In[ ]:


# Encode the dato as 0, 1 and 2
all_data["Embarked"] = all_data["Embarked"].map({"S":0, "C":1, "Q":2})


# ## Overview the data after the process

# In[ ]:


all_data.head()


# In[ ]:


all_data.drop(labels = ["PassengerId"], axis = 1, inplace = True)


# In[ ]:


# creat categorical value for data with character value
all_data = pd.get_dummies(all_data, columns = ["Ticket"], prefix="T")
all_data = pd.get_dummies(all_data, columns = ["Cabin"], prefix="C")


# In[ ]:


all_data.head()


# # Modeling

# ## 1. Separat the data as training and testing

# In[ ]:


train_data = all_data[:train_len]
test_data = all_data[train_len:]

test_info = test_data.drop(labels=["Survived"],axis = 1)

train_label = train_data["Survived"].astype(int)
train_info = train_data.drop(labels = ["Survived"],axis = 1)


# In[ ]:


print("The # of training data: {}".format(train_label.shape[0]))
print("The # of training data: {}\nThe dimension of training data: {}".format(train_info.shape[0],train_info.shape[1]))
print("The # of testing data: {}\nThe dimension of testing data: {}".format(test_info.shape[0],test_info.shape[1]))


# ## 2. Define the Classifiers

# In[ ]:


classifiers = []
classifiers.append(SVC(random_state = 2))
classifiers.append(AdaBoostClassifier(random_state = 2))
classifiers.append(KNeighborsClassifier())
classifiers.append(LinearDiscriminantAnalysis())
classifiers.append(MLPClassifier(random_state = 2))
classifiers.append(RandomForestClassifier(random_state = 2))
classifiers.append(DecisionTreeClassifier(random_state = 2))


# In[ ]:


kfold = StratifiedKFold(n_splits=5)


# In[ ]:


# evaluate and fit the classifiers
result = []
for i in classifiers:
    result.append(cross_val_score(i, train_info, y = train_label, scoring = "accuracy", cv = kfold, n_jobs=1))
    i = i.fit(train_info, train_label)
means = []
for i in result:
    means.append(i.mean())
    


# In[ ]:


res = pd.DataFrame({"Means":means,"Methods":['SVC','AdaBoost','KNeighbors','LDA',
                                             'MLP','RandomForest','DecisionTree']})
g = sns.barplot("Means","Methods",data = res, palette="muted")
title = g.set_xlabel("Mean Accuracy")


# All the classifier's accuracy is around 0.8. So, we can use all of them.

# ## 3. predict

# In[ ]:


predicts = np.empty((7,418))
row = 0
for i in classifiers:
    predicts[row,:] = i.predict(test_info)
    row += 1


# In[ ]:


print(predicts.shape)


# Generate the final result

# In[ ]:


res = []
for col in range(418):
    mean = np.mean(predicts[:,col])
    res.append(0 if mean <= 0.5 else 1)


# # 3. Generate the submission file

# In[ ]:


result = pd.Series(res, name="Survived")
submit = pd.concat([test_ID,result],axis=1)
submit.to_csv("submission.csv",index=False)

