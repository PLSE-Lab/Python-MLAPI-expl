#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_c
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
print("Setup Complete")


# # Acquiring Data

# In[ ]:


from sklearn.model_selection import train_test_split
#Read Train and Test Data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

y=train_data["Survived"]


# # Exploration, Analysis and Feature Engineering

# In[ ]:


train_data.info()
test_data.info()


# In[ ]:


print("The shape of train_data:", train_data.shape)
print("The shape of test_data:", test_data.shape)


# In[ ]:


train_data.head()


# In[ ]:


train_data.tail()


# In[ ]:


columns = train_data.columns
for col in columns:
    print("unique values in {} column is: {}". format(col, train_data[col].value_counts().size))


# In[ ]:


print("Total number of male passengers:")
print(train_data.loc[train_data.Sex=="male"].Sex.size)
print("Total number of female passengers:")
print(train_data.loc[train_data.Sex=="female"].Sex.size)


# In[ ]:


print("The percentage of survived with respect to Sex:")
print(100 * train_data.groupby("Sex").Survived.mean())


# In[ ]:


print("The percentage of survived with respect to Pclass:")
print(100 * train_data.groupby("Pclass").Survived.mean())


# In[ ]:


print("The percentage of survived with respect to Age:")
print(100 * train_data.groupby("Age").Survived.mean())


# As you can see it is difficult to grasp an idea about the relation between survived and age features. In this case, data visualization will be used.

# In[ ]:


g = sns.FacetGrid(col="Survived", data=train_data, height = 2, aspect=3)
g.map(sns.distplot, "Age", kde=False, bins=80)


# The above graph tells us that children had a higher survival chance. To get a more clear prediction, one way is to divide the age feature into categories. pandas.cut() function will be used to achieve this goal. Since xgboost will be used, missing values will not be imputed instead they will be categorized as missing.

# In[ ]:


def cut_age(df, cut_values, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age"]=pd.cut(df["Age"], bins=cut_values, labels=label_names)
    return df
    
cut_values=[-1, 0, 3, 12, 19, 35, 60, 80]
label_names=["Missing", "Infants", "Children", "Teenagers", "Young Adults", "Middle-Age Adults", "Seniors"]
train_data=cut_age(train_data, cut_values, label_names)
test_data=cut_age(test_data, cut_values, label_names)


# In[ ]:


sns.catplot(x="Age", row="Survived", kind="count", height=3, aspect=4, data=train_data)


# In[ ]:


print(100 * train_data.groupby("Age").Survived.mean())


# The results prove that infants and children had survived more than other age groups.

# In[ ]:


print("The percentage of survived with respect to SibSp:")
print(100 * train_data.groupby("SibSp").Survived.mean())


# In[ ]:


print("The percentage of survived with respect to Parch:")
print(100 * train_data.groupby("Parch").Survived.mean())


# What is interesting in these findings is that it seems that individuals who are alone (no family members) had less chance of survival than if they had 1 to 3 family members on board. But when the family members are more than 3 the survival chance drops. To be more precise, Parch and SibSp featurs will be combined into one feature to indicate the number of family members of each passengers on board. The new feature will be called Fam_membs, and Parch and SibSp features will be dropped.

# In[ ]:


def Cr_fam_membs(df):
    df["FamMembs"]= df["Parch"] + df["SibSp"]
    df=df.drop(["SibSp", "Parch"], axis=1)
    return df

train_data=Cr_fam_membs(train_data)
test_data=Cr_fam_membs(test_data)


# In[ ]:


print(100 * train_data.groupby("FamMembs").Survived.mean())


# To limit the number of categories in Fam_membs features, it will be divided into 4 categories as following:
# * IsAlone: 0 fam_membs
# * Small family: 1-3 fam_membs
# * Meduim family: 4-6 fam_membs
# * Large family: 7-10 fam_membs

# In[ ]:


train_data["FamMembs"].unique()


# In[ ]:


test_data["FamMembs"].unique()


# In[ ]:


train_data["FamMembs"]=train_data["FamMembs"].apply(lambda s: "IsAlone" if s==0 else s)
train_data["FamMembs"]=train_data["FamMembs"].apply(lambda s: "Small family" if (s==1 or s==2 or s==3) else s)
train_data["FamMembs"]=train_data["FamMembs"].apply(lambda s: "Meduim family" if (s==4 or s==5 or s==6) else s)
train_data["FamMembs"]=train_data["FamMembs"].apply(lambda s: "Large family" if (s==7 or s==10) else s)


# In[ ]:


test_data["FamMembs"]=test_data["FamMembs"].apply(lambda s: "IsAlone" if s==0 else s)
test_data["FamMembs"]=test_data["FamMembs"].apply(lambda s: "Small family" if (s==1 or s==2 or s==3) else s)
test_data["FamMembs"]=test_data["FamMembs"].apply(lambda s: "Meduim family" if (s==4 or s==5 or s==6) else s)
test_data["FamMembs"]=test_data["FamMembs"].apply(lambda s: "Large family" if (s==7 or s==10) else s)


# In[ ]:


train_data["FamMembs"].value_counts()


# In[ ]:


test_data["FamMembs"].value_counts()


# In[ ]:


print("The percentage of survived with respect to Fam_membs:")
print(100 * train_data.groupby("FamMembs").Survived.mean())


# In[ ]:


sns.catplot(x="FamMembs", row="Survived", kind="count", height=3, aspect=4, data=train_data)


# In[ ]:


print("The percentage of survived with respect to Embarked:")
print(100 * train_data.groupby("Embarked").Survived.mean())


# Since there is only two missing values in the train_data Embarked feature, they will be filled with the most frequent value.

# In[ ]:


train_data["Embarked"]=train_data["Embarked"].fillna(train_data["Embarked"].mode()[0])
test_data["Embarked"]=test_data["Embarked"].fillna(test_data["Embarked"].mode()[0])


# In[ ]:


g = sns.FacetGrid(col="Survived", data=train_data, height = 2, aspect=3)
g.map(sns.distplot, "Fare", kde=False, bins=100)


# Here we face the same issue as the age feature. To be more clear pandas.cut() or pandas.qcut() will be used to divide the fare into categories. Further study will be done on the Fare feature to decide which to use  pandas.cut() or pandas.qcut().

# In[ ]:


sns.catplot(x="Pclass", y="Fare", kind="bar", data=train_data)


# In[ ]:


bins=np.arange(0, 600, 50)
g=sns.FacetGrid(row="Pclass", data=train_data, height = 3, aspect=5)
g.map(sns.distplot, "Fare", kde=False, bins=bins, color="b")


# The values differ from what is expected, as there are people who are in Pclass 1 but paid low to no fare. This can be due to several reasons such as if the passenger is an employee or not.
# Before dividing the fare, its relation with embarked port will be checked. And the relation of Embarked port with Pclass will also be checked.

# In[ ]:


bins=np.arange(0, 600, 50)
g=sns.FacetGrid(col="Embarked", data=train_data, height = 3, aspect=2)
g.map(sns.distplot, "Fare", kde=False, bins=bins, color="b")


# Individuals who embarked through port S paid the lowest fare.[](http://)

# In[ ]:


sns.catplot(x="Pclass", hue="Embarked", kind="count", data=train_data)


# This explains why a high number of passengers who embarked from port S paid the low fares, as this port is mostly used by Pclass 3. It also showed the lowest survival mean. Port C showed passengers who paid higher fares, as it is mostly used by Pclass 1. Also, port C showed the highest survival mean. Port Q was embarked by individuals who paid low fares and mainly belonged to Pclass 3. But why did Port S show less survival mean than Port Q? This might be because Port S was used by most of the passengers (higher number of passengers embarked through this port).

# In[ ]:


#Check number of passengers who embarked at each port
print(train_data.loc[train_data["Embarked"]=="S"].PassengerId.value_counts().sum())
print(train_data.loc[train_data["Embarked"]=="Q"].PassengerId.value_counts().sum())
print(train_data.loc[train_data["Embarked"]=="C"].PassengerId.value_counts().sum())


# In the test_data there is 1 empty value in the Fare feature. I will fill it with -1. 

# In[ ]:


test_data["Fare"] = test_data["Fare"].fillna(-1)


# In[ ]:


train_data["Fare"].describe()


# In[ ]:


test_data["Fare"].describe()


# pandas.qcut() will be used to divide the Fare into four categories with equal distributions. pandas.cut() was not used since it is not clear how the fare was assigned and how it relates with other features. 

# In[ ]:


def qcut_fare(df, q, labels):
    df["Fare"]=pd.qcut(df["Fare"], q, labels=labels)
    return df

labels=["range1", "range2", "range3", "range4"]
train_data=qcut_fare(train_data, 4, labels)
test_data=qcut_fare(test_data, 4, labels)


# The reason q=4 was used is becaused the data will be divided according to the values that .desctibe() show you (min, 25%, 50%, 75%, max)

# In[ ]:


sns.catplot(x="Fare", data=train_data, kind="count", height=2, aspect=3)


# In[ ]:


sns.catplot(x="Fare", data=test_data, kind="count", height=2, aspect=3)


# In[ ]:


train_data["Name"]


# At the beginning I started with the assumption that the Name feature does not serve useful since every individual has a unique name. What is interesting here is that names have titles. These titles can be useful for the predictions.

# In[ ]:


train_data["Name"]=train_data["Name"].apply(lambda s: s.split(', ')[1].split('.')[0])
test_data["Name"]=test_data["Name"].apply(lambda s: s.split(', ')[1].split('.')[0])


# In[ ]:


train_data["Name"].unique()


# In[ ]:


test_data["Name"].unique()


# In[ ]:


train_data["Name"].value_counts()


# In[ ]:


test_data["Name"].value_counts()


# In[ ]:


train_data["Name"]=train_data["Name"].replace(["Ms", "Mlle"], "Miss")
train_data["Name"]=train_data["Name"].replace(["Sir"], "Mr")
train_data["Name"]=train_data["Name"].replace(["Mme"], "Mrs")
train_data["Name"]=train_data["Name"].replace(["Dr", "Rev", "Col", "Major", "Capt", "Master", 
                                             "Lady", "the Countess", "Don", "Dona", "Jonkheer"], "Other")


# In[ ]:


test_data["Name"]=test_data["Name"].replace(["Ms", "Mlle"], "Miss")
test_data["Name"]=test_data["Name"].replace(["Sir"], "Mr")
test_data["Name"]=test_data["Name"].replace(["Mme"], "Mrs")
test_data["Name"]=test_data["Name"].replace(["Dr", "Rev", "Col", "Major", "Capt", "Master", 
                                             "Lady", "the Countess", "Don", "Dona", "Jonkheer"], "Other")


# In[ ]:


train_data["Name"].unique()


# In[ ]:


train_data["Name"].value_counts()


# In[ ]:


test_data["Name"].value_counts()


# In[ ]:


sns.catplot(x="Name", hue="Survived", kind="count", data=train_data)


# In[ ]:


train_data.Cabin.describe()


# In[ ]:


train_data.Cabin.unique()


# At first I began with the idea of dropping the Cabin feature, as it has a lot of missing values. However, since I will be using Xgboost in my model, which is known to be good with dealing with missing values, I will make use of this feature. 
# 
# Instead of having a Cabin feature, a Deck feature will be created. Cabins were located on different decks on the ship. These decks were: A, B, C, D, E, F, G, T.
# 
# "Unknown" will designate NaN values.

# In[ ]:


train_data["Cabin"]=train_data["Cabin"].fillna("Unknown")


# In[ ]:


test_data["Cabin"]=test_data["Cabin"].fillna("Unknown")


# In[ ]:


train_data["Cabin"].unique()


# In[ ]:


test_data["Cabin"].unique()


# In[ ]:


train_data["Deck"]=train_data["Cabin"].str.replace("([0-9\s])+","")


# In[ ]:


test_data["Deck"]=test_data["Cabin"].str.replace("([0-9\s])+","")


# In[ ]:


test_data["Deck"].value_counts()


# In[ ]:


train_data["Deck"].value_counts()


# The data reveals that some passangers had more than one cabin. Some of them were on the same deck and some were on different decks. To deal with this, a new feature will be created that indicates the total number of cabins per passenger. If the passenger's cabin is unknown it will give 0. For decks which consitute of more than one letter, the first letter will be taken.

# In[ ]:


def total_cabins(row):
    if row.Deck == "Unknown":
        row["TotalCab"] = 0
    elif len(row.Deck) > 1:
        row["TotalCab"] = len(row.Deck)
    else:
        row["TotalCab"] = 1
    return row

train_data=train_data.apply(total_cabins, axis=1)
test_data=test_data.apply(total_cabins,axis=1)
        


# In[ ]:


train_data["TotalCab"].value_counts()


# In[ ]:


test_data["TotalCab"].value_counts()


# In[ ]:


train_data["Deck"]=train_data["Deck"].apply(lambda s: s[0] if s != "Unknown" else s)


# In[ ]:


test_data["Deck"]=test_data["Deck"].apply(lambda s: s[0] if s != "Unknown" else s)


# In[ ]:


test_data["Deck"].value_counts()


# In[ ]:


train_data["Deck"].value_counts()


# In[ ]:


train_data=train_data.drop(["Survived", "Cabin", "Ticket"], axis=1)
test_data=test_data.drop(["Cabin", "Ticket"], axis=1)


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# # Deal with categorical values

# In[ ]:



from sklearn.preprocessing import OneHotEncoder

OHE = OneHotEncoder(handle_unknown='ignore', sparse=False)

features = ["Pclass", "Name", "Sex", "Age", "Fare", "Embarked", "FamMembs", "Deck", "TotalCab"]
OHE_train_cols = pd.DataFrame(OHE.fit_transform(train_data[features]))
OHE_test_cols = pd.DataFrame(OHE.transform(test_data[features]))

OHE_train_cols.index = train_data.index
OHE_test_cols.index = test_data.index

num_train=train_data.drop(features, axis=1)
num_test=test_data.drop(features, axis=1)

train_data = pd.concat([num_train, OHE_train_cols], axis=1)
test_data = pd.concat([num_test, OHE_test_cols], axis=1)


# In[ ]:


print(train_data.shape, test_data.shape)


# # XGBoost Parameter Tuning & RandomizedSearchCV

# In[ ]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
xgb=XGBClassifier(objective='reg:logistic')

params={
    'n_estimators': [200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 9],
    'colsample_bytree': [ 0.4, 0.6, 0.8],
    'subsample': [0.8, 0.9, 1],
    'gamma': [0, 0.5, 1]
}

clf=RandomizedSearchCV(xgb, param_distributions=params, n_iter=50, n_jobs=-1, verbose=1)
clf.fit(train_data, y)


# In[ ]:


score=clf.best_score_
params=clf.best_params_
print("Best score: ",score)
print("Best parameters: ", params)


# In[ ]:


final_predictions = clf.predict(test_data)


# In[ ]:


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': final_predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")

