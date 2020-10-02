#!/usr/bin/env python
# coding: utf-8

# # Cleaning and preparing the Titanic Dataset for Scikit-Learn
# * first import the data and take a look at its structure

# In[ ]:


#!/usr/bin/python3

import pandas as pd
import numpy as np

train=pd.read_csv("../input/train.csv", sep=",", header=0, index_col=0)
test=pd.read_csv("../input/test.csv", sep=",", header=0, index_col=0)

data = train.append(test)

print("Data structure:")
print("***************")
print(data.columns)
print(data.dtypes)
print("\nExample:")
print("**********")
print(data.head())
print("\nStatistics:")
print("*************")
print(data.describe())
print("Correlations:")
print(data.corr())
print("*************")
print("Columns with <10 categories:")
for i in data.columns:
    catdat = pd.Categorical(data[i])
    if len(catdat.categories)>9:
        continue

    print(i," ",pd.Categorical(data[i]))


# * Handle NA values and replace by mean or "usual" value

# In[ ]:


data.Age.fillna(value=data.Age.mean(), inplace=True)
data.Fare.fillna(value=data.Fare.mean(), inplace=True)
data.Embarked.fillna(value=(data.Embarked.value_counts().idxmax()), inplace=True)
data.Survived.fillna(value=-1, inplace=True) # the test data have NA


# * Extract title/salutation from name string

# In[ ]:


# extract title and salutation
print("Extracting titles and adding column...")
titles = pd.DataFrame(data.apply(lambda x: x.Name.split(",")[1].split(".")[0], axis=1), columns=["Title"])
print(pd.Categorical(titles.Title))
data = data.join(titles)


# * Add family size as a combination of the other 2 columns

# In[ ]:


# family size
print("Calculating family size and adding column...")
fsiz = pd.DataFrame(data.apply(lambda x: x.SibSp+x.Parch, axis=1), columns=["FSize"])
data = data.join(fsiz)


# * replace columns that are not usable by numeric algorithms and have no use (cabin, ticket...) or have been substituted (parch, sibsp)

# In[ ]:


# drop useless columns
data.drop('Name', axis=1, inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)

# no need for the following as the sum is used
data.drop('Parch', axis=1, inplace=True)
data.drop('SibSp', axis=1, inplace=True)


# * finally generate a numerical outupt from categories (like sex, title...) in an automatic way
# * write cleaned data to output csv

# In[ ]:


# generate numerical output
print("Conveting to numerical output...")

for col in data.select_dtypes(exclude=["number"]).columns:
    print("Converting column "+col+"...")
    data[col] = data[col].astype('category')    
    print(data[col].cat.categories)
    data[col] = data[col].cat.codes   

train = data[data['Survived']!=-1]
train.to_csv("train-clean.csv")

test = data[data['Survived']==-1]
test.drop('Survived', axis=1, inplace=True)
test.to_csv("test-clean.csv")

