#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import random
import seaborn as sns
import matplotlib.pyplot as plt
import math
import sklearn.ensemble as skl
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Step 1: Actually Look at the Data**
# 
# First things first, we need to see what the dataset looks like- what information we have, and what needs to be filled in

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
combined = [train,test]
train.head()


# In[ ]:


test.describe()


# **So, some off-the-cuff thoughts**
# * Full name is almost certainly useless
# * ...that being said, I wonder if LENGTH of the name will do something? I'll keep that in mind for future analysis
# * Cabin number appears to contrain many nulls, possibly a category to drop entirely
# * Embarkation location has two nulls: I might just drop these from the dataset
# * Age has quite a few null values, enough that some sort of interpolation/guessing would be useful

# In[ ]:



#:ets convert sex to a numerical value for simplicity
for ds in combined:
    ds['Sex'] = ds['Sex'].map({'female': 1, 'male': 0})
    # Next, let's fill in the Embarkation point values. Shamelessly borrowed from https://www.kaggle.com/startupsci/titanic-data-science-solutions

    freq_port = ds.Embarked.dropna().mode()[0]
    ds['Embarked'] = ds['Embarked'].fillna(freq_port)
    ds['Embarked'] = ds['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    

    # Let's simplify the Titles/Honorifics
    # In theory, the Age values can be interpreted from these
    ds['Title'] = ds.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
#NOTE: Will reuse following code a lot
train = combined[0]
test = combined[1]


# In[ ]:


#train.head()
pd.crosstab(train["Sex"],train['Title'])


# In[ ]:


#train.head()
pd.crosstab(test["Sex"],test['Title'])


# **So what do we do with all of these?**
# 
# I don't think giving each of these a unique title value will be terribly helpful, so let's look for a better categorization. We can group these into five distinct categories by the looks of it:
# 1. Master (Presumably younger male passengers)
# 2. Mr. (Older male)
# 3. Mrs./Mme (Married women)
# 4. Miss/Mlle/Ms. (I.e. unmarried women) (Not sure if this is distinct from 3. in analysis)
# 5. All the honorific titles (Capt,Col,Jonkheer, etc.)

# In[ ]:


for ds in combined:
    honorifics = ['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Rev','Sir','Dona']
    ds["Title"] = ds["Title"].replace(honorifics, 'Titled')
    
    ds["Title"] = ds["Title"].replace(["Miss","Mlle"],'Ms')
    ds["Title"] = ds["Title"].replace(["Mme"],'Mrs')
    
    ds["Title"]  = ds["Title"].map( {'Master': 0, 'Ms': 1, 'Mr': 2, 'Mrs' : 3, 'Titled' : 4})
    
#NOTE: Will reuse following code a lot
train = combined[0]
test = combined[1]


# **So how do we deal with that age column?**
# 
# Well,
# * The ages of all passengers should follow some (wide) distribution
# * Within the title classes, there should be significantly tighter distributions
# * So if we assume Gaussians in all title classes, we can fill in the ageless entries with guesses on that distribution
# 

# In[ ]:


counts = np.zeros((5,1))
means = np.zeros((5,1))
st_devs = np.zeros((5,1))

#First thing, go over the completed rows for the age distributions
merged = pd.concat(combined, sort=True) #NOTE TO SELF: Go back over the earlier processing steps later maybe
merged = merged.dropna(subset=["Age"],axis = "rows")

for i in range(5):
    a = merged.loc[merged["Title"] == i]
    means[i] = a.loc[:,"Age"].mean()
    st_devs[i] = a.loc[:,"Age"].std()


# In[ ]:


#Now that we have the descriptive statistics, we will assume gaussian distributions for all of them and fill in accordingly
i = 0
for ds in combined:
    for index, row in ds.iterrows():
        if math.isnan(row["Age"]):
            age = -2
            title = int(row["Title"])
            mu = means[title]
            spread = st_devs[title]
            while (age < 0):
                age = np.random.normal(loc = mu,scale = spread)
            i = i + 1
            row["Age"] = age
            ds.loc[index,:] = row
#NOTE: Will reuse following code a lot
train = combined[0]
test = combined[1]            


# In[ ]:


#Just making sure nothing broke
train.tail()


# In[ ]:


#Now to keep paring it down, we'll cut the unimportant variables and extract the survived column
for ds in combined:
    ds.drop(['Ticket', 'Cabin','Name','PassengerId'], axis=1,inplace = True)
#NOTE: Will reuse following code a lot
train = combined[0]
test = combined[1]

survival = train["Survived"]
train.drop(["Survived"],axis = 1,inplace = True)


# In[ ]:


#So it turns out there's a nan in the fares for the test set. Let's fix that
#Shouldn't make things too bad if we just use mean for this one
mean_fare = merged["Fare"].mean()
test["Fare"] = test["Fare"].fillna(mean_fare)


# In[ ]:


#So I guess we try some classification now

clf = skl.RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
clf.fit(train,survival)


# In[ ]:


predictions = clf.predict(test)


# In[ ]:


#Now to package this all up
IDs  = range(892,1310) #Because I forgot about this
d = {'PassengerId': IDs, 'Survived': predictions}
final = pd.DataFrame(data = d)
final.to_csv('Final.csv',index = False)

