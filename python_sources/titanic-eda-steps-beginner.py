#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# #### Exploratory Data Analysis
# This notebook contains my understanding on the titanic data set, the objective is to draw as many valuable insights as possible. Do upvote it if you like.
# 
# Lets begin now
# 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train=pd.read_csv('../input/titanic/train.csv',na_values="NAN")
test=pd.read_csv('../input/titanic/test.csv',na_values="NAN")


# In[ ]:


train.head()


# In[ ]:


test.head()


# ### Find out if any null values exist 

# In[ ]:


train.isnull().sum()


# ##### Age , cabin and Embarked have missing values. We will deal with the missing values later in this notebook
# 
# To visualize it we can use heatmap for better understanding 

# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='plasma')


# the yellow lines show the data that is missing , If you will observe Cabin has most of the data that is missing therefore we can think about removing the entire column in the future.
# For Age we will deal with it later in this notebook.

# In[ ]:


train.shape


# In[ ]:


test.shape


# (There are more rows in train data than in test data )

# In[ ]:


train.columns


# In[ ]:


train.info()


# Save all the survivor data in another data frame called survivor

# In[ ]:


survived=train[train["Survived"]==1]
survived.head()


# Get the stats about total male and female survivors

# In[ ]:


train['Sex'].value_counts()


# In[ ]:


female_s=survived[survived["Sex"]=="female"]
female_s.head()


# In[ ]:


female_s.shape


# In[ ]:


male_s=survived[survived['Sex']=="male"]
male_s.head()


# In[ ]:


male_s.shape


# In[ ]:


print("Total number of female who did not survive the crash is 81 out of 314 ")
print("Total number of male who did not survive the crash is 468 out of 577")


# In[ ]:


survived.shape


# take a look at the .shape above where the rows show the total female and male survivors

# Visualize the above data to understand better 
# 
# 1) first graph shows that if there were more deaths as compared to survivors

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x="Survived",data=train)


# 2) the next graph shows the comparison of how many male and female survived and who unfortunately did not

# In[ ]:


sns.set_style("whitegrid")
sns.countplot(x="Survived",hue=train.Sex,data=train)


# 3) This graph show the number of male and females in particulatar ticket class 
# 
# where 1st = Upper
# 2nd = Middle
# 3rd = Lower

# In[ ]:


sns.countplot(x="Pclass",hue=train.Sex,data=train)


# 4)The graph shows the most deaths were from people who were in lower class ie 3rd

# In[ ]:


sns.set_style("whitegrid")
sns.countplot(x="Survived",hue="Pclass",data=train)


# 5) Observe the distribution in Ages of all the passengers can be shown.
# It can be seen that the range of ages were more between 20 to 40

# In[ ]:


sns.distplot(train["Age"].dropna(),kde=False,color="darkred",bins=40)


# 6) distribution for ages for pleaople who survived the crash

# In[ ]:


sns.distplot(survived["Age"].dropna(),kde=False,color="darkred",bins=40)


# 7)The graph represents the siblings or spouse
# 
# maximums passengers were traveling withoutsiblings or spouse. 
# 
# 1 mostly represents spouse

# In[ ]:


sns.countplot(x="SibSp",data=train)


# 8) the graph show that the people who were traveling without spouse or siblings.

# In[ ]:


sns.countplot(x="SibSp",data=survived)


# 9) Fare

# In[ ]:


train["Fare"].hist(bins=40,figsize=(8,4))


# 10) Boxplot is very significant kind of plot. It shows the average age of passengers of each ticket class.

# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x="Pclass",y="Age",data=train)


# Now we can replace the missing age values with the average age for depending on the ticket class they belong to.

# In[ ]:


def replace_age(cols):
    age=cols[0]
    pclass=cols[1]
    if pd.isnull(age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 29
        elif pclass==3:
            return 24
    else:
        return age


# In[ ]:


train["Age"]=train[["Age","Pclass"]].apply(replace_age,axis=1)


# In[ ]:


train.isnull().sum()


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="plasma")


# Since the age columns contain no missing value , you can see the difference in the heatmaps above.

# In[ ]:


train.drop("Cabin",axis=1,inplace=True)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap="plasma")


# Do upvote if this notebook helped in someway .

# Notebook will be updated keep checking.

# In[ ]:




