#!/usr/bin/env python
# coding: utf-8

# In[457]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[458]:


train = pd.read_csv(r"../input/train.csv")
test= pd.read_csv(r"../input/test.csv")

test['Fare'].fillna(test.Fare.mean(),inplace=True)


# In[459]:


df = pd.concat([train, test], axis=0)


# In[460]:


#Drop columns which won't be useful
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)


# In[461]:


df.isnull().sum()


# In[462]:


df.head()


# In[463]:


#Encode categorica; variables
df.Embarked.fillna('S',inplace=True)
df.Cabin.fillna('U',inplace=True)

df = pd.concat([df.drop('Sex', axis=1), pd.get_dummies(df['Sex'],drop_first=True)], axis=1)
df = pd.concat([df.drop('Embarked', axis=1), pd.get_dummies(df['Embarked'],drop_first=True)], axis=1)


# In[464]:


#Findings based on name titles
df['Name']= df['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())


# In[465]:


unique_titles=df.Name.unique()
unique_titles


# In[466]:


#fill age acc to people belonging to same title
medians_acc_age=dict()
df['Age'].fillna(-1,inplace=True)

for title in unique_titles:
    median_value=df.Age[(df.Age!=-1) & (df['Name']==title)].median()
    medians_acc_age[title]=median_value


# In[467]:


for index,row in df.iterrows():
    if row['Age']==-1 :
        df.loc[index,'Age']=medians_acc_age[row['Name']]


# In[468]:


df.Cabin=df['Cabin'].apply(lambda x:x[0])


# In[469]:


df.head()


# In[470]:


unique_cabins=df['Cabin'].unique()
unique_cabins


# In[471]:


fig=plt.figure(figsize=(6,10))
i=1
for u in unique_cabins:
    fig.add_subplot(3,3,i)
    plt.title(u)
    df.Survived[df.Cabin==u].value_counts().plot(kind='pie')
    i=i+1


# 

# In[472]:


#Encode values to know which Cabin has higher chance of people surviving
values_to_replace= {
    'T': 0,
    'U': 1,
    'A': 2,
    'G': 3,
    'C': 4,
    'F': 5,
    'B': 6,
    'E': 7,
    'D': 8
}

df['Cabin'] = df['Cabin'].apply(lambda x: values_to_replace.get(x))


# In[473]:


df.drop('Name',axis=1,inplace=True)


# In[474]:


#Create x train and y train by filtering the Survived who have NaN meaning test data

x = df[df['Survived'].notnull()]
y_train=x.iloc[:,6]

x.drop('Survived',axis=1,inplace=True)


x_t=df[df['Survived'].isnull()]
x_t.drop('Survived',axis=1,inplace=True)


# In[475]:


from sklearn.ensemble import RandomForestClassifier


# In[476]:


models=[
   RandomForestClassifier(n_estimators=1000),
  
]


# In[477]:


for m in models:
    m.fit(x, y_train)
    y_pred=m.predict(x_t)
   


# In[478]:


#Accuracy Achieved: Approx 77%


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




