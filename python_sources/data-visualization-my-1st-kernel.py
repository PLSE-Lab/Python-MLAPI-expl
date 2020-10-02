#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../input/xAPI-Edu-Data.csv")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.


# In[ ]:



df.head()


# In[ ]:


df.info()


# In[ ]:


# Visualize classes categorically
sns.countplot(x="Class", data=df,palette="muted")


# In[ ]:


# Visualization of all the topics with class segregation
g=sns.countplot(x="Topic", data=df,hue="Class", palette="muted");
#g.set_xticklabels(labels = df["Topic"].value_counts().index.tolist(),rotation=90)
plt.xticks(rotation=90)


# In[ ]:


# Gender wise class segregation
sns.countplot(x="Class", data=df,hue="gender", palette="muted")


# In[ ]:


#Adding one column for status of pass or fail
y=[]
for x in df['Class']:
    if x=='L':
      y.append('Fail')
    else:
       y.append('Pass')
df['Stats']=y


    
    


# In[ ]:


df.PlaceofBirth.value_counts()


# In[ ]:


sns.countplot(x="PlaceofBirth", data=df,hue="Stats", palette="muted")
plt.xticks(rotation=90)


# In[ ]:


sns.countplot(x="Stats", data=df,hue="Relation", palette="muted")


# In[ ]:


sns.countplot(x="ParentAnsweringSurvey", data=df,hue="Stats", edgecolor=sns.color_palette("dark", 3))


# In[ ]:


sns.pairplot(df, hue="Stats")


# In[ ]:


# Adding one column to differentiate among active and dull students based on raisedhands
h=[]
for m in df['raisedhands']:
    if m>30:
      h.append('Active')
    else:
       h.append('Dull')
df['Cat']=h


# In[ ]:


#Boxplot showing Active & Dull students categorization based on Pass/Fail and Discussion

sns.boxplot(x='Cat', y='Discussion',hue='Stats',data=df,palette="Set3")


# In[ ]:


sns.countplot(x="StudentAbsenceDays", data=df,hue="Stats")


# In[ ]:




