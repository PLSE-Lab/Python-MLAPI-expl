#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import altair as alt
alt.renderers.enable('notebook')

import os
print(os.listdir("../input"))


# ## Importing Data 

# In[ ]:


#Import dataset and print the structure of the dataset
df = pd.read_csv('../input/heart.csv')
print('\nShape of Dataset: {}'.format(df.shape))


# In[ ]:


#view the data
df.head()


# # Exploratory Data Analysis (EDA)

# In[ ]:


#Checking the count of samples across each person's Age
df["age"].value_counts().head(10)


# In[ ]:


# Adding a new column - 'age_group'. Calculation logic - eg) If age is between 40 to 49, age_group will be 40

# start stop and step variables 
start, stop, step = 0, 1, 1  
# converting to string data type 
df["age_str"]= df["age"].astype(str) 
# slicing till 2nd last element 
df["age_group"]= df["age_str"].str.slice(start, stop, step) 
# concatenate zero at the end
df['age_group'] = df['age_group'] + '0'
#converting to int
df['age_group'] = df['age_group'].astype(int)


# ## Heart Diseases among different Age groups

# In[ ]:


df2 = df.groupby(['age_group','target'])['age_group'].count().unstack('target').fillna(0)
df2.plot(kind='bar', stacked=True, color=['green', 'red'])


# In[ ]:


df = df.drop(columns=['age_group','age_str'])

As per the above graph, the highest number of heart diseases are observed in people when their age is around 50's. 
When compared to the total volume of data in each age groups, people in their 40's are highly susceptible to heart problems since the volume of red color (which denotes heart disease) is greater than green color (who doesn't have heart disease)
# ## Comparison of Heart Diease seen between Males and Females

# In[ ]:


#sex 1- male, 0-female
df[['age','sex','target']].groupby(['sex','target']).count()


# In[ ]:


#Chart showing the comparison heart diseases between Males & Females.
#sex 1- male, 0-female
df2 = df.groupby(['sex','target'])['sex'].count().unstack('target').fillna(0)
df2.plot(kind='bar', stacked=False, color=['limegreen', 'orangered'])
plt.show()

Females are highly prone to heart diease compared to Males as the red bar is longer than the green one for Sex = 0 (Females) compared to Sex = 1 (Males) 

# ## Identifying features that are highly correlated to the heart disease

# In[ ]:


corr = df.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
plt.show()


# The possibility of heart disease is highly related to the features - 'chest pain type' (CP), maximum heart rate achieved (thalach). It is surprising that the column 'age' is not very correlated to 'target'. This tells us that heart dieases are common in all age groups

# ## Which chest pain types attribute to heart disease ?

# In[ ]:


# Grouping by 'cp' (Chest Pain Type)
df[['target','cp']].groupby(['cp']).count()


# In[ ]:


alt.Chart(df).mark_bar().encode(
    x='count(target):Q',
    y=alt.Y(
        'cp:N',
        sort=alt.EncodingSortField(
            field="target",  # The field to use for the sort
            op="count",  # The operation to run on the field prior to sorting
            order="descending"  # The order to sort in
        )
    )
)

Persons having chest pain type '0' has maximum possibility of getting a heart disease followed by chest pain type - '2'. Therefore, whomever has type '0' heart pain should immediately visit the nearest hospital. The detailed description for different chest pains is not provided in the dataset.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ### Since the dataset is very small, applying any Machine Learning Algorithm will result in overfitting and therefore cannot predict accurate results. This can be used for carrying out basic data analysis thereby make anyone familiar with few of the data vizualisation libraries and how it can be applied to the dataset.
