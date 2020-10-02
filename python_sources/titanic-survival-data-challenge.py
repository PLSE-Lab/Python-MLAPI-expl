#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas import Series, DataFrame


# In[ ]:


titanic_df = pd.read_csv('train.csv')


# In[ ]:


titanic_df.head()


# In[ ]:





# In[ ]:


#Exporing the columns

titanic_df.info()


# In[ ]:





# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Let's first check gender
sns.factorplot('Sex', data=titanic_df,kind='count')


# In[ ]:


sns.factorplot('Sex', data=titanic_df,kind='count',hue='Pclass')


# In[ ]:


sns.factorplot('Pclass', data=titanic_df,kind='count',hue='Sex')


# In[ ]:


This tells us that there was more males in the 3rd class compartment 


# In[ ]:


# First let's make a function to sort through the sex 
def male_female_child(passenger):
    # Take the Age and Sex
    age,sex = passenger
    # Compare the age, otherwise leave the sex
    if age < 16:
        return 'child'
    else:
        return sex
    

# We'll define a new column called 'person', remember to specify axis=1 for columns and not index
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[ ]:


# Let's see if this worked, check out the first ten rows
titanic_df[0:10]


# In[ ]:


sns.factorplot('Pclass', data=titanic_df,kind='count',hue='person')


# In[ ]:


#Lets see the histogram of the Age column

titanic_df['Age'].hist(bins=70)


# In[ ]:


#Average Age on the Ship was ---
titanic_df['Age'].mean()


# In[ ]:


# Lets see how many women, men and children were on board
titanic_df['person'].value_counts()


# In[ ]:


fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age', shade=True)

oldest =  titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[ ]:


# Kernel density plot by male, female and child

fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)
fig.map(sns.kdeplot,'Age', shade=True)

oldest =  titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[ ]:


#Distribution by Class

fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age', shade=True)

oldest =  titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[ ]:


titanic_df.head()


# In[ ]:


deck = titanic_df['Cabin'].dropna()


# In[ ]:


deck.head()


# In[ ]:





# In[ ]:


# So let's grab that letter for the deck level with a simple for loop

# Set empty list
levels = []

# Loop to grab first letter
for level in deck:
    levels.append(level[0])    

# Reset DataFrame and use factor plot
cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot('Cabin',data=cabin_df,kind='count',palette='winter_d')


# In[ ]:


cabin_df = cabin_df[cabin_df.Cabin != 'T']


# In[ ]:


sns.factorplot('Cabin',data=cabin_df,kind='count',palette='summer')


# In[ ]:


titanic_df.head()


# In[ ]:


#Let's research about the ports embarked

sns.factorplot('Embarked',data=titanic_df,kind='count',hue='Pclass')


# In[ ]:


#Lets see how was the gender distributed

sns.factorplot('Embarked',data=titanic_df,kind='count',hue='person')


# In[ ]:


# Who was alone  and who was with family
titanic_df.head()


# In[ ]:


# Lets mine the SibSp and Parch to understand passenger who were alone


# In[ ]:


titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch


# In[ ]:


titanic_df.head()


# In[ ]:





# In[ ]:


titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'with family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


# In[ ]:


titanic_df.head()


# In[ ]:


titanic_df[titanic_df['Alone'] == 'Alone'].count()


# In[ ]:


sns.factorplot('Alone',data=titanic_df,palette='Blues', kind='count')


# In[ ]:


#Lets see if they survived the fateful day
titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})


# In[ ]:


titanic_df.head()


# In[ ]:


sns.factorplot('Survivor',data=titanic_df,palette='Set1',kind='count')


# In[ ]:


sns.factorplot('Pclass','Survived',hue='person',data=titanic_df)


# In[ ]:


sns.lmplot('Age','Survived',titanic_df)


# In[ ]:


sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter')


# In[ ]:


generations = [10,20,40,60,80]

sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)


# In[ ]:


sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)

