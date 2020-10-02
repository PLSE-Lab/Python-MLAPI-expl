#!/usr/bin/env python
# coding: utf-8

# <h1>FIFA 19 Data Analysis</h1>
# FIFA 19 is the latest buzz in the gaming world right now.
# Using this data set, I will try to find the top players according to their different abilites. We will try to pick the GOAT among the world's best using visualizations.

# In[ ]:


#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#now we will load the FIFA 19 dataset
data=pd.read_csv("../input/data.csv")


# In[ ]:


#lets see the summary of the dataset
data.describe()


# In[ ]:


#first three rows of the dataset
data.head(3)


# In[ ]:


#last three rows
data.tail(3)


# In[ ]:


#lets see how many rows and columns we have in our dataset
print("Number of (rows,columns):",data.shape)


# In[ ]:


#checking if there is any NULL value in the dataset
data.isna().sum()


# In[ ]:


#we saaw that most of the data in 'Loaned From' column is not assigned, hence we will drop it
data.drop('Loaned From',axis=1,inplace=True)


# In[ ]:


#now the data which have NA values, we will fill them with the mean value of that column
data.fillna(data.mean(),inplace=True)


# In[ ]:


#we will check again if after assigning the mean value to the cells of the originally NA values; if there is any cell which has NA value
data.isna().sum()


# In[ ]:


#there are still cells in which the mean value could not be assigned. This may be because those columns have strings. So we will assign a value "Unassigned" to the dataset
data.fillna("Unassigned",inplace=True)


# In[ ]:


#after assigning the term, we shall check again whether we have attained a clean data set or not
data.isna().sum()


# <h2>Exploratory Data Analysis</h2>
# So far till now, we have checked the dataset and have made a clean dataset. Now let's begin with the interesting part of the analysis.

# In[ ]:


#as we started our analysis with the summary of the dataset. We will make a heatmap for the same.
plt.figure(figsize=(50,50))
p=sns.heatmap(data.corr(),annot=True)


# We are done with the summary. Now lets play with the data. We can find answers to some questions, like
# <h3> Which countries have the highest overall scores?</h3>

# In[ ]:


# Lets see the top 15 country-wise distribution of players
fif_countries = data['Nationality'].value_counts().head(15).index.values
fif_countries_data = data.loc[data['Nationality'].isin(fif_countries),:]


# In[ ]:


#we will make a simple visualization for the 15 countries data
#We will make a basic Bar Plot
sns.set(style="dark")
plt.figure(figsize=(25,10))
p=sns.barplot(x='Nationality',y='Overall',data=fif_countries_data)
p.set(xlabel='Country', ylabel='Total')


# A better visual for the above will be a boxplot. We will make the same.

# In[ ]:


#Box Plot
sns.set(style="ticks")
plt.figure(figsize=(25,10))
p=sns.boxplot(x='Nationality',y='Overall',data=fif_countries_data)
p.set(xlabel='Country', ylabel='Total')


# From the above two visuals, we can say that Argentina, Brazil and Spain have the highest overall scores..

# <h3>Potential of Players from top 10 countries</h3>

# In[ ]:


ten_countries = data['Nationality'].value_counts().head(10).index.values
ten_countries_data = data.loc[data['Nationality'].isin(ten_countries),:]
sns.set(style="ticks")
plt.figure(figsize=(15,10))
p=sns.boxplot(x='Nationality',y='Potential',data=ten_countries_data)


# Spain has the highest potential of players.

# In[ ]:




