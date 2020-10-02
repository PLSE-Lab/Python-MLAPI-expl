#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:


# Read data from gun-violence-data_01-2013_03-2018.csv file
data = pd.read_csv('../input/gun-violence-data_01-2013_03-2018.csv')
data


# In[ ]:


data.columns


# In[ ]:


# Data type and summary
data.info()


# In[ ]:


data.describe()


# In[ ]:


# How many rows?
data.shape[0]


# In[ ]:


# How many columns?
data.shape[1]


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


# Values in state column
data['state'].value_counts()


# In[ ]:


data['state'].value_counts().nlargest(10)


# In[ ]:


data['n_killed'].value_counts().unique()


# In[ ]:


# Converting date calumn format to datetime format
data['date'] = pd.to_datetime(data['date'])


# In[ ]:


# Adding new columns year and month
data['year'] = data['date'].dt.year


# In[ ]:


data['month'] = data['date'].dt.month


# In[ ]:


# Verifying year and month columns
data.head()


# In[ ]:


data['gun_type'].value_counts()


# In[ ]:


sc = data['state'].value_counts()
sc
#sns.barplot(data=sc)
#sns.barplot(x='state', y='aa', data = sc)

#sns.distplot(x)


# In[ ]:


st = data['state'].unique()
st
#sns.barplot(data=sc)
#sns.barplot(x='state', y='aa', data = sc)
#sns.distplot(st)


# In[ ]:


cnt = data['state'].value_counts().unique()
cnt


# In[ ]:


sns.barplot(x='state', y = 'n_killed', data = data)


# In[ ]:


sns.distplot(cnt)


# In[ ]:


sns.jointplot(x='n_injured', y = 'n_killed', data = data, size=10)


# In[ ]:


#Ploting graph for year wise incedence
sns.countplot(x='year', data=data)


# In[ ]:


#Ploting graph for state wise incedence
sns.countplot(x='state', data=data)


# In[ ]:


#Violin plot for year wise killed
sns.violinplot('year', 'n_killed',data=data)


# In[ ]:


sns.distplot(data['year'], kde=False)


# In[ ]:


plt.plot(data.year)


# In[ ]:


datagrid = ['state', 'n_killed', 'n_injured', 'year']
g = sns.PairGrid(data, vars=datagrid, hue='state')
g.map_offdiag(plt.scatter)
g.add_legend()


# In[ ]:


# creating a dataframe for pie graph
state_count = data['state'].value_counts()
state_cases = pd.DataFrame({'labels':state_count.index, 'values':state_count.values})


# In[ ]:


state_cases.iloc[:20,]


# In[ ]:


plt.pie(state_cases['values'], labels = state_cases['labels'], autopct = '%1.1f%%')


# In[ ]:


# Find data for Chicago
ch = data.loc[data.city_or_county =='Chicago']
ch.head(5)


# In[ ]:


#Box Plot for n_killed or n_injured for chicago
sns.boxplot('n_killed', 'n_injured', data=ch)


# In[ ]:


#Scatter plot month wise n_killed on each year on Chicago
g = sns.FacetGrid(ch, col = "year")
g.map(plt.scatter, "month", "n_killed", alpha = .7)
g.set(xlim=(0,12))


# In[ ]:


#bar plot month wise n_killed on each year on Chicago
g = sns.FacetGrid(ch, col = "year")
g.map(sns.barplot, "month", "n_killed", alpha = 1)
g.set(xlim=(0,12))
g.set(ylim=(0,2))


# In[ ]:


g = sns.FacetGrid(ch, col = "year")
g.map(sns.barplot, "month", "n_injured")
g.set(xlim=(0,12))
#g.set(ylim=(0,2))


# In[ ]:


# density plot: killed and injured on Chicago
sns.kdeplot(ch.n_killed[ch.year ==2013], label = 'killed', shade = True)


# In[ ]:


sns.kdeplot(ch.n_injured[ch.year ==2013], label = 'injured', shade = True)


# In[ ]:


# Group by n_killed and n_injured for state
data_st = data.groupby('state').aggregate({'n_killed':np.sum, 'n_injured':np.sum}).reset_index()


# In[ ]:


data_st


# In[ ]:


# Plot graph for no of killed vs no of injured State wise.
plt.plot(data_st.state, data_st.n_killed, 'r')


# In[ ]:


plt.plot(data_st.state.head(7), data_st.n_killed.head(7), 'r' )


# In[ ]:


plt.plot(data_st.state.head(7), data_st.n_injured.head(7), 'b' )
plt.xlabel('State Name >>')
plt.suptitle('state wise case details')


# In[ ]:


sns.jointplot("n_killed","n_injured", data_st, )


# In[ ]:




