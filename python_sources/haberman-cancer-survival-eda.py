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


# ## Title: Haberman's Survival Data
# 
# #### Sources: 
# (a) Donor: Tjen-Sien Lim (limt@stat.wisc.edu) 
# (b) Date: March 4, 1999
# 
# #### Number of Instances: 306
# 
# #### Number of Attributes:
# 4 (including the class attribute)
# 
# #### Attribute Information:
# 
# 1. Age of patient at time of operation (numerical)
# 2. Patient's year of operation (year - 1900, numerical)
# 3. Number of positive axillary nodes detected (numerical)
# 4. Survival status (class attribute) 1 = the patient survived 5 years or longer 2 = the patient died within 5 year
# 
# #### Missing Attribute Values: None

# In[ ]:


df = pd.read_csv('/kaggle/input/habermans-survival-data-set/haberman.csv')
df.head()


# Columns names should be renamed so it gets easier to understand

# In[ ]:


df = df.rename(columns={'30': 'Age','64': 'year','1':'p_axl_nodes','1.1':'Survival_status'})
df.head()


# #### Importing the necessary libraries for visulizing and plotting

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


print('Value counts of survival status')
df['Survival_status'].value_counts()


# Most of the patients survived within 5 years of the operation/treatment.

# In[ ]:


df.groupby('Survival_status')['p_axl_nodes','Age'].mean()


# In[ ]:


sns.FacetGrid(df, hue = 'Survival_status', height = 6)     .map(sns.distplot, 'p_axl_nodes')     .add_legend()


# 1. It is clear that about 90% of the patients has less than 25 positive axilliarynodes.
# 2. Patients with nodes between 0-3 have higher chance of survival.
# 3. Clear conclusion can not be drawn as the data is overlapping

# In[ ]:


g = sns.FacetGrid(df, hue='Survival_status',height = 6)
g = g.map(sns.distplot, 'Age').add_legend()


# 1. The major area of  PDF is overlapping, hence proper conclusion cannot be detected
# 2. Patients below 40 years have higher chance of survival while patients above 70 years don't.

# In[ ]:


sns.jointplot("Age","p_axl_nodes", data=df,
                  kind="reg", truncate=True,
                  xlim=(25,85), ylim=(0, 60),
                  color="b", height=7)

plt.show()


# 1. Irrespective of age,around 90% of positive nodes detected are between 0 - 10 in numbers.
# 2. No relation can be drawn.

# In[ ]:


sns.FacetGrid(df, hue='Survival_status', height=6)     .map(plt.scatter, 'p_axl_nodes', 'Age')     .add_legend()
plt.show()


# 1. Irrespective of the age with respect to number of positive axillary nodes, survival status varies .

# In[ ]:


sns.FacetGrid(df, hue = 'Survival_status', height =5)     .map(sns.distplot, 'year')     .add_legend()
plt.show()
    


# 1. Most of the operations are done between 1958 and 1968.
# 2. Highly overlaped data no clear conclusions can be made.

# In[ ]:


new = df.groupby(['year'])['Survival_status'].sum()
new.head()


# In[ ]:


dummy = pd.get_dummies(df['Survival_status'])
dummy = dummy.rename(columns = {1:'Survived_5yrs',2:'Died_within_5yrs'})


# In[ ]:


plot = pd.concat([df,dummy],axis =1)
plot.head()


# In[ ]:


plot_year = plot.groupby(['year'])['Survival_status','Survived_5yrs','Died_within_5yrs'].sum()
plot_year = plot_year.reset_index()
plot_year


# In[ ]:


plot_year2 = plot.groupby(['year'])['Age'].count()
plot_year2 = plot_year2.reset_index()
plot_year2


# In[ ]:


sns.barplot(data=plot_year2, x='year', y='Age',color='b').set(ylabel='patients')


# Maximum numbe of cases were in year 1959

# In[ ]:


plt.plot(plot_year['year'],plot_year['Survival_status'], label = 'Total cases')
plt.plot(plot_year['year'],plot_year['Survived_5yrs'], label = 'Survived 5 years')
plt.plot(plot_year['year'],plot_year['Died_within_5yrs'], label = 'Died within 5 years')
plt.rcParams["figure.figsize"] = [16,6]
plt.legend()
plt.xlabel('Year 1900')
plt.ylabel('Survival')

plt.show()


# Total cases with total survived and total died over the years.
# 1. Fron the year 1964 to 1966, mortality of people rised to its peak.

# In[ ]:


age_plot = plot.groupby('Age')['Survival_status','Survived_5yrs','Died_within_5yrs'].sum()
age_plot = age_plot.reset_index()
age_plot.head()


# In[ ]:


plt.subplot(121)
plt.plot(age_plot['Age'],age_plot['Survived_5yrs']/sum(age_plot['Survived_5yrs']), label = 'Survived 5 years')
y2 = np.cumsum(age_plot['Survived_5yrs'])/sum(age_plot['Survived_5yrs'])
plt.plot(age_plot['Age'],y2,label = 'cdf Survived')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Survival')

plt.subplot(122)
plt.plot(age_plot['Age'],age_plot['Died_within_5yrs'], label = 'Died within 5 years')
y = np.cumsum(age_plot['Died_within_5yrs'])
plt.plot(age_plot['Age'],y,label = 'cdf Died')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Survival')
#plt.rcParams["figure.figsize"] = [10,6]


plt.show()


# In[ ]:


Survived= df.loc[df["Survival_status"]== 1]
Died = df.loc[df["Survival_status"]==2]


plt.figure(figsize=(20,5))
i=1
for state in (list(df.columns)[:-1]):
#survived
    plt.subplot(1,3,i)
    Counts , bin_edges = np.histogram(Survived[state],bins=20,density=True)
    pdf=Counts/sum(Counts)
    cdf = np.cumsum(Counts)
    plt.plot(bin_edges[1:],cdf,label="cdf of survived",color="red")
    plt.plot(bin_edges[1:],pdf,label="pdf of survived",color="black")

#Death
    Counts , bin_edges = np.histogram(Died[state],bins=20,density=True)
    pdf=Counts/sum(Counts)
    cdf = np.cumsum(Counts)
    plt.plot(bin_edges[1:],cdf,label="cdf of Death")
    plt.plot(bin_edges[1:],pdf,label="pdf of Death")
    plt.xlabel(state)
    plt.grid()
    plt.legend()
    i+=1
plt.show()


# 1. People less than the age of 36 have definitely survived and peopole above 75 years definitely did not survived.
# 2. No clear conclusion as majority of the data is over lapping and evenly distributed. Except in the year 1965 chances of surviving was    very low, while higher survival rate in the year between 1961 to 1965.
# 3. People with the positive axillary node between 0 to 5 had high survival rate.

# In[ ]:


sns.pairplot(df, hue = 'Survival_status', height = 4, vars= ['Age','p_axl_nodes','year'])
plt.show()


# # Observation
# The majority of the data is overlapping and it will be hard to develop a seperating line between the features to determine the survival status. 
# More complex model will be required or maybe different features.
