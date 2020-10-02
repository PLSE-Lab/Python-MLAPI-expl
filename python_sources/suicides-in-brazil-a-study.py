#!/usr/bin/env python
# coding: utf-8

# <h2> Suicides in Brazil: a more in-depth study on the number of suicides in Brazil </h2>

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Reading the dataset
suicides = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')
suicides.head()


# In[ ]:


# Selecting only data from Brazil and assigning it to a variable
suicides_brazil = suicides[suicides.country == 'Brazil'].sort_values(by='year', ascending=False)
suicides_brazil.head()


# <h2>Plotting a comparison chart</h2>

# In[ ]:


# Setting the size of the figure
plt.figure(figsize=(10, 6))

#Plotting the comparison lines between suicides in Brazil and in the world
sns.lineplot(x=suicides_brazil['year'], y=suicides_brazil['suicides/100k pop'], color='b', label='Brazil')
sns.lineplot(x=suicides['year'],  y=suicides['suicides/100k pop'], color='r', label='World')

plt.title('Comparison of suicides between Brazil and the world')

# Defining the limit of the figure to show the years from 1987 to 2015, since Brazil does not have data for 2016
plt.xlim(1987, 2015)

plt.legend();


# > 
# It is interesting to note that as of 1995, the suicide rate per 100k inhabitants decreased, while the same rate increased in Brazil

# <h2>Suicides between men and women</h2>

# In[ ]:


# Separating and adding the number of suicides by sex
sex_suicides_percent = suicides_brazil.groupby('sex')['suicides_no'].sum()

sex_suicides_percent


# In[ ]:


# Plotting a pie chart with the number of suicides by sex
colors_pie = ['red', 'cyan']
plt.pie(sex_suicides_percent, 
        labels=sex_suicides_percent.index,
        autopct='%.1f%%',
        shadow=True,
        colors=colors_pie,
        explode=[0.1, 0]);


# > There is much research to explain this overwhelming percentage of suicides among men

# <h2>Number of suicides by age</h2>

# In[ ]:


# Separating and adding the number of suicides by age
colors_barh = ['y', 'm', 'b', 'c', 'g', 'r']
age = suicides_brazil.groupby('age')['suicides_no'].sum().sort_values()
x = age.index
age


# In[ ]:


# Setting the size of the figure
plt.figure(figsize=(10, 8))

# Plotting a horizontal bar chart
plt.barh(x, age.values,
        color=colors_barh);


# > 35-54 is the age group with the most suicides. Something that worries me a lot is the high suicide rate among people so young (15-24 years old).

# <h2>Number of suicides by generation</h2>

# In[ ]:


# Separating the number of suicides by generation
colors_bar = ['y', 'm', 'b', 'c', 'g', 'r']
generation = suicides_brazil.groupby('generation')['suicides_no'].sum().sort_values()
generation


# In[ ]:


# Separating the number of suicides by generation
gen = generation.index

plt.figure(figsize=(10, 8))

# Plotting a graph showing the total number of suicides per generation
plt.bar(gen, generation.values,
       color=colors_bar);


# > The Boomers (1946 - 1964) and Generation X (1961 - 1981) have the highest number of suicides. It is interesting to note that the Millennial Generation (1981 - 1996) made up of young people has such a high number of suicides.
