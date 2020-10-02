#!/usr/bin/env python
# coding: utf-8

# # Evaluation of Olympic Games

# Hey Thanks for viewing my kernel.
# 
# Today, we will evaluate modern day of Olympic Games from 1896 to 2016.
# 
# **Content**
# 
# The file athlete_events.csv contains 271116 rows and 15 columns; Each row corresponds to an individual athlete competing in an individual Olympic event (athlete-events). The columns are the following:
# * ID - Unique number for each athlete
# * Name - Athlete's name
# * Sex - M or F
# * Age - Integer;
# * Height - In centimeters;
# * Weight - In kilograms;
# * Team - Team name;
# * NOC - National Olympic Committee 3-letter code;
# * Games - Year and season;
# * Year - Integer;
# * Season - Summer or Winter;
# * City - Host city;
# * Sport - Sport;
# * Event - Event;
# * Medal - Gold, Silver, Bronze, or NA.

# # Index
# * Importing the modules.
# * Data importing.
# * Collecting information about the two dataset.
# * Joining the dataframes.
# * Distribution of the age of gold medalists.
# * Women in Athletics.
# * Medals per country.
# * Disciplines with the greatest number of Gold Medals.
# * What is the median height/weight of an Olympic medalist?
# * Evolution of the Olympics over time.

# ##  Importing Modules

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))


# ## Data Importing

# In[ ]:


data = pd.read_csv('../input/athlete_events.csv')
regions = pd.read_csv('../input/noc_regions.csv')


# ## Collection Information about two DataFrame

# we are going to:
#  1. Review first line of data
#  2. Use describe and info functions to collect statistical information, data types, column names and other information

# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.info()


# In[ ]:


regions.head()


# ## Joining two DataFrames

# ##### We can now join two dataframes using as key the NOC column with the pandas Merge function [See Doumentation](http://https://pandas.pydata.org/pandas-docs/stable/merging.html)

# In[ ]:


combined_df = pd.merge(data, regions, on = 'NOC', how = 'left')


# In[ ]:


combined_df.head()


# ## Distribution of the age of Gold Medalists

# ##### Create New Dataframe with contains only Gold Medalists

# In[ ]:


gold_medalist = combined_df[combined_df['Medal'] == 'Gold']
gold_medalist.head()


# ##### Before make distribution plot we have to check is there any NaN values in Age Columns

# In[ ]:


gold_medalist.isnull().any()


# In[ ]:


gold_medalist.shape


# #####  Drop na based on Age Columns

# In[ ]:


gold_medalist = gold_medalist.dropna(subset = ['Age'])
gold_medalist.shape


# ##### Now we can create Counter plot 

# In[ ]:


plt.figure(figsize = (20,5))
plt.title('Distibution of Gold Medalist')
plt.tight_layout()
sns.countplot(x = 'Age', data = gold_medalist)

plt.show()


# ##### It seems some people hit gold metal at their age of 50+

# In[ ]:


gold_medalist['Age'][gold_medalist['Age']>50].count()


# ##### 65 people: Great! But which disciplines allows you to land a gold medal after your fifties?
# 
# We will now create a new dataframe called masterDisciplines in which we will insert this new set of people and then create a visualization with it.

# In[ ]:


disciplines = gold_medalist['Sport'][gold_medalist['Age']>50]


# In[ ]:


plt.figure(figsize = (10,5))
plt.tight_layout()
plt.title('Disciplines')
sns.countplot(disciplines)
plt.show()


# ## Women in Athletes

# ##### Studying the data we can try to understand how much medals we have only for women in the recent history of the Summer Games.

# dd

# In[ ]:


combined_df.head(2)


# In[ ]:


women_athletes = combined_df[(combined_df['Sex'] == 'F') & (combined_df['Season'] == 'Summer')]


# ##### Done. Let's review on our work

# In[ ]:


women_athletes.head()


# ##### Visualize Women Athletes per year.

# In[ ]:


plt.figure(figsize = (15,5))
plt.title('Women Athlets')
plt.tight_layout()
sns.countplot(x = 'Year', data = women_athletes)
plt.show()


# ## 

# ##### To review count of women athlets in 2016 Summer edition

# In[ ]:


women_athletes['Sex'][women_athletes['Year'] == 2016].count()


# ## Gold Medals per country

#  ##### Count Gole Medals per Country
#  

# In[ ]:


golds = combined_df[(combined_df['Medal'] == 'Gold')]


# In[ ]:


total_golds = golds['region'].value_counts().reset_index(name = 'Medal')
total_golds.head(10)


# ###### Let's Plot only top 10 Countries

# In[ ]:


top10_country = total_golds.head(10)
sns.catplot(x = 'index',y = 'Medal', data = top10_country,kind = 'bar', height = 8)
plt.title('Medals per Country')
plt.xlabel('Top10 Countries')
plt.ylabel('Number of Medals')
plt.show()


# ##### USA seems to be most winning country 

# ## Disciplines with the greatest number of Gold Medals

# ##### Let's Create a dataframe which contains gold medalist only for USA.

# In[ ]:


USA_Goldlist = combined_df[(combined_df['Medal'] == 'Gold') & (combined_df['region'] == 'USA')]

USA_Goldlist.head()


# In[ ]:


sports = USA_Goldlist['Event'].value_counts().reset_index(name = 'Medal')
sports.head(5)


# ##### Of course, BasketBall is the leading discipline!
# 
# Counting the medal of each member of the team instead of counting the medals per team.
# 
# Let's slice the dataframe using only the data of male athletes to better review it:

# In[ ]:


combined_df.head(2)


# In[ ]:


BasketBall_USA = combined_df[(combined_df['Sport'] == 'Basketball') & (combined_df['Sex'] == 'M') & 
                         (combined_df['region'] == 'USA')].sort_values(['Year'])
BasketBall_USA.head(5)


# ##   What is the median height/weight of an Olympic medalist?
# 

# ##### Let's try to plot a scatterplot of height vs weight to see the distribution of values.
# ##### First of all, we have to take gold_medalist dataframe

# In[ ]:


gold_medalist.head(5)


# ##### Let's get info about gold_medalist

# In[ ]:


gold_medalist.info()


# ##### Both Height and Weight Columns contain NaN. We can drop those NaN from our dataframe

# In[ ]:


NotNullMedals = gold_medalist[(gold_medalist['Height'].notnull()) & (gold_medalist['Weight'].notnull())]
NotNullMedals.count()


# ##### Here we have 10000 rows. let's create scatterplot

# In[ ]:


plt.figure(figsize = (10,5))
sns.scatterplot(x = 'Height', y = 'Weight', data = NotNullMedals)
plt.title('Height vs Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()


# ##### As per the plot, Relationship between Height and Weight is linear.
# ##### Let's look which is athlete that weighs more than 160Kg and their sports.

# In[ ]:


NotNullMedals[['Name','Sport']][NotNullMedals['Weight']>160]


# ##### Weightlifting make sense

# ## Evaluation of Olympics over time

# ###### Now we will try to answer the following questions
# 1. How the propostion of Men / Women varied with time?
# 2. How about the mean age  along time?
# 

# #### 1.How the propostion of Men / Women varied with time?

# In[ ]:


MaleAthletes = combined_df[['Year', 'Sex']][(combined_df['Sex'] == 'M') & (combined_df['Season'] == 'Summer')]
FemaleAthletes = combined_df[['Year','Sex']][(combined_df['Sex'] == 'F') & (combined_df['Season'] == 'Summer')]


# Let's create sSeperate DataFrame for male and female

# In[ ]:


v1 = MaleAthletes['Year'].value_counts().reset_index(name = 'Male_Count')
v2 = FemaleAthletes['Year'].value_counts().reset_index(name = 'Female_Count')


# Visualise varience berween male and women athletes

# In[ ]:


plt.figure(figsize = (10,5))
sns.lineplot(x = 'index', y = 'Male_Count', data  = v1)
sns.lineplot(x = 'index', y = 'Female_Count', data  = v2)
plt.title('Male vs Women Contribution')
plt.xlabel('Year')
plt.ylabel('Male vs Female count')
plt.show()


# ### Varience of Age along Time

# #### Let's create box plot 
#    MaleAthelets vs Age

# In[ ]:


plt.figure(figsize = (20,10))
plt.tight_layout()
sns.boxplot(x = 'Year', y = 'Age' ,data = combined_df[combined_df['Sex']== 'M'])
plt.show()


# Female vs Age 

# In[ ]:


plt.figure(figsize = (20,10))
plt.tight_layout()
sns.boxplot(x = 'Year', y = 'Age', data = combined_df[combined_df['Sex'] == 'F'])
plt.show()


# ## Conclusion

# #### First of all, thank you so much for reading! If you liked my work, please, do not forget to leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)
