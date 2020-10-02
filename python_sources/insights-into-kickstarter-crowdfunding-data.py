#!/usr/bin/env python
# coding: utf-8

# # **Insights into the kickstarter crowdfunding data**

# ## Motivation
# In the crowdfunding world, it is interesting to see why projects succeed and fail in collecting the funds. I have interest in exploring the croudfunding model in future and would like to learn more about the same. My interest in the Film & Video category. This data set provides a good start to explore and investigate if there are certain factors/trends that lead to a successful project. 

# ## Project Aim
# The primary scope of this project is to apply learnings from the course of Python for Datascience course on Edx from UCSD. The attempt is to get familar with working using Pandas dataframes, and plotting features of Matplotlib and Seaborn.  
# 
# ## Research Questions
# - What is the percentage of projects that are successful? 
# - How does success rate vary with:
#     - Length of the name
#     - Year/Month/Date of launch
#     - Duration
#     - Country 
#     - Categories
#     - Goal amount
# - How does the funding vary with the different factors for the Film & Video category? Do all the general trends hold?
# - What are the best performing sub categories?
# - Explore any correlation amongst the factors.

# ## Loading Modules and Environment Setup

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import os
from datetime import datetime
import seaborn as sns


# ## Reading Data

# In[ ]:


df = pd.read_csv("../input/ks-projects-201801.csv")
df.head()


# ## Dataset Information
# 
# 

# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# ## Dataset Preperation and Cleaning

# In[ ]:


# drop the nan rows 
df.dropna
df.shape


# In[ ]:


# drop the columns 'goal','pledged','usd pledged','launched','deadline','ID' 
df_cleaned = df.drop(['goal','pledged','usd pledged','launched','deadline','ID'], axis = 1)


# In[ ]:


#convert the time to date time format
df_cleaned['launched_date'] = pd.to_datetime(df['launched'], format='%Y-%m-%d %H:%M:%S')
df_cleaned['deadline_date'] = pd.to_datetime(df['deadline'], format='%Y-%m-%d %H:%M:%S')


# In[ ]:


# calculate the duration of the project in days. This is the difference between the deadline date and the launched date
df_cleaned['duration'] = df_cleaned['deadline_date'] - df_cleaned['launched_date']
df_cleaned['duration'] = df_cleaned['duration'].dt.days


# In[ ]:


df_cleaned[df_cleaned['state'] == 'successful']['state'].value_counts()


# ### Percentage of projects that are successful 

# In[ ]:


# successful projects / total projects * 100
success_percentage = df_cleaned[df_cleaned['state'] == 'successful']['state'].value_counts() / len(df_cleaned["state"])
print('Sucess Percent = {0:2.2%}' .format(success_percentage[0]))


# In[ ]:


# percentages of all different states
state_percentage = round(df_cleaned['state'].value_counts() / len(df_cleaned["state"]) * 100,1)
print(state_percentage)


# Removing the canceled, undefined, live, and suspended cases. Also saving the live cases in a seperate dataframe.

# In[ ]:


df_test = df_cleaned[(df_cleaned['state'] == 'successful') | (df_cleaned['state'] == 'failed')]
df_test_live = df_cleaned[df_cleaned['state'] == 'live']

print (df_test.shape)
print (df_test_live.shape)


# In[ ]:


#length of the name of the projects
df_test['length_name'] = df_test.loc[:,'name'].str.len()


# In[ ]:


# Launch year, month, date
df_test['launch_year'] = pd.DatetimeIndex(df_test.loc[:,'launched_date']).year
df_test['launch_month'] = pd.DatetimeIndex(df_test.loc[:,'launched_date']).month
df_test['launch_date'] = pd.DatetimeIndex(df_test.loc[:,'launched_date']).day
df_test= df_test.sort_values('launched_date',ascending=True)
df_test = df_test.drop(['launched_date','deadline_date'], axis = 1)


# In[ ]:


df_test.columns


# ## Initial Data Investigation

# In[ ]:


# Generate plots of projects per year, month, date, country, currency
# Generate plots to see the sucess percentages
# Observe any trends

plot_columns = ['launch_year','launch_month','launch_date','country','currency'] 
x_label = ['Launch Year','Launch Month','Launch Date', 'Country','Currency'] 
y_label = ['Number of Projects', 'Sucess Percentage(%)']
fig, axarr = plt.subplots(len(plot_columns), len(y_label), figsize=(20,5*len(plot_columns)))
for i in range(len(plot_columns)):
    a = None
    a = round(100 * df_test[df_test.state == "successful"][plot_columns[i]].value_counts()/df_test[plot_columns[i]].value_counts(), 1)
    # taking the column data/variable for example launch year and count the sucessful number of projects for each year and divide by the total projects in that year. 
    #print (a)
    sns.countplot(df_test[plot_columns[i]],ax=axarr[i][0])
    axarr[i][0].set_xlabel(x_label[i])
    axarr[i][0].set_ylabel(y_label[0])
    sns.barplot(x = a.index,y = a.values,ax=axarr[i][1])
    axarr[i][1].set_xlabel(x_label[i])
    axarr[i][1].set_ylabel(y_label[1])


# In[ ]:


for i in range(len(plot_columns)):
    a = None
    a = round(100 * df_test[df_test.state == "successful"][plot_columns[i]].value_counts()/df_test[plot_columns[i]].value_counts(), 1)
    # taking the column data/variable for example launch year and count the sucessful number of projects for each year and divide by the total projects in that year. 
    print ( a, '\n')


# In[ ]:


# percentage of the main categories in the data
round(100* df_test['main_category'].value_counts()/df_test['main_category'].value_counts().sum(),2)


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(20, 5))
sns.countplot(df_test['main_category'])
ax.set_xlabel ('Main Category')
ax.set_ylabel ('Number of Projects')


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(30, 5))
sns.barplot(df_test['duration'].value_counts().index,df_test['duration'].value_counts().values)
ax.set_xlabel ('Duration')
ax.set_ylabel ('Number of Projects')


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(20, 5))
df_test['length_name'].value_counts().plot(kind='bar', figsize=(20,5))
ax.set_xlabel ('Length of the Project Name')
ax.set_ylabel ('Number of Projects')


# ## Here is the summary from initial data investigation:
# - 2015 is the year with most completed projects in the dataset
# - Film & Video is the main category with most projects
# - Duration of 29 days has the most number of projects
# - 60 is the length of project name for maximum projects
# - Most of the projects in the data set are from US and the currency is USD. 
# 
# In the further analysis a subset will be created comprising of only projects from US and with currency of USD. This will also reduce the number of columns. 
# 

# ## Investigation - Data Subset/Data of Interest 
# - Country - US
# - Category - Video & Film
# 
# We will also drop the columns of - country, currency, category. Will analyze this in futures studies. 
# We will also check if all the previous findings will hold.

# In[ ]:


df_US = df_test[(df_test.country == "US") & df_test.main_category.str.contains('Film & Video')]
df_US['state_val'] = (df_US['state'] == 'successful') * 1
df_US = df_US.drop(['country','currency','main_category','state'], axis = 1)
print(df_US.shape)
df_US.category.value_counts().index[0:10]


# Create the dataframe with the top 10 of the categories in the Film & Video 

# In[ ]:


list_cat = list(df_US.category.value_counts().index[0:10]) # Top 10 of the categories in the Film & Video 
df_US_sub = None
for j in range(len(list_cat)):
    #print (j)
    if j == 0:
        #print (list_cat[j])
        df_US_sub = df_US[df_US.category.str.contains(list_cat[j])]
        #print (df_US_sub.shape)
    else:
        #print (list_cat[j])
        new_portion = df_US[df_US.category.str.contains(list_cat[j])]
        df_US_sub = pd.concat([df_US_sub,new_portion])
        #print (df_US_sub.shape)
        #print (new_portion.shape)
df_US_sub.shape


# In[ ]:


# Observing the number of projects in each of the categories
df_US_sub.category.value_counts()


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(15, 5))
sns.countplot(df_US_sub['category'])
ax.set_xlabel ('Category')
ax.set_ylabel ('Number of Projects')


# In[ ]:


# sucess percentage of the Film&Video
print ("Film & Video sucess percentage is : ", round(100 * df_US_sub[df_US_sub.state_val == 1].size/df_US_sub.size, 1))


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(15, 5))
a = None
a = round(100 * df_US_sub[df_US_sub.state_val == 1].category.value_counts()/df_US_sub.category.value_counts(), 1)
sns.barplot(x = a.index,y = a.values)
ax.set_xlabel('Category')
ax.set_ylabel('Success Percentage')


# In[ ]:


print ('Category.......Sucess Percentage')
print (a)


# In[ ]:


plot_columns = ['launch_year','launch_month','launch_date'] 
x_label = ['Launch Year','Launch Month','Launch Date'] 
y_label = ['Count', 'Sucess Percentage']
fig, axarr = plt.subplots(len(plot_columns), len(y_label), figsize=(20,5*len(plot_columns)))
for i in range(len(plot_columns)):
    a = None
    a = round(100 * df_US_sub[df_US_sub.state_val == 1][plot_columns[i]].value_counts()/df_US_sub[plot_columns[i]].value_counts(), 1)
    # taking the column data/variable for example launch year and count the sucessful number of projects for each year and divide by the total projects in that year. 
    #print (i)
    sns.countplot(df_US_sub[plot_columns[i]],ax=axarr[i][0])
    axarr[i][0].set_xlabel(x_label[i])
    axarr[i][0].set_ylabel(y_label[0])
    sns.barplot(x = a.index,y = a.values,ax=axarr[i][1])
    axarr[i][1].set_xlabel(x_label[i])
    axarr[i][1].set_ylabel(y_label[1])


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(25, 5))
sns.barplot(df_US_sub['duration'].value_counts().index,df_US_sub['duration'].value_counts().values)
ax.set_xlabel('Duration')
ax.set_ylabel('Number of Projects')


# In[ ]:


fig, ax = plt.subplots(1,1, figsize=(20, 5))
df_US_sub['length_name'].value_counts().plot(kind='bar', figsize=(20,5))
ax.set_xlabel('Length of the Project Name')
ax.set_ylabel('Number of Projects')


# ## Conclusion from the "Film & Video" Main_Category
# This portion only considers projects in US.
# 
# - Based on the intial data analysis, most of the projects in Kickstarter are from US. 
# - The highest number of projects are in the categories of [Documentary, Shorts, Film&Video]. Note that the Film&Video category is also sub category under the main_cateory of Film&Video.
# - The most sucessful category is Shorts followed by Narratives and Comedy.
# - The trends in sucess percentage per year, per month and per date shows a similar behavior as the overall data. It does not provide any clear trends.
# - The success percentage of the Film&Video Main_category is 42.8% while the overall sucess percentage was 35.8%.
# - From the duration data i.e., the difference between launch date and end date, 29 and 59 days seem the duration with most projects.
# 

# ## Future Work
# 
# Since this was a week long project work for the course, here are some ideas that I would like to explore in the future:
# - Create the variables corresponding to backers per dollar amount, ratio of pledged to the goal amount and observe the trends with the projects.
# - Normalize the data and generate correlations.
# - Create a predictive model and check its accuracy. 
# - Use the model on the live projects data. Since the live projects data is from 2018, try to find newer data from Kickstarter website and check model accuracy.
# 
# 
# This is my first submission and I welcome all comments! :)
# 

# #### Creating additional two columns:
# - backer_per_usd = backer per every USD of the goal amount 
# - pledge_to_goal = ratio of pledge amount to goal amount (less than 1 would be failed and >=1 would be sucessful)

# In[ ]:


df_US_sub['backer_per_usd'] = (df_US_sub.backers/df_US_sub.usd_goal_real)
print (df_US_sub.backer_per_usd.round(2).value_counts().head())
df_US_sub['pledge_to_goal'] = (df_US_sub.usd_pledged_real/df_US_sub.usd_goal_real)
print (df_US_sub.pledge_to_goal.round(2).value_counts().head())


# In[ ]:


backers = df_US_sub['backers'].unique()
launch_year = df_US_sub['launch_year'].unique()
launch_month = df_US_sub['launch_month'].unique()
launch_date = df_US_sub['launch_date'].unique()
state = df_US_sub['state_val'].unique()


# In[ ]:


df_US_sub.columns


# ## Check for correlation

# In[ ]:


corr = df_US_sub.corr(method = 'pearson')


# In[ ]:


# plot the heatmap
fig, ax = plt.subplots(1,1, figsize=(10, 10))
fig.suptitle('Correlation heat map', fontsize = 15)
sns.set(font_scale=1)  
sns.heatmap(corr, 
            cmap = 'coolwarm',
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            annot = True,
            fmt = '.2f',
            linewidths = 0.25,
            cbar_kws={"orientation": "vertical"})


# In[ ]:


df_group = df_US_sub.groupby(['launch_year'])


# In[ ]:


df_group.head()


# In[ ]:


df_group.state_val.value_counts().plot(kind='barh', figsize = (15,10))


# In[ ]:


a = df_test.groupby(['launch_year']).state.value_counts()


# In[ ]:


year_data=[]
year_sucess_percentage = []
for launch_year, state in df_test.groupby(['launch_year']).state.value_counts().groupby(level=0):
    #print(state[0])
    #print (state[1])        
    year_data.append(launch_year)
    year_sucess_percentage.append(round((state[0]/(state[0]+state[1]))*100))


# In[ ]:


df_year_percent = pd.DataFrame.from_dict({'Year':year_data, 'Sucess_percentage':year_sucess_percentage}).set_index('Year')
df_year_percent['Failed_percentage'] = 100 - df_year_percent['Sucess_percentage']
print ("Success percentages of projects over the years:")
print (df_year_percent)
df_year_percent.plot(kind='barh',stacked = True, legend = 'True', figsize=(15,10),title='Project percentages for each year')


# In[ ]:


cat_data=[]
cat_sucess_percentage = []
for category, state in df_test.groupby(['main_category']).state.value_counts().groupby(level=0):
    #print (state[0])
    #print (state[1]) 
    #print (category)
    cat_data.append(category)
    cat_sucess_percentage.append(round((state[0]/(state[0]+state[1]))*100))


# In[ ]:


df_cat_percent = pd.DataFrame.from_dict({'Category':cat_data, 'Sucess_percentage':cat_sucess_percentage}).set_index('Category')
df_cat_percent['Failed_percentage'] = 100 - df_cat_percent['Sucess_percentage']
print ("Success percentages of projects for the main categories:")
print (df_cat_percent)
df_cat_percent.plot(kind='barh',stacked = True, legend = 'best', figsize=(15,10),title='Sucess percentages for main categories')


# In[ ]:


country_data=[]
country_sucess_percentage = []
for country, state in df_test.groupby(['country']).state.value_counts().groupby(level=0):
    #print (state[0])
    #print (state[1]) 
    #print (category)
    country_data.append(country)
    country_sucess_percentage.append(round((state[0]/(state[0]+state[1]))*100))


# In[ ]:


df_country_percent = pd.DataFrame.from_dict({'Country':country_data, 'Sucess_percentage':country_sucess_percentage}).set_index('Country')
print ("Success percentages of projects for different countries:")
print (df_country_percent)
df_country_percent.plot(kind='barh',legend = None, figsize=(10,10),title='Sucess percentages for countries')

