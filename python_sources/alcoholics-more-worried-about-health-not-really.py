#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# The information in this analysis is gathered from the kaggle data "Young people survey",  a dataset that looks at different hobbies and interests of young people. Throughout this article I will analyze two particular variables, Alcohol and Health rating. Health rating in this dataset is rated from "1-5", "1" being least concerned about Health and "5" being highly concerned. Alcohol usage is categorized as "drink a lot", "social drinker" and "never". I will look at counts and proportions by grouping the two variables together and looking at the data from different perspectives. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math as mt 
get_ipython().magic(u'pylab inline')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

responses_data = pd.read_csv('../input/responses.csv') 
columns_data = pd.read_csv('../input/columns.csv')

# Any results you write to the current directory are saved as output.

## Load columns from csv file to dataframe to make pandas analysis simpler and get an idea of which 
## columns to analyze 
df = pd.DataFrame({'Alcohol':responses_data['Alcohol'], 
                   "Loneliness":responses_data['Loneliness'],
                   "Health":responses_data['Health'],"Age":responses_data['Age'] })
Alcohol = df['Alcohol']
Health = df['Health']
## Rough look at the dataframe 
## Count the different types of alcoholics there are 
print("Different types of Alcohol users and count \n")
group_alcohol = df.Alcohol.value_counts()
print(group_alcohol)

## See statistics in Alcohol vs Health (People that worry about their health on a scale of 1-5) 
grouped_data = df.groupby(['Alcohol','Health']).size()


# In[ ]:



## Table counting how many people there are in each health rating by alcohol category
print(df.pivot_table(index = ['Health'], columns='Alcohol', values='Loneliness', aggfunc=len)) 

## Proportion/percentage of people in each Health rating by Alcohol usage category, e.g. 
## there are 30 people in Health 1.0 and Alcohol "drink a lot", so that is 13% out of everyone 
## in the "drink a lot" column 
print(" ")
print("Percentage of each Health rating by Alcohol usage category \n")
health_alcohol_percent = df.pivot_table(index = ['Health'], columns='Alcohol', values='Loneliness', aggfunc=len).apply(lambda y: y/y.sum())
print(health_alcohol_percent.applymap(lambda x: "{:.0f}%".format(100*x))) 


# The following two bar plots are a visualization of the two tables above. The first bar plot shows how many different types of Alcohol users there are in each Health rating and the second shows the proportion of people in each Health rating by Alcohol usage. That is, the percentage of people in each rating grouped by Alcohol user. So for example, if you look at the second table where Health rating is "1.0" and Alcohol usage is "drink a lot" you will see 14%, which means 14% of everyone that "drinks a lot" rate their "concerns about health" at "1.0". In other words, the proportions are by column. 
# 
# We see in the first plot that the green bars (social drinkers) are dominant in each Health rating, followed by the blue bars(drink a lot) and orange bars(never). From the very first chart where we counted how many different people there are in each Alcohol usage category, we know "social drinker" and "drink a lot" have  the highest counts. So we next compute the proportions to see how many people in those categories rate their health. 
# 
# We see that the tallest bars lie at 3.0. That is, a Health rating of 3.0 has the highest proportion among all types of Alcohol users. This could be because regardless of whether people drink or not, everyone **is moderately** concerned about their health. 
# 
# 

# In[ ]:


## Bar plot for previous pivot table (count of people in each rating by alcohol usage category)
import matplotlib.gridspec as gridspec

fig1 = plt.figure(figsize=[15,8])
gs = GridSpec(100,100)
ax1 = fig1.add_subplot(gs[:70,0:40])
ax2 = fig1.add_subplot(gs[:70,60:100])

plot_1 = df.pivot_table(index = ['Health'], columns='Alcohol', values='Loneliness', aggfunc=len).plot(kind='bar',stacked=False,ax=ax1).set_ylabel("Count")
plot_2 = health_alcohol_percent.plot(kind='bar',grid=True,ax=ax2).set_ylabel("Proportion") 


# Next I made two columns or categories. I sorted everyone that described their drinking as "drink a lot" and "social drinker" into a column called "drinkers" and everyone that said "never" as "non-drinkers". I again computed the count by the different health ratings and the proportion of people in each Health rating by category. 
# 
# 
# 
# 

# In[ ]:


drinkers = np.array(Alcohol[(Alcohol == 'drink a lot') | (Alcohol == 'social drinker')].values)

df_1 = pd.DataFrame({'Drinkers':Alcohol.isin(drinkers),
                     "Non-drinkers":df['Alcohol'] == 'never',"Health":df['Health']})
df_1_count = df_1.groupby('Health').sum()
df_1_totals = df_1.groupby('Health').sum()
df_1_totals.loc['Total']= df_1_totals.sum()
print(df_1_totals)

print(" ")
print("Percent of people in each Health rating by Drinking category")
df_1_percent = df_1_count.apply(lambda y: y/y.sum())
print(df_1_percent.applymap(lambda x: "{:.0f}%".format(100*x)))


# In[ ]:


fig2 = plt.figure(figsize=[15,8])
gs = GridSpec(100,100)
ax3 = fig2.add_subplot(gs[:70,0:40])
ax4 = fig2.add_subplot(gs[:70,60:100])
print("Drinkers and Non-drinkers by Health Rating")
df_1_count.plot(kind='bar',ax=ax3).set_ylabel("Count") 
df_1_percent.plot(kind='bar',ax=ax4).set_ylabel("Proportion")


# First of all we see that there are many more drinkers than non-drinkers. There are a total of 880 Drinkers and 124 Non-drinkers. To see how many people from each category rate their Health I find the proportion of people in each category by Health rating. And once again we see that the tallest bars a.k.a the highest proportion of people in both categories (Drinkers and Non-drinkers) rate their concern about Health at 3.0. Why might this be? 
# 
# Well without going too much into detail, remember that Alcohol is only one part out of many other health factors such as diet, excercise and disease. There may be many other factors the people filling out this survey are considering or it might just be they already care too much or care too little about their health regardless of whether they drink or not.
# 
# For the next charts and plots, I will make three categories for the Health ratings. Everyone that rated their Health from "1-2" will be mapped as "Not very concerned", "3" will be "Moderate" and "4-5" will be "Very concerned". I will once again find the the counts and proportion. 
# 
# "1-2" => "Not-very-concerned", 
# "3" => "Moderate" , 
# "4-5" => "Very concerned" 
# 

# In[ ]:



def label_rating(row):
    if row['Health'] == 1 or row['Health'] == 2:
        return "Not-concerned"
    if row['Health'] == 3:
        return "Moderate"
    if row['Health'] == 4 or row['Health'] == 5:
        return "Very-concerned"

def label_alcohol(row):
    if row['Alcohol'] == 'never':
        return "Non-drinker"
    if row['Alcohol'] == 'social drinker' or row['Alcohol'] == 'drink a lot':
        return "Drinker"
    
df['health_label'] = df.apply (lambda row: label_rating (row),axis=1)
df['Alcohol_usage'] = df.apply (lambda row:label_alcohol (row), axis = 1)

df_2 = pd.DataFrame({'Drinkers':Alcohol.isin(drinkers),"Non-drinkers":df['Alcohol'] == 'never',"Health_labels":df['health_label']})

df_2_count = df_2.groupby('Health_labels').sum()
df_2_totals = df_2.groupby('Health_labels').sum()
df_2_totals.loc['Total']= df_2_totals.sum()
print(df_2_totals)

print(" ")
print("Percent of people in each Health rating by Drinking category")
df_2_percent = df_2_count.apply(lambda y: y/y.sum())
print(df_2_percent.applymap(lambda x: "{:.0f}%".format(100*x)))



# In[ ]:


fig3 = plt.figure(figsize=[15,8])
gs = GridSpec(100,100)
ax5 = fig3.add_subplot(gs[:70,0:40])
ax6 = fig3.add_subplot(gs[:70,60:100])
df_2_count.plot(kind='bar',ax=ax5).set_ylabel("Count") 
df_2_percent.plot(kind='bar',ax=ax6).set_ylabel("Proportion")


# Among all the non-drinkers the highest proportion of people are "Very-concerned" about their health, followed by "Moderate" and "Not-concerned". In the drinkers column, 41% are moderately concerned about their health while 39% are very concerned. And among all the drinkers the highest proportion of people are "Moderate" concerned about their health followed by "Very concerned" and "Not-concerned". In the non-drinkers column, 32% are moderately concerned about their health and 48% are very concerned. 
# 
# **Flipping the coin** 
# 
# So far we've analyzed the data by grouping it with Alcohol usage category. That is we grouped people by Alcohol usage and then counted the number of people in each Health rating. We also found the proportion of people in each Health rating by Alcohol category. So now lets flip the coin. 
# 
# Next we will group people by Health rating and count the number of people in each Alcohol category. And we will also find the proportion of people in each Alcohol usage category grouped by Health rating. In other words, we are switching the rows and columns to look at the data from a different perspective. Still confused? Just read on. 

# In[ ]:


## Table counting how many people there are in each health rating by alcohol category
print(df.pivot_table(index = ['Alcohol'], columns='Health', values='Loneliness', aggfunc=len)) 

## Proportion/percentage of people in each Health rating by Alcohol usage category, e.g. 
## there are 30 people in Health 1.0 and Alcohol "drink a lot", so that is 13% out of everyone 
##  in the drink a lot column 
print(" \n Percentage of people in each Alcohol usage category by Health Rating \n")
health_alcohol_percent_1 = df.pivot_table(index = ['Alcohol'], columns='Health', values='Loneliness', aggfunc=len).apply(lambda y: y/y.sum())
print(health_alcohol_percent_1.applymap(lambda x: "{:.0f}%".format(100*x)))


# In[ ]:


fig4 = plt.figure(figsize=[15,8])
gs = GridSpec(100,100)
ax7 = fig4.add_subplot(gs[:70,0:40])
ax8 = fig4.add_subplot(gs[:70,60:100])

df.pivot_table(index = ['Alcohol'], columns='Health', values='Loneliness', aggfunc=len).plot(kind='bar',ax=ax7).set_ylabel("Count")
health_alcohol_percent_1.plot(kind='bar',ax=ax8,title='Proportion of Alcohol users in each Health rating').set_ylabel("Proportion")
print(" ")


# We see that among people who rate their worriness at 1.0 are  41% "drink a lot" and 51% "social drinkers". Only 8% of people who rated their health concern at 1.0 is 8%. In addition, if we take a quick glance at the second plot above or the second chart we see that the "never" row has the smallest percent of people amongst all Health ratings and "social drinker" has the highest.   
# 
# Next I will again sort "drink a lot" and "social drinker" into a category called "Drinkers" and "never" into "Non-drinker". 

# In[ ]:


df_4 = pd.DataFrame({"Alcohol_usage":df['Alcohol_usage'], "Health":df['Health'],"Loneliness":df['Loneliness']})
## Table for Drinkers and non-drinkers 
df_4_count = df_4.pivot_table(index = ['Alcohol_usage'], columns='Health', values='Loneliness', aggfunc=len)
print(df_4_count)

print(" ")
## Shows percents of previous table 
print("Percent of people in each Alcohol usage by Health rating")
df_4_percent = df_4_count.apply(lambda y: y/y.sum())
print(df_4_percent.applymap(lambda x: "{:.0f}%".format(100*x)))

fig5 = plt.figure(figsize=[15,8])
gs = GridSpec(100,100)
ax9 = fig5.add_subplot(gs[:70,0:40])
ax10 = fig5.add_subplot(gs[:70,60:100])


df_4_count.plot(kind='bar',ax=ax9,title="Alcohol usage by Health Rating").set_ylabel("Count")
plt.ylabel("Proportion")
df_4_percent.plot(kind='bar',ax=ax10,title="Proportion of each Health rating").set_ylabel("Proportion")



# The most interesting thing to note here is the "Drinker" proportions are much bigger in every health rating than the non-drinker column. Out of everyone who rated their Health concerns at 1.0, 92% are drinkers while only 8% are non-drinker. Out of everyone who rated their Health concern  at 5.0, 82% are drinkers while 18% are non-drinkers. 
# 

# **So whats the conclusion?**
# 
# Regardless of whether people are heavy users, occasional users or not alcohol users at all everyone has concerns about their health. We looked at the data from two perspectives first how people among each Alcohol usage category rate their Health and second how people among each Health rating categorize their Alcohol usage. We saw that across all Health ratings most people are "Drinkers" and among different categories of drinking, people can be rating their concerns about their health  anywhere from 1-5. As mentioned earlier, there are perhaps other factors such diet, age, excercise and whether or not the person is facing any other disease. And so... are you an Alcohol user? Don't worry too much about it. 
