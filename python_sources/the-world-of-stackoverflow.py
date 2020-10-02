#!/usr/bin/env python
# coding: utf-8

# ![**STACKOVERFLOW**](http://img.devrant.com/devrant/rant/r_384080_DhzML.gif)
# 
# Hello everyone!! Enjoying Kaggling? I bet you must be. Apart from the competitions,  there are certain datasets which I found very cool. Most of these are `dirty` datasets and I think, the more you engage yourself in such datasets, the better you become. After all, in the real world, no one is going to give you a cleaned dataset. Cleansing is an important process and apart from other pre-processing stuff, I think data cleansing eats 70% of the time. Playing with messy datasets can help you become better at this task.
# 
# StackOverflow survey 2018 is another such kind of dataset. Though I won't say that it's too messy but yeah it is enough to make you sweat to figure out important features for your ML model that you might want to put on some of the subsets of this survey. So, let's dive in!!
# 
# PS: This will be a long kernel and I will be updating it from time to time. So look out for new things whenever you visit this dataset page. Also, don't forget to **up vote** if you like it.

# In[51]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import re
import glob
import shutil
import altair as alt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(font_scale=2)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))
np.random.seed(111)
# Any results you write to the current directory are saved as output.


# In[52]:


# Read the CSV file
data = pd.read_csv('../input/survey_results_public.csv')
data.head()


# Woahh.. That's a lot of columns!!. We will analyze things column by column. I may not analyze the columns in the same order as in the CSV, rather I would look up for columns first that are more interesting than others(TBH, I would go for random :))

# In[53]:


# How many entries are there?
print("Total number of responses: ", data.shape[0])

# How many columns are there?
print("Number of columns in the dataset: ", data.shape[1])

# What are the column names?
print("Columns are: ")
print(list(data.columns))


# Before we start, let's define some handy-dandy functions. These functions will be used over the time at different places in the notebook.  Always remember **Repeating same code is a bad practice**. So, until unless you are too lazy, just keep things in functions/classes. 

# In[54]:


# A handy dandy function for making a bar plot. You can make it as flexible as much as you want!!
def do_barplot(df, figsize=(20,8), plt_title=None, xlabel=None, ylabel=None, title_fontsize=20, fontsize=16, orient='v', clr=None, max_counts=None):
    # Get the value counts 
    df_counts = df.value_counts()
    total = df.shape[0]
    
    # If there are too many values, limit the amount of information for display purpose
    if max_counts:
        df_counts = df_counts[:max_counts]
    
    # Print the values along with their counts and overall %age
    for i, idx in enumerate(df_counts.index):
        val = df_counts.values[i]
        percentage = (val/total)*100
        print("{:<20s}    {}  or roughly {:.2f}% ".format(idx, val, percentage))
    
    # Plot the results 
    plt.figure(figsize=figsize)
    if orient=='h':
        if clr:
            sns.barplot(y=df_counts.index, x=df_counts.values, orient='h', color=color[clr])
        else:
            sns.barplot(y=df_counts.index, x=df_counts.values, orient='h')
    else:
        if color:
            sns.barplot(x=df_counts.index, y=df_counts.values, orient='v', color=color[clr])
        else:
            sns.barplot(x=df_counts.index, y=df_counts.values, orient='v')
            
    plt.title(plt_title, fontsize=title_fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    
    if orient=='h':
        plt.yticks(range(len(df_counts.index)), df_counts.index)
    else:
        plt.xticks(range(len(df_counts.index)), df_counts.index)
    plt.show()
    del df_counts


# In[55]:


# A handy dandy function for countplot. (I may or may not use it very often but let's define it anyways)
def do_countplot(df, yval=None, xval=None, hueval=None, axs=[0,0], hue_ord=None):
    if df is None or (xval is None and yval is None):
        print("Either data or the axis values is missing")
        return
    if yval:
        sns.countplot(y=yval, data=df, hue=hueval, ax=axs,hue_order=hue_ord)
    else:
        sns.countplot(x=xval, data=df, hue=hueval, ax=axs,hue_order=hue_ord)


# For each column, we will get the view from the original dataframe and store it in a variable. We will not be changing anything in the original data.  For thr purpose of analysis, I would be dropping the null values as of now. It's upto you what you want to do with them.
# 
# ## Respondent
# This column contains a unique id corresponding to the respondents. We will leave this column as this is least informative
# 
# 
# ## Coding- Do you code as a hobby?

# In[56]:


# Get the hobby column 
hobby = data['Hobby'].dropna()

# Visualize the results. (You see, how handy our function is!!)
do_barplot(df=hobby, figsize=(10,8), 
           fontsize=16, title_fontsize=20, 
           xlabel='Hobby?', ylabel='Count', 
           plt_title="Coding as a hobby",
           orient='v', clr=5)
del hobby


# 80% of the developers say that coding is their hobby. If you are a serious developer, then if not initially but after a certain period of time, coding eventually becomes your hobby. Of course there will be exceptions but in general you start loving things you do regularly.

# ## OpenSource: Do you contribute to open source projects?

# In[57]:


# Get the corresponding column and drop the null values
opensource = data['OpenSource'].dropna()

#visualize
do_barplot(df=opensource, figsize=(10,8), 
           fontsize=16, title_fontsize=20, 
           xlabel="Type", ylabel='Count', 
           plt_title="Contribution to OpenSource",
           orient='v', clr=3)
del opensource


# Contributing to oprnsource is one of the most important things. **We should always support Open Source**. But there are situations when contributing to open source isn't possible. For example, some companies don't allow their employee to contribute, sometimes the developer doesn't know how to contribute or where to start from or sometimes people are just too busy. But no matter what, you should always try to contribute to open source. 

# ## Country: In which country do you currently reside?

# In[58]:


# Get the country column and do a value counts
country = data['Country'].dropna()
country_counts = country.value_counts()

# Get the countries with maximum and minimum number of developers
max_count = country_counts.max()
min_count = country_counts.min()

print("Total number of countries: ", len(country_counts))
print("Country with maximum number of developers: {}     #Developers: {}".format(country_counts.index[country_counts.values==max_count][0], max_count))
print("")
print("Country with least number of developers: {}     #Developers: {}".format(list(country_counts.index[country_counts.values==min_count]), min_count))
print("==========================================================================\n")

# As there are developers from 183 countries(woahh...), for the sake of plotting we will choose the top 50 countries
max_counts = 50

# visualize(check the max counts argument this time)
do_barplot(df=country, figsize=(30,30), 
           fontsize=16, title_fontsize=20, 
           xlabel="Count", ylabel="Country", 
           plt_title="Country where the developers reside",
           orient='h', max_counts=max_counts)

del country_counts
del country    


# Before we move on to some other things, let's explore the ** $100,000** worth questions. These questions are all about **AI**. So, they are more important than others for sure.
# 
# ## Is AI Dangerous?

# In[59]:


aidanger = data['AIDangerous'].dropna()
do_barplot(df=aidanger, figsize=(20,10), 
           fontsize=16, title_fontsize=20, 
           xlabel="Count", ylabel="Reasoning", 
           plt_title="What do developers fear about AI?",
           orient='h')

del aidanger


# The best part is that people embrace automation. Again, not all but as per our survey, automations is least of the worries. For the singularity thing, I can only say one thing: Our algorithms make so naive mistakes that sometimes ROFL isn't enough. Also, singularity is a big thing. Many people just read some shitty stuff over the internet and start taking that information as granted. I don't see singularity happening at least for the next 2-4 decades.

# ## Is AI interesting and how?

# In[60]:


# Get the column data
aiinterest = data['AIInteresting'].dropna()

# Visualize
do_barplot(df=aiinterest, figsize=(20,10), 
           fontsize=16, title_fontsize=20, 
           xlabel="Count", ylabel="Reasoning", 
           plt_title="What's interesting about AI?",
           orient='h')

del aiinterest


# ## Who should be responsible for regulating AI things?

# In[61]:


# Same thing..get the column and just use our handy dandy function. Life is easy!!
airesp = data['AIResponsible'].dropna()
do_barplot(df=airesp, figsize=(20,10), 
           fontsize=16, title_fontsize=20, 
           xlabel="Count", ylabel="Who?", 
           plt_title="Who should bear the burden of responsibilites in AI?",
           orient='h', clr=5)
del airesp


# 
# This result is truly amazing. I totally agree with the fact that it is our responsibility first to make build ethical things and for doing good for the society. But it is also true that not everyone wants to follow this path. For example, **DeepFakes** was totally an unacceptable thing as people started using public figure faces in porn without their consent. That is purely evil and unethical. Hence we also need a regulatory body that is above the industries. Whether that should be government itself or an independent body, that is a different thing to discuss.

# ## What developers think about future of AI?

# In[62]:


# Get the column
aifuture = data['AIFuture'].dropna()

#visualize 
do_barplot(df=aifuture,figsize=(20,8), 
           fontsize=16, title_fontsize=20, 
           xlabel='Count', ylabel='What?', 
           plt_title="Opininon about future of AI?",
           orient='h', clr=1)
del aifuture


# Such a good thing!! Developers across the world are much more exxcited about the new possibilities with AI rather than the dangers. A tech person always knows what is more important. **It's the business people who are fearmongers!!**

# 
# ## Gender ratio among developers in top four countries
# 
# We all are aware of gender imbalanced ratio in developers all across the world. Good thing is that this situation is improving now, bad things is that the rate is still very slow.  We wil look at the employment part among the developers, taking into accoun the gender ratio within, for the top foru countries which contains maximum number of developers 

# In[63]:


# A handy dandy function for returning a grouby object 
def return_grouped_data(df, group_by =None, group=None):
    if group_by is None or group is None:
        print("ValueError: You mist provide the groupby and group name")
        return
    
    grouped_data = df.groupby(group_by).get_group(group).reset_index(drop=True)
    return grouped_data


# In[64]:


# Get the revelvant columns and drop null values
country_gender = data[['Country', 'Gender', 'Employment']].dropna()

# Do some cleaning on the gender column. People fill multiple values for this columns. I never get the logic  behind that. It's a survey
# You fill it up the wrong way, thigs are never gonna improve then.
country_gender['Gender'] = country_gender['Gender'].apply(lambda x: x if x in ['Male', 'Female'] else 'Other')

# Get the groupby object for the top foir countries
country_US = return_grouped_data(df=country_gender,group='United States', group_by='Country')
country_India = return_grouped_data(df=country_gender,group='India', group_by='Country')
country_Germany = return_grouped_data(df=country_gender,group='Germany', group_by='Country')
country_UK = return_grouped_data(df=country_gender,group='United Kingdom', group_by='Country')

# Plot the results
f, axs = plt.subplots(2,2, figsize=(35,30), sharey=True, sharex=True)
sns.countplot(y=country_US['Employment'], data=country_US, hue='Gender', ax=axs[0,0], hue_order=['Male', 'Female', 'Other'])
axs[0,0].set_title('US', fontsize=20)

sns.countplot(y=country_India['Employment'], data=country_India, hue='Gender', ax=axs[0,1], hue_order=['Male', 'Female', 'Other'])
axs[0,1].set_title('India', fontsize=20)

sns.countplot(y=country_Germany['Employment'], data=country_Germany, hue='Gender', ax=axs[1,0], hue_order=['Male', 'Female', 'Other'])
axs[1,0].set_title('Germany', fontsize=20)

sns.countplot(y=country_UK['Employment'], data=country_UK, hue='Gender', ax=axs[1,1], hue_order=['Male', 'Female', 'Other'])
axs[1,1].set_title('UK', fontsize=20)

plt.show()

del country_gender, country_US, country_Germany, country_India, country_UK


# If we just focus on the full time employed developers in all the four countries, we can see how horrible the siituations is. **Where are the female developers?**  If you are reading this kernel, and if you know any female  in your contacts who  is interested in Data Science, AI/ML/DL, or pure coding, just encourage them more and more. **This situation needs to be improved if we really want our algorithms to be unbiased**

# ## What type of developer you are?
# 
# This is a multiplt choice question. The values are separated by a semi-colon. Hence we require a little bit of preprocessing first.

# In[65]:


# A handy-dandy function to process the values for multiple choice questions
def split_values(x, samples_dict):
    # Split values based on semi-colon
    items = re.split(r';', x)
    for item in items:
        samples_dict[item] +=1


# In[66]:


dev = data['DevType'].dropna()

# Create a new dictionary
samples_dict = defaultdict(int)

# Apply the fucntion to each row of the series
dev = dev.apply(split_values, args=(samples_dict,))

# Sort the dictionay based on its values
samples_dict = dict(sorted(samples_dict.items(), key=lambda x: x[1], reverse=True))

devtype = list(samples_dict.keys())
count = list(samples_dict.values())

# Fancy way of showing percentage in pie chart
#Courtesy: StackOverflow
def show_autopct(values):
    def my_autopct(pct):
        total = len(dev)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%'.format(p=pct)
    return my_autopct


plt.figure(figsize=(40,30))
patches, text, autotext = plt.pie(count, labels=devtype, autopct=show_autopct(count))
plt.title("Type of developers", fontsize=20)
plt.show()

del dev, samples_dict


# **Only 2.5% of the developers are Data Scientits/ML engineers**. After all ML isn't that easy. You can't be an ML engineer just because you know how to **fit** a model in scikit-learn or Keras. Majority of the developes are either Back-end developers or Full stack developers followed by Front-end developers and Mobile developers.
# 
# 
# ## Since how many days/months/years are you coding?
# 
# I will do something very carzy here. Let's try to build a stacked plot withot using it out of the box.

# In[67]:


# Get the data
yearsOfCoding = data[['YearsCodingProf', 'YearsCoding']].dropna()

# Remove everything after numeric value
yearsOfCoding = yearsOfCoding.applymap(lambda x: x.split(' ')[0])

# Labels we are going to use for our stacked plot
xlabels = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30']

# Count of years coding and coding professionally
years_count = yearsOfCoding['YearsCoding'].value_counts()
year_count = dict(zip(list(years_count.index), list(years_count.values)))

years_count_prof = yearsOfCoding['YearsCodingProf'].value_counts()
years_count_prof = dict(zip(list(years_count_prof.index), list(years_count_prof.values)))

# List to store values corresponding to each type
y1, y2=[], []

# Get the count for each xlabel
for item in xlabels:
    y1.append(years_count[item])
    y2.append(years_count_prof[item])
    
f, axs = plt.subplots(1,1, figsize=(30,10))
# Create a twin yaxis sharing the same x-axis
ax2 = axs.twinx()
sns.barplot(x=xlabels, y=y1, color=color[1], ax=axs)
sns.barplot(x=xlabels, y=y2, color=color[2], ax=ax2)
axs.set_ylabel('YearsCoding', fontsize=20)
ax2.set_ylabel('YearsCodingProf', fontsize=20)
plt.xlabel('Years')
plt.title("Coding Experience in years: Normal and Professional", fontsize=20)
plt.show()


# So majority of our developers coding experience lies in the range 3-8 years. I am very glad to see the numbers on the rightmost side. The number of developers coding professionally for more than 30 years is amazing. Like you can see how much coding matters in the world!!

# ## Job Satisafction? 
# Again, a million dollar question...haha!!

# In[68]:


job_sat = data['JobSatisfaction'].dropna()

do_barplot(df=job_sat, figsize=(12,8), 
          fontsize=16, title_fontsize=20, 
          xlabel='Count', ylabel='Satisafction level', 
          plt_title="Job Satisfaction check",
           orient='h', clr=2)
del job_sat


# ## JobSearchStatus- How many are actively looking for a job?

# In[69]:


job_stat = data['JobSearchStatus'].dropna()

do_barplot(df=job_stat, figsize=(12,8), 
          fontsize=16, title_fontsize=20, 
          xlabel='Count', ylabel='Status', 
          plt_title="Job Search Status",
           orient='h', clr=5)
del job_stat


# ## FormalEducation
# Does school/college really matters? if yes, to what extent? Let's find out

# In[71]:


education = data['FormalEducation'].dropna()

do_barplot(df=education, figsize=(12,8), 
          fontsize=16, title_fontsize=20, 
          ylabel='Count', xlabel='Education', 
          plt_title="Highest level of formal eduaction",
           orient='h', clr=5)
del education


# **Bachelors** run the world!! Period.

# ## Salary
# We will be looking at the converted salary represnted in USD and that too yearly

# In[ ]:




