#!/usr/bin/env python
# coding: utf-8

# # Mental Health Survey in the Tech Industry
# 
# <p align="center">
#   <img src="https://qz.com/wp-content/uploads/2017/07/david-mao-7091-e1499867773401.jpg?quality=80&strip=all&w=1600"/>
# </p>
# 
# Mental health is a level of psychological well-being or an absence of mental illness. It is the "psychological state of someone who is functioning at a satisfactory level of emotional and behavioural adjustment"
# 
# In this **kernel**, I hope to explore some relevant stats extracted from the **Mental Health Survey - Tech Industry, 2014"**. And since the data was found to be extremely messy & largely categorical in nature so therefore **data cleaning** was an important part of this analysis.
# 
# 
# #### Necessary Library Imports & Data Loading

# In[ ]:


import os
import gc
import warnings
import re
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from wordcloud import WordCloud
#matplotlib.rc['font.size'] = 9.0
matplotlib.rc('font', size=20)
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=20)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('legend', fontsize=20)
matplotlib.rc('figure', titlesize=20)
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/survey_2014.csv')
data.shape


# #### Let us take a brief look at the data to get an idea - 

# In[ ]:


data.sample(10)


# In[ ]:


data.nunique()


# **Gender** oddly has 49 unique values & therefore might require some cleaning!

# In[ ]:


gender_clean = {
    "female":"Female",
    "male":"Male",
    "Male":"Male",
    "male-ish":"Male",
    "maile":"Male",
    "trans-female":"Female",
    "cis female":"Female",
    "f":"Female",
    "m":"Male",
    "M":"Male",
    "something kinda male?":"Male",
    "cis male":"Male",
    "woman":"Female",
    "mal":"Male",
    "male (cis)":"Male",
    "queer/she/they":"Female",
    "non-binary":"Unspecified",
    "femake":"Female",
    "make":"Male",
    "nah":"Unspecified",
    "all":"Unspecified",
    "enby":"Unspecified",
    "fluid":"Unspecified",
    "genderqueer":"Unspecified",
    "androgyne":"Unspecified",
    "agender":"Unspecified",
    "cis-female/femme":"Female",
    "guy (-ish) ^_^":"Male",
    "male leaning androgynous":"Male",
    "man":"Male",
    "male ":"Male",
    "trans woman":"Female",
    "msle":"Male",
    "neuter":"Unspecified",
    "female (trans)":"Female",
    "queer":"Unspecified",
    "female (cis)":"Female",
    "mail":"Male",
    "a little about you":"Unspecified",
    "malr":"Male",
    "p":"Unspecified",
    "femail":"Female",
    "cis man":"Male",
    "ostensibly male, unsure what that really means":"Male",
    "female ":"Female",
    "Female":"Female",
    "Male-ish":"Male"
}

data.Gender = data.Gender.str.lower()
data.Gender = data.Gender.apply(lambda x: gender_clean[x])


# ## Some basic employment statistics
# 
# - Distribution on the basis of **Gender**. (Need more women in tech...surprised,eh?)
# - How does the age vary in the professional industry?

# In[ ]:


f, ax = plt.subplots(1,2, figsize=(15,7))
ax1 = ax[0].pie(list(data['Gender'].value_counts()), 
                   labels=['Male','Female','Unspecified'],
                  autopct='%1.1f%%', shadow=True, startangle=90,
             colors=['#66b3ff','#ff9999','#99ff99'])
ax[0].set_title("Gender Distribution")
ax[1].set_title("Distribution of Ages")
ax2 = sns.distplot(data.Age.clip(15,70), ax=ax[1])


# I thought it would also be interesting to extract other basic stats about **Age** here like *mean, standard dev, quartile values* etc. The average age of an IT employee stands at only **32**, quite surprising!

# In[ ]:


#Extraction of basic stats from all numeric columns
pd.DataFrame(data.Age.clip(15,60).describe())


# ## Participation in the Survery - by Country
# 
# Although, the United States dominates this category it would've been great if developing nations such as **India**, **Russia** & **Israel** had more participants since little is known about the working conditions in these countries & health issues that working professionals from these countries face.
# 
# Another thing, due to this extreme domination of the US in this survey, it has kind-of rendered it useless to do a country-wise analysis since there are *<50* participants from a majority of the countries.

# In[ ]:


sns.set_style("darkgrid")
plt.figure(figsize=(15,20))
sns.countplot(y='Country', data=data, 
              orient='h', order=data.Country.value_counts().index)
plt.show()


# ## How big/small is your company?
# 
# About 75% of the employees belong to the companies with less than 500 employees deeming them as very small ventures. Quite typical of the tech-industry.

# In[ ]:


f, ax = plt.subplots(1,2, figsize=(15,10))
patches, texts, autotexts = ax[0].pie(list(data['no_employees'].value_counts()), 
                   labels=['6-25', '26-100', '>1000', '100-500', '1-5', '500-1000'],
                  autopct='%1.1f%%', shadow=True, startangle=90)
new = ax[1].pie(list(data['remote_work'].value_counts()),
                                     labels=['Non-Remote', 'Remote Work'],
                                     autopct='%1.1f%%', shadow=True, startangle=0,
                                        colors=['#66b3ff','#ff9999'])


# ## How easy/difficult is it to take a leave?
# 
# This one caught me by surprise. Nearly *40%* of the total respondents are unsure about their company's policies on taking leaves. The trend is consistent quite consistent in Tech-NonTech companies, and with both Males & Females.

# In[ ]:


sns.set_style("whitegrid")
plt.figure(figsize=(15,7))
sns.countplot(x='leave', data=data, order=data.leave.value_counts().index, 
              hue='tech_company', color='r')
plt.xlabel("How East/Difficult is it to take a leave?")
plt.ylabel("# Reponses")


# # Let's explore some Company Policies & their correlation
# 
# I'm taking the following columnar attributes as an indicator of the company policies towards their employees (all of them take binary values i.e. Yes/No) --
# 
# 1. Treatment 
# 2. Benefits 
# 3. Care Options
# 4. Wellness Program (whether it exists or not)
# 
# Since all of these are categorical in nature, in order to calculate their correlation, we must **factorize** them to convert them to numeric. `pd.factorize` of Pandas comes in handy to this cause.

# In[ ]:


company_characs = [
    "treatment",
    "benefits",
    "care_options",
    "wellness_program",
]


# In[ ]:


sns.set_style("darkgrid")
company_chars_corr = data[company_characs].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', 
                                                                            min_periods=1)
plt.figure(figsize = (8, 6))

# Heatmap of correlations
sns.heatmap(company_chars_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title("Correlation Heatmap of Company Policies \ntowards employee's wellness");


# # Well Being Indicators
# 
# There are also certain binary attributes in this dataset that help describe the state of well being of an individual, namely -- 
# 
# 1. Seeking Help 
# 2. Mental Health Consequences (due to job)
# 3. Physical Health Consequences (due to job)
# 4. Observed Consequences
# 5. Mental Health Interview
# 
# All of these are also categorical in nature so we simply factorize them as before.

# In[ ]:


wellbeing_indicators = [
    'seek_help',
    'mental_health_consequence',
    'obs_consequence',
    'mental_health_interview',
    'phys_health_consequence'
]


# In[ ]:


wellbeing_indicators_corr = data[wellbeing_indicators].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', 
                                                                            min_periods=1)
plt.figure(figsize = (12, 9))

# Heatmap of correlations
sns.heatmap(wellbeing_indicators_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title("Correlation Heatmap of Well Being\n indicators of Employees");


# # How do company policies relate to their employee's well-being?
# 
# Here, we simple put together the features indicating company policies with those indicating the overall well-being of the employee. The results are also quite interesting, if you look closely

# In[ ]:


wellbeing_policy_corr = data[wellbeing_indicators + company_characs].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', 
                                                                            min_periods=1)
plt.figure(figsize = (12, 9))

# Heatmap of correlations
sns.heatmap(wellbeing_policy_corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title("Correlation Heatmap between Employee Well Being \nand Company Policies");


# # A (Powerful) Wordcloud
# 
# A word cloud of what the employees have to say in regard to this issue of Mental Health!

# In[ ]:


plt.figure(figsize=(20,20))
wordcloud = WordCloud(
                          background_color='white',
                          width=1024,
                          height=1024,
                         ).generate(re.sub(r'[^\w\s]',''," ".join(list(data.comments.unique()[1:]))))
plt.imshow(wordcloud)


# ### That'd be all for now, I'll continue adding some more visuals as I explore this data. 
#  
# **Let me know what you guys think in the comments below!**

# In[ ]:




