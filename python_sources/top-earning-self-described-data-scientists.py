#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# For notebook plotting
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


Dir = '../input/'
freeform_df = pd.read_csv(Dir + 'freeFormResponses.csv', low_memory=False, header=[0,1])
multi_df = pd.read_csv(Dir + 'multipleChoiceResponses.csv', low_memory=False, header=[0,1])
SurveySchema = pd.read_csv(Dir + 'SurveySchema.csv', low_memory=False, header=[0,1])
freeform_df.columns = freeform_df.columns.map('_'.join)
multi_df.columns = multi_df.columns.map('_'.join)
SurveySchema.columns = SurveySchema.columns.map('_'.join)


# In[ ]:


data_scientist = multi_df['Q26_Do you consider yourself to be a data scientist?'] == "Definitely yes"
#datasci_df = multi_df[data_scientist]
top_earning_sal = {"100-125,000","125-150,000","150-200,000","200-250,000","250-300,000","300-400,000","400-500,000","500,000+"}
top_earn = multi_df['Q9_What is your current yearly compensation (approximate $USD)?'].isin(top_earning_sal)
top_dsci_df  = multi_df[data_scientist & top_earn]


# ___

# # Top Earning Data Scientists
# ## Some Insight for Aspiring Data Scientists
# 

# # Table of Contents
# - <a href='#Introduction'> Introduction </a>
# - <a href='#First_Insight'> First Insight </a>
# - <a href='#Second_Insight'> Second Insight </a>
# - <a href='#Third_Insight'> Third Insight </a>
# - <a href='#India_Insight'> Top Earning Data Scientists in India </a>

# <a id='Introduction'></a>

# # Introduction 
# 

# Taking a look at the top earning self-described data scientists, we will explore what these folk do, say and recommend and try to see if there is any significant difference from those self-described (definitely yes) data scientists and those who are less certain. These recommendations will give some insight into what the top earners believe and do.
# 
# The main focus of here is to focus on the top quarter of earners (earning 100k+) who took the survey who are also self-described data scientists, a group of around 759 who took the survey.
# 
# This group is primarily made up of men with 645 responding and 102 women responding with 12 either chosing not to disclose or being self-described. This does seem to indicate a significant between men and women, although compared to the total respondent it appears as though it could demonstrate some bias in the sample as 84.9% responded as men.
# 
# The age breakdown also showed that these earners tended to be a bit older than the general sample population with most being around 30-34 compared to 25-29. Both data sets are skewed slightly to a younger crowd.

# In[ ]:


age_df = multi_df['Q2_What is your age (# years)?'].value_counts()
age_df.index.name = 'Age'
age_df.sort_index(inplace=True)
age_df.plot(kind='bar',rot=20, title='Age distribution of Kagglers',figsize=(14,5));


# In[ ]:


age_teds_df = top_dsci_df['Q2_What is your age (# years)?'].value_counts()
age_teds_df.index.name = 'Age'
age_teds_df.sort_index(inplace=True)
age_teds_df.plot(kind='bar',rot=20, title='Age distribution of Top Earning Data Scientist Kagglers',figsize=(14,5));


# Of these top earners, over half come from the United States with 515 of the 759 coming from the US with India and Canada having the next highest with 29, although people from the EU represent 91 of the responses from these top earners. The US currently seems to pay self-described data scientists the most.
# 
# This does differ significantly from the average Kaggler though, this might be how the question focused on people earning in USD and differences in standard cost of living, average pay, etc.
# As the wage measured in USD is a primary means of determining 'top earners' for this report, the insight provided here will have a bit of a Western bias and may not be suitable for audiences outside of the US, Canada, and the EU.

# In[ ]:


country_df = multi_df['Q3_In which country do you currently reside?'].value_counts()
country_df.index.name = 'Country'
#country_df.sort_index(inplace=True)
country_df.plot(kind='barh',rot=0, title='Countries of Kagglers',figsize=(15,15), colormap = 'Paired');


# In[ ]:


country_teds_df = top_dsci_df['Q3_In which country do you currently reside?'].value_counts()
country_teds_df.index.name = 'Country'
#country_teds_df.sort_index(inplace=True)
country_teds_df.plot(kind='barh',rot=0, title='Countries of Top Earning Data Science Kagglers',figsize=(15,15), colormap = 'Paired');


# When compared to their fellow Kagglers, the top earning data scientists tend to have higher education, particularly more doctoral degrees.

# In[ ]:


edu_df = multi_df['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts()
edu_df.index.name = 'Education'
#edu_df.sort_index(inplace=True)
edu_df.plot(kind='barh',rot=0, title='Education of Kagglers',figsize=(15,5), colormap = 'Paired');


# In[ ]:


edu_teds_df = top_dsci_df['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts()
edu_teds_df.index.name = 'Education'
#edu_teds_df.sort_index(inplace=True)
edu_teds_df.plot(kind='barh',rot=0, title='Education of Top Earning Data Scientists Kagglers',figsize=(15,5), colormap = 'Paired');


# ## What Aspirational Data Scientists should know

# <a id='First_Insight'></a>

# ## First Insight 
# 
# 
# One of the most important questions any aspiring data scientist asks is where to start. As one of the survey questions was "What programming language would you recommend an aspiring data scientist to learn first?" we find that among this crowd of top earners the overwhelming recommendation is Python with 581 of 759 suggesting it with R followed up with 91 and SQL with 49. Python is overwhelmingly recommended despite a significantly smaller amount of these top users using Python most often, only 420 of 667 reported, with the spread redistributing around the other options.  One thing interesting to note here is that this isn't significantly different between genders either, men and women both seem to recommend and use Python with 75 of 102 top earning women recommending Python and 500 of 645 men recommending Python.
# 
# Compared to the community as a whole, we see Python still having a commanding lead with similar results with 14181 (of 18788) of respondents saying that they recommend apsiring data scientists use Python followed by 2342 suggesting R, and 914 saying SQL. Within the total respondents 8180 (of 15222) use Python, followed by 2046 using R, and 1211 using SQL as the language they use most often. 
# 
# ### While there is a significant portion of survey takers and top earners that use Python, it is overwhelmingly recommended that first time data scientists learn Python first.

# In[ ]:


reclang_teds_df = top_dsci_df['Q18_What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts()
reclang_teds_df.index.name = 'Recommendation'
#reclang_teds_df.sort_index(inplace=True)
reclang_teds_df.plot(kind='bar',rot=80, title='Recommended Programming Language of Top Earning Data Science Kagglers',figsize=(15,5));


# <a id='Second_Insight'></a>

# ## Second Insight 
# "How do you perceive the quality of online learning platforms and in-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions? - Online learning platforms and MOOCs"
# 
# Among these self-described data scientists who are earning 100k+ there is not clear consenus as to whether or not online learning platforms are better than traditional brick and mortar institutions. While only a small group (67 of 692) believe these alternative educations are much worse than traditional brick and mortar schools, a plurality of these top earners believed that they were neither bettor nor worse than traditional schooling with 171 of 692 feeling that way. This becomes even more interesting when 133 and 111 feel that these alternative teaching methods are either much better or at least slightly better respectively. This represents around 35.2% of these respondents saying that these options are better than more traditional options representing a plurality of respondents.
# 
# ### Aspiring data scientists may want to consider looking into online learning platforms to see where they want to learn their skills in data science.
# 

# In[ ]:


reclang_teds_df = top_dsci_df['Q39_Part_1_How do you perceive the quality of online learning platforms and in-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions? - Online learning platforms and MOOCs:'].value_counts()
reclang_teds_df.index.name = 'MOOCS Better or Worse'
#reclang_teds_df.sort_index(inplace=True)
temp_new_df = reclang_teds_df.reindex(["Much better", "Slightly better", "Neither better nor worse", "Slightly worse", "Much worse", "No opinion; I do not know"])
temp_new_df.plot(kind='barh',rot=40, title='What Top Earning Data Science Kagglers Think of MOOCS and Online Learning Platforms',figsize=(15,5));


# <a id='Third_Insight'></a>

# ## Third Insight 
# 
# ### The Prefered Online Learning Platforms

# In[ ]:


onlinelearn_teds_df = top_dsci_df['Q37_On which online platform have you spent the most amount of time? - Selected Choice'].value_counts()
onlinelearn_teds_df.index.name = 'MOOCS and Online Learning Platforms'
#onlinelearn_teds_df.sort_index(inplace=True)
onlinelearn_teds_df.plot(kind='barh',rot=40, title='MOOCS and Online Learning Platforms the Top Earning Data Scientists Use',figsize=(15,5), colormap = 'Spectral');


# The main community looks similar but the top earning data scientists seem to prefer Udacity, DataCamp, and Online University Courses more than the average Kaggler. 
# 
# Coursera is loved universally.

# In[ ]:


edu_online_df = multi_df['Q37_On which online platform have you spent the most amount of time? - Selected Choice'].value_counts()
edu_online_df.index.name = 'MOOCS and Online Learning Platforms'
#edu_online_df.sort_index(inplace=True)
edu_online_df.plot(kind='barh',rot=40, title='MOOCS and Online Learning Platforms of Kagglers',figsize=(15,5), colormap = 'Spectral');


# As the whole community shows, there is a lot of love for Coursera courses with most people spending the most time on Coursera regardless of whether they are top earning data scientists or not, however, the top earnering self-described data scientists seem to not like Udemy as much while liking Udacity, DataCamp, and Online University Courses more.
# 
# ### Top Earners seem to have a preference to using Coursera, followed by Udacity, Udemy, and then Online University Courses. Coursera seems to be a good place to look for those MOOCs if you take the advice from Insight Two!

# ___

# <a id='India_Insight'></a>

# # Top Earning Data Scientists in India 
# 
# 
# As noted, the top quarter of earners were primarily from the EU, the US, and Canada, however, many Kagglers come from India. Since there is a significant difference in earnings between India and these nations, it seems important to include the takeaway from top earners in India so Kagglers in India can see what the top earners currently residing in India think. The top quartile of earners appears to be those making somewhere over 15-20,000 USD in India, so this analysis will focus on those self-described data scientists who are currently residing in India and reported earning over 20,000 USD.
# 
# Using the same methods, we can see some similarities with the top earners residing in India versus the global top earners.

# In[ ]:


top_earning_india = {"20-30,000","30-40,000","40-50,000","50-60,000","60-70,000","70-80,000","80-90,000","90-100,000","100-125,000","125-150,000","150-200,000","200-250,000","250-300,000","300-400,000","400-500,000","500,000+"}
top_earn_india = multi_df['Q9_What is your current yearly compensation (approximate $USD)?'].isin(top_earning_india)
india_res = multi_df['Q3_In which country do you currently reside?'] == "India"
indiatop_dsci_df  = multi_df[data_scientist & top_earn_india & india_res]


# In[ ]:


age_indiatop_df = indiatop_dsci_df['Q2_What is your age (# years)?'].value_counts()
age_indiatop_df.index.name = 'Age'
age_indiatop_df.sort_index(inplace=True)
age_indiatop_df.plot(kind='bar',rot=20, title='Age distribution of Top Earning Data Scientist Kagglers in India',figsize=(14,5));


# In[ ]:


edu_indiatop_df = indiatop_dsci_df['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts()
edu_indiatop_df.index.name = 'Education'
#edu_teds_df.sort_index(inplace=True)
edu_indiatop_df.plot(kind='barh',rot=0, title='Education of Top Earning Data Scientists Kagglers in India',figsize=(15,5), colormap = 'Paired');


# The breakdown is fairly similar to the global distribution, although it seems as though those top earners residing in India may be slightly younger than their counterparts globally. It also appears that these top earners, while still favoring master's degrees also tend to have more bachelor's degree instead of doctoral degrees.

# In[ ]:


reclang_indiateds_df = indiatop_dsci_df['Q18_What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice'].value_counts()
reclang_indiateds_df.index.name = 'Recommendation'
#reclang_teds_df.sort_index(inplace=True)
reclang_indiateds_df.plot(kind='bar',rot=80, title='Recommended Programming Language of Top Earning Data Science Kagglers in India',figsize=(15,5));


# Python is once again a highly recommended language with a similar breakdown in popularity for R and SQL for languages recommended for aspiring data scientists to learn first.
# 
# However, a major difference is seen in how these top earners view MOOCs and Online Learning Platforms versus Brick and Mortar Institutions.

# In[ ]:


reclang_indiateds_df = indiatop_dsci_df['Q39_Part_1_How do you perceive the quality of online learning platforms and in-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions? - Online learning platforms and MOOCs:'].value_counts()
reclang_indiateds_df.index.name = 'MOOCS Better or Worse'
#reclang_teds_df.sort_index(inplace=True)
temp_indianew_df = reclang_indiateds_df.reindex(["Much better", "Slightly better", "Neither better nor worse", "Slightly worse", "Much worse", "No opinion; I do not know"])
temp_indianew_df.plot(kind='barh',rot=40, title='What Top Earning Data Science Kagglers in India Think of MOOCS and Online Learning Platforms',figsize=(15,5));


# The top earners residing in India value MOOCs and Online Learning Platforms far more than those globally. It isn't just a plurality of respondents but an overwhelming majority of these respondents view MOOCs and Online Learning Platforms as more valuable than traditional education.
# 
# These individuals also tend to differ on what platforms they like to learn from as well.

# In[ ]:


indiaonlinelearn_teds_df = indiatop_dsci_df['Q37_On which online platform have you spent the most amount of time? - Selected Choice'].value_counts()
indiaonlinelearn_teds_df.index.name = 'MOOCS and Online Learning Platforms'
#onlinelearn_teds_df.sort_index(inplace=True)
indiaonlinelearn_teds_df.plot(kind='barh',rot=40, title='MOOCS and Online Learning Platforms the Top Earning Data Scientists in India Use',figsize=(15,5), colormap = 'Spectral');


# While Coursera still remains a favorite, Udemy got a bit of a boost over Udacity and edX and DataCamp are tied. While a bit of a small sample size it is interesting to note that Online University Courses have fallen in the ranks which could be tied to how this particular population views traditional education.
