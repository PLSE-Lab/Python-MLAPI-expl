#!/usr/bin/env python
# coding: utf-8

# **Summary**
# 
# I would like to dig this survey and find insights related to **non-Data scientists who are leveraging ML and DS in their day to day jobs**. What I mean by Non-Data Scientists? Those who do not earn their salaries doing ML work but rather leverage ML to improve the quality of their work. 
# *I belong to this category of non- data scientists where I work in Corporate Strategy and do not need to use ML in my job but we do use ML to improve the quality of decision making.*
# 
# 1. **ML and DS is not only a young woman (or man's) game anymore**- almost 25% of the respondents are above 40. This population might not necessarily be having data scientists jobs in the true sense but more of leveraging ML in their work.
# 
# 2. Respondents are from diverse backgrounds (other than computers and mathematics)- almost **8% are from arts background. **
# 
# 3. 40% respondents are not from Computer industry or Academia.
# 
# 4. Majority of popular work with text, numerical and categorical data.
# 
# 5. **Importance of traditional educational institutions and academic achievements is decreasing** with 75% of respondents believing that MOOCs are at least on par or better than brick and mortar institutions.
# 
# 6. 70% of respondents believe that either they are experts can explain the model insights- **ML models are not longer black boxes.**
# 
# Note: I am not an expert in coding, please pardon me for using inelegant code snippets that somehow quickly work but are not necessarily the best approach. Also, I have tried to look at data and come up with hypothesis- it could be or could not be correct. 
# 
# I would love to hear from you all on the comments on what different would you do and what do you think of this segment of population.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


surveySchema=pd.read_csv('../input/SurveySchema.csv')
freeFormResponses=pd.read_csv('../input/freeFormResponses.csv')
multipleChoiceResponses=pd.read_csv('../input/multipleChoiceResponses.csv')


# Starting with the basics- **81% of respondents are Male while 17% are female**, unsurprisingly. However, I am sure that these % would be more equitable in few years. Let's find out the various countries and age groups.

# In[ ]:


df_temp=multipleChoiceResponses['Q2']
df_temp.value_counts(normalize=True).plot(kind='bar',figsize=(8,8))


# The top most age category is 25-29, sugesting that **young working professionals are most active on Kaggle**. Even 22-24 (second largest category) would be comprising of post graduate students or people just entering the workforce. Data science being the hot job, unsurprisingly this age group uses Kaggle the most.
# Interestingly for me, above 40 age group comprises of roughly 18% of the population and above 50 age is around 7-8%. So yes, younger people are into data Science but even the older generation is taking a lot of interest in AI. I am assuming this **older generation might not be data scientists by profession but more of hobbyists looking to learn something new and stay relevant in the job market.**

# In[ ]:


df_temp=multipleChoiceResponses['Q3']
df_temp.value_counts(normalize=True).plot(kind='bar',figsize=(20,10))


# Top 3 countries with data science professionals, unsurprisingly are: USA (roughly 20%), India (roughly 18%), China (roughly 7.3%). US is where most of the advancements in data science have occured, so makes sense to have that as the top country. Indian, China with their talented workforce are next, however there is a substantial difference in number of users from India and China. China has more AI related startups than India, so my hypothesis would be that **Chinese use less of Kaggle than assuming that there are lesser data scientist in China.**

# In[ ]:


df_temp=multipleChoiceResponses['Q4']
df_temp.value_counts(normalize=True).plot(kind='barh')


# In[ ]:


df_temp=multipleChoiceResponses['Q5']
df_temp.value_counts(normalize=True).plot(kind='barh')


# Computer science focused education is on the top-expected. However, **what's interesting is in other fields.** Now, roughly 7-8% of respondents are from non-science background (social science, fine arts etc.) and rougly 30% of respondents are from non-computer science/ mathematics/ engineering focused education. This implies that **Data science is not only a computer science phenomenon- wonderful open source libraries and platforms like Kaggle are enabling non-engineers to actively leverage Data science in their individual discipline**. These might not be hard coders but people leveraging data to help in their fields.

# In[ ]:


df_temp=multipleChoiceResponses['Q6']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(8,8))


# On the profession front, again my focus would be non-data scientists. **Roughy 15 -18% of the job titles suggest that they do not earn a living using ML but leverage ML to improve decision making in their day-to-day job** (myself included in this 15-18%). This category would include titles like Manager, Consultant, salesperson, business analyst, marketing analyst, project manager, chief officer etc. 

# In[ ]:


df_temp=multipleChoiceResponses['Q7']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(8,8))


# Almost **40% of respondents are not from Computer industry or Academia** (including students). Again, a very good indicator of broad leverage of ML across different industries.

# In[ ]:


df_temp=multipleChoiceResponses.loc[:,'Q11_Part_1':'Q11_Part_7']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="Selected Choice -",expand=True)
df_temp.columns=[column_names[1]]
(df_temp.count()/len(df_temp)).plot(kind='barh',figsize=(4,4))


# I am unable to get any meaningful insights from the above graph, so will leave at that.

# In[ ]:


df_temp=multipleChoiceResponses['Q23']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))


# Almost, **23% of the respondents spend less than 25% of their time coding**-  we can infer that these 23% respondents are not primarily software programmers but leverage data science in their jobs.

# In[ ]:


df_temp=multipleChoiceResponses['Q26']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))


# Almost, **25% of respondents do not consider themselves as Data scientists** (Probably not and Definitely not). This is the population of interest for this kernel.

# In[ ]:


df_temp=multipleChoiceResponses.loc[:,'Q31_Part_1':'Q31_Part_12']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="Selected Choice -",expand=True)
df_temp.columns=[column_names[1]]
(df_temp.count()/len(df_temp)).plot(kind='barh',figsize=(6,6))


# Very interesting chart.. Clearly shows that a **majority of respondents work with regular data** like text, timeseries, numerical and categorical. Respondents using more specialzied data like video, sensor, audio, geopspatial are more likely to be data scientists by profession.

# In[ ]:


df_temp=multipleChoiceResponses.loc[:,'Q33_Part_1':'Q33_Part_11']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="Selected Choice -",expand=True)
df_temp.columns=[column_names[1]]
(df_temp.count()/len(df_temp)).plot(kind='barh',figsize=(6,6))


# Around **10% of the respondents do not use public data**- effectively implying that they use their specific company data for ML and DS work. 

# In[ ]:


df_temp=multipleChoiceResponses.loc[:,'Q34_Part_1':'Q34_Part_6']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="-",expand=True)
df_temp.columns=[column_names[1]]

df_temp=df_temp.iloc[1:]
df_temp=df_temp.apply(pd.to_numeric)
df_temp.mean().plot(kind='barh',figsize=(6,6))


# From the above graph, difficult to put any hypothesis about respondents who are not data scientists. However, **12% of time getting spent in communicating with the stakeholders most likely would be a job of non-data scientists**. I know in smaller setups, this won't be true but in my experience in larger corporates, there would be a layer between data scientists and business team (business analyst or some similar title) who would be performing this task.

# In[ ]:


df_temp=multipleChoiceResponses.loc[:,'Q35_Part_1':'Q35_Part_6']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="-",expand=True)
df_temp.columns=[column_names[1]]

df_temp=df_temp.iloc[1:]
df_temp=df_temp.apply(pd.to_numeric)
df_temp.mean().plot(kind='barh',figsize=(6,6))


# Above chart cannot comment on non-data scientists. But clearly shows the **lesser importance of University courses in a new field like ML**. Growing importance of MOOCs and on the job learning (Work) is clearly visible in the above chart.

# In[ ]:


df_temp=multipleChoiceResponses['Q39_Part_1']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))


# In[ ]:


df_temp=multipleChoiceResponses['Q39_Part_2']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))


# Warning signs for traditional institutions- An **overwhelming majorty (almost 75%) feel that MOOCs are at least on par** with brick and mortal institutions but the verdict is more fragmented with regards to in-person bootcamps. This is likely due to the fact that scaling of in-person bootcamps is a big challenge, hence more than 30% of respondents might not have tried it.

# In[ ]:


df_temp=multipleChoiceResponses['Q40']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))


# Above, we saw the decreasing importance of traditional universities and this chart shows the **decreasing importance of academic achievements**. Around, 85% of respondents feel that independent projects are more  important than academic achievements. 

# In[ ]:


df_temp=multipleChoiceResponses.loc[:,'Q42_Part_1':'Q42_Part_5']
column_names= df_temp.iloc[0,:]
column_names=column_names.str.split(pat="Choice -",expand=True)
df_temp.columns=[column_names[1]]
(df_temp.count()/len(df_temp)).plot(kind='barh',figsize=(6,6))


# Almost **23% of respondents consider revenue/ business goals** as a determinant for the success of their ML models. What else could show better of how ML has moved from academica to computer science practitioners and now to the regular business users.

# In[ ]:


df_temp=multipleChoiceResponses['Q48']
df_temp.value_counts(normalize=True).plot(kind='barh',figsize=(6,6))


# A large proportion of respondents feel confident (roughly 70%) that either they themselves or the experts can explain the ML models. I am sure this would be a **big improvement from previous years** where ML models were considered to be more of black boxes.
