#!/usr/bin/env python
# coding: utf-8

# # **Introduction**

# In[ ]:


from IPython.display import Image


# In[ ]:


Image("../input/kaggle-survey-image/Kaggle.png", width ='1000')


# **Kaggle** is the world's largest data science platform and great playground where many people can learn and grow about data & data science & statistics.
# It's also a great platform for statisticians, data analysts, and data scientists in the real world industry to share their analytical and machine learning methods.
# The competition is survey data published by kaggle in 2019 and began in 2017.
# In the three years from 2017 to 2019, many analytical & machine learning technologies were developed and further enhanced.
# That's why I want to use the data for 2017, 2018 and 2019 to find time-to-time changes in the overall area.
# And the final analysis will be to analyze what data scientists have answered in 2019.
# 
# **There are several limitations in this analysis.** 
# * First. The questions aren't the same for every year. 
#     * So I did a comparison of very similar questions.
# * Second, Since this is a survey, not everyone will answer with the appropriate credentials and there may be a fake response.
# 
# Let's get some insights from this analysis.

# # How did it change over time?

# Over the course of three years, much of society is changing as interest in data analysis and data science grows.
# What has changed in the last three years?

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import squarify
plt.style.use('fivethirtyeight')
# You Can Change 
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import base64
import io
from scipy.misc import imread
import codecs
from IPython.display import HTML
from matplotlib_venn import venn3
import re


# In[ ]:


survey_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')
question_2019 = pd.read_csv('../input/kaggle-survey-2019/questions_only.csv')
columns_multiple_2019 = [col for col in list(survey_2019.columns) if re.search('Part_\d{1,2}$', col)]
multiple_columns_list_2019 = [ [col]+col.split('_') for col in columns_multiple_2019 ]
qa_multiple_2019 = pd.DataFrame(multiple_columns_list_2019).groupby([1])[0].apply(list)
question_numbers_list_2019 = sorted([int(i.split('Q')[1]) for i in list(qa_multiple_2019.index)])
question_list_2019 = [ 'Q{}'.format(i) for i in question_numbers_list_2019]
#questions_2019 = ''.join([f'<li>{i}</li>' for i in question_list_2019])
survey_2019['year'] = '2019'

survey_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')
question_2018 = pd.read_csv('../input/kaggle-survey-2018/SurveySchema.csv').iloc[0:1]
del question_2018['2018 Kaggle Machine Learning and Data Science Survey']
columns_multiple_2018 = [col for col in list(survey_2018.columns) if re.search('Part_\d{1,2}$', col)]
multiple_columns_list_2018 = [ [col]+col.split('_') for col in columns_multiple_2018 ]
qa_multiple_2018 = pd.DataFrame(multiple_columns_list_2018).groupby([1])[0].apply(list)
question_numbers_list_2018 = sorted([int(i.split('Q')[1]) for i in list(qa_multiple_2018.index)])
question_list_2018 = [ 'Q{}'.format(i) for i in question_numbers_list_2018]
#questions_2018 = ''.join([f'<li>{i}</li>' for i in question_list_2018])
survey_2018['year'] = '2018'

survey_2017 = pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')
survey_2017['year'] = '2017'


# # Simple Visualization 

# ## Gender

# In[ ]:


plt.subplots(figsize = (20, 20))
gender_2019 = survey_2019[['year','Q2']].rename(columns = {'Q2' : 'GenderSelect'}).iloc[1:,]
gender_2018 = survey_2018[['year','Q1']].rename(columns = {'Q1' : 'GenderSelect'}).iloc[1:,]
gender_2017 = survey_2017[['year','GenderSelect']]
gender_data = pd.concat([gender_2019,gender_2018,gender_2017])
gender_data_prop = gender_data['GenderSelect'].groupby(gender_data['year']).value_counts(normalize = True).rename ('Prop').reset_index()

sns.barplot(gender_data_prop['Prop'], gender_data_prop['GenderSelect'],palette='inferno_r', hue =gender_data_prop['year'])
plt.legend()
plt.show()


# If you look at it as a whole, you can see that men are overwhelmingly more important than women. <br>
# Also, the difference between 2017 and 2019 is not clear. <br>
# There are a few guesses about this. The part about the data is either that men are interested or that men are responding a lot.... <br>
# **I don't know exactly...haha**

# ## Country

# In[ ]:


# tranform percentage
country_2019 = survey_2019[['year','Q3']].rename(columns = {'Q3' : 'Country'}).iloc[1:,]
country_2018 = survey_2018[['year','Q3']].rename(columns = {'Q3' : 'Country'}).iloc[1:,]
country_2017 = survey_2017[['year','Country']]
country_2019_prop = country_2019['Country'].groupby(country_2019['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending=False)[:10].reset_index()
country_2018_prop = country_2018['Country'].groupby(country_2018['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending=False)[:10].reset_index()
country_2017_prop = country_2017['Country'].groupby(country_2017['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending=False)[:10].reset_index()

# plot
f,ax=plt.subplots(1,3,figsize=(25,15))
sns.barplot('Prop','Country', data=country_2019_prop , palette='inferno',ax=ax[0])
ax[0].set_xlabel('')
ax[0].set_title('Top 10 Countries by number of Response 2019')
sns.barplot('Prop','Country', data=country_2018_prop , palette='inferno',ax=ax[1])
ax[1].set_xlabel('')
ax[1].set_ylabel('')
ax[1].set_title('Top 10 Countries by number of Response 2018')
sns.barplot('Prop','Country', data=country_2017_prop , palette='inferno',ax=ax[2])
ax[2].set_xlabel('')
ax[2].set_ylabel('')
ax[2].set_title('Top 10 Countries by number of Response 2017')
plt.subplots_adjust(wspace=1.0)
plt.show()


# The above plot shows that the top countries from 2017 to 2019 are the United States and India.<br>
# However, when Kaggle's users think of more than a million, it is unlikely that most users will see the U.S. and India. <br>
# "Where is Korea?....."

# ## Age

# In[ ]:


age = {}
for i in survey_2019['Q1'].iloc[1:,].unique()[:-1]:
    min = int(i.split('-')[0])
    max = int(i.split('-')[1])
    age.update({i : list(range(min,max+1))})
    
def chage_categori_age(x):
    for i in age.items():
        if x in i[1]:
            return i[0]
        
survey_2018['Q2'] = survey_2018['Q2'].iloc[1:,].apply(lambda x: '70+' if (x == '80+') | (x == '70-79') else x)
survey_2017['Age'] = survey_2017['Age'].apply(chage_categori_age)


# In[ ]:


# tranform percentage
age_2019_prop = survey_2019['Q1'].groupby(country_2019['year']).value_counts(normalize = True).rename ('Prop').reset_index().rename(columns = {'Q1' : 'Age'})
age_2018_prop = survey_2018['Q2'].groupby(country_2018['year']).value_counts(normalize = True).rename ('Prop').reset_index().rename(columns = {'Q2' : 'Age'})
age_2017_prop = survey_2017['Age'].groupby(country_2017['year']).value_counts(normalize = True).rename ('Prop').reset_index()

# plot
f,ax=plt.subplots(1,3,figsize=(25,15))
sns.barplot('Prop','Age', data=age_2019_prop , palette='summer',ax=ax[0])
ax[0].set_xlabel('')
ax[0].set_title('Age 2019')
ax[0].axvline(0.25, linestyle='dashed')
ax[0].axvline(0.10, linestyle='dashed', color = 'r')
ax[0].axhspan(2.5,3.5 ,facecolor='Blue', alpha=0.2) # hilight space

sns.barplot('Prop','Age', data=age_2018_prop , palette='summer',ax=ax[1])
ax[1].set_xlabel('')
ax[1].set_ylabel('')
ax[1].set_title('Age 2018')
ax[1].axvline(0.25, linestyle='dashed')
ax[1].axvline(0.10, linestyle='dashed', color = 'r')
ax[1].axhspan(2.5,3.5 ,facecolor='Blue', alpha=0.2) # hilight space

sns.barplot('Prop','Age', data=age_2017_prop , palette='summer',ax=ax[2])
ax[2].set_xlabel('')
ax[2].set_ylabel('')
ax[2].set_title('Age 2017')
ax[2].axvline(0.25, linestyle='dashed')
ax[2].axvline(0.10, linestyle='dashed', color = 'r')
ax[2].axhspan(3.5,4.5 ,facecolor='Blue', alpha=0.2) # hilight space

plt.subplots_adjust(wspace=0.6)
plt.show()


# The most to responses believed to have people in their 20s and 30s.
# Also, people between the ages of 18 and 21 from 2018 can see their response rate rise.
# If so, I think the response rate of students is likely to increase as well.

# ## CurrentJob 

# In[ ]:


currentjob_2019_prop = survey_2019['Q5'].groupby(survey_2019['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending = False).reset_index().rename(columns = {'Q5' : 'CurrentJobTitleSelect'})[:10]
currentjob_2018_prop = survey_2018['Q6'].groupby(survey_2018['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending = False).reset_index().rename(columns = {'Q6' : 'CurrentJobTitleSelect'})[:10]
currentjob_2017_prop = survey_2017['CurrentJobTitleSelect'].groupby(survey_2017['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending = False).reset_index()[:10]

# plot
f,ax=plt.subplots(1,3,figsize=(25,15))

sns.barplot('Prop','CurrentJobTitleSelect', data=currentjob_2019_prop , palette='BrBG',ax=ax[0])
ax[0].set_xlabel('')
ax[0].set_title('Top10 CurrentJobTitle 2019')
ax[0].axvline(0.20, linestyle='dashed')
ax[0].axvline(0.15, linestyle='dashed', color = 'r')
ax[0].axhspan(0.5,1.5 ,facecolor='Red', alpha=0.5) # hilight space

sns.barplot('Prop','CurrentJobTitleSelect', data=currentjob_2018_prop , palette='BrBG',ax=ax[1])
ax[1].set_xlabel('')
ax[1].set_ylabel('')
ax[1].set_title('Top10 CurrentJobTitle 2018')
ax[1].axvline(0.20, linestyle='dashed')
ax[1].axvline(0.15, linestyle='dashed', color = 'r')
ax[1].axhspan(-0.5,0.5 ,facecolor='Red', alpha=0.5) # hilight space

sns.barplot('Prop','CurrentJobTitleSelect', data=currentjob_2017_prop , palette='BrBG',ax=ax[2])
ax[2].set_xlabel('')
ax[2].set_ylabel('')
ax[2].set_title('Top10 CurrentJobTitle 2017')
ax[2].axvline(0.20, linestyle='dashed')
ax[2].axvline(0.15, linestyle='dashed', color = 'r')

plt.subplots_adjust(wspace=0.6)
plt.show()


# Looking at the current job of TOP10 you can see that the data scientist has had a high rate for three years.
# However, what's unusual is that starting in 2018, the percentage of study has suddenly risen.
# It can be seen that many students are starting to pay a lot of attention to ML or DS technology.
# 
# **So what would happen to the degrees of those who answered that they were students?**
# I want to know why many students answered that they are interested and therefore eager to get a higher education to become an expert.

# ### Student Education

# In[ ]:


survey_2019_e = survey_2019[survey_2019['Q5'] == 'Student'].iloc[1:,]
survey_2018_e = survey_2018[survey_2018['Q6'] == 'Student'].iloc[1:,]
education_2019_prop = survey_2019_e['Q4'].groupby(survey_2019_e['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending = False).reset_index().rename(columns = {'Q4' : 'Formal Education'})[:10]
education_2018_prop = survey_2018_e['Q4'].groupby(survey_2018_e['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending = False).reset_index().rename(columns = {'Q4' : 'Formal Education'})[:10]

# plot
f,ax=plt.subplots(1,2,figsize=(25,15))

sns.barplot('Prop','Formal Education', data=education_2019_prop , palette='RdYlGn',ax=ax[0])
ax[0].set_xlabel(' ')
ax[0].set_ylabel(' ')
ax[0].set_title('Formal Education 2019')
ax[0].axvline(0.40, linestyle='dashed')
ax[0].axhspan(-0.5,0.5 ,facecolor='Gray', alpha=0.5) # hilight space

sns.barplot('Prop','Formal Education', data=education_2018_prop , palette='RdYlGn',ax=ax[1])
ax[1].set_xlabel(' ')
ax[1].set_ylabel(' ')
ax[1].set_title('Formal Education 2018')
ax[1].axvline(0.40, linestyle='dashed')
ax[1].axhspan(0.5,1.5 ,facecolor='Gray', alpha=0.5) # hilight space

plt.subplots_adjust(wspace=1.0)
plt.show()


# Well, interestingly, it seems that the response rate of the Bachelor has increased, while that of the masters has decreased. Well, if we can make a funny guess from the above information, maybe it's because the masters is busy..

# # Platform To Laern

# In[ ]:


question = 'Q13' # On which platforms have you begun or completed data science courses?
columns_list_2019 = qa_multiple_2019[question]
survey_2019 ['LearningPlatformSelect'] = survey_2019[columns_list_2019].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)

question = 'Q36' # On which platforms have you begun or completed data science courses?
columns_list_2018 = qa_multiple_2018[question]
survey_2018['LearningPlatformSelect'] = survey_2018[columns_list_2018].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)


# In[ ]:


learn_2019 = survey_2019['LearningPlatformSelect'].iloc[1:,].str.split(',')
learn_2018 = survey_2018['LearningPlatformSelect'].iloc[1:,].str.split(',')
learn_2017 = survey_2017['LearningPlatformSelect'].iloc[1:,].str.split(',')

platform_2019 = []
platform_2018 = []
platform_2017 = []

for i in learn_2019.dropna():
    platform_2019.extend(i)
    
for i in learn_2018.dropna():
    platform_2018.extend(i)
    
for i in learn_2017.dropna():
    platform_2017.extend(i)

    
f, ax = plt.subplots(1,3, figsize = (18,8))

pd.Series(platform_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('winter',15),ax = ax[0])
ax[0].set_title('Top 10 Platforms to Learn 2019', size = 15)
pd.Series(platform_2018).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('winter',15),ax = ax[1])
ax[1].set_title('Top 10 Platforms to Learn 2018', size = 15)
pd.Series(platform_2017).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('winter',15),ax = ax[2])
ax[2].set_title('Top 10 Platforms to Learn 2017', size = 15)
plt.show()


# The overall weight indicates that the percentage mentioned in Coursera will increase from 2018.
# In fact, many are being created and I also lectures mooc coursera or knowledge relating to data and statistics through.
# The certificate of qualification for completion is also attractive.

# # Machine Learning

# ## Machine Learning Alforithm

# In[ ]:


question = 'Q24' # Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice
columns_list_2019 = qa_multiple_2019[question]
survey_2019 ['MLTechniquesSelect'] = survey_2019[columns_list_2019].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)


# In[ ]:


mlTech_2019 = survey_2019['MLTechniquesSelect'].iloc[1:,].str.split('?')
mlTech_2017 = survey_2017['MLTechniquesSelect'].iloc[1:,].str.split(',')
mlTech_ds_2019 = survey_2019[survey_2019['Q5'] =='Data Scientist']['MLTechniquesSelect'].iloc[1:,].str.split('?')
mlTech_ds_2017 = survey_2017[survey_2017['CurrentJobTitleSelect'] =='Data Scientist']['MLTechniquesSelect'].iloc[1:,].str.split(',')


ml_2019 = []
ml_2017 = []
ml_ds_2019 = []
ml_ds_2017 = []

for i in mlTech_2019.dropna():
    ml_2019.extend(i)
    
for i in mlTech_2017.dropna():
    ml_2017.extend(i)
    
for i in mlTech_ds_2019.dropna():
    ml_ds_2019.extend(i)
    
for i in mlTech_ds_2017.dropna():
    ml_ds_2017.extend(i)
    
    
f, ax = plt.subplots(2,2, figsize = (25,15))
pd.Series(ml_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('spring',15),ax = ax[0][0])
ax[0][0].set_title('Top 10 MLTech 2019', size = 15)
ax[0][0].axhspan(6.5,7.5 ,facecolor='Gray', alpha=0.5) # hilight space
ax[0][0].axhspan(5.5,6.5 ,facecolor='Gray', alpha=0.5) # hilight space
ax[0][0].axvline(0.10, linestyle='dashed', color= 'r')
pd.Series(ml_2017).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('spring',15),ax = ax[0][1])
ax[0][1].set_title('Top 10 MLTech 2017', size = 15)
ax[0][1].axhspan(1.5,2.5 ,facecolor='Gray', alpha=0.5) # hilight space
ax[0][1].axhspan(3.5,4.5 ,facecolor='Gray', alpha=0.5) # hilight space
ax[0][1].axvline(0.10, linestyle='dashed', color= 'r')
pd.Series(ml_ds_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('winter',15),ax = ax[1][0])
ax[1][0].set_title('Top 10 Data Scientist MLTech 2019', size = 15)
ax[1][0].axhspan(6.5,7.5 ,facecolor='Gray', alpha=0.5) # hilight space
ax[1][0].axhspan(5.5,6.5 ,facecolor='Gray', alpha=0.5) # hilight space
ax[1][0].axvline(0.10, linestyle='dashed', color= 'r')
pd.Series(ml_ds_2017).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('winter',15),ax = ax[1][1])
ax[1][1].set_title('Top 10 Data Scientist MLTech 2017', size = 15)
ax[1][1].axhspan(1.5,2.5 ,facecolor='Gray', alpha=0.5) # hilight space
ax[1][1].axhspan(3.5,4.5 ,facecolor='Gray', alpha=0.5) # hilight space
ax[1][1].axvline(0.10, linestyle='dashed', color= 'r')
plt.subplots_adjust(wspace=.6)
plt.show()


# Machine Learning Algorithm, which is used most frequently in 2017 and 2019, is Logistic (Linear) Regression. It is the most basic ML method, but it is also an easy to interpret and powerful algorithm, so it is an algorithm that you will try to use first at the start of the project. What we have looked carefully at is that the Boasting method is ranked higher than it was in 2017. In Kaggle, we can see that Xgboost and Catboost are very popular, and these results are shown in the above plot. In addition, it is possible to see that CNN and RNN are higher than 2017 and higher than other top-tier algorithms because Deep Learning is a hot algorithm.
# 
# When looking at data scientists, it seems that they are using more Gradient Boosting techniques for the whole thing.

# ## Machine Learning FrameWork

# In[ ]:


question = 'Q28' # Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice
columns_list_2019 = qa_multiple_2019[question]
survey_2019 ['MLFramework'] = survey_2019[columns_list_2019].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)

question = 'Q19' # What machine learning frameworks have you used in the past 5 years?
columns_list_2018 = qa_multiple_2018[question]
survey_2018['MLFramework'] = survey_2018[columns_list_2018].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)


# In[ ]:


mlFrame_2019 = survey_2019['MLFramework'].iloc[1:,].str.split(',')
mlFrame_2018 = survey_2018['MLFramework'].iloc[1:,].str.split(',')

mlfr_2019 = []
mlfr_2018 = []

for i in mlFrame_2019.dropna():
    mlfr_2019.extend(i)
    
for i in mlFrame_2018.dropna():
    mlfr_2018.extend(i)
    
f, ax = plt.subplots(1,2, figsize = (18,8))

pd.Series(mlfr_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('inferno_r',15),ax = ax[0])
ax[0].set_title('Top 10 ML Framework 2019', size = 15)
pd.Series(mlfr_2018).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('inferno_r',15),ax = ax[1])
ax[1].set_title('Top 10 ML Framework 2018', size = 15)

plt.show()


# Overall, Machine Learning FrameWork is showing similar performance in 2018 and 2019.
# The Skikit-Learn package is basically the FrameWork that you use when you study and learn about data because of the Framework that carries a lot of powerful functions.
# Also, Tensorflow is Deep Learning OpenSource, a FrameWork used by many artificial intelligence engineers.

# ## Cloud Computing

# In[ ]:


question = 'Q29' # Which categories of computer vision methods do you use on a regular basis?  (Select all that apply) - Selected Choice
columns_list_2019 = qa_multiple_2019[question]
survey_2019 ['Cloud'] = survey_2019[columns_list_2019].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)

question = 'Q15' # What machine learning frameworks have you used in the past 5 years?
columns_list_2018 = qa_multiple_2018[question]
survey_2018['Cloud'] = survey_2018[columns_list_2018].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)


# In[ ]:


cloud_2019 = survey_2019['Cloud'].iloc[1:,].str.split('?')
cloud_2018 = survey_2018['Cloud'].iloc[1:,].str.split('?')

cl_2019 = []
cl_2018 = []

for i in cloud_2019.dropna():
    cl_2019.extend(i)
    
for i in cloud_2018.dropna():
    cl_2018.extend(i)
    
f, ax = plt.subplots(1,2, figsize = (18,15))

pd.Series(cl_2019).value_counts(normalize=True)[:5].sort_values(ascending=True).plot.barh(width = 0.9,ax = ax[0])
ax[0].set_title('Top 5 Cloud 2019', size = 15)
ax[0].axvline(0.2, linestyle='dashed', color= 'r')
pd.Series(cl_2018).value_counts(normalize=True)[:5].sort_values(ascending=True).plot.barh(width = 0.9, ax = ax[1])
ax[1].set_title('Top 5 Cloud  2018', size = 15)
ax[1].axvline(0.2, linestyle='dashed', color= 'r')
plt.subplots_adjust(wspace=.6)
plt.show()


# As data grows, many people or businesses are using cloud services.
# The most commonly used support appears to be Amazon Web Service, followed by Google Cloud Service.
# Recently, Kagle also introduced Google services.
# We also see many cases of using Google Cloud as a Data Lake.
# **"How will change occur next year? "**

# ## Automated machine learning tools

# In[ ]:


question = 'Q33'# Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?
columns_list_2019 = qa_multiple_2019[question]
survey_2019 ['AutoML'] = survey_2019[columns_list_2019].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)


# In[ ]:


auto_2019 = survey_2019['AutoML'].iloc[1:,].str.split('?')

aml_2019 = []

for i in auto_2019.dropna():
    aml_2019.extend(i)
    
pd.Series(aml_2019).value_counts(normalize=True)[:5].sort_values(ascending=True).plot.barh(width = 0.9)
plt.title('Top 5 Cloud 2019', size = 15)
plt.show()


# Recently, **"Artificial Intelligence Makes Artificial Intelligence"** came along with the introduction of Automated Machine Learning.
# It was interesting to see that the FE and FS sections were able to save a lot of time and create models that are optimized for a given data. And I think it's a very efficient tool for real companies because it makes it easy to deploy models.
# However, so far, only the reform and classification are supported, and Google has its disadvantages.
# As technology becomes more advanced over time, many analysts are likely to use it as well.

# ## Programming

# In[ ]:


question = 'Q18' # What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice 
columns_list_2019 = qa_multiple_2019[question]
survey_2019 ['WorkToolsSelect'] = survey_2019[columns_list_2019].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)

question = 'Q16' # What programming languages do you use on a regular basis? (Select all that apply) - Selected Choice 
columns_list_2018 = qa_multiple_2018[question]
survey_2018['WorkToolsSelect'] = survey_2018[columns_list_2018].apply(lambda x: ",".join(x.dropna().astype(str)), axis=1).replace('',np.nan)


# In[ ]:


programm_2019 = survey_2019.dropna(subset = ['WorkToolsSelect']).iloc[1:,]
programm_2018 = survey_2018.dropna(subset = ['WorkToolsSelect']).iloc[1:,]
programm_2017 = survey_2017.dropna(subset = ['WorkToolsSelect'])


# In[ ]:


python_2019 = programm_2019[(programm_2019['WorkToolsSelect'].str.contains('Python')) & (~programm_2019['WorkToolsSelect'].str.contains('R')) & (~programm_2019['WorkToolsSelect'].str.contains('SQL'))]
R_2019 = programm_2019[(~programm_2019['WorkToolsSelect'].str.contains('Python')) & (programm_2019['WorkToolsSelect'].str.contains('R')) & (~programm_2019['WorkToolsSelect'].str.contains('SQL'))]
SQL_2019 = programm_2019[(~programm_2019['WorkToolsSelect'].str.contains('Python')) & (~programm_2019['WorkToolsSelect'].str.contains('R'))& (programm_2019['WorkToolsSelect'].str.contains('SQL'))]
python_R_2019 = programm_2019[(programm_2019['WorkToolsSelect'].str.contains('Python')) & (programm_2019['WorkToolsSelect'].str.contains('R'))& (~programm_2019['WorkToolsSelect'].str.contains('SQL'))]
python_SQL_2019 = programm_2019[(programm_2019['WorkToolsSelect'].str.contains('Python')) & (~programm_2019['WorkToolsSelect'].str.contains('R'))& (programm_2019['WorkToolsSelect'].str.contains('SQL'))]
R_SQL_2019 = programm_2019[(~programm_2019['WorkToolsSelect'].str.contains('Python')) & (programm_2019['WorkToolsSelect'].str.contains('R'))& (programm_2019['WorkToolsSelect'].str.contains('SQL'))]
ALL_2019 = programm_2019[(programm_2019['WorkToolsSelect'].str.contains('Python')) & (programm_2019['WorkToolsSelect'].str.contains('R'))& (programm_2019['WorkToolsSelect'].str.contains('SQL'))]
OTHER_2019 = programm_2019[(~programm_2019['WorkToolsSelect'].str.contains('Python')) & (~programm_2019['WorkToolsSelect'].str.contains('R'))& (~programm_2019['WorkToolsSelect'].str.contains('SQL'))]

python_2018 = programm_2018[(programm_2018['WorkToolsSelect'].str.contains('Python')) & (~programm_2018['WorkToolsSelect'].str.contains('R')) & (~programm_2018['WorkToolsSelect'].str.contains('SQL'))]
R_2018 = programm_2018[(~programm_2018['WorkToolsSelect'].str.contains('Python')) & (programm_2018['WorkToolsSelect'].str.contains('R')) & (~programm_2018['WorkToolsSelect'].str.contains('SQL'))]
SQL_2018 = programm_2018[(~programm_2018['WorkToolsSelect'].str.contains('Python')) & (~programm_2018['WorkToolsSelect'].str.contains('R'))& (programm_2018['WorkToolsSelect'].str.contains('SQL'))]
python_R_2018 = programm_2018[(programm_2018['WorkToolsSelect'].str.contains('Python')) & (programm_2018['WorkToolsSelect'].str.contains('R'))& (~programm_2018['WorkToolsSelect'].str.contains('SQL'))]
python_SQL_2018 = programm_2018[(programm_2018['WorkToolsSelect'].str.contains('Python')) & (~programm_2018['WorkToolsSelect'].str.contains('R'))& (programm_2018['WorkToolsSelect'].str.contains('SQL'))]
R_SQL_2018 = programm_2018[(~programm_2018['WorkToolsSelect'].str.contains('Python')) & (programm_2018['WorkToolsSelect'].str.contains('R'))& (programm_2018['WorkToolsSelect'].str.contains('SQL'))]
ALL_2018 = programm_2018[(programm_2018['WorkToolsSelect'].str.contains('Python')) & (programm_2018['WorkToolsSelect'].str.contains('R'))& (programm_2018['WorkToolsSelect'].str.contains('SQL'))]
OTHER_2018 = programm_2018[(~programm_2018['WorkToolsSelect'].str.contains('Python')) & (~programm_2018['WorkToolsSelect'].str.contains('R'))& (~programm_2018['WorkToolsSelect'].str.contains('SQL'))]

python_2017 = programm_2017[(programm_2017['WorkToolsSelect'].str.contains('Python')) & (~programm_2017['WorkToolsSelect'].str.contains('R')) & (~programm_2017['WorkToolsSelect'].str.contains('SQL'))]
R_2017 = programm_2017[(~programm_2017['WorkToolsSelect'].str.contains('Python')) & (programm_2017['WorkToolsSelect'].str.contains('R')) & (~programm_2017['WorkToolsSelect'].str.contains('SQL'))]
SQL_2017 = programm_2017[(~programm_2017['WorkToolsSelect'].str.contains('Python')) & (~programm_2017['WorkToolsSelect'].str.contains('R'))& (programm_2017['WorkToolsSelect'].str.contains('SQL'))]
python_R_2017 = programm_2017[(programm_2017['WorkToolsSelect'].str.contains('Python')) & (programm_2017['WorkToolsSelect'].str.contains('R'))& (~programm_2017['WorkToolsSelect'].str.contains('SQL'))]
python_SQL_2017 = programm_2017[(programm_2017['WorkToolsSelect'].str.contains('Python')) & (~programm_2017['WorkToolsSelect'].str.contains('R'))& (programm_2017['WorkToolsSelect'].str.contains('SQL'))]
R_SQL_2017 = programm_2017[(~programm_2017['WorkToolsSelect'].str.contains('Python')) & (programm_2017['WorkToolsSelect'].str.contains('R'))& (programm_2017['WorkToolsSelect'].str.contains('SQL'))]
ALL_2017 = programm_2017[(programm_2017['WorkToolsSelect'].str.contains('Python')) & (programm_2017['WorkToolsSelect'].str.contains('R'))& (programm_2017['WorkToolsSelect'].str.contains('SQL'))]
OTHER_2017 = programm_2017[(~programm_2017['WorkToolsSelect'].str.contains('Python')) & (~programm_2017['WorkToolsSelect'].str.contains('R'))& (~programm_2017['WorkToolsSelect'].str.contains('SQL'))]


# In[ ]:


f, ax = plt.subplots(1,3, figsize = (18,8))

venn3(subsets = (round(python_2019.shape[0]/len(programm_2019),2) , 
                 round(R_2019.shape[0]/len(programm_2019),2), 
                 round(SQL_2019.shape[0]/len(programm_2019),2) ,
                 round(python_R_2019.shape[0]/len(programm_2019),2) ,
                 round(python_SQL_2019.shape[0]/len(programm_2019),2) ,
                 round(R_SQL_2019.shape[0]/len(programm_2019) ,2) ,
                 round(ALL_2019.shape[0]/len(programm_2019) ,2)),
       set_labels = ('Python','R','SQL','Python + R','Python + SQL' , 'R + SQL', 'ALL' ) , ax = ax[0] )
ax[0].set_title('Percent of Users 2019')

venn3(subsets = (round(python_2018.shape[0]/len(programm_2018),2) , 
                 round(R_2018.shape[0]/len(programm_2018),2), 
                 round(SQL_2018.shape[0]/len(programm_2018),2) ,
                 round(python_R_2018.shape[0]/len(programm_2018),2) ,
                 round(python_SQL_2018.shape[0]/len(programm_2018),2) ,
                 round(R_SQL_2018.shape[0]/len(programm_2018) ,2) ,
                 round(ALL_2018.shape[0]/len(programm_2018) ,2)),
       set_labels = ('Python','R','SQL','Python + R','Python + SQL' , 'R + SQL', 'ALL') , ax = ax[1])
ax[1].set_title('Percent of Users 2018')

venn3(subsets = (round(python_2017.shape[0]/len(programm_2017),2) , 
                 round(R_2017.shape[0]/len(programm_2017),2), 
                 round(SQL_2017.shape[0]/len(programm_2017),2) ,
                 round(python_R_2017.shape[0]/len(programm_2017),2) ,
                 round(python_SQL_2017.shape[0]/len(programm_2017),2) ,
                 round(R_SQL_2017.shape[0]/len(programm_2017) ,2) ,
                 round(ALL_2017.shape[0]/len(programm_2017) ,2)),
       set_labels = ('Python','R','SQL','Python + R','Python + SQL' , 'R + SQL', 'ALL') , ax = ax[2])
ax[2].set_title('Percent of Users 2017')
plt.show()

print('2019 OTHER Percentage : ' + str(round(OTHER_2019.shape[0]/len(programm_2019) ,2)))
print('2018 OTHER Percentage : ' + str(round(OTHER_2018.shape[0]/len(programm_2018) ,2)))
print('2017 OTHER Percentage : ' + str(round(OTHER_2017.shape[0]/len(programm_2017) ,2)))


# Compared to 2017, the percentage of people who enjoy all three languages will decrease in 2018 and 2019.
# Strangely, the proportion of people using R & SQL decreases over time, while the proportion of people using Python & SQL increases.
# This is the first language that Python is highly recommended for beginners and is also the first language to use when learning many data science basics. I'm using SQL, R, and Python. Because each language has its advantages and disadvantages, it is recommended to use all three.
# Based on the CNN article, if you look at the strengths and weaknesses of language..

# **1. Learning curve (R)**
# * To use python numpy, pandas, including matplotlib to learn a lot of data. But r in built-in graphics and basic matrix type by default.
# 
# **2. Available libraries (no winning)**
# * Python packages index (pypi), with a package of more than 183,000 cran have a package of more than 12,000 are (compreged rarchive network).
# 
# **3. Machine Running (Python)**
# * Python's tremendous growth in recent years has been partly affected by the rise in machine learning and artificial intelligence. Python offers many finely tuned libraries for image recognition, such as AlexNet, but R versions are also easy to develop.
# 
# **4. Statistical accuracy (R)**
# *  Matloff pointed out that machine learning experts who advocate Python sometimes do not understand the statistical problems involved. R, on the other hand, was written by statisticians, it added.
# 
# **5. Parallel operation (no win)**
# *  The basic versions of R and Python have weak support for multi-core operations. Python's multiprocessing package is not a good solution to other issues, nor is R's parallel package.
# 
# **6. Uniformity of Language (Python)**
# *  Python won't be much of a mess if the version changes. However, R is changing into two different languages under the influence of RStudio. (R, Tidyverse)

# ## Recommend Programming

# In[ ]:


f, ax = plt.subplots(1,2, figsize = (18,8))

# 2019-Q19 or 2018-Q18 : What programming language would you recommend an aspiring data scientist to learn first? - Selected Choice 
survey_2019.iloc[1:,].dropna(subset =['Q19'])['Q19'].value_counts(normalize = True, ascending = True).plot.barh(width = 0.9, color =sns.color_palette('inferno_r',15) ,ax = ax[0])
ax[0].set_title('Recommend Programming Tool 2019', size = 15)
survey_2018.iloc[1:,].dropna(subset =['Q18'])['Q18'].value_counts(normalize = True, ascending = True).plot.barh(width = 0.9, color =sns.color_palette('inferno_r',15) ,ax = ax[1])
ax[1].set_title('Recommend Programming Tool 2018', size = 15)
plt.show()


# We can see that Python is the first language to recommend for the data scientist's introduction.
# Many people are recommending Python for the first entrance and it is as powerful as it is, so I think we can boost our commitment to the Ben Diagram above.

# ## Programming Used By Professionals

# In[ ]:


#2019
python_2019_1 = python_2019.copy()
r_2019_1 = R_2019.copy()
sql_2019_1 = SQL_2019.copy()

python_2019_1['WorkToolsSelect_1'] = 'Python'
r_2019_1['WorkToolsSelect_1']='R'
sql_2019_1['WorkToolsSelect_1']='SQL'

python_r_sql_2019 = pd.concat([python_2019_1,r_2019_1,sql_2019_1]).rename(columns = {'Q5' : 'CurrentJobTitleSelect'})
python_r_sql_2019 = python_r_sql_2019['WorkToolsSelect_1'].groupby(python_r_sql_2019['CurrentJobTitleSelect']).value_counts(normalize = True).rename ('Prop').reset_index()


#2018
python_2018_1 = python_2018.copy()
r_2018_1 = R_2018.copy()
sql_2018_1 = SQL_2018.copy()

python_2018_1['WorkToolsSelect_1'] = 'Python'
r_2018_1['WorkToolsSelect_1']='R'
sql_2018_1['WorkToolsSelect_1']='SQL'

python_r_sql_2018 = pd.concat([python_2018_1,r_2018_1,sql_2018_1]).rename(columns = {'Q6' : 'CurrentJobTitleSelect'})
python_r_sql_2018 = python_r_sql_2018['WorkToolsSelect_1'].groupby(python_r_sql_2018['CurrentJobTitleSelect']).value_counts(normalize = True).rename ('Prop').reset_index()

#2017
python_2017_1 = python_2017.copy()
r_2017_1 = R_2017.copy()
sql_2017_1 = SQL_2017.copy()

python_2017_1['WorkToolsSelect_1'] = 'Python'
r_2017_1['WorkToolsSelect_1']='R'
sql_2017_1['WorkToolsSelect_1']='SQL'

python_r_sql_2017 = pd.concat([python_2017_1,r_2017_1,sql_2017_1])
python_r_sql_2017 = python_r_sql_2017['WorkToolsSelect_1'].groupby(python_r_sql_2017['CurrentJobTitleSelect']).value_counts(normalize = True).rename ('Prop').reset_index()


#plot
f, ax = plt.subplots(1,3, figsize = (25,15))
python_r_sql_2019.pivot('CurrentJobTitleSelect','WorkToolsSelect_1','Prop').plot.barh(width=0.8, ax = ax[0])
ax[0].set_title('Percent Programmin Per Current Job 2019')
ax[0].axhspan(1.5,2.5 ,facecolor='Orange', alpha=0.25) # hilight space
ax[0].axhspan(9.5,10.5 ,facecolor='Orange', alpha=0.25) # hilight space

python_r_sql_2018.pivot('CurrentJobTitleSelect','WorkToolsSelect_1','Prop').plot.barh(width=0.8, ax = ax[1])
ax[1].set_title('Percent Programmin Per Current Job 2018')
ax[1].set_ylabel('')
ax[1].axhspan(3.5,4.5 ,facecolor='Orange', alpha=0.25) # hilight space
ax[1].axhspan(18.5,19.5 ,facecolor='Orange', alpha=0.25) # hilight space

python_r_sql_2017.pivot('CurrentJobTitleSelect','WorkToolsSelect_1','Prop').plot.barh(width=0.8, ax = ax[2])
ax[2].set_title('Percent Programmin Per Current Job 2017')
ax[2].set_ylabel('')
ax[2].axhspan(2.5,3.5 ,facecolor='Orange', alpha=0.25) # hilight space
ax[2].axhspan(14.5,15.5 ,facecolor='Orange', alpha=0.25) # hilight space

plt.subplots_adjust(wspace=1.0)
plt.show()


# Python is rapidly increasing in 2018 and 2019 compared to 2017.
# Noticeably, data analysts seemed to prefer R and SQL to Python in 2017, but by 2018 and 2019, Python was very dominant.
# Statisticians can also see a rise of approximately 20% in 2018 and 2019, although Python use was less than 10% in 2017.
# It seems that those involved in the analysis also prefer Python to R.

# # About Data Scientist Answer

# I'd like to ask some questions to the data scientist. 
# * "What is important part of your role at work?"
# * "What platform do you get the data science news from?"
# * "How long have you used the code?"
# * "How many years have you used the machine learning methodology?"
# * "Do you have to spend a lot of money to be a data scientist?" 
# 
# These questions stem from my curiosity to become a data scientist.

# In[ ]:


ds_data = survey_2019[survey_2019['Q5'] == 'Data Scientist'].iloc[1:,]


# ## First. *What is important part of your role at work ?*

# In[ ]:


question = 'Q9' # Select any activities that make up an important part of your role at work: (Select all that apply) 
columns_list_2019 = qa_multiple_2019[question]
ds_data ['Activites'] = ds_data[columns_list_2019].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)
activities_2019 = ds_data['Activites'].str.split('?')

at_2019 = []
for i in activities_2019.dropna():
    at_2019.extend(i)

plt.figure(figsize = (15,10))
pd.Series(at_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('inferno_r',15))
plt.title('Data Scientist Importance Role', size = 15)

plt.show()


# Many data scientists understand their data and play a role in influencing product or business decisions. Many data scientists understand their data and play a role in influencing product or business decisions. I think there are many roles for data scientists because they need to learn and know knowledge about various fields such as math, computer, and business.
# 
# In other words, you need to build up domain knowledge of the data you see.

# ## Second. *What platform do you get the data science news from?*

# In[ ]:


question = 'Q12' # Select any activities that make up an important part of your role at work: (Select all that apply) 
columns_list_2019 = qa_multiple_2019[question]
ds_data ['Media_Source'] = ds_data[columns_list_2019].apply(lambda x: "?".join(x.dropna().astype(str)), axis=1).replace('',np.nan)
media_2019 = ds_data['Media_Source'].str.split('?')

md_2019 = []
for i in media_2019.dropna():
    md_2019.extend(i)

plt.figure(figsize = (15,10))
pd.Series(md_2019).value_counts(normalize=True)[:10].sort_values(ascending=True).plot.barh(width = 0.9, color  = sns.color_palette('inferno',15))
plt.title('DataScientist Media Source', size = 15)

plt.show()


# Data scientists answered a lot of news from Blog or Kagle.
# In fact, there are Discuss channels in Kaggle with many questions and answers, and I am also going to try to get into Kaggle with this competition. haha

# ## Third. *How long have you used the code?*

# In[ ]:


plt.figure(figsize=(10,8))
code_2019 = ds_data['Q15'].value_counts(normalize = True).rename ('Prop').plot.pie(autopct='%1.1f%%',explode=[0.1,0,0,0,0,0,0], shadow=True,)
plt.title('DataScientist Code Time', size = 15)
plt.show()


# The data scientists said they usually typed the code for three to five years.
# If you ask the surrounding analysts and scientists, they say, "The foundation of the code begins in two years."
# **I don't think it's wrong...OTL**

# ## Fourth. *How many years have you used the machine learning methods?*

# In[ ]:


plt.figure(figsize=(10,8))
code_2019 = ds_data['Q23'].value_counts(normalize = True).rename ('Prop').plot.pie(autopct='%1.1f%%',explode=[0.2,0.1,0,0,0,0,0,0], shadow=True,)
plt.title('DataScientist Code Time', size = 15)
plt.show()


# ## Fifth. *Do you have to spend a lot of money to be a data scientist?*

# In[ ]:


money_2019 = ds_data['Q11'].groupby(ds_data['year']).value_counts(normalize = True).rename ('Prop').sort_values(ascending=False).reset_index()
# plot
plt.figure(figsize=(25,15))
sns.barplot('Prop','Q11', data=money_2019 , palette=sns.color_palette('viridis',15))
plt.title('DataScientist Spent Money', size = 15)
plt.show()


# Different people may have different answers, **but the most answers are $0!!**
# This may be because many people interact with other data scientists through published information such as Kaggle or Blog.

# # Conclusion

# Through this analysis, we have learned some facts.
# 
# ** Simple Data Analytis** 
# 
# 1) Most of the respondents came from the United States, followed by India. The United States also had the largest number of data scientists after India.
# 
# 2) Most of the respondents are 20-35 years old, indicating that data science is quite popular with young people.
# 
# 3) Most of the respondents were students.It is thought that younger friends will learn and be interested in data science in the future.
# 
# 4) Many respondents are learning about data science using Mooc lectures like Coursera and Kaggle.
# 
# **  Machine Laerning ** 
# 
# 1) It is good to learn (Boasting techniques and Deep Learning).
# 
# 2) I recommend Python for the first time. Plus, I think SQL will be good. R is also recommended for convenience in statistical analysis if opportunity arises.
# 
# 3) To become a data scientist, increase your domain knowledge of that data.
# 
# 4) Let's listen to and learn from various data scientists through blogs or Kaggle
# 
# 5) From now on, let's practice code and machine learning methodology.

# It's not enough yet, but thank you for watching. Please give us a lot of feedback. Also, please follow me from Kaggle. **I want to learn from you.**

# In[ ]:




