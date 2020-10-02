#!/usr/bin/env python
# coding: utf-8

# # Let's explore the ML and DS survey results for the 2017 and 2018 surveys 
# ## In this notebook, we will explore both free form and multiple choice questions survey done in 2017 and 2018. We will specifically explore following:
# * Where do the survey takers live? Was their a sigificant change in the kaggle user's demographics from 2017 to 2018?
# * What is the gender distribution of the participants?
# * What is the age distribution of the participants?
# * What is the change in compensation of Data scientists?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# 
# **Convert data into pandas format for both freeform responses and multiplechoice responses**
# 
# 

# In[ ]:


ffr_pd_2017 = pd.read_csv("../input/kaggle-survey-2017/freeformResponses.csv")
ffr_pd_2017.head(3)


# In[ ]:


ffr_pd_2018 = pd.read_csv("../input/kaggle-survey-2018/freeFormResponses.csv")
ffr_pd_2018.head(3)


# In[ ]:


mcr_pd_2017 = pd.read_csv("../input//kaggle-survey-2017/multipleChoiceResponses.csv",encoding='latin-1')
mcr_pd_2017.head(3)


# In[ ]:


mcr_pd_2018 = pd.read_csv("../input//kaggle-survey-2018/multipleChoiceResponses.csv")
mcr_pd_2018.head(3)


# ## Let's analyze results of Multiple choice Response survey first

# In[ ]:


print("Total MCR Responses in 2017 = ", mcr_pd_2017.shape[0])
print("Total MCR Responses in 2018 = ", mcr_pd_2018.shape[0])


# ## More candidates participated in 2018 survey
# 

# ## What is the distribution of survey particiant's gender?
# 
# 

# In[ ]:


gender_distr_2017 = mcr_pd_2017['GenderSelect'].value_counts()
labels = ['Male', 'Female', 'Unknown']
sizes_2017 = [gender_distr_2017[0], gender_distr_2017[1], (gender_distr_2017[2] + gender_distr_2017[3])]
colors = ['blue', 'coral', 'yellowgreen']
explode = (0.1, 0.1, 0) 


gender_distr_2018 = mcr_pd_2018['Q1'].value_counts()
labels = ['Male', 'Female', 'Unknown']
sizes_2018 = [gender_distr_2018[0], gender_distr_2018[1], (gender_distr_2018[2] + gender_distr_2018[3])]
colors = ['cornflowerblue', 'coral', 'yellowgreen']
explode = (0.1, 0.1, 0) 
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(25,12))
plt.subplot(1, 2, 1)

patches, texts, pct = plt.pie(sizes_2017, explode = explode, colors = colors, 
                         shadow = True, autopct='%1.1f%%', startangle = 90)
plt.axis('equal')
plt.tight_layout()
plt.title("Gender Distribution in 2017")


plt.subplot(1, 2, 2)
patches, texts, pct = plt.pie(sizes_2018, explode = explode, colors = colors, 
                         shadow = True, autopct='%1.1f%%', startangle = 90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.title("Gender Distribution in 2018")
plt.show()


# ## Overwhelming number of Kaggle user are Males in both 2017 and 2018.
# ### The result shows the Data Science Diversity Gap and forces us to think that How diverse will a lucrative, growing field like data science be in the future? Will it end up like computer science today (not very diverse).  There are few proposed reasons for this gender gap. 
# *  A lack of STEM education for women early on in life.
# *  Lack of mentorship for women in data science, and human resouces rules and regulations not catching up to gender balance policies, to name a few.
# * Males outnumbered Females by a significant marging. almost 3 to 1 ratio.  

# ## Let's find top 10 countries

# In[ ]:


country_distr_2017 = mcr_pd_2017['Country'].value_counts()
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(25,12))
labels = list(country_distr_2017.keys())[:10]
sizes = country_distr_2017.values[:10]
plt.subplot(1, 2, 1)
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.45,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.title("Top 10 Countries in 2017")

country_distr_2018 = mcr_pd_2018['Q3'].value_counts()
if 'In which country do you currently reside?' in country_distr_2018: 
    del country_distr_2018['In which country do you currently reside?']

plt.subplot(1, 2, 2)
labels = list(country_distr_2018.keys())[:10]
sizes = country_distr_2018.values[:10]

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        
#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.40,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.title("Top 10 Countries in 2018")
plt.show();


# ## ** Significant number of Kaggle users are based in USA, India and China. However, it is noteworthy that there was significant bump in survey takes from China from 2017 to 2018.**
# ### Given the fact that survey user count increased significantly in 2018, It would be fair to assume that Kaggle use has exploded in China in 2018, while usage in USA with relatively flat or US users did not have motivation to take the survey in 2018.

# # Age group of survey participants
# 

# In[ ]:


# Calculate Age range for 2017
conditions = [(mcr_pd_2017['Age']<18),
(mcr_pd_2017['Age']>=18) & (mcr_pd_2017['Age']<=21),
(mcr_pd_2017['Age']>=22) & (mcr_pd_2017['Age']<=24),
(mcr_pd_2017['Age']>=25) & (mcr_pd_2017['Age']<=29),
(mcr_pd_2017['Age']>=30) & (mcr_pd_2017['Age']<=34),
(mcr_pd_2017['Age']>=35) & (mcr_pd_2017['Age']<=39),
(mcr_pd_2017['Age']>=40) & (mcr_pd_2017['Age']<=44),
(mcr_pd_2017['Age']>=44) & (mcr_pd_2017['Age']<=49),
(mcr_pd_2017['Age']>=50) & (mcr_pd_2017['Age']<=54),
(mcr_pd_2017['Age']>=55) & (mcr_pd_2017['Age']<=59),
(mcr_pd_2017['Age']>=60) & (mcr_pd_2017['Age']<=69),
(mcr_pd_2017['Age']>=70) & (mcr_pd_2017['Age']<=79),
(mcr_pd_2017['Age'] >= 80)]

choices = ['18-','18-21','22-24','25-29','30-34','35-39','40-44','44-49','50-54','55-59','60-69','70-79','80+']

mcr_pd_2017['age_range'] = np.select(conditions, choices, default='')


# ## Kagglers age group in2017, 2018

# In[ ]:


age_group_2017 = mcr_pd_2017['age_range'].value_counts()
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(25,12))
plt.subplot(1, 2, 1)
sns.barplot(y=age_group_2017.index, x=age_group_2017.values, alpha=0.6)
plt.title("Age Group - 2017",fontsize=20)
plt.xlabel("Number of participants", fontsize=16)
plt.ylabel("Age Range", fontsize=16)

age_group_2018 = mcr_pd_2018['Q2'].value_counts()
if 'What is your age (# years)?' in age_group_2018: 
    del age_group_2018['What is your age (# years)?']
keys, counts = np.unique(age_group_2018, return_counts=True)
pos = np.arange(len( age_group_2018.keys()))
plt.subplot(1, 2, 2)
sns.barplot(y=age_group_2018.index, x=age_group_2018.values, alpha=0.6)
plt.xlabel('Number of participants', fontsize=16)
plt.ylabel('Age Range', fontsize=16)
plt.title("Age Group - 2018",fontsize=20)
plt.show()


# ##  The majority of participants in surveys were from 25-29 age group. 22-24 age group was second most active group in 2018. This group is likely to be university students. A lot more students took survey in 2018 as compared to 2017.

# # Education level
# 

# In[ ]:


edu_distr_2017 = mcr_pd_2017['FormalEducation'].value_counts()
edu_distr_2018 = mcr_pd_2018['Q4'].value_counts()
if 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?' in edu_distr_2018: 
    del edu_distr_2018['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?']

# set width of bar
barWidth = 0.25
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(25, 18))

# set height of bar
bars1 = edu_distr_2017.values
bars2 = edu_distr_2018.values
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Make the plot
plt.bar(r1, bars1, color='orange', width=barWidth, edgecolor='white', label='2017')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='2018')
 
# Add xticks on the middle of the group bars
#plt.xlabel('Degree', fontweight='bold')
labels = [ '\n'.join(wrap(l, 20)) for l in list(edu_distr_2018.keys()) ]

plt.xticks([r + barWidth for r in range(len(bars1))], labels, rotation='vertical')
 
# Create legend & Show graphic
plt.legend()

plt.title("Education Level")
plt.show()


# ## More than 90% of Kaggle users hold Bachelors, Masters of Doctorate degree. The spread among these three degree remained consistent and there was proportionate increase in users.

# # Education in Females
# 

# In[ ]:


is_female_2017 =  mcr_pd_2017['GenderSelect'] == 'Female'
is_female_2018 =  mcr_pd_2018['Q1'] == 'Female'
mcr_pd_2017_f = mcr_pd_2017[is_female_2017]
mcr_pd_2018_f = mcr_pd_2018[is_female_2018]
#print(mcr_pd_2017_f.shape, mcr_pd_2018_f.shape )
edu_distr_2017 = mcr_pd_2017_f['FormalEducation'].value_counts()
edu_distr_2018 = mcr_pd_2018_f['Q4'].value_counts()
if 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?' in edu_distr_2018: 
    del edu_distr_2018['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?']
# set width of bar
barWidth = 0.25
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(25, 18))
# set height of bar
bars1 = edu_distr_2017.values
bars2 = edu_distr_2018.values
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Make the plot
plt.bar(r1, bars1, color='pink', width=barWidth, edgecolor='white', label='2017')
plt.bar(r2, bars2, color='orchid', width=barWidth, edgecolor='white', label='2018')
 
# Add xticks on the middle of the group bars
#plt.xlabel('Degree', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], labels, rotation=90)
 
# Create legend & Show graphic
plt.legend()
plt.title("Education Level in Females")
plt.show()


# ## The numbers females with Masters degree participating in survey in 2018 doubled compared to 2017.
# 

# # Compensation Analysis

# In[ ]:


mcr_pd_2017['CompensationAmount']=mcr_pd_2017['CompensationAmount'].str.replace(',','')
mcr_pd_2017['CompensationAmount']=mcr_pd_2017['CompensationAmount'].str.replace('-','')
rates=pd.read_csv('../input/kaggle-survey-2017/conversionRates.csv')
rates.drop('Unnamed: 0',axis=1,inplace=True)
salary=mcr_pd_2017[['CompensationAmount','CompensationCurrency','GenderSelect','Country','CurrentJobTitleSelect']].dropna()
salary=salary.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')
salary['Salary']=pd.to_numeric(salary['CompensationAmount'])*salary['exchangeRate']
print('Maximum Salary in 2017 is USD $',salary['Salary'].dropna().astype(int).max())
print('Minimum Salary in 2017 is USD $',salary['Salary'].dropna().astype(int).min())
print('Median Salary in 2017 is USD $',salary['Salary'].dropna().astype(int).median())


# In[ ]:


resp_coun=mcr_pd_2017['Country'].value_counts()[:10].to_frame()
f,ax=plt.subplots(1,2,figsize=(28,18))
sal_coun=salary.groupby('Country')['Salary'].median().sort_values(ascending=False)[:10].to_frame()
sns.barplot('Salary',sal_coun.index,data=sal_coun,palette='RdYlGn',ax=ax[0])
ax[0].axvline(salary['Salary'].median(),linestyle='dashed')
ax[0].set_title('Highest Salary Paying Countries in 2017')
ax[0].set_xlabel('')
max_coun=salary.groupby('Country')['Salary'].median().to_frame()
max_coun=max_coun[max_coun.index.isin(resp_coun.index)]
max_coun.sort_values(by='Salary',ascending=True).plot.barh(width=0.8,ax=ax[1],color=sns.color_palette('RdYlGn'))
ax[1].axvline(salary['Salary'].median(),linestyle='dashed')
ax[1].set_title('Compensation of Top 10 Respondent Countries in 2017')
ax[1].set_xlabel('')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()


# In[ ]:


mcr_pd_2018['Salary']=(mcr_pd_2018['Q9'].str.split("-", n = 1, expand = True)[1].str.replace(',',''))
mcr_pd_2018['Salary']=pd.to_numeric(mcr_pd_2018['Salary'])
salary=mcr_pd_2018[['Q3','Salary']].dropna()
compensation_2018 = salary['Salary'].value_counts()
salary.drop(salary.index[0], inplace=True)
print('Maximum Salary in 2018 is USD $',salary['Salary'].dropna().astype(int).max())
print('Minimum Salary in 2018 is USD $',salary['Salary'].dropna().astype(int).min())
print('Median Salary in 2018 is USD $',salary['Salary'].dropna().astype(int).median())
resp_coun=mcr_pd_2018['Q3'].value_counts()[:10].to_frame()
f,ax=plt.subplots(1,2,figsize=(28,18))
sal_coun=salary.groupby('Q3')['Salary'].median().sort_values(ascending=False)[:10].to_frame()
sns.barplot('Salary',sal_coun.index,data=sal_coun,palette='RdYlGn',ax=ax[0])
ax[0].axvline(salary['Salary'].median(),linestyle='dashed')
ax[0].set_title('Highest Salary Paying Countries in 2018')
ax[0].set_xlabel('')
ax[0].set_ylabel('Country')
max_coun=salary.groupby('Q3')['Salary'].median().to_frame()
max_coun=max_coun[max_coun.index.isin(resp_coun.index)]
max_coun.sort_values(by='Salary',ascending=True).plot.barh(width=0.8,ax=ax[1],color=sns.color_palette('RdYlGn'))
ax[1].axvline(salary['Salary'].median(),linestyle='dashed')
ax[1].set_title('Compensation of Top 10 Respondent Countries in 2018')
ax[1].set_xlabel('')
ax[1].set_ylabel('')
plt.subplots_adjust(wspace=0.8)
plt.show()


# ## Switzerland paid higher salary than USA in 2018, while Israel paid significantly more compensation in 2018 vs. 2017.

# # Job Titles

# In[ ]:


emp_status_2017 = mcr_pd_2017['CurrentJobTitleSelect'].value_counts()
emp_status_2018 = mcr_pd_2018['Q6'].value_counts()

def pltfn():
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    patches, texts, pct = plt.pie(emp_status_2017.values[:15], autopct='%1.1f%%', shadow = True, startangle = 90)
    plt.legend(patches, list(emp_status_2017.keys())[:15], bbox_to_anchor=(1, 1), loc="best")
    plt.axis('equal')
    plt.title("Titles in 2017", fontsize=20)

    plt.subplot(1, 2, 2)
    patches, texts, pct = plt.pie(emp_status_2018.values[:15], autopct='%1.1f%%', shadow = True, startangle = 90)
    plt.legend(patches, list(emp_status_2018.keys())[:15], bbox_to_anchor=(1, 1), loc="best")
    plt.axis('equal')
    plt.title("Titles in 2018", fontsize=20)
    plt.show()


# In[ ]:


import squarify
heatmap_colors = ["red","violet","blue", "grey", "orange", 
                     "pink", "dodgerblue", "sandybrown", "turquoise", 
                     "olive", "lightsalmon", "limegreen", "forestgreen", "dimgrey"]
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(35, 15))
plt.subplot(1, 2, 1)
squarify.plot(sizes=emp_status_2017.values[:15], 
              label=list(emp_status_2017.keys())[:15], 
              color= heatmap_colors,
              alpha=.4 )
plt.title("Titles in 2017", fontsize=30)
plt.axis('off')
plt.subplot(1, 2, 2)
squarify.plot(sizes=emp_status_2018.values[:15], 
              label=list(emp_status_2018.keys())[:15], 
              color=heatmap_colors,
              alpha=.4 )
plt.title("Titles in 2018", fontsize=30)
plt.axis('off')
plt.show()


# # Source of Information and Knowledge about DS
# 

# In[ ]:


edu_source_2017 = mcr_pd_2017['BlogsPodcastsNewslettersSelect'].value_counts()
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 12))
patches, texts, pct = plt.pie(edu_source_2017.values[:10], autopct='%1.1f%%', shadow = True, startangle = 90)
plt.legend(patches, list(edu_source_2017.keys())[:10], bbox_to_anchor=(1, 1), loc="upper left")
plt.axis('equal')
plt.title("Info Source in 2017", fontsize=20)
plt.show()


# ## The question was not asked in 2018 survey. In 2017, people relied heavily on blogs than videos for reliable information.

# # Primary tool at work
# 

# In[ ]:


tool_distr_2017 = mcr_pd_2017['WorkToolsSelect'].value_counts()
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 12))
patches, texts, pct = plt.pie(tool_distr_2017.values[:10], autopct='%1.1f%%', shadow = True, startangle = 90)
plt.legend(patches, list(tool_distr_2017.keys())[:10], bbox_to_anchor=(1, 1), loc="upper left")
plt.axis('equal')
#plt.tight_layout()
plt.title("Primary Tool used at Work in 2017",fontsize=20)
plt.show()


# In[ ]:


tool_distr_2018 = mcr_pd_2018['Q12_MULTIPLE_CHOICE'].value_counts()
if 'What is the primary tool that you use at work or school to analyze data? (include text response) - Selected Choice' in tool_distr_2018: 
    del tool_distr_2018['What is the primary tool that you use at work or school to analyze data? (include text response) - Selected Choice']
plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(12, 12))
patches, texts, pct = plt.pie(tool_distr_2018.values, autopct='%1.1f%%', shadow = True, startangle = 90)
plt.legend(patches, list(tool_distr_2018.keys()), bbox_to_anchor=(1, 1), loc="upper left")
plt.axis('equal')
plt.title("Primary Tool used at Work in 2018", fontsize=20)
plt.show()


# ## Questions regrading primary tool at work were posed slightly differently in two surveys. Still, it is abudantly clear that Python and R languages rule the datascience world.

# # Free text survey 
# 

# In[ ]:


print("Total MCR Responses = ", ffr_pd_2017.shape[0])
print("Total Columns = ", ffr_pd_2017.shape[1])
null_values = ffr_pd_2017.isnull().sum(axis = 0)
objects = null_values.keys().tolist()
y_pos = np.arange(len(objects))
performance = null_values.values
plt.figure(figsize=(20, 8))
plt.xticks(rotation=90)

plt.bar(objects, performance, align='center', width = 0.9, alpha=0.5, color='lime')
plt.axhline(y=ffr_pd_2017.shape[0],linewidth=2, color='k', linestyle='--')
plt.ylim(0, ffr_pd_2017.shape[0] + 5000)
plt.ylabel('Count')
plt.title(' Null Values in 2017', fontsize=20)

plt.show()


# In[ ]:


print("Total MCR Responses = ", ffr_pd_2018.shape[0])
print("Total Columns = ", ffr_pd_2018.shape[1])
null_values = ffr_pd_2018.isnull().sum(axis = 0)
objects = null_values.keys().tolist()
y_pos = np.arange(len(objects))
performance = null_values.values
plt.figure(figsize=(20, 8))
plt.xticks(rotation=90)

plt.bar(objects, performance, align='center', width = 0.9, alpha=0.5, color='lime')
plt.axhline(y=ffr_pd_2018.shape[0],linewidth=2, color='k', linestyle='--')
plt.ylim(0, ffr_pd_2018.shape[0] + 5000)
plt.ylabel('Count')
plt.title(' Null Values in 2018', fontsize=20)

plt.show()


# **Most columns appear to have Null values. The dashed line represents total responses and height represents number of NULL values for a given column. **
# 
# ** Let's explore these some of columns with lower NULL values**

# In[ ]:


q12_part4_distr = ffr_pd_2017['ImpactfulAlgorithmFreeForm'].value_counts()
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(12, 12))
patches, texts, pct = plt.pie(q12_part4_distr.values[:15], autopct='%1.1f%%', shadow = True, startangle = 90)
plt.legend(patches, list(q12_part4_distr.keys())[:15], bbox_to_anchor=(1, 1), loc="upper left")
plt.axis('equal')
#plt.tight_layout()
plt.title("Impact Algo. (free form text) 2017", fontsize=20)
plt.show()


# In[ ]:


q12_part4_distr = ffr_pd_2018['Q12_Part_4_TEXT'].value_counts()
#q34_other_distr = ffr_pd_2018['Q34_OTHER_TEXT'].value_counts()
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(12, 12))
patches, texts, pct = plt.pie(q12_part4_distr.values[:15], autopct='%1.1f%%', shadow = True, startangle = 90)
plt.legend(patches, list(q12_part4_distr.keys())[:15], bbox_to_anchor=(1, 1), loc="upper left")
plt.axis('equal')
plt.title("Primary Tool used at Work (free form text) 2018", fontsize=20)
plt.show()


# 
# ## **As expected, lot of variations of same options. Jupyter, RStudio, Python and R seems to be the favorite tools.**
# 
# ## Quality of results from free form results were generally poor with lot of empty/Null responses.
# 

# # References
# ## Python documentation for SNS and Matplotlib
# 
# # Conclusions
# 
# ##  Kaggle team should focus on increasing the female user base. It is significantly lower and showed no significant improvement from 2017 to 2018. It was encouraging to see that number of female survey takers with master's degree almost doubled from 2017 to 2018.
# 
# ## There was a significant change in demographics of  Kaggle users. More people from developing countries are using Kaggle. 
# 
# ## People with Master's or higher degree are the majority of the users. Simple tools  should be offered to make it easier for others to use this platform.
# 
# ## Students in the age group of 22-24 are enthusiastically adopting this platform. In a year, they became second largest age group of kaggle users.
# 
# ## Most people rely on blogs for knowledge. Kaggle should offer high quality education resources. 

# In[ ]:




