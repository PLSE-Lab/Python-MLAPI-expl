#!/usr/bin/env python
# coding: utf-8

# # Comparison of the state of Data Science and Machine Learning in India and USA

# In this notebook,the state of Data Science and Machine Learning in the world is analysed and a etailed comparison is made between the two leading countries - India and USA

# In[ ]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# read the data
mcr =pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv',encoding='ISO-8859-1')
mcr.head()


# In[ ]:


mcr.info()


# In[ ]:


cols_to_drop = mcr[mcr.columns[pd.Series(mcr.columns).str.contains('OTHER_TEXT')]].columns.tolist()
mcr.drop(cols_to_drop,axis=1,inplace=True)


# ## EDA

# In[ ]:


# function to plot barchart
from textwrap import wrap
def plot_bar(col,title,label='',rot=0):                        
    plt.figure(figsize=(16,6), dpi=80, facecolor='w')
    df1 = mcr[col][1:].value_counts()
    ax=round(100*df1/sum(df1),2).sort_values(ascending=False).plot.bar(stacked=True)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ["\n".join(wrap(l,15)) for l in labels]
    plt.title(title)
    if label=='': label=mcr[col][0]
    plt.xlabel(label)
    plt.grid(b=True,which='major', color='#666666', linestyle='-',alpha=0.5)
    ax.set(ylabel='Percentage')
    ax.set_xticklabels(labels,rotation=rot)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.tight_layout()


# ## Understanding the distribution of all the respondents

# ### Country

# In[ ]:


plot_bar('Q3','Country',label='Country',rot=90)


# Approximately one-fourth of the Kagglers in the world come from India and around 16% from USA

# ### Age Distribution

# In[ ]:


plot_bar('Q1','Age',label='Age groups')


# More than 50% of the respondents are within 40 years of age
# 

# ### Education

# In[ ]:


plot_bar('Q4','Education')


# Around 44 % of participants have a Master's degree

# ### Gender distribution

# In[ ]:


plot_bar('Q2','Gender')


# Around 80 % of respondents are Male 

# ### Current Role

# In[ ]:


plot_bar('Q5','Current role')


# Data scientist and Students together make 40% of the respodents approximately

# ### Size of the Company

# In[ ]:


plot_bar('Q6','Size of the company')


# Most of the respondents work in companies with less than 50 employees

# ### Number of individuals responsible for Data Science wworkload in the organisation

# In[ ]:


plot_bar('Q7','#Data Scientist')


# It is surprising to note that nearly 64 % of the respondents worked in organisation that had less than 10 individuals responsible for data science workloads

# ### Incorporation of Machine Leraning methods in business

# In[ ]:


plot_bar('Q8','Incorporate machine learning in business?')


# While 1/5 th of the surveyed population population work in industries that incorporat3 well established ML methods into their business, 1/2 of the population have just begun exploring or using them in the business and other's haven't started yet.

# ### Yearly Compensation

# In[ ]:


plot_bar('Q10','Yearly Compensation',rot=60)


# Though nearly 3% of respondents draw a whooping salary of > $ 200,000 per year, majority of the respondents (30%) receive < $7500 as yearly compensation

# #### Money Spent on ML and CC platforms in last 5 years

# In[ ]:


plot_bar('Q11','Money Spent on ML and CC platforms in last 5 years',label='Amount in $' )


# Around 50% of the respondents work in organization that has spent less than 100 USD for ML and CC platforms in the last 5 years. This is not surprising, as most of the respondents work in organizations that has just begun using ML methods

# ### Coding experience

# In[ ]:


plot_bar('Q15','Coding experience',label='# of Years' )


# 56 % of respondents have a coding experience of 2 years or less. 

# ### TPU Usage

# In[ ]:


plot_bar('Q22','Used TPU?' )


# A tensor processing unit (TPU) is an AI accelerator application-specific integrated circuit (ASIC) developed by Google specifically for neural network machine learning(Source : Wikipedia).
# 80 % of the respondents have never used Tensor Processing Unit. 

# ### Experience in using Machine Learning methods

# In[ ]:


plot_bar('Q23','Machine Learning experience',label='# of Years' )


# Approximately 62 % of the population have  less than 2  years of experience in Machine Learning

# # Comparison between India & USA

# In[ ]:


def plot_bar_stacked(col,title,label='',rot=0):                        
                     
    plt.figure(figsize=(16,6), dpi=80, facecolor='w')
    #ax=sns.countplot(x=mcr[col][1:], data=mcr, order = mcr[col][1:].value_counts().index)
    df1 = mcr.groupby(col)['Q1'].value_counts()
    ax=round(100*df1/sum(df1),2).unstack().plot.bar(stacked=True)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ["\n".join(wrap(l,15)) for l in labels]
    plt.title(title)
    if label=='': label=mcr[col][0]
    plt.xlabel(label)
    plt.grid(b=True,which='major', color='#666666', linestyle='-',alpha=0.5)
    ax.set(ylabel='Percentage')
    ax.set_xticklabels(labels,rotation=rot)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.tight_layout()

      


# In[ ]:


India=mcr.loc[mcr['Q3']=='India']
USA=mcr.loc[mcr['Q3']=='United States of America']
India.head()


# In[ ]:


# Gender
plt.figure(figsize=(16,6), dpi=80, facecolor='w')
plt.subplot(121)
val = India['Q2'].value_counts()
label=list(India['Q2'].unique())
my_circle = plt.Circle((0, 0), 0.7, color = 'white')
plt.pie(val,autopct = '%.2f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.legend(label,loc=2)
plt.title('Gender Distribution in India')
plt.subplot(122)
val = USA['Q2'].value_counts()
label=list(USA['Q2'].unique())
my_circle = plt.Circle((0, 0), 0.7, color = 'white')
plt.pie(val,autopct = '%.2f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.legend(label,loc=2)
plt.title('Gender Distribution in USA')


# There is a huge difference in gender distribution between India and USA. While the M:F is 82:16 in India, it is the opposite in USA where M:F = 20:77.

# In[ ]:


# function to concat two dfs and plot
def func_plt_2df(col,tit,rot=0):
    fig, ax = plt.subplots(figsize=(16,10))
    ax= pd.concat({'India': 100*India[col].value_counts()/len(India[col]), 'USA': 100*USA[col].value_counts()/len(USA[col])}, axis=1).plot.bar(ax=ax)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ["\n".join(wrap(l,15)) for l in labels]
    plt.grid(b=True,which='major', color='#666666', linestyle='-',alpha=0.5)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax.set(xlabel=mcr[col][0])
    ax.set_xticklabels(labels,rotation=rot)
    ax.set(ylabel='Percentage')
    plt.title(tit)


# In[ ]:


#Age groups
func_plt_2df('Q1','Age')


# A highly contratsing trend is oberved in the age group of respondents. While 75% of the respondents in India are within 30 years of age,only 35% respondents from USA are within 30 years of age. 
# Age distribution is thus highly skewed in India than in the USA.

# In[ ]:


# Current Roles
func_plt_2df('Q5','Current role')


# * Students, Software engineers and Data Scientists form around 65 % of Kaggle population in India.
# * Proportion of people occupying higher ranks like Reserch Scientist,Datascientist, Data Analyst and Product Manager are higher in United States thank in India.
# * These results suggest India still has to go along way in developing its man power 
# 
# 

# In[ ]:


# Education
func_plt_2df('Q4','Education')


# 48 % of Indian respondents have a Bachelors degree while 49% of Americans have a MAster's degree and 20 %  have a Doctoral Degree

# In[ ]:


# Company Size 
func_plt_2df('Q6','Company Size ')


# Cleraly Americans working in large sized companies are higher than their counterparts.

# In[ ]:


# individuals responsible for data science workloads
func_plt_2df('Q7','Number of individuals responsible for data science workloads')


# Number of individuals responsible for data science workloads are higher in USA than in India

# In[ ]:


# Incorporate ML metohods
func_plt_2df('Q8','Incorporate ML metohods')


# While India has just started exploring/using ML methods, these are well established and used for a longer time by Americans than Indians for making Business Decisions.

# In[ ]:


func_plt_2df('Q10','Yearly Compensation',rot=90)


# There exists a huge difference in yearly compensation received by Indians  adn Americans.
# * 25 % of Indiand receive < 7500 USD per year
# *  Around 40 % of Americans receive a compensation 100,000 USD per year
# 

# In[ ]:


func_plt_2df('Q11','Money spent on Machine Learning and Cloud Computing platforms at work')


# Again a stark contrast is observed in the money spent by an organisation of ML and CC products at work place. Most of the Indians work in Organisation that spend less than 1000 USD, while most of the Americans work in organizations spending more than 1000 USd on these products 

# In[ ]:


func_plt_2df('Q14','Primary tool use for Analysing Data')


# In[ ]:


func_plt_2df('Q15','Coding experience')


# Most of the Indian respondents have a coding experience of 2 years or less, while the opposite is true for Americans

# In[ ]:


func_plt_2df('Q19','Programming Language recommended to aspiring Data Scientist')


# Python is the most preferred language in both the countries.
# Americans recommend R and SQL also

# In[ ]:


func_plt_2df('Q22','TPU Usage')


# TPU is not very popular in both the countries

# In[ ]:


func_plt_2df('Q23','Experience in using Machine Learning methods')


# American repsondents are more experienced in using Machine Learning methods than Indian counterparts

# In[ ]:


# func to plot questions with subdividsion
def func_grp_plt(column,tit):  
    cols = [col for col in mcr if col.startswith(column)]
    col=mcr[cols].iloc[0]
    col_key = [x.split('-')[2] for x in col]
        
    col_val=India[cols].count(axis=0)
    d=  {k:v for (k,v) in zip(col_key,col_val)}
    lists = sorted(d.items(),key=lambda x: x[1], reverse=True) # sorted by key, return a list of tuples
    x, y = zip(*lists)
    fig, ax = plt.subplots(figsize=(8,8))
    plt.barh(x, y)
    plt.xticks(rotation=0)   
    plt.xlabel('Frequency')
    plt.title(tit + ' - India')
    
    col_key = [x.split('-')[2] for x in col]
    col_val=USA[cols].count(axis=0)
    d=  {k:v for (k,v) in zip(col_key,col_val)}
    lists = sorted(d.items(),key=lambda x: x[1], reverse=True) # sorted by key, return a list of tuples
    x, y = zip(*lists)
    fig, ax = plt.subplots(figsize=(8,8))
    plt.barh(x, y)
    plt.xticks(rotation=0) 
    plt.xlabel('Frequency')
    plt.title(tit + ' - USA')


    


# In[ ]:


func_grp_plt('Q9','Imporatnt role at work')


# In[ ]:


func_grp_plt('Q13','Platforms in which data science courses were completed')


# In[ ]:


cols_Q14 = [col for col in mcr if col.startswith('Q14')]
col=mcr[cols_Q14].iloc[0]


# In[ ]:


col_key = [x.split('-')[1] for x in col]


# In[ ]:


col_val=India[cols_Q14].count(axis=0)
d=  {k:v for (k,v) in zip(col_key,col_val)}
lists = sorted(d.items(),key=lambda x: x[1], reverse=True) # sorted by key, return a list of tuples
x, y = zip(*lists)
fig, ax = plt.subplots(figsize=(8,8))
plt.barh(x, y)


# In[ ]:


col_val=USA[cols_Q14].count(axis=0)
d1=  {k:v for (k,v) in zip(col_key,col_val)}
lists = sorted(d.items(),key=lambda x: x[1], reverse=True) # sorted by key, return a list of tuples
x1, y1 = zip(*lists)
fig, ax = plt.subplots(figsize=(8,8))
#sns.barplot(x,y)
plt.barh(x, y)

plt.xticks(rotation=0)    


# In[ ]:


func_grp_plt('Q16','IDEs used regularly')


# In[ ]:


func_grp_plt('Q17','Hosted notebook products used regularly')


# In[ ]:


func_grp_plt('Q18','programming Languages used Regularly')


# In[ ]:


func_grp_plt('Q20','Data visulaisation libraries used regularly')


# In[ ]:


func_grp_plt('Q21','Specialised hardware used regularly')


# In[ ]:


func_grp_plt('Q24','ML algorithms used regularly')


# In[ ]:


func_grp_plt('Q25','ML tools used regularly')


# In[ ]:


func_grp_plt('Q26','Computer vision methods used regularly')


# In[ ]:


func_grp_plt('Q27','NLP methods used regularly')


# In[ ]:


func_grp_plt('Q28','Machine learning Frameworks used regularly')


# In[ ]:


func_grp_plt('Q29','Cloud computing platforms used regularly')


# In[ ]:


func_grp_plt('Q30','Specific cloud computing products used regularly')


# In[ ]:


func_grp_plt('Q31','Specific Big data/ analytics products used regularly')


# In[ ]:


func_grp_plt('Q32','Machine Learning Products used regularly')


# In[ ]:


func_grp_plt('Q33','Automated learning tools used regularly')


# In[ ]:


func_grp_plt('Q34','Relational database products used regularly')


# Thus the most commonly use Ml tools, programming languages are same in both the countries

# The comparison results suggests that the investmentts made for ML products are more in USA than in India. The field of Data Science is more developed in USA and India should start embracing ML methods more to take Business decisions 
