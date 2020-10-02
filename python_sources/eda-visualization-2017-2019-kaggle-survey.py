#!/usr/bin/env python
# coding: utf-8

# # Simple Analysis
# 
# 
# ![](https://miro.medium.com/max/1336/1*GZrTyTz0OKMbxnO5Trhcew.png)
# 
# 
# The trend of Machine Learning & Data Science is one of our hottest topic. At this notebook I'll make some simple EDA and Data Visualization.

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

qo_19 = pd.read_csv('../input/kaggle-survey-2019/questions_only.csv')
mcr_19 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')
otr_19 = pd.read_csv('../input/kaggle-survey-2019/other_text_responses.csv')
ss_19 = pd.read_csv('../input/kaggle-survey-2019/survey_schema.csv')
ffr_18 = pd.read_csv("../input/kaggle-survey-2018/freeFormResponses.csv")
mcr_18 = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv")
ss_18 = pd.read_csv("../input/kaggle-survey-2018/SurveySchema.csv")
ffr_17 = pd.read_csv("../input/kaggle-survey-2017/freeformResponses.csv")
mcr_17 = pd.read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding='latin-1')
ss_17 = pd.read_csv("../input/kaggle-survey-2017/schema.csv")


# In[ ]:


print('     2019         2018         2017')
print('mcr',mcr_19.shape,mcr_18.shape,mcr_17.shape)
print('ss ',ss_19.shape,'   ',ss_18.shape,'   ',ss_17.shape)


# # Summary of 35 Questions 2019
# 
# #Q0 : Total time to answer all the Questions
# #Q1 : Age
# #Q2 : Gender
# #Q3 : Country 
# #Q4 : Educational Background
# #Q5 : Job Name
# #Q6 : Size of Workplace
# #Q7 : Number of Colleague Data Scientists
# #Q8 : If ML is in company's business
# #Q9 : Main activity at work
# #Q10 : Yearly compensation
# #Q11 : Money spent on ML/Cloud computing
# #Q12 : Favorite media sources (person/platform)
# #Q13 : Where to begin DS
# #Q14 : Tool to EDA
# #Q15 : How long to write DS code
# #Q16 : Which IDE
# #Q17 : Which notebook product to use
# #Q18 : What language for DS
# #Q19 : What language to recommend for beginners
# #Q20 : Tool to visualize
# #Q21 : Which Hardware
# #Q22 : If used TPU
# #Q23 : How long to use ML
# #Q24 : ML algorithms mainly using
# #Q25 : Category of ML mainly using
# #Q26 : Category of CV model mainly using
# #Q27 : NLP methods mainly using
# #Q28 : ML frameworks mainly using
# #Q29 : Cloud computing platforms mainly using
# #Q30 : Cloud computing products mainly using
# #Q31 : Analytics products mainly using
# #Q32 : ML products mainly using
# #Q33 : AutoML tools mainly using
# #Q34 : Database products mainly using

# # Let's start from individual information

# # Q1 : Age

# In[ ]:


age_x = ['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70+']
mean = []
for ran in age_x:
    if ran == '70+':
        min_ran = 70
        mean.append(75)
    else:
        min_ran, max_ran = int(ran.split('-')[0]), int(ran.split('-')[1])
        mean.append((min_ran+max_ran)/2)

ave = {}
year = 2017
col = 2
age_y = []
param = mcr_17.iloc[1:,col].values
for ran in age_x:
    if ran == '70+':
        min_ran = 70
        age_y.append(sum((min_ran <= param)))
    else:
        min_ran, max_ran = int(ran.split('-')[0]), int(ran.split('-')[1])
        age_y.append(sum((min_ran <= param)&(param <= max_ran)))
ave[year] = sum(np.array(mean)*np.array(age_y))/sum(age_y)
fig = plt.figure(figsize = (14,9))
plt.bar(age_x,age_y,color = 'g')
plt.xlabel('age',fontsize=40)
plt.ylabel('number of people',fontsize=40)
plt.title(str(year),fontsize=40)
plt.show()

year = 2018
col = 3

param = mcr_18.iloc[1:,col].value_counts()
age_y = []
for ran in age_x:
    if ran == '70+':
        age_y.append(param['70-79']+param['80+'])
    else:
        age_y.append(param[ran])
ave[year] = sum(np.array(mean)*np.array(age_y))/sum(age_y)
fig = plt.figure(figsize = (14,9))
plt.bar(age_x,age_y)
plt.xlabel('age',fontsize=40)
plt.ylabel('number of people',fontsize=40)
plt.title(str(year),fontsize=40)
plt.show()

year = 2019
col = 1
param = mcr_19.iloc[1:,col].value_counts()
age_y = []
for ran in age_x:
    age_y.append(param[ran])
ave[year] = sum(np.array(mean)*np.array(age_y))/sum(age_y)
fig = plt.figure(figsize = (14,9))
plt.bar(age_x,age_y,color = 'orange')
plt.xlabel('age',fontsize=40)
plt.ylabel('number of people',fontsize=40)
plt.title(str(year),fontsize=40)
plt.show()


# ## It seems younger generations are gradually increasing in this 3 years

# # Change of Average Age

# In[ ]:


fig, ax = plt.subplots(figsize=(14,9))
plt.title('Change of average age',fontsize=40)
plt.xlabel('Year',fontsize=40)
plt.ylabel('Average age',fontsize=40)
ax.plot(list(ave.keys()),list(ave.values()),linewidth=10)
ax.set_ylim(0, 50)
plt.xticks(rotation=30,fontsize=30)
plt.yticks(fontsize=30)
plt.show()


# # Q2 : Gender 
# (This time I checked only male/female because of data duplicate)

# ## Ratio of Male/Female for each year

# In[ ]:


r_men = {2017:0,2018:0,2019:0}
for year in range(2017,2020):
    if year == 2019:
        r_men[year] = (sum((mcr_19['Q2'][1:]=='Male'))/sum((mcr_19['Q2'][1:]=='Male')|(mcr_19['Q2'][1:]=='Female')))*100
    if year == 2018:
        r_men[year] = (sum((mcr_18['Q1'][1:]=='Male'))/sum((mcr_18['Q1'][1:]=='Male')|(mcr_18['Q1'][1:]=='Female')))*100
    if year == 2017:
        r_men[year] = (sum((mcr_17['GenderSelect'][1:]=='Male'))/sum((mcr_17['GenderSelect'][1:]=='Male')|(mcr_17['GenderSelect'][1:]=='Female')))*100
r_women = {2017:0,2018:0,2019:0}
for year in range(2017,2020):
    r_women[year] = 100 - r_men[year]

for year in range(2017,2020):
    label = ['Male', 'Female']
    fig, ax = plt.subplots(figsize=(5,5))
    plt.title(year,fontsize=20)
    xs = [r_men[year],r_women[year]]
    plt.pie(xs,labels = label)
    plt.show()


# ## About 80% of respondents are Male and only 20 are Female
# #### I was so surprised that the ratio of Male/Female is always almost the same for each year.

# # Q3 : Mother Country

# ## Top5 kaggler's mother country of each year

# In[ ]:


year = 2017
for i in range(3):
    if year == 2017:
        col = 1
        idx = 0
        param = mcr_17.iloc[idx:,col].value_counts()
    if year == 2018:
        col = 4
        idx = 1
        param = mcr_18.iloc[idx:,col].value_counts()
    else:
        col = 4
        idx = 0
        param = mcr_19.iloc[idx:,col].value_counts()
    x_5 = list(param.index)[:5]
    y_5 = list(param.values)[:5]
    fig = plt.figure(figsize = (14,9))
    plt.plot(x_5,y_5,linewidth=8)
    plt.title(str(year),fontsize=40)
    plt.xlabel('Country',fontsize=40)
    plt.ylabel('Number of People',fontsize=40)
    plt.xticks(rotation=30,fontsize=30)
    plt.yticks(fontsize=30)
    plt.show()
    year += 1


# ### People from India & US are domaining a lot of part in respondents

# # Q4 : Educational Background (From Here 2018 & 2019)

# In[ ]:


x_7 = [
    'Master',
    'Bachelor',
    'Doctor',
    'Retired Univ',
    'Professional',
    'Not answer',
    'High school'
]
param = mcr_18.iloc[1:,5].value_counts()
y_7 = list(param.values)
fig = plt.figure(figsize = (14,9))
plt.xticks(rotation=30,fontsize=40)
plt.yticks(fontsize=40)
plt.plot(x_7,np.array(y_7)*(100/sum(y_7)),linewidth=8,label = '2018')
param = mcr_19.iloc[1:,5].value_counts()
y_7 = list(param.values)
plt.plot(x_7,np.array(y_7)*(100/sum(y_7)),linewidth=8,label = '2019')
plt.title(str(2018),fontsize=40)
plt.xlabel('Final Education',fontsize=40)
plt.ylabel('%',fontsize=40)
plt.legend()
plt.show()


# ### Master, Bachelor, Doctor degrees are the most domaining parts of respondents

# # Top10 places of  currently kagglers live in

# In[ ]:


param = mcr_18['Q3'][1:].value_counts()[:10]
x_7, y_7 = list(param.index), list(param.values)
x_7[0], x_7[7] = 'US', 'UK'
fig = plt.figure(figsize = (14,9))
plt.xticks(rotation=35,fontsize=35)
plt.yticks(fontsize=40)
plt.xlabel('Top10 Country',fontsize=40)
plt.ylabel('Number of People',fontsize=40)
#plt.plot(x_7,y_7,linewidth=8,label = '2018',color='g')
plt.bar(x_7,y_7,linewidth=8,label = '2018',color='g')
plt.title(str(2018),fontsize=40)
plt.show()
param = mcr_19['Q3'][1:].value_counts()[:10]
x_7, y_7 = list(param.index), list(param.values)
x_7[1], x_7[8] = 'US', 'UK'
fig = plt.figure(figsize = (14,9))
plt.xticks(rotation=35,fontsize=35)
plt.yticks(fontsize=40)
plt.xlabel('Top10 Country',fontsize=40)
plt.ylabel('Number of People',fontsize=40)
#plt.plot(x_7,y_7,linewidth=8,label = '2019',color='g')
plt.bar(x_7,y_7,linewidth=8,label = '2019')
plt.title(str(2019),fontsize=40)
plt.show()


# ### US, India are the quite top of each year based on these plots

# # Top10 jobs of kagglers

# In[ ]:


param = mcr_18['Q6'].value_counts()[:10]
x_7, y_7 = list(param.index), list(param.values)
fig = plt.figure(figsize = (14,9))
plt.xticks(rotation=30,fontsize=25)
plt.yticks(fontsize=40)
plt.xlabel('Country',fontsize=40)
plt.ylabel('Number of People',fontsize=40)
plt.plot(x_7,y_7,linewidth=8,label = '2018',color='orange')
plt.title(str(2018))
plt.show()
param = mcr_19['Q5'].value_counts()[:10]
x_7, y_7 = list(param.index), list(param.values)
fig = plt.figure(figsize = (14,9))
plt.xticks(rotation=30,fontsize=25)
plt.yticks(fontsize=40)
plt.xlabel('Country',fontsize=40)
plt.ylabel('Number of People',fontsize=40)
plt.plot(x_7,y_7,linewidth=8,label = '2019',color='orange')
plt.title(str(2019))
plt.show()


# ### Data Scientists & Students are the most domaining part of respondents, I think that's why Kaggle is one of the best platform for employing DS Human Resources.
# ### and Software Engineer is the next major part.

# # Q5 : Kaggler's job trend for each country (2019)

# In[ ]:


param = mcr_19['Q3'][1:].value_counts()[:10]
x_7, y_7 = list(param.index), list(param.values)
fig, ax = plt.subplots(figsize = (14,9))
btm = np.zeros(len(x_7))
for job in mcr_19['Q5'][1:].value_counts().index:
    bottom_param = []
    for j in x_7:
        name = j
        if name == 'US':
            name = 'United States of America'
        if name == 'UK':
            name = 'United Kingdom of Great Britain and Northern Ireland'
        bottom_param.append(sum((mcr_19['Q3']==name)&(mcr_19['Q5']==job)))
    bottom_param = np.array(bottom_param)
    ax.bar(x_7, bottom_param,bottom = btm,label = job)
    plt.xticks(rotation=30,fontsize=30)
    plt.yticks(fontsize=40)
    btm += bottom_param
plt.xlabel('Number of People',fontsize=40)
plt.ylabel('Country',fontsize=40)
plt.legend(prop={'size': 18})
plt.plot()


# ### There are lots of students in each country, espessially India, but Data Scientists are the most part totally.

# # Q6 : Size of Workplace

# In[ ]:


nums = ['0-49', '50-249', '250-999', '1000-9999', '>10000']
keys =['0-49 employees','50-249 employees','250-999 employees','1000-9,999 employees','> 10,000 employees']
ys = []
for i in range(len(nums)):
    key = keys[i]
    ys.append(mcr_19['Q6'].value_counts()[key])
fig, ax = plt.subplots(figsize = (14,9))
plt.plot(nums,ys, linewidth = 10)
plt.title('Size of Workplace',fontsize=40)
plt.xticks(rotation=30,fontsize=30)
plt.yticks(fontsize=40)
plt.ylabel('Number of People',fontsize=40)
plt.xlabel('Size of Workplace',fontsize=40)
plt.show()


# ### It seems there are lots of people working at small company like startup, and next big company(maybe it's just because members of company are lot.)

# # Q7 : Number of colleage Data Scientists

# In[ ]:


xs = ['0','1-2','3-4','5-9','10-14','15-19','20+']
ys = []
for ran in xs:
    ys.append(mcr_19['Q7'][1:].value_counts()[ran])
fig, ax = plt.subplots(figsize = (14,9))
plt.plot(xs,ys, linewidth = 10,color = 'brown')
plt.title('Number of colleage Data Scientists',fontsize=40)
plt.xticks(rotation=30,fontsize=30)
plt.yticks(fontsize=40)
plt.ylabel('Number of People',fontsize=40)
plt.xlabel('Colleage Data Scientists',fontsize=40)
plt.show()


# # Q8 : If ML is in Company's business 

# In[ ]:


xs = list(mcr_18['Q10'][1:].value_counts().index)
ys = list(mcr_18['Q10'][1:].value_counts().values)
year = 2018
#ys = []
#for x in xs:
#    ys.append(mcr_18['Q10'][1:].value_counts()[x])
#plt.plot(xs,ys, linewidth = 10,color = 'green')
fig, ax = plt.subplots(figsize=(5,5))
plt.pie(ys,labels = xs)
plt.title(str(year),fontsize = 40)
plt.show()
xs = list(mcr_19['Q8'][1:].value_counts().index)
ys = list(mcr_19['Q8'][1:].value_counts().values)
year = 2019
fig, ax = plt.subplots(figsize=(5,5))
plt.pie(ys,labels = xs)
plt.title(str(year),fontsize = 40)
plt.show()


# # Q9 : Main activity at work

# In[ ]:


xs = []
ys = []
for i in range(1,9):
    xs.append(mcr_19['Q9_Part_'+str(i)].value_counts().index[1].split('-')[-1])
    ys.append(mcr_19['Q9_Part_'+str(i)].value_counts().values[0])
xs = ['Analyse for business decisions','Make ML for new fields','Other','Experiments to improve ML','Make Data Infrastructure','SoTA Research','Make ML for product','No ML/DS at Work']
fig, ax = plt.subplots(figsize = (14,9))
plt.bar(xs,ys)
plt.title('Main Activity',fontsize = 40)
plt.xticks(rotation=40,fontsize=30)
plt.yticks(fontsize=40)
plt.ylabel('Number of People',fontsize=40)
plt.xlabel('Size of Workplace',fontsize=40)
plt.show()


# # Q10 : Yearly Income

# In[ ]:


year = 2018
cols = {}
for i in mcr_18['Q9'][1:].value_counts().index:
    if i == '$0-999':
        cols[i] = 500
    else:
        if i == '500,000+':
            cols[i] = 500000
        else:
            if i == 'I do not wish to disclose my approximate yearly compensation':
                cols[i] = np.nan
            else:
                cols[i] = (int(i.split('-')[0].replace(',',''))+int(i.split('-')[1].replace(',','')))/2
print("median of respondents' income {} : $".format(year),int(mcr_18['Q9'][1:].copy().replace(cols).median()))
print("mean of respondents' income {}: $".format(year),int(mcr_18['Q9'][1:].copy().replace(cols).mean()))
idx = np.argsort(mcr_18['Q9'][1:].copy().replace(cols).value_counts().index)
xs = list(mcr_18['Q9'][1:].copy().replace(cols).value_counts().index[idx])
ys = list(mcr_18['Q9'][1:].copy().replace(cols).value_counts().values[idx])
fig, ax = plt.subplots(figsize = (14,9))
plt.plot(xs,ys, linewidth = 10)
plt.title(str(year),fontsize = 40)
plt.xticks(rotation=40,fontsize=30)
plt.yticks(fontsize=40)
plt.ylabel('Number of People',fontsize=40)
plt.xlabel('$',fontsize=40)
plt.show()
year = 2019
cols = {}
for i in mcr_19['Q10'][1:].value_counts().index:
    if i == '$0-999':
        cols[i] = 500
    else:
        if i == '> $500,000':
            cols[i] = 500000
        else:
            cols[i] = (int(i.split('-')[0].replace(',',''))+int(i.split('-')[1].replace(',','')))/2
back_cols = {}
back_cols = {cols[i]:i for i in cols}
print("median of respondents' income {}: $".format(year),int(mcr_19['Q10'][1:].copy().replace(cols).median()))
print("mean of respondents' income {}: $".format(year),int(mcr_19['Q10'][1:].copy().replace(cols).mean()))
idx = np.argsort(mcr_19['Q10'][1:].copy().replace(cols).value_counts().index)
xs = np.array(mcr_19['Q10'][1:].copy().replace(cols).value_counts().index)[idx]
ys = np.array(mcr_19['Q10'][1:].copy().replace(cols).value_counts().values)[idx]
fig, ax = plt.subplots(figsize = (14,9))
plt.plot(xs,ys, linewidth = 10)
plt.title(str(year),fontsize = 40)
plt.xticks(rotation=40,fontsize=30)
plt.yticks(fontsize=40)
plt.ylabel('Number of People',fontsize=40)
plt.xlabel('$',fontsize=40)
plt.show()


# # In 2019 there are much more rich people in respondents than 2018

# ![](https://i.ytimg.com/vi/fcxm5ZKkMKQ/maxresdefault.jpg)
