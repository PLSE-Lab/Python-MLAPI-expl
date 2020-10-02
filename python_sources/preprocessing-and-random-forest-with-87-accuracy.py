#!/usr/bin/env python
# coding: utf-8

#  This is Python 2 Notebook working on **OSMI Mental Health in Tech Survey 2016**
#  Main focus is on preprocessing the data. The simple thumbrule of Machine Learning is Quality of Data and then even a simple algorithm works  quite well.
#  Cells are explained whenever required. 
#  Inside some cell you will find commented statements that are done for analysis of problem for better understanding at any instance.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('Set2')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


response=pd.read_csv('../input/mental-heath-in-tech-2016_20161114.csv')


# In[ ]:


#response.describe()#(include='all')
#response.info()
#response.get_index()
features=(list(response))
print(len(features))
response.shape
# for val in features:
#     print(val)
#     print('\n')


# On analysing Dataset, I found that out of 63 questions asked in survey and answered by 1433 people. 13 questions are there which are not answered by 50% of participants. So I drop that questions

# In[ ]:


count=0
major_not_ans=[]
for col in features:
    #print(col,sum(pd.isnull(response[col])))
    if(sum(pd.isnull(response[col]))>721):
        count=count+1
        major_not_ans.append(col)
        #response.drop([col],axis=1,inplace=True)
print(len(major_not_ans))


# In[ ]:


response.drop([i for i in major_not_ans],axis=1,inplace=True)


# Its not necessary that every participant lives or works in US territory.
# 'Why or why not?','Why or why not?.1' these answers are explaining the reason why or why not you would share  your health problem in interview. Well what matters to us is " Will a person share his mental health problem?(Y/N)" and not its explaination.
# 'Which of the following best describes your work position?' this feature contain very different roles. So, for the sake of simplicity, I drop this feature as well.

# In[ ]:


extra_feature=['What US state or territory do you work in?','Why or why not?','Why or why not?.1','What US state or territory do you live in?']
one_more=['Which of the following best describes your work position?']


# In[ ]:


response.drop([i for i in extra_feature],axis=1,inplace=True)


# In[ ]:


response.drop(one_more,axis=1,inplace=True)


# In[ ]:


response.drop(['What country do you live in?'],axis=1,inplace=True)


# There are 287 people who are self-employed and they didn't answer various questions leading to NaN value. I find that these survey questions are not for self-employed people. After I removed the NaN records of these 287 participants, I found only Non-self-employed persons are left. So, this feature will not add any value to the model. 4th cell below this will explain you the reason why many features have 287 NaN values and it is observed that all these are self employed.

# In[ ]:


response.drop(['Are you self-employed?'],axis=1,inplace=True)


# In[ ]:


response.drop(['Do you have previous employers?'],axis=1,inplace=True)


# In[ ]:


real_features=(list(response))
print(len(real_features))


# In[ ]:


# REMOVING SELF EMPLOYED 287 NAN
count=0
for index,col in enumerate(real_features):
    idx=response.index[response[col].isnull()]
    #response.drop(idx,inplace=True)
    if(len(idx)==287):
        #print(index,idx)
        k=idx
        count+=1
#print(count)    12
#print(k)
#idx=response.index[response['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].isnull()]
response.drop(k,inplace=True)
response.shape


# After dropping above(287) rows there are 131 common NaN values. So they are also dropped.

# In[ ]:


# Group of people not answering the same questions So removing them
count=0
for index,col in enumerate(real_features):
    idx=response.index[response[col].isnull()]
    #response.drop(idx,inplace=True)
    if(len(idx)==131):
        #print(index,idx)
        k=idx
        count+=1
# print(count) 11
# print(k)
#idx=response.index[response['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].isnull()]
response.drop(k,inplace=True)
response.shape


# Since it is a survey, evryone has different way of saying same thing e.g. Male,M,MALE,man......
# the below cell solves this.

# In[ ]:


# clean the genders by grouping the genders into 3 categories: Female, Male, Genderqueer/Other
response['What is your gender?'] = response['What is your gender?'].replace([
    'male', 'Male ', 'M', 'm', 'man', 'Cis male',
    'Male.', 'Male (cis)', 'Man', 'Sex is male',
    'cis male', 'Malr', 'Dude', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
    'mail', 'M|', 'male ', 'Cis Male', 'Male (trans, FtM)',
    'cisdude', 'cis man', 'MALE'], 'Male')
response['What is your gender?'] = response['What is your gender?'].replace([
    'female', 'I identify as female.', 'female ',
    'Female assigned at birth ', 'F', 'Woman', 'fm', 'f',
    'Cis female', 'Transitioned, M2F', 'Female or Multi-Gender Femme',
    'Female ', 'woman', 'female/woman', 'Cisgender Female', 
    'mtf', 'fem', 'Female (props for making this a freeform field, though)',
    ' Female', 'Cis-woman', 'AFAB', 'Transgender woman',
    'Cis female '], 'Female')
response['What is your gender?'] = response['What is your gender?'].replace([
    'Bigender', 'non-binary,', 'Genderfluid (born female)',
    'Other/Transfeminine', 'Androgynous', 'male 9:1 female, roughly',
    'nb masculine', 'genderqueer', 'Human', 'Genderfluid',
    'Enby', 'genderqueer woman', 'Queer', 'Agender', 'Fluid',
    'Genderflux demi-girl', 'female-bodied; no feelings about gender',
    'non-binary', 'Male/genderqueer', 'Nonbinary', 'Other', 'none of your business',
    'Unicorn', 'human', 'Genderqueer'], 'Genderqueer/Other')


# In[ ]:


# clean the ages by replacing the weird ages with the mean age
# min age was 3 and max was 393 
response.loc[(response['What is your age?'] > 90), 'What is your age?'] = 34
response.loc[(response['What is your age?'] < 10), 'What is your age?'] = 34
# replace the one null with Male, the mode gender, so we don't have to drop the row
response['What is your gender?'] = response['What is your gender?'].replace(np.NaN, 'Male')
response['What is your gender?'].value_counts()


# By Now, only one feature is having NaN values. So filling them.

# In[ ]:


response.fillna(method='ffill',inplace=True)
response.fillna(value='Yes', limit=1,inplace=True)
#print response['Do you know the options for mental health care available under your employer-provided coverage?']
g = sns.countplot(x='Do you know the options for mental health care available under your employer-provided coverage?',data=response)


# Now analysing all the 42 real questions and what possible values they take.

# In[ ]:


for index,val in enumerate(real_features):
    p=response[val].unique()
    print(index,val)
    print(p)
    print('\n')
    #print(response[val].isnull().sum())
    #print("\n")


# Machine Learning Model wants numeric representation of each feature. So our main task now is to convert all the answers to a numeric representation like "Yes->1", "No->0" and so on.

# In[ ]:


country=(response[real_features[40]].unique())
num_rep=[]    #numeric representation with there index
alp_rep=[]    # name of country
#print(type(country))
for index,val in enumerate(country):
    num_rep.append(index)
    alp_rep.append(val)
print(len(num_rep),len(alp_rep))
response[real_features[40]].replace(alp_rep, num_rep,inplace=True)  # Replacing country name with the index


# In[ ]:


response['Does your employer provide mental health benefits as part of healthcare coverage?'] = response['Does your employer provide mental health benefits as part of healthcare coverage?'].replace('Not eligible for coverage / N/A','No')
g = sns.countplot(x='Does your employer provide mental health benefits as part of healthcare coverage?',data=response)


# In[ ]:


response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('1-5', 5)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('6-25',25)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('26-100', 100)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('100-500',500)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('500-1000',1000)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace('More than 1000',5000)
response['How many employees does your company or organization have?'] = response['How many employees does your company or organization have?'].replace(np.nan,5)
# Replacing NaN values wd range 26-100


# In[ ]:


# USED DIRECTLY
# #response[real_features[7]]
# g = sns.countplot(x=response[real_features[7]],data=response)
g = sns.countplot(x=response[real_features[14]],data=response)


# Here all possible answers are mapped to a numeric value, considering usniversal 0 for "No", 1 for "Yes", I don't know/May be/ I am not sure all represent same, so replaced them by 2 and so on. 

# In[ ]:


numeric = {real_features[2]:     {'No':0, 'Yes':1, "I don't know":2},
                real_features[3]: {'Yes':1, 'I am not sure':2, 'No':0},
                 real_features[4]:{'No':0, 'Yes':1, "I don't know":2},
                  real_features[5]:{'No':0, 'Yes':1, "I don't know":2},
                   real_features[6]:{"I don't know":2, 'Yes':1, 'No':0},
                    real_features[7]:{'Very easy':0 ,'Somewhat easy':1, 'Neither easy nor difficult':2,'Very difficult':-1,
 'Somewhat difficult':-2, "I don't know":2}, #### MODIFIED DIRECTLY
                real_features[8]:{'No':0, 'Maybe':2, 'Yes':1},
                real_features[9]:{'No':0, 'Maybe':2, 'Yes':1},
                 real_features[10]:{'No':0, 'Maybe':2, 'Yes':1},
                 real_features[11]:{'No':0, 'Maybe':2, 'Yes':1},
                 real_features[12]:{"I don't know":2, 'Yes':1, 'No':0},
                 real_features[13]:{'No':0, 'Yes':1},
                 real_features[14]:{'No, none did':0, 'Yes, they all did':1, "I don't know":2, 'Some did':3},
                 real_features[15]:{'N/A (not currently aware)':0, 'I was aware of some':1,
 'Yes, I was aware of all of them':1, 'No, I only became aware later':0},  ### MODIFIED DIRECTLY
                real_features[16]:{"I don't know":2, 'None did':0, 'Some did':3,'Yes, they all did':1},
                real_features[17]:{'None did':0, 'Some did':3, 'Yes, they all did':1},
                real_features[18]:{"I don't know":2, 'Yes, always':1, 'Sometimes':3, 'No':0},
                real_features[19]:{'Some of them':3, 'None of them':0, "I don't know":2, 'Yes, all of them':1},
                real_features[20]:{'None of them':0, 'Some of them':3, 'Yes, all of them':1},
                real_features[21]:{'Some of my previous employers':3, 'No, at none of my previous employers':0,
 'Yes, at all of my previous employers':1},
                real_features[22]:{'Some of my previous employers':3, "I don't know":2, 'No, at none of my previous employers':0,
 'Yes, at all of my previous employers':1},
                real_features[23]:{"I don't know":2, 'Some did':3, 'None did':0, 'Yes, they all did':1},
                real_features[24]:{'None of them':0, 'Some of them':3, 'Yes, all of them':1},
                real_features[25]:{'Maybe':2, 'Yes':1, 'No':0},
                real_features[26]:{'Maybe':2, 'Yes':1, 'No':0},
                real_features[27]:{'Maybe':2, "No, I don't think it would":0, 'Yes, I think it would':1,
 'No, it has not':0, 'Yes, it has':1},  ### MODIFIED DIRECTLY
                real_features[28]:{"No, I don't think they would":0, 'Maybe':2, 'Yes, they do':1,'Yes, I think they would':1, 'No, they do not':0},  ## MODIFIED DIRECTLY
                real_features[29]:{'Somewhat open':1, 'Not applicable to me (I do not have a mental illness)':4,
 'Very open':2, 'Not open at all':-2 ,'Neutral':0, 'Somewhat not open':-1}, ### MODIFIED DIRECTLY
                real_features[30]:{'No':0, 'Maybe/Not sure':2, 'Yes, I experienced':1, 'Yes, I observed':1},
                real_features[31]:{'No':0, 'Yes':1, "I don't know":2},
                real_features[32]:{'Yes':1, 'Maybe':2, 'No':0},
                real_features[33]:{'Yes':1, 'Maybe':2, 'No':0},
                real_features[34]:{'Yes':1, 'No':0},
                real_features[36]:{'Not applicable to me':4, 'Rarely':0, 'Sometimes':3, 'Never':0, 'Often':1},
                real_features[37]:{'Not applicable to me':4, 'Sometimes':3, 'Often':1, 'Rarely':0, 'Never':0},
                real_features[39]:{'Male':1, 'Female':0, 'Genderqueer/Other':2},
                real_features[41]:{'Sometimes':3, 'Never':0, 'Always':1}
          }


# In[ ]:


response.replace(numeric, inplace=True)
response.head()


# Now, the data is processed and all in numric representation. So, we are ready to start Machine Learning. Since, we processed it to best of our understanding I directly applied Random Forest on it. Although, I believe that we can process it more.
# # Random Forest
# 
# 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


target=response['Have you ever sought treatment for a mental health issue from a mental health professional?']
response.drop(['Have you ever sought treatment for a mental health issue from a mental health professional?'],axis=1,inplace=True)


# In[ ]:


#target.unique()
#response.shape
X=response


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,target, test_size=0.2, random_state=0)


# In[ ]:


clf = RandomForestClassifier(max_depth=14, random_state=0)
clf.fit(X_train, y_train)


# In[ ]:


print(clf.score(X_test,y_test))


# The below cells are  extra showing the different trials with data. Its always best to play with data as much as you can. I hope, this notebook will be helpful.**Please Upvote the notebook if you find it useful.**

# In[ ]:


# NOT USED
# count=0
# # print(match,unmatch)
# k=response['Have you been diagnosed with a mental health condition by a medical professional?'].tolist()
# p=response['Have you ever sought treatment for a mental health issue from a mental health professional?'].tolist()

# print(k[0],p[0])
# for i,val in enumerate(k):
#     if(val==p[i]):
#         count+=1
# print(count)


# In[ ]:


#g = sns.countplot(x='How many employees does your company or organization have?',data=response)


# In[ ]:


# for i in _notans169:
#     print(response.iloc[i].values)
#     print('\n')


# In[ ]:


# match=0
# unmatch=0
# np.where(response['What country do you live in?']==response['What country do you work in?'], match+=1,unmatch+=1)
#count=0
# print(match,unmatch)
# k=response['What country do you live in?'].tolist()
# p=response['What country do you work in?'].tolist()

# #print(k)
# for i,val in enumerate(k):
#     if(val==p[i]):
#         count+=1
# print(count)
#1407 people works on same place where they live
#print(response[:]['What country do you work in?'],response[:]['What country do you live in?'])


# In[ ]:


# USEFUL
# count=0
# for i,col in enumerate(real_features):
#     idx=response.index[response[col].isnull()]
#     #response.drop(idx,inplace=True)
#     if(len(idx)==169):
#         print(col)
#         print(i,idx)
#         _notans169=idx
#         count+=1
# print(count,_notans169)
#idx=response.index[response['Does your employer offer resources to learn more about mental health concerns and options for seeking help?'].isnull()]
#response.drop(idx,inplace=True)


# In[ ]:


# USEFUL
# count=0
# tcount=0
# selfnotans=[]
# for index,col in enumerate(features):
#     idx=response.index[response[col].isnull()]
#     tcount+=1
#     #response.drop(idx,inplace=True)
#     if(len(idx)==287):
#         #print(features[index])
#         selfempl=idx
#         selfnotans.append(features[index])
#         #print('\n')
#         #print(index,idx)
#         count+=1
# #print(count,tcount)

# print((selfempl[:]))


# In[ ]:


# for i in selfempl:
#     print(response.loc[i,'Do you currently have a mental health disorder?'])

