#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Business problem
# We want to Identify wheater the job posting given to us is the geniune or a fraudulent

# # Machine learning problem
# This one is a classical Classifiation problem were we have to predict weater the given input belongs to the class 0 or class 1 here the input is the job post and the we have to clasify is it a geniune opening or a fraud opening.

# # Performance metric
# The performance metric that I am going to use it in this is F1 score.

# # importing the libraries

# In[ ]:


import pandas as pd
import seaborn as sns
from tqdm import tqdm
from prettytable import PrettyTable
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier


# # Loading the data

# In[ ]:


job_data = pd.read_csv('../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv', index_col = 'job_id')
job_data


# # EDA

# In[ ]:


# checking the shape of the data
job_data.shape


# ### Observation
# There are around 18k data with 17 feature

# ### How balance the dataset is

# In[ ]:


# 0 is not fake and 1 is fake
# checking for Imbalance
job_data['fraudulent'].value_counts()


# In[ ]:


print("percentage of data with class a 0: ",job_data['fraudulent'].value_counts()[0] /job_data.shape[0] *100)
print("percentage of data with class a 1: ",job_data['fraudulent'].value_counts()[1] /job_data.shape[0] *100)
sns.set(style="darkgrid")
ax = sns.countplot(x="fraudulent", data=job_data)
ax.set_title("count plot of the classes")


# By looking at the count of number of data point belongs to each class we can clearly see that the dataset is highly imbalaced in nature and hence we have to balance it using various balancing techniques

# In[ ]:


job_data.info()


# ### Checking the missing value 

# In[ ]:


import matplotlib.pyplot as plt
total= job_data.isnull().sum()
missing_percent =  job_data.isnull().sum()* 100 / len(job_data)
missing_data = pd.concat([total,missing_percent],axis=1,keys=['Total','Percentage'])
f,ax = plt.subplots(figsize=(15,6))
xlocs=plt.xticks(rotation='90')
bars = plt.bar(missing_data.index,missing_data['Total'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, round((yval*100)/len(job_data),2), color='red',fontweight='bold')


# Mejority of the columns in this data has missing value as we can see the percent of missing value in each columns/feature so we have to decide either we have to imuputate the missing data or we have to drop the feature

# Now since the dataset contains missing value and it has 17 columns and feature we have to first identitfy which of them is catgorical and which of them are not to do so I am going to count the number of unique value in the cols and will set some threshold and if the col have unique values less than the threshold means that column is categorical

# ### Identiyfing the categorical features 

# In[ ]:


categorical_feature = []
for col in job_data.columns:
    print(f'Unique rows in {col}:', job_data[col].nunique())
    if job_data[col].nunique() < 15:
        categorical_feature.append(col)
print('Categorical feature:',categorical_feature)
print('Total cat feature',len(categorical_feature))


# we can se that we have total of 6 categorical feature I am not counting the fraudulent col as that is our target class.

# ## Imputation of missing values

# Now It is time to imputate the missing value data. You must remember that we have calculated the percenteage of mising value of data so we set a threshold using that and drop the column having percentage more than threshold.

# In[ ]:


# job_data.dropna(thresh = threshold, axis=1)
job_data.drop(columns = ['salary_range'],axis=1,inplace = True)
job_data['department'].fillna('other',inplace=True)
job_data.head()


# In[ ]:


job_data.shape


# In[ ]:


# total= job_data.isnull().sum()
# missing_percent =  job_data.isnull().sum()* 100 / len(job_data)
# missing_data = pd.concat([total,missing_percent],axis=1,keys=['Total','Percentage'])
# f,ax = plt.subplots(figsize=(15,6))
# xlocs=plt.xticks(rotation='90')
# bars = plt.bar(missing_data.index,missing_data['Total'])

# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x(), yval + .005, round((yval*100)/len(job_data),2), color='red',fontweight='bold')


# In[ ]:


# empty_columns = job_data.loc[:,job_data.isnull().sum()>0]
# empty_columns


# ### Imputating the categorical feature
# now to all the categorical data we will replace it with the most frequent occuring data

# In[ ]:


# filling he categorical missing data
for feature in categorical_feature:
#     print("feature name: ",feature)
    if job_data[feature].isnull().sum() > 0 :
        print("find the empyt value in feature:",feature)
        job_data[feature].fillna(value= job_data[feature].mode()[0],inplace=True)


# ### Imputating the non categorical feature

# In[ ]:


# filling the non categorical data
non_categorical_data =  list(set(job_data.columns) - set(categorical_feature))
for feature in non_categorical_data:
    if job_data[feature].isnull().sum() > 0 :
        print("find the empyt value in feature:",feature)
        job_data[feature].fillna(value= 'Not specified',inplace=True)


# ### Checking the data balance after Imputation

# In[ ]:


job_data.isnull().sum(), job_data.shape


# Now we are done with our missing value and now we have no missing data feature so we can move futher in our analysis

# ## Univarient Analysis

# In[ ]:


job_data.describe()


# In[ ]:


job_data.head()


# In[ ]:


categorical_feature.pop()


# ### Employment type on job post fraud

# In[ ]:


# categorical feature effect on the fraudulent classes
plt.figure(1,figsize=(20,8))
sns.countplot(hue=job_data.fraudulent,x=job_data.employment_type);
plt.title('Which type of jobs have more fraudulent postings');


# ### Observation
# 
# By observing the count plot of the employment type we can make a conclusion that expect of employment type full time there is no other types that contribute to the fraudulent job post.

# ### Required Experience on job post fraud

# In[ ]:


plt.figure(1,figsize=(20,8))
sns.countplot(hue=job_data.fraudulent,x=job_data.required_experience);
plt.title('Which required experience of jobs have more fraudulent postings');


# ## Observation
# 
# The mid senior level work exprience job posting have more fraudulent job post then any other

# ### Required Education on job post fraud

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=job_data.fraudulent,x=job_data.required_education)
plt.legend(loc='upper right')
plt.title('Which required education of jobs have more fraudulent postings')


# ## Observation
# 
# We can see in the plot that the job post which have education requirement as bachelors degree contribute more to the fraudulent post

# ### Telecommuting Education on job post fraud

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=job_data.fraudulent,x=job_data.telecommuting)
plt.legend(loc='upper right')
plt.title('How telecommuting jobs effect contribute towards the fraudulent postings.')


# ### Observation
# For the non telecomunicating position there is fraudulent post then the telecomunicating position.

# ### Presence of company logo on job post on fraud post

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=job_data.fraudulent,x=job_data.has_company_logo)
plt.legend(loc='upper right')
plt.title('Company logo presence effect on fraudulent postings')


# ### Observation
# Job post which have the company logo in it has less number of faudulent casses then the one which do not have the company logo which is like a natural thing to see.

# ### Presence of screening question on job post fraud

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=job_data.fraudulent,x=job_data.has_questions)
plt.legend(loc='upper right')
plt.title('Screening Question effect fraudulent postings')


# ### Observation
# We can see that is the screening questions are present then there is less number of fraudulent job compare to the job posting where not screening questions are present.

# ### DATA CLEANING AND FEATURE ENGINEEING OF LOCATION COLUMN

# #### Getting the country name from the job location

# In[ ]:


processed_country = []
# not_specified_country_index = []
# processed_state = []
# not_specified_state_index = []
# processed_city = []
def location_separator(splitted_location_name):
    for idx, j in enumerate(splitted_location_name):
        if j.isspace() :
            if idx == 0:
                processed_country.append('not given')
#             elif idx == 1:
# #                 not_specified_state_index.append(idx)
#                 processed_state.append('not given')
#             else:
#                 processed_city.append('not given')
        else:
            if idx == 0:
                processed_country.append(j.replace(" ", ""))
#             elif idx == 1:
#                 processed_state.append(j.replace(" ", ""))
#             else:
#                 processed_city.append(j.replace(" ", "")) 


# In[ ]:


processed_location = []
for idx,value in enumerate(job_data.location.to_list()):
    if '\t' in value:
        processed_location_name = value.replace('\t','')
        processed_location.append(processed_location_name)
        splitted_location_name = processed_location_name.split(',')
        location_separator(splitted_location_name) 
    else:
        
        if value == 'Not specified':
            value = 'NA, NA, NA'
        processed_location.append(value)
        location_separator(value.split(','))


# In[ ]:


len(processed_country),len(processed_location)


# #### Getting the state name from the location

# In[ ]:


state_name = []
import re
for idx,value in enumerate(job_data.location.to_list()):
    value = value.replace('\t','')
    if len(value.split(',')) > 3:
        state_value = list(filter(str.strip, value.split(',')))
        state_value = [re.sub('[0-9]','',word) for word in state_value if len(word.replace(" ", ""))>1]
        state_value = list(filter(str.strip,state_value))
        state_value = list(set(state_value))
        state_value = state_value[:3]
        state_name.append(state_value[1].replace(" ", ""))  
    elif len(value.split(',')) == 1:
        state_name.append('NOT GIVEN')      
    else:
        if len(value.split(',')[1].replace(" ", ""))==1 or value.split(',')[1].isspace():
#             print("original_value: ",value,"<---list_value:--> ",value.split(','))
#             print('idx',idx,'lenght: ',len(value.split(',')))
            state_name.append('NOT GIVEN')
        elif value.split(',')[1].replace(" ", "").isdigit():
            state_name.append('NOT GIVEN')
        else:
#             print("original_value: ",value,"<---list_value:--> ",value.split(','))
#             print('idx',idx,'lenght: ',len(value.split(',')))
            
            state_name.append(value.split(',')[1].replace(" ", "")) 


# In[ ]:


len(state_name)


# In[ ]:


# job_data[job_data.duplicated(keep=False)].reset_index()
# state_name
# len('01')


# #### Getting the city name from the location

# In[ ]:


city_name = []
import re
for idx,value in enumerate(job_data.location.to_list()):
    value = value.replace('\t','')
    if len(value.split(',')) > 3:
        state_value = list(filter(str.strip, value.split(',')))
        state_value = [re.sub('[0-9]','',word) for word in state_value if len(word.replace(" ", ""))>1]
        state_value = list(filter(str.strip,state_value))
        state_value = list(set(state_value))
        city_name.append(state_value[2].replace(" ", ""))  
    elif len(value.split(',')) == 1:
        city_name.append('NOT GIVEN')      
    else:
        if len(value.split(',')[2].replace(" ", ""))==1 or value.split(',')[2].isspace():
            city_name.append('NOT GIVEN')
        elif value.split(',')[2].replace(" ", "").isdigit():
            city_name.append('NOT GIVEN')
        else:
            city_name.append(value.split(',')[2].replace(" ", ""))
    


# In[ ]:


len(processed_country), len(state_name),len(city_name)


# In[ ]:


processed_job_data = job_data.copy()
processed_job_data['country'] = processed_country
processed_job_data['state'] = state_name
processed_job_data['city'] = city_name
processed_job_data.drop(columns=['location'],axis=1,inplace=True)


# ### Separating all the fraud and no fraud data for some analysis on our newly created feature

# In[ ]:


fraud_data = processed_job_data[processed_job_data.fraudulent==1]
# fraud_data


# #### Which Department is the most common in the fraud job posting

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=fraud_data.fraudulent,x=fraud_data.department, data=fraud_data, order= fraud_data.department.value_counts().iloc[:].index)
plt.legend(loc='upper right')
plt.title('which department contribute towards the fraudulent postings.')


# ### Observation
# By lookig at the plot of department of fraudulent post we can clearly see that the most fraud post there is no deparment specified more than 64% of total fraudlent post dont specify there department.

# In[ ]:


no_fraud_data = processed_job_data[processed_job_data.fraudulent==0]
# no_fraud_data


# ### Which department is most common in the non fraud job post

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=no_fraud_data.fraudulent,x=no_fraud_data.department, data=no_fraud_data, order= no_fraud_data.department.value_counts().iloc[:100].index)
plt.legend(loc='upper right')
plt.title('which department contribute towards the non fraudulent postings.')


# ### Observation
# In the non fraudulent job post also more than 59% of total job post do not specify there department so here we cannot clearly say that if the department is specified as other then the job post is fraud.

# ### Which industry has most common in fraud job post

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=no_fraud_data.fraudulent,x=no_fraud_data.industry,data=no_fraud_data,order=no_fraud_data.industry.value_counts().iloc[:50].index)
plt.legend(loc='upper right')
plt.title('which department contribute towards the non fraudulent postings.')


# ### Observation
# 
# Looking at the count plot of the industry type in the non fraudulent job post we can see that the high number of job post does not specify the industry count.

# ### Which industry is most common in the non fraud job post

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=fraud_data.fraudulent,x=fraud_data.industry,data=fraud_data,order=fraud_data.industry.value_counts().iloc[:50].index)
plt.legend(loc='upper right')
plt.title('which department contribute towards the non fraudulent postings.')


# ### Observation
# The fraudulent post also do not specify the industry type and it is vast in number but if we see carefully the number of post with oil and energy , Accounting etc is higher than that of non fraudulent post hence we can use industry as feature.

# In[ ]:


processed_job_data.columns


# ### Which function is most common in the fraud job post and non fraud job post

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=processed_job_data.fraudulent,x=processed_job_data.function ,data=processed_job_data, order = processed_job_data.function.value_counts().iloc[:].index )
plt.legend(loc='upper right')
plt.title('function effect on fraudulent postings')


# ### Observation
# We cannot make a descision after looking at the countplot of the function in case of the job post to be fraud or geniuine.

# ### Which country is most common in the fraud job post

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=fraud_data.fraudulent,x=fraud_data.country ,data=fraud_data, order = fraud_data.country.value_counts().iloc[:].index )
plt.legend(loc='upper right')
plt.title('state effect on fraudulent postings')


# ### Observation
# US is the country with maximum fraud post.

# ### Which state is the most common in the fraud job post

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=fraud_data.fraudulent,x=fraud_data.state ,data=fraud_data, order = fraud_data.state.value_counts().iloc[:].index )
plt.legend(loc='upper right')
plt.title('state effect on fraudulent postings')


# ### Observation
# In US also the texas is the state were most fraud job post were present

# ### Which country is most common in the non fraud job post

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=no_fraud_data.fraudulent,x=no_fraud_data.country ,data=no_fraud_data, order = no_fraud_data.country.value_counts().iloc[:].index )
plt.legend(loc='upper right')
plt.title('Country effect on non fraudulent postings')


# ### Observation
#  Also US is the country with most geniune job post hence we can say that most of the job post were posted in the US then any other country in the world.

# ### Which state is the most common in the non fraud job post

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=no_fraud_data.fraudulent,x=no_fraud_data.state ,data=no_fraud_data, order = no_fraud_data.state.value_counts().iloc[:100].index )
plt.legend(loc='upper right')
plt.title('state effect on non fraudulent postings')


# ### Observation
# This is intereseting the to see that most of the authentic job post in the us does not specify the state in there location.

# ### Which title of the job is common in the most fraud job post

# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=fraud_data.fraudulent,x=fraud_data.title ,data=fraud_data, order = fraud_data.title.value_counts().iloc[:100].index )
plt.legend(loc='upper right')
plt.title('title effect on fraudulent postings')


# ### Observation
# In the plot of titles in the fraud post we can see the titles like URGENT, Data Entry or customer care have most the count towards fraud post.

# ### Which title of the job is common in the most non fraud job post

# In[ ]:


no_fraud_data.title.value_counts()


# ### Which title of the job is common in the most fraud job post

# In[ ]:


fraud_data.title.value_counts()


# In[ ]:


plt.figure(1,figsize=(20,8))
plt.xticks(rotation='90')
sns.countplot(hue=no_fraud_data.fraudulent,x=no_fraud_data.title ,data=no_fraud_data, order = no_fraud_data.title.value_counts().iloc[:100].index )
plt.legend(loc='upper right')
plt.title('title effect on non fraudulent postings')


# ### Observation
# Most geniune job post have the title as the english teacher.

# ## Text Preprocessing

# In[ ]:


# import re
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# In[ ]:


# https://gist.github.com/sebleier/554280
# we are removing the words from the stop words list: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',             'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',             'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',             'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',             'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very',             's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',             've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",             'won', "won't", 'wouldn', "wouldn't"]


# In[ ]:


processed_job_data.columns


# ### Description

# In[ ]:


preprocessed_description = []
for sentance in tqdm(processed_job_data['description'].values):
    sent = sentance.lower()
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = re.sub(r'\w*[0-9]\w*', '', sent, flags=re.MULTILINE)
    sent = re.sub('[0-9]', ' ', sent)
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    
    sent = ' '.join(e for e in sent.split(' ') if e not in stopwords)
    sent = decontracted(sent)
    # https://gist.github.com/sebleier/554280
    preprocessed_description.append(sent)  


# In[ ]:


processed_job_data.drop(['description'], axis=1,inplace=True)
processed_job_data['preprocessed_description'] = preprocessed_description


# ### Benefits

# In[ ]:


preprocessed_benefits = []
# tqdm is for printing the status bar
for sentance in tqdm(processed_job_data['benefits'].values):
    sent = sentance.lower()
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = re.sub(r'\w*[0-9]\w*', '', sent, flags=re.MULTILINE)
    sent = re.sub('[0-9]', ' ', sent)
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    
    sent = ' '.join(e for e in sent.split(' ') if e not in stopwords)
    sent = decontracted(sent)
    # https://gist.github.com/sebleier/554280
#     sent = ' '.join(e for e in sent.split() if e not in stopwords)
#     preprocessed_essays.append(sent.lower().strip())
    preprocessed_benefits.append(sent)  


# In[ ]:


processed_job_data.drop(['benefits'], axis=1,inplace=True)
processed_job_data['preprocessed_benefits'] = preprocessed_benefits


# ### Requirements

# In[ ]:


preprocessed_requirements = []
# tqdm is for printing the status bar
for sentance in tqdm(processed_job_data['requirements'].values):
    sent = sentance.lower()
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = re.sub(r'\w*[0-9]\w*', '', sent, flags=re.MULTILINE)
    sent = re.sub('[0-9]', ' ', sent)
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    
    sent = ' '.join(e for e in sent.split(' ') if e not in stopwords)
    sent = decontracted(sent)
    # https://gist.github.com/sebleier/554280
#     sent = ' '.join(e for e in sent.split() if e not in stopwords)
#     preprocessed_essays.append(sent.lower().strip())
    preprocessed_requirements.append(sent)  


# In[ ]:


processed_job_data.drop(['requirements'], axis=1,inplace=True)
processed_job_data['preprocessed_requirements'] = preprocessed_requirements


# ### Title

# In[ ]:


preprocessed_title = []
# tqdm is for printing the status bar
for sentance in tqdm(processed_job_data['title'].values):
    sent = sentance.lower()
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = re.sub(r'\w*[0-9]\w*', '', sent, flags=re.MULTILINE)
    sent = re.sub('[0-9]', ' ', sent)
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent = ' '.join(e for e in sent.split(' ') if e not in stopwords)
    sent = sent.strip()
    sent = decontracted(sent)
    # https://gist.github.com/sebleier/554280
#     sent = ' '.join(e for e in sent.split() if e not in stopwords)
#     preprocessed_essays.append(sent.lower().strip())
    preprocessed_title.append(sent)  


# In[ ]:


processed_job_data.drop(['title'], axis=1,inplace=True)
processed_job_data['preprocessed_title'] = preprocessed_title


# ### Company Profile

# In[ ]:


preprocessed_company_profile = []
# tqdm is for printing the status bar
for sentance in tqdm(processed_job_data['company_profile'].values):
    sent = sentance.lower()
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = re.sub(r'\w*[0-9]\w*', '', sent, flags=re.MULTILINE)
    sent = re.sub('[0-9]', ' ', sent)
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent = ' '.join(e for e in sent.split(' ') if e not in stopwords)
    sent = sent.strip()
    sent = decontracted(sent)
    # https://gist.github.com/sebleier/554280
#     sent = ' '.join(e for e in sent.split() if e not in stopwords)
#     preprocessed_essays.append(sent.lower().strip())
    preprocessed_company_profile.append(sent)  


# In[ ]:


processed_job_data.drop(['company_profile'], axis=1,inplace=True)
processed_job_data['preprocessed_company_profile'] = preprocessed_company_profile


# In[ ]:


processed_job_data


# In[ ]:


processed_job_data.drop(['department'],axis=1,inplace=True)


# In[ ]:


# processed_job_data
processed_job_data.reset_index(inplace=True)
processed_job_data.drop(['job_id'],axis=1,inplace=True)


# ## Splitting the data

# In[ ]:


y = processed_job_data['fraudulent'].values
X = processed_job_data.drop(['fraudulent'], axis=1)
X.head()


# In[ ]:


get_ipython().system('pip install imbalanced-learn')


# In[ ]:


# check version number
import imblearn
print(imblearn.__version__)


# In[ ]:


# # balancig the Imbalanced dataset https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)
# X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)

# print(X_train.shape)
# print(y_train.shape)
# print(X_cv.shape)
# print(y_cv.shape)
# print(X_test.shape)
# print(y_test.shape)


# ## Onehot encoding of the categorical data

# ### Employment type

# In[ ]:


# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(vocabulary=set(processed_job_data['employment_type']),lowercase=False, binary=True)
# X_train_employment_type_one_hot = vectorizer.fit_transform(X_train['employment_type'].values)
# X_cv_employment_type_one_hot = vectorizer.transform(X_cv['employment_type'].values)
# X_test_employment_type_one_hot = vectorizer.transform(X_test['employment_type'].values)

# print(vectorizer.get_feature_names())
# print("Shape of X_train matrix after one hot encodig ",X_train_employment_type_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_employment_type_one_hot.shape)
# print("Shape of X_test matrix after one hot encodig ",X_test_employment_type_one_hot.shape)


# ### Required Experience

# In[ ]:


# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(vocabulary=set(processed_job_data['required_experience']),lowercase=False, binary=True)
# X_train_required_experience_one_hot = vectorizer.fit_transform(X_train['required_experience'].values)
# X_cv_required_experience_one_hot = vectorizer.transform(X_cv['required_experience'].values)
# X_test_required_experience_one_hot = vectorizer.transform(X_test['required_experience'].values)

# print(vectorizer.get_feature_names())
# print("Shape of X_train matrix after one hot encodig ",X_train_required_experience_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_required_experience_one_hot.shape)
# print("Shape of X_test matrix after one hot encodig ",X_test_required_experience_one_hot.shape)


# ### Required Education

# In[ ]:


# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(vocabulary=set(processed_job_data['required_education']),lowercase=False, binary=True)
# X_train_required_education_one_hot = vectorizer.fit_transform(X_train['required_education'].values)
# X_cv_required_education_one_hot = vectorizer.transform(X_cv['required_education'].values)
# X_test_required_education_one_hot = vectorizer.transform(X_test['required_education'].values)

# print(vectorizer.get_feature_names())
# print("Shape of X_train matrix after one hot encodig ",X_train_required_education_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_required_education_one_hot.shape)
# print("Shape of X_test matrix after one hot encodig ",X_test_required_education_one_hot.shape)


# ### Industry

# In[ ]:


# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(vocabulary=set(processed_job_data['industry']),lowercase=False, binary=True)
# X_train_industry_one_hot = vectorizer.fit_transform(X_train['industry'].values)
# X_cv_industry_one_hot = vectorizer.transform(X_cv['industry'].values)
# X_test_industry_one_hot = vectorizer.transform(X_test['industry'].values)

# print(vectorizer.get_feature_names())
# print("Shape of X_train matrix after one hot encodig ",X_train_industry_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_industry_one_hot.shape)
# print("Shape of X_test matrix after one hot encodig ",X_test_industry_one_hot.shape)


# ### function

# In[ ]:


# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(vocabulary=set(processed_job_data['function']),lowercase=False, binary=True)
# X_train_function_one_hot = vectorizer.fit_transform(X_train['function'].values)
# X_cv_function_one_hot = vectorizer.transform(X_cv['function'].values)
# X_test_function_one_hot = vectorizer.transform(X_test['function'].values)

# print(vectorizer.get_feature_names())
# print("Shape of X_train matrix after one hot encodig ",X_train_function_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_function_one_hot.shape)
# print("Shape of X_test matrix after one hot encodig ",X_test_function_one_hot.shape)


# ### Country

# In[ ]:


# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(vocabulary=set(processed_job_data['country']),lowercase=False, binary=True)
# X_train_country_one_hot = vectorizer.fit_transform(X_train['country'].values)
# X_cv_country_one_hot = vectorizer.transform(X_cv['country'].values)
# X_test_country_one_hot = vectorizer.transform(X_test['country'].values)

# print(vectorizer.get_feature_names())
# print("Shape of X_train matrix after one hot encodig ",X_train_country_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_country_one_hot.shape)
# print("Shape of X_test matrix after one hot encodig ",X_test_country_one_hot.shape)


# ### State

# In[ ]:


# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(vocabulary=set(processed_job_data['state']),lowercase=False, binary=True)
# X_train_state_one_hot = vectorizer.fit_transform(X_train['state'].values)
# X_cv_state_one_hot = vectorizer.transform(X_cv['state'].values)
# X_test_state_one_hot = vectorizer.transform(X_test['state'].values)

# print(vectorizer.get_feature_names())
# print("Shape of X_train matrix after one hot encodig ",X_train_state_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_state_one_hot.shape)
# print("Shape of X_test matrix after one hot encodig ",X_test_state_one_hot.shape)


# ## Vectorizing the preprocessed text data

# ### Description

# In[ ]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=10)
# X_train_description_tfidf = vectorizer.fit_transform(X_train['preprocessed_description'])
# X_cv_description_tfidf = vectorizer.transform(X_cv['preprocessed_description'])
# X_test_description_tfidf = vectorizer.transform(X_test['preprocessed_description'])
# print("Shape of X_train_essay_tfidf matrix after ",X_train_description_tfidf.shape)
# print("Shape of X_cv_essay_tfidf matrix after ",X_cv_description_tfidf.shape)
# print("Shape of X_test_essay_tfidf matrix after ",X_test_description_tfidf.shape)


# ### Benifits

# In[ ]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=10)
# X_train_denifits_tfidf = vectorizer.fit_transform(X_train['preprocessed_benefits'])
# X_cv_denifits_tfidf = vectorizer.transform(X_cv['preprocessed_benefits'])
# X_test_denifits_tfidf = vectorizer.transform(X_test['preprocessed_benefits'])
# print("Shape of X_train_essay_tfidf matrix after ",X_train_denifits_tfidf.shape)
# print("Shape of X_cv_essay_tfidf matrix after ",X_cv_denifits_tfidf.shape)
# print("Shape of X_test_essay_tfidf matrix after ",X_test_denifits_tfidf.shape)


# ### Requirement

# In[ ]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=10)
# X_train_requirements_tfidf = vectorizer.fit_transform(X_train['preprocessed_requirements'])
# X_cv_requirements_tfidf = vectorizer.transform(X_cv['preprocessed_requirements'])
# X_test_requirements_tfidf = vectorizer.transform(X_test['preprocessed_requirements'])
# print("Shape of X_train_essay_tfidf matrix after ",X_train_requirements_tfidf.shape)
# print("Shape of X_cv_essay_tfidf matrix after ",X_cv_requirements_tfidf.shape)
# print("Shape of X_test_essay_tfidf matrix after ",X_test_requirements_tfidf.shape)


# ### title

# In[ ]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=10)
# X_train_title_tfidf = vectorizer.fit_transform(X_train['preprocessed_title'])
# X_cv_title_tfidf = vectorizer.transform(X_cv['preprocessed_title'])
# X_test_title_tfidf = vectorizer.transform(X_test['preprocessed_title'])
# print("Shape of X_train_essay_tfidf matrix after ",X_train_title_tfidf.shape)
# print("Shape of X_cv_essay_tfidf matrix after ",X_cv_title_tfidf.shape)
# print("Shape of X_test_essay_tfidf matrix after ",X_test_title_tfidf.shape)


# ### Company profile

# In[ ]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(min_df=10)
# X_train_profile_tfidf = vectorizer.fit_transform(X_train['preprocessed_company_profile'])
# X_cv_profile_tfidf = vectorizer.transform(X_cv['preprocessed_company_profile'])
# X_test_profile_tfidf = vectorizer.transform(X_test['preprocessed_company_profile'])
# print("Shape of X_train_essay_tfidf matrix after ",X_train_profile_tfidf.shape)
# print("Shape of X_cv_essay_tfidf matrix after ",X_cv_profile_tfidf.shape)
# print("Shape of X_test_essay_tfidf matrix after ",X_test_profile_tfidf.shape)


# In[ ]:


X.columns


# In[ ]:


# # merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039

# # with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
# # X = hstack((categories_one_hot, sub_categories_one_hot, text_bow, price_standardized))
# X_tr = hstack((X_train_employment_type_one_hot,X_train_required_experience_one_hot,X_train_required_education_one_hot,X_train_industry_one_hot,
#               X_train_function_one_hot,X_train_country_one_hot,X_train_state_one_hot,X_train_description_tfidf,
#               X_train_denifits_tfidf,X_train_requirements_tfidf,X_train_title_tfidf,X_train_profile_tfidf,
#               X_train['telecommuting'].values.reshape(-1,1),X_train['has_company_logo'].values.reshape(-1,1),
#                X_train['has_questions'].values.reshape(-1,1))).tocsr()

# X_cr = hstack((X_cv_employment_type_one_hot,X_cv_required_experience_one_hot,X_cv_required_education_one_hot,
#               X_cv_industry_one_hot,X_cv_function_one_hot,X_cv_country_one_hot,X_cv_state_one_hot,
#               X_cv_description_tfidf,X_cv_denifits_tfidf,X_cv_requirements_tfidf,X_cv_title_tfidf,
#               X_cv_profile_tfidf,X_cv['telecommuting'].values.reshape(-1,1),
#               X_cv['has_company_logo'].values.reshape(-1,1),X_cv['has_questions'].values.reshape(-1,1))).tocsr()

# X_te = hstack((X_test_employment_type_one_hot,X_test_required_experience_one_hot,X_test_required_education_one_hot,
#               X_test_industry_one_hot,X_test_function_one_hot,X_test_country_one_hot,X_test_state_one_hot,
#               X_test_description_tfidf,X_test_denifits_tfidf,X_test_requirements_tfidf,X_test_title_tfidf,
#               X_test_profile_tfidf,X_test['telecommuting'].values.reshape(-1,1),
#               X_test['has_company_logo'].values.reshape(-1,1),X_test['has_questions'].values.reshape(-1,1))).tocsr()


# In[ ]:


# #For memory issue batch wise prediction
# def batch_predict(clf, data):
#     y_data_pred = []
#     tr_loop = data.shape[0] - data.shape[0]%1000
#     for i in range(0, tr_loop, 1000):
#         y_data_pred.extend(clf.predict(data[i:i+1000])[:,1])
#     if data.shape[0]%1000 !=0:
#         y_data_pred.extend(clf.predict(data[tr_loop:])[:,1])
#     return y_data_pred

# def find_best_threshold(threshould, fpr, tpr):
#     t = threshould[np.argmax(tpr*(1-fpr))]
#     # (tpr*(1-fpr)) will be maximum if your fpr is very low and tpr is very high
#     print("the maximum value of tpr*(1-fpr)", max(tpr*(1-fpr)), "for threshold", np.round(t,3))
#     return t

# def predict_with_best_t(proba, threshould):
#     predictions = []
#     for i in proba:
#         if i>=threshould:
#             predictions.append(1)
#         else:
#             predictions.append(0)
#     return predictions


# ### Logistic regression

# In[ ]:


# # Please write all the code with proper documentation
# #selecting the hyperparameter using RandomSearch
# from sklearn.metrics import f1_score, make_scorer
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import roc_curve
# from xgboost import XGBClassifier
# from sklearn.linear_model import SGDClassifier

# # from sklearn.model_selection import GridSearchCV

# log = SGDClassifier(loss = 'log', class_weight= 'balanced')
# # parameters = {'alpha':sp_randint(50, 100)}
# parameters = {'alpha':[10**x for x in range(-4,4)]}
# clf = GridSearchCV(log, parameters, cv=3, scoring=make_scorer(f1_score),return_train_score=True)
# clf.fit(X_tr, y_train)

# results = pd.DataFrame.from_dict(clf.cv_results_)
# # results3 = results3.sort_values(['param_alpha'])

# train_f1= results['mean_train_score']
# train_f1_std= results['std_train_score']
# cv_f1 = results['mean_test_score'] 
# cv_f1_std= results['std_test_score']
# # alpha3 =  results3['param_alpha']
# # alpha3 = alpha3.astype(np.int64) # https://stackoverflow.com/questions/46995041/why-does-this-array-has-no-attribute-log10?rq=1
# alpha = np.log10(parameters['alpha'])

# plt.plot(alpha, train_f1, label='Train F1')
# # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
# # plt.gca().fill_between(K, train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

# plt.plot(alpha, cv_f1, label='CV F1')
# # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
# # plt.gca().fill_between(K, cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

# plt.scatter(alpha, train_f1, label='Train F1 points')
# plt.scatter(alpha, cv_f1, label='CV F1 points')



# plt.legend()
# plt.xlabel("alpha: hyperparameter")
# plt.ylabel("f1")
# plt.title("Hyper parameter Vs f1 score plot")
# plt.grid()
# plt.show()

# results.head()


# In[ ]:


# # Please write all the code with proper documentation
# best_alpha = 0.001


# log_reg = SGDClassifier(loss = 'log',alpha=best_alpha, class_weight= 'balanced')
# log_reg.fit(X_tr, y_train)
# # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# # not the predicted outputs

# y_train_pred = log_reg.predict(X_tr) 
# y_cv_pred = log_reg.predict(X_cr)    
# y_test_pred = log_reg.predict(X_te)

# # train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
# # test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)

# # plt.plot(train_fpr_3, train_tpr_3, label="train AUC ="+str(auc(train_fpr_3, train_tpr_3)))
# # plt.plot(test_fpr_3, test_tpr_3, label="test AUC ="+str(auc(test_fpr_3, test_tpr_3)))
# # plt.legend()
# # plt.xlabel("alpha_3: hyperparameter")
# # plt.ylabel("AUC")
# # plt.title("ERROR PLOTS")
# # plt.grid()
# # plt.show()


# In[ ]:


# y_train_pred


# In[ ]:


# # best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)

# print("Train confusion matrix")

# f1_score_ = f1_score(y_train,y_train_pred)
# print("F1_score",f1_score_)
# cm_train_set = confusion_matrix(y_train, y_train_pred)
# print(cm_train_set)
# df_cm = pd.DataFrame(cm_train_set, columns=np.unique(y_test), index = np.unique(y_test))
# df_cm.index.name = 'Actual'
# df_cm.columns.name = 'Predicted'
# plt.figure(figsize = (10,7))
# sns.set(font_scale=1.4)#for label size
# sns.heatmap(df_cm,fmt='d', cmap="Blues", annot=True,annot_kws={"size": 16})# font size


# In[ ]:


# # best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)

# print("cv confusion matrix")
# f1_score_ = f1_score(y_cv,y_cv_pred)
# print("F1_score",f1_score_)
# cm_cv_set = confusion_matrix(y_cv,y_cv_pred)
# print(cm_cv_set)
# df_cm = pd.DataFrame(cm_cv_set, columns=np.unique(y_cv), index = np.unique(y_cv))
# df_cm.index.name = 'Actual'
# df_cm.columns.name = 'Predicted'
# plt.figure(figsize = (10,7))
# sns.set(font_scale=1.4)#for label size
# sns.heatmap(df_cm,fmt='d', cmap="Blues", annot=True,annot_kws={"size": 16})# font size


# In[ ]:


# # best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)

# print("Test confusion matrix")
# f1_score_ = f1_score(y_test,y_test_pred)
# print("F1_score",f1_score_)
# cm_test_set = confusion_matrix(y_test,y_test_pred)
# print(cm_test_set)
# df_cm = pd.DataFrame(cm_test_set, columns=np.unique(y_test), index = np.unique(y_test))
# df_cm.index.name = 'Actual'
# df_cm.columns.name = 'Predicted'
# plt.figure(figsize = (10,7))
# sns.set(font_scale=1.4)#for label size
# sns.heatmap(df_cm,fmt='d', cmap="Blues", annot=True,annot_kws={"size": 16})# font size


# ### XGBOOST Classifier

# In[ ]:


# #selecting the hyperparameter using GridSearch
# # https://www.kaggle.com/arindambanerjee/grid-search-simplified
# from sklearn.metrics import f1_score, make_scorer
# from sklearn.model_selection import GridSearchCV

# xg_clf = XGBClassifier()
# parameters = {'n_estimators':[5, 10, 50, 100, 200],'max_depth':[2,3, 4, 5, 6, 7]}
# xg_clf_ = GridSearchCV(xg_clf, parameters, n_jobs= -1, verbose=10, cv=2, scoring=make_scorer(f1_score),return_train_score=True)
# xg_clf_.fit(X_tr, y_train)

# results2 = pd.DataFrame.from_dict(xg_clf_.cv_results_)
# # results4 = results4.sort_values(['param_alpha'])
# max_depth_list = list(xg_clf_.cv_results_['param_max_depth'].data)
# n_estimator_list = list(xg_clf_.cv_results_['param_n_estimators'].data)


# sns.set_style("whitegrid")
# plt.figure(figsize=(16,6))
# plt.subplot(1,2,1)

# data = pd.DataFrame(data={'Number of Estimator':n_estimator_list, 'Max Depth':max_depth_list, 'f1_score':xg_clf_.cv_results_['mean_train_score']})
# data = data.reset_index().pivot_table(index='Max Depth', columns='Number of Estimator', values='f1_score')
# sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('AUC for Training data')
# plt.subplot(1,2,2)

# # Testing Heatmap
# data = pd.DataFrame(data={'Number of Estimator':n_estimator_list, 'Max Depth':max_depth_list, 'f1_score':xg_clf_.cv_results_['mean_test_score']})
# data = data.reset_index().pivot_table(index='Max Depth', columns='Number of Estimator', values='f1_score')
# sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('f1_score for Test data')
# plt.show()

# results2.head()


# In[ ]:


# # Please write all the code with proper documentation
# n_estimators = 50
# max_depth = 6


# xg_clf = XGBClassifier(max_depth = max_depth, n_estimators = n_estimators , n_jobs=1)
# xg_clf.fit(X_tr, y_train)
# # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
# # not the predicted outputs

# y_train_pred = xg_clf.predict(X_tr)    
# y_cv_pred = xg_clf.predict(X_cr)    
# y_test_pred = xg_clf.predict(X_te)

# # train_fpr, train_tpr, tr_thresholds = roc_curve(y_train, y_train_pred)
# # test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_test_pred)


# In[ ]:


# # best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)

# print("Train confusion matrix")
# f1_score_ = f1_score(y_train,y_train_pred)
# print("F1_score",f1_score_)
# cm_train_set = confusion_matrix(y_train, y_train_pred)
# print(cm_train_set)
# df_cm = pd.DataFrame(cm_train_set)
# df_cm.index.name = 'Actual'
# df_cm.columns.name = 'Predicted'
# plt.figure(figsize = (10,7))
# sns.set(font_scale=1.4)#for label size
# sns.heatmap(df_cm,fmt='d', cmap="Blues", annot=True,annot_kws={"size": 16})# font size


# In[ ]:


# # best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)

# print("cv confusion matrix")
# f1_score_ = f1_score(y_cv,y_cv_pred)
# print("F1_score",f1_score_)
# cm_train_set = confusion_matrix(y_cv, y_cv_pred)
# print(cm_train_set)
# df_cm = pd.DataFrame(cm_train_set)
# df_cm.index.name = 'Actual'
# df_cm.columns.name = 'Predicted'
# plt.figure(figsize = (10,7))
# sns.set(font_scale=1.4)#for label size
# sns.heatmap(df_cm,fmt='d', cmap="Blues", annot=True,annot_kws={"size": 16})# font size


# In[ ]:


# # best_t = find_best_threshold(tr_thresholds, train_fpr, train_tpr)

# print("Test confusion matrix")
# f1_score_1 = f1_score(y_test,y_test_pred)
# print("F1_score",f1_score_1)
# cm_test_set = confusion_matrix(y_test, y_test_pred)
# print(cm_test_set)
# df_cm = pd.DataFrame(cm_test_set)
# df_cm.index.name = 'Actual'
# df_cm.columns.name = 'Predicted'
# plt.figure(figsize = (10,7))
# sns.set(font_scale=1.4)#for label size
# sns.heatmap(df_cm,fmt='d', cmap="Blues", annot=True,annot_kws={"size": 16})# font size


# # Lets try balancing the data

# In[ ]:


# balancig the Imbalanced dataset https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18
from sklearn.model_selection import train_test_split
X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.33, stratify=y)
# X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33, stratify=y_train)

print(X_train_.shape)
print(y_train_.shape)
# print(X_cv.shape)
# print(y_cv.shape)
print(X_test_.shape)
print(y_test_.shape)


# In[ ]:


# X_train_


# ## Onehot encoding 

# ### Eemployment type

# In[ ]:





# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=set(processed_job_data['employment_type']),lowercase=False, binary=True)
X_train_employment_type_one_hot = vectorizer.fit_transform(X_train_['employment_type'].values)
# X_cv_employment_type_one_hot = vectorizer.transform(X_cv['employment_type'].values)
X_test_employment_type_one_hot = vectorizer.transform(X_test_['employment_type'].values)

print(vectorizer.get_feature_names())
print("Shape of X_train matrix after one hot encodig ",X_train_employment_type_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_employment_type_one_hot.shape)
print("Shape of X_test matrix after one hot encodig ",X_test_employment_type_one_hot.shape)


# ### required_experience

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=set(processed_job_data['required_experience']),lowercase=False, binary=True)
X_train_required_experience_one_hot = vectorizer.fit_transform(X_train_['required_experience'].values)
# X_cv_required_experience_one_hot = vectorizer.transform(X_cv['required_experience'].values)
X_test_required_experience_one_hot = vectorizer.transform(X_test_['required_experience'].values)

print(vectorizer.get_feature_names())
print("Shape of X_train matrix after one hot encodig ",X_train_required_experience_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_required_experience_one_hot.shape)
print("Shape of X_test matrix after one hot encodig ",X_test_required_experience_one_hot.shape)


# ### required_education

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=set(processed_job_data['required_education']),lowercase=False, binary=True)
X_train_required_education_one_hot = vectorizer.fit_transform(X_train_['required_education'].values)
# X_cv_required_education_one_hot = vectorizer.transform(X_cv['required_education'].values)
X_test_required_education_one_hot = vectorizer.transform(X_test_['required_education'].values)

print(vectorizer.get_feature_names())
print("Shape of X_train matrix after one hot encodig ",X_train_required_education_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_required_education_one_hot.shape)
print("Shape of X_test matrix after one hot encodig ",X_test_required_education_one_hot.shape)


# ### industry

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=set(processed_job_data['industry']),lowercase=False, binary=True)
X_train_industry_one_hot = vectorizer.fit_transform(X_train_['industry'].values)
# X_cv_industry_one_hot = vectorizer.transform(X_cv['industry'].values)
X_test_industry_one_hot = vectorizer.transform(X_test_['industry'].values)

print(vectorizer.get_feature_names())
print("Shape of X_train matrix after one hot encodig ",X_train_industry_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_industry_one_hot.shape)
print("Shape of X_test matrix after one hot encodig ",X_test_industry_one_hot.shape)


# ### function

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=set(processed_job_data['function']),lowercase=False, binary=True)
X_train_function_one_hot = vectorizer.fit_transform(X_train_['function'].values)
# X_cv_function_one_hot = vectorizer.transform(X_cv['function'].values)
X_test_function_one_hot = vectorizer.transform(X_test_['function'].values)

print(vectorizer.get_feature_names())
print("Shape of X_train matrix after one hot encodig ",X_train_function_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_function_one_hot.shape)
print("Shape of X_test matrix after one hot encodig ",X_test_function_one_hot.shape)


# ### country

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=set(processed_job_data['country']),lowercase=False, binary=True)
X_train_country_one_hot = vectorizer.fit_transform(X_train_['country'].values)
# X_cv_country_one_hot = vectorizer.transform(X_cv['country'].values)
X_test_country_one_hot = vectorizer.transform(X_test_['country'].values)

print(vectorizer.get_feature_names())
print("Shape of X_train matrix after one hot encodig ",X_train_country_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_country_one_hot.shape)
print("Shape of X_test matrix after one hot encodig ",X_test_country_one_hot.shape)


# ### state

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=set(processed_job_data['state']),lowercase=False, binary=True)
X_train_state_one_hot = vectorizer.fit_transform(X_train_['state'].values)
# X_cv_state_one_hot = vectorizer.transform(X_cv['state'].values)
X_test_state_one_hot = vectorizer.transform(X_test_['state'].values)

print(vectorizer.get_feature_names())
print("Shape of X_train matrix after one hot encodig ",X_train_state_one_hot.shape)
# print("Shape of X_cv matrix after one hot encodig ",X_cv_state_one_hot.shape)
print("Shape of X_test matrix after one hot encodig ",X_test_state_one_hot.shape)


# ## Vectorizing the processed text data

# ### preprocessed_description

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10)
X_train_description_tfidf = vectorizer.fit_transform(X_train_['preprocessed_description'])
# X_cv_description_tfidf = vectorizer.transform(X_cv['preprocessed_description'])
X_test_description_tfidf = vectorizer.transform(X_test_['preprocessed_description'])
print("Shape of X_train_essay_tfidf matrix after ",X_train_description_tfidf.shape)
# print("Shape of X_cv_essay_tfidf matrix after ",X_cv_description_tfidf.shape)
print("Shape of X_test_essay_tfidf matrix after ",X_test_description_tfidf.shape)


# ### Benifits

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10)
X_train_denifits_tfidf = vectorizer.fit_transform(X_train_['preprocessed_benefits'])
# X_cv_denifits_tfidf = vectorizer.transform(X_cv['preprocessed_benefits'])
X_test_denifits_tfidf = vectorizer.transform(X_test_['preprocessed_benefits'])
print("Shape of X_train_essay_tfidf matrix after ",X_train_denifits_tfidf.shape)
# print("Shape of X_cv_essay_tfidf matrix after ",X_cv_denifits_tfidf.shape)
print("Shape of X_test_essay_tfidf matrix after ",X_test_denifits_tfidf.shape)


# ### Requirements

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10)
X_train_requirements_tfidf = vectorizer.fit_transform(X_train_['preprocessed_requirements'])
# X_cv_requirements_tfidf = vectorizer.transform(X_cv['preprocessed_requirements'])
X_test_requirements_tfidf = vectorizer.transform(X_test_['preprocessed_requirements'])
print("Shape of X_train_essay_tfidf matrix after ",X_train_requirements_tfidf.shape)
# print("Shape of X_cv_essay_tfidf matrix after ",X_cv_requirements_tfidf.shape)
print("Shape of X_test_essay_tfidf matrix after ",X_test_requirements_tfidf.shape)


# ### Title

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10)
X_train_title_tfidf = vectorizer.fit_transform(X_train_['preprocessed_title'])
# X_cv_title_tfidf = vectorizer.transform(X_cv['preprocessed_title'])
X_test_title_tfidf = vectorizer.transform(X_test_['preprocessed_title'])
print("Shape of X_train_essay_tfidf matrix after ",X_train_title_tfidf.shape)
# print("Shape of X_cv_essay_tfidf matrix after ",X_cv_title_tfidf.shape)
print("Shape of X_test_essay_tfidf matrix after ",X_test_title_tfidf.shape)


# ### company profile

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=10)
X_train_profile_tfidf = vectorizer.fit_transform(X_train_['preprocessed_company_profile'])
# X_cv_profile_tfidf = vectorizer.transform(X_cv['preprocessed_company_profile'])
X_test_profile_tfidf = vectorizer.transform(X_test_['preprocessed_company_profile'])
print("Shape of X_train_essay_tfidf matrix after ",X_train_profile_tfidf.shape)
# print("Shape of X_cv_essay_tfidf matrix after ",X_cv_profile_tfidf.shape)
print("Shape of X_test_essay_tfidf matrix after ",X_test_profile_tfidf.shape)


# In[ ]:


# merge two sparse matrices: https://stackoverflow.com/a/19710648/4084039

# with the same hstack function we are concatinating a sparse matrix and a dense matirx :)
# X = hstack((categories_one_hot, sub_categories_one_hot, text_bow, price_standardized))
X_tr = hstack((X_train_employment_type_one_hot,X_train_required_experience_one_hot,X_train_required_education_one_hot,X_train_industry_one_hot,
              X_train_function_one_hot,X_train_country_one_hot,X_train_state_one_hot,X_train_description_tfidf,
              X_train_denifits_tfidf,X_train_requirements_tfidf,X_train_title_tfidf,X_train_profile_tfidf,
              X_train_['telecommuting'].values.reshape(-1,1),X_train_['has_company_logo'].values.reshape(-1,1),
               X_train_['has_questions'].values.reshape(-1,1))).tocsr()

X_te = hstack((X_test_employment_type_one_hot,X_test_required_experience_one_hot,X_test_required_education_one_hot,
              X_test_industry_one_hot,X_test_function_one_hot,X_test_country_one_hot,X_test_state_one_hot,
              X_test_description_tfidf,X_test_denifits_tfidf,X_test_requirements_tfidf,X_test_title_tfidf,
              X_test_profile_tfidf,X_test_['telecommuting'].values.reshape(-1,1),
              X_test_['has_company_logo'].values.reshape(-1,1),X_test_['has_questions'].values.reshape(-1,1))).tocsr()


# In[ ]:


X_train_upsample, y_train_upsample = SMOTE(random_state=42).fit_sample(X_tr, y_train_)
y_train_upsample.mean()


# In[ ]:


from collections import Counter
print(Counter(y_train_upsample))


# In[ ]:


kf = KFold(n_splits=10, random_state=42, shuffle=False)


# In[ ]:


params = {'alpha':[0.0001,0.001,0.01,0.1,1]}
def score_model(model, params, cv=None):
    """
    Creates folds manually, and upsamples within each fold.
    Returns an array of validation (recall) scores
    """
    if cv is None:
        cv = KFold(n_splits=5, random_state=42)

    smoter = SMOTE(random_state=42)
    
    scores = []

    for train_fold_index, val_fold_index in cv.split(X_tr, y_train_):
        # Get the training data
        X_train_fold, y_train_fold = X_tr[train_fold_index], y_train_[train_fold_index]
        # Get the validation data
        X_val_fold, y_val_fold = X_tr[val_fold_index], y_train_[val_fold_index]

        # Upsample only the data in the training section
        X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,
                                                                           y_train_fold)
        # Fit the model on the upsampled training data
        model_obj = model(**params).fit(X_train_fold_upsample, y_train_fold_upsample)
        # Score the model on the (non-upsampled) validation data
        score = f1_score(y_val_fold, model_obj.predict(X_val_fold))
        scores.append(score)
    return np.array(scores)


# Example of the model in action
# score_model(RandomForestClassifier, example_params, cv=kf)


# In[ ]:


score_tracker = []
for alpha in params['alpha']:
        example_params = {
            'alpha': alpha,
            'random_state': 13,
            'loss': 'log',
        }
        example_params['f1_score'] = score_model(SGDClassifier, 
                                               example_params, cv=kf).mean()
        score_tracker.append(example_params)
     
# What's the best model?
# print(score_tracker)
sorted(score_tracker, key=lambda x: x['f1_score'], reverse=True)[0]


# In[ ]:


# clf = SGDClassifier(alpha=0.0001,loss='log',random_state=13)
# clf.fit(X_train_upsample, y_train_upsample)
print("Train f1 score",f1_score(y_train_,clf.predict(X_tr))),("Test f1 score: ",f1_score(y_test_,clf.predict(X_te)))


# In[ ]:


params = {'n_estimators':[5, 10, 50, 100, 200],'max_depth':[2,3, 4, 5, 6, 7]}
def score_model(model, params, cv=None):
    """
    Creates folds manually, and upsamples within each fold.
    Returns an array of validation (recall) scores
    """
    if cv is None:
        cv = KFold(n_splits=5, random_state=42)

    smoter = SMOTE(random_state=42)
    
    scores = []

    for train_fold_index, val_fold_index in cv.split(X_tr, y_train_):
        # Get the training data
        X_train_fold, y_train_fold = X_tr[train_fold_index], y_train_[train_fold_index]
#         print('X_train_fold',X_train_fold.shape)
        # Get the validation data
        X_val_fold, y_val_fold = X_tr[val_fold_index], y_train_[val_fold_index]
#         print('X_val_fold',X_val_fold.shape)

        # Upsample only the data in the training section
        X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,
                                                                           y_train_fold)
#         print("X_train_fold_upsample",X_train_fold_upsample.shape)
        # Fit the model on the upsampled training data
#         print(model)
#         clf = model()
#         print(params)
        model_obj = model(**params).fit(X_train_fold_upsample, y_train_fold_upsample)
        # Score the model on the (non-upsampled) validation data
#         print('X_val_fold',X_val_fold.shape)
#         print('y_val_fold',y_val_fold.shape)
        score = f1_score(y_val_fold, model_obj.predict(X_val_fold))
        scores.append(score)
    return np.array(scores)

# Example of the model in action
# score_model(RandomForestClassifier, example_params, cv=kf)


# In[ ]:


# score_tracker = []
# for n_estimator in params['n_estimators']:
#     for max_depth in params['max_depth']:
#         example_params = {
#             'n_estimators': n_estimator,
#             'max_depth': max_depth,
#             'random_state': 13,
            
#         }
#         example_params['f1_score'] = score_model(XGBClassifier, 
#                                                example_params, cv=kf).mean()
#         score_tracker.append(example_params)
     
# # What's the best model?
# print(score_tracker)
# sorted(score_tracker, key=lambda x: x['f1_score'], reverse=True)[0]


# In[ ]:


from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold


# In[ ]:


kf = KFold(n_splits=5, shuffle=False)
# from sklearn.neighbors import KNeighborsClassifier
imba_pipeline = Pipeline([('smote',SMOTE(random_state=42)),('classifier',SGDClassifier())])

imba_pipeline


# In[ ]:


imba_pipeline.get_params().keys()


# In[ ]:


# y_train_upsample,clf.predict(X_train_upsample)
# cross_val_score(imba_pipeline, X_tr, y_train_, scoring=make_scorer(f1_score), cv=kf)


# In[ ]:


params = {
    'alpha': [0.0001,0.001,0.01,0.1,1],
    'loss':['log'],
    'penalty':['l1','l2'],
    'random_state': [13]
}
new_params = {'classifier__' + key: params[key] for key in params}
grid_imba = GridSearchCV(imba_pipeline, param_grid=new_params, cv=kf, scoring=make_scorer(f1_score),
                        return_train_score=True)

grid_imba.fit(X_tr, y_train_)


# In[ ]:


# estimator.get_params().keys()


# In[ ]:


grid_imba.cv_results_['mean_test_score'], grid_imba.cv_results_['mean_train_score']


# In[ ]:


#selecting the hyperparameter using GridSearch
# https://www.kaggle.com/arindambanerjee/grid-search-simplified
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

results = pd.DataFrame.from_dict(grid_imba.cv_results_)

# results4 = results4.sort_values(['param_alpha'])
alpha = list(grid_imba.cv_results_['param_classifier__alpha'].data)
# loss = list(grid_imba.cv_results_['param_classifier__loss'].data)

train_f1= results['mean_train_score']
# train_auc_std= results['std_train_score']
cv_f1 = results['mean_test_score'] 
# cv_auc_std= results['std_test_score']
# K =  results['param_classifier__alpha']
alpha = np.log10(alpha)
print(params['alpha'])
print(alpha_)
print(train_f1)

plt.plot(alpha, train_f1, label='Train f1 score')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
# plt.gca().fill_between(K, train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

plt.plot(alpha, cv_f1, label='CV f1 score')
# this code is copied from here: https://stackoverflow.com/a/48803361/4084039
# plt.gca().fill_between(K, cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')

plt.scatter(alpha, train_f1, label='Train f1 score')
plt.scatter(alpha, cv_f1, label='CV f1 score')


plt.legend()
plt.xlabel("alpha: hyperparameter")
plt.ylabel("F1 SCORE")
plt.title("Hyper parameter Vs F1 plot")
plt.grid()
plt.show()


sns.set_style("whitegrid")
# plt.figure(figsize=(16,6))
# plt.subplot(1,2,1)

# data = pd.DataFrame(data={'alpha':alpha, 'loss':loss, 'f1_score':grid_imba.cv_results_['mean_train_score']})

# data = data.reset_index().pivot_table(index='alpha', columns='loss', values='f1_score')
# sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('f1_score for Training data')
# plt.subplot(1,2,2)

# # Testing Heatmap
# data = pd.DataFrame(data={'alpha':alpha, 'loss':loss, 'f1_score':grid_imba.cv_results_['mean_test_score']})
# data = data.reset_index().pivot_table(index='alpha', columns='loss', values='f1_score')
# sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('f1_score for Test data')
# plt.show()

results.head()


# In[ ]:


grid_imba.best_score_


# In[ ]:


grid_imba.best_params_


# In[ ]:


clf = SGDClassifier(alpha=0.001,loss='log',random_state=13)
clf.fit(X_train_upsample, y_train_upsample)
print("Train f1 score",f1_score(y_train_upsample,clf.predict(X_train_upsample))),("Test f1 score: ",f1_score(y_test_,clf.predict(X_te)))


# In[ ]:


y_test_predict = grid_imba.best_estimator_.predict(X_te)
y_train_predict = grid_imba.best_estimator_.predict(X_tr)


# In[ ]:


f1_score(y_test_, y_test_predict)


# In[ ]:


f1_score(y_train_, y_train_predict)


# In[ ]:


f1_score(y_train_upsample,clf.predict(X_train_upsample))


# In[ ]:


kf = KFold(n_splits=5, shuffle=False)
from sklearn.neighbors import KNeighborsClassifier
imba_pipeline = Pipeline([('smote',SMOTE(random_state=42)),('xgbclassifier',XGBClassifier())])

imba_pipeline


# In[ ]:


params = {
    'n_estimators':[5, 10, 50, 100, 200],
    'max_depth':[2,3, 4, 5, 6, 7]
}
new_params = {'xgbclassifier__' + key: params[key] for key in params}
grid_imba = GridSearchCV(imba_pipeline, param_grid=new_params, cv=kf, scoring=make_scorer(f1_score),
                        return_train_score=True)

grid_imba.fit(X_tr, y_train_)


# In[ ]:


grid_imba.cv_results_['mean_test_score'], grid_imba.cv_results_['mean_train_score']


# In[ ]:


#selecting the hyperparameter using GridSearch
# https://www.kaggle.com/arindambanerjee/grid-search-simplified
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

results2 = pd.DataFrame.from_dict(grid_imba.cv_results_)

# results4 = results4.sort_values(['param_alpha'])
max_depth_list = list(grid_imba.cv_results_['param_xgbclassifier__max_depth'].data)
n_estimator_list = list(grid_imba.cv_results_['param_xgbclassifier__n_estimators'].data)


sns.set_style("whitegrid")
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)

data = pd.DataFrame(data={'Number of Estimator':n_estimator_list, 'Max Depth':max_depth_list, 'f1_score':grid_imba.cv_results_['mean_train_score']})
data = data.reset_index().pivot_table(index='Max Depth', columns='Number of Estimator', values='f1_score')
sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('f1_score for Training data')
plt.subplot(1,2,2)

# Testing Heatmap
data = pd.DataFrame(data={'Number of Estimator':n_estimator_list, 'Max Depth':max_depth_list, 'f1_score':grid_imba.cv_results_['mean_test_score']})
data = data.reset_index().pivot_table(index='Max Depth', columns='Number of Estimator', values='f1_score')
sns.heatmap(data, annot=True, cmap="YlGnBu").set_title('f1_score for Test data')
plt.show()

results2.head()


# In[ ]:


grid_imba.best_score_


# In[ ]:


grid_imba.best_params_


# In[ ]:


y_test_predict = grid_imba.best_estimator_.predict(X_te)
y_train_predict = grid_imba.best_estimator_.predict(X_tr)


# In[ ]:


f1_score(y_test_, y_test_predict)


# In[ ]:


f1_score(y_train_, y_train_predict)


# In[ ]:


f1_score(y_train_upsample,clf.predict(X_train_upsample))


# In[ ]:


from sklearn.metrics import recall_score, roc_auc_score
print(roc_auc_score(y_train_,grid_imba.predict_proba(X_tr)[:,1]))
# print ('Cross validation AUC for Random Forest model : ',np.mean(cross_val_score(grid_imba,X_tr,y_train_,scoring='roc_auc',cv=10)))
print(roc_auc_score(y_test_,grid_imba.predict_proba(X_te)[:,1]))


# ### lightgbm

# In[ ]:


kf = KFold(n_splits=5, shuffle=False)
# from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
imba_pipeline = Pipeline([('smote',SMOTE(random_state=42)),('lgbclassifier',lgb.LGBMClassifier())])

imba_pipeline


# In[ ]:


params = {'num_leaves':[6,8,12,16],'learning_rate':[0.01,0.03,0.05,0.1,0.15,0.2],
          'n_estimators':[5, 10, 50, 100, 200],'max_depth':[3,5,10]}

# params = {
#     'n_estimators':[5, 10, 50, 100, 200],
#     'max_depth':[2,3, 4, 5, 6, 7]
# }
new_params = {'lgbclassifier__' + key: params[key] for key in params}
grid_imba = GridSearchCV(imba_pipeline, param_grid=new_params, cv=kf, scoring=make_scorer(f1_score),
                        return_train_score=True)

grid_imba.fit(X_tr, y_train_)


# In[ ]:




