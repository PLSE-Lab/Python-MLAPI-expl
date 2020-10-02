#!/usr/bin/env python
# coding: utf-8

# ### Kaggle ML and DS Survey for 2019
# 
# As a Data Science Enthusiast I am very exicited to explore Kaggle survery 2019 to build new and/or strengthen my skills.In this notebook I explored various aspects of Data Science and Machine Learning mainly based on top 8 countires have more than 500 participants,Designation,Education,Company manpower,ML/DS Implementation Status and varity of DS/ML skills.My keen focus is on how Data Scientists and DS Enthusiasts working towards achieving common goals.
# As we all know that most of the time Job postings inlcude varity of skills and Education that are not really relavent for ML/DS role ,Infact this really holds back for Job Seekers like me.In order to clearly understand the basic and actual skills which are practically necesssary therefore I am comparing in each findings against actual Data Scientists by designation to do so used simple bar plots to measure the findings. 
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


questions = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')
questions_ = questions.T
questions_.head()


# In[ ]:


# Multiple Choice Response
multiple_resp = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
#multiple_resp.head()


# In[ ]:


multiple_resp.drop(columns=['Time from Start to Finish (seconds)'],inplace=True)
multiple_resp.drop(multiple_resp.index[0],inplace=True)
#multiple_resp.head()


# ### Age Distribution
# From below plot following age group participants in descending order
# 25-29 -> Typically 4 to 9 years of experience professionals   
# 22-24 -> 1 to 3 years of experience 
# 30-34 -> above 9+ years of experience   
# 18-21 -> most probably freshers who recently joined or students   
# 35-39 -> above 14+ years of experience and so on
# 
# 

# In[ ]:


# Age Group Distribution
fig,ax = plt.subplots(1)
multiple_resp.Q1.value_counts().plot(kind='bar',ax=ax)
ax.set_title("Age Distribution")
ax.set(xlabel='Age Bins',ylabel='Frequency')
plt.show()


# ### Age vs Gender Distribution
# 
# Male gender participants are higher in the age group 21 to 44 in a range 1000+ to 3500+ and female are in a range 500+ to 800.

# In[ ]:


# Age vs Gender comparision
plt.figure(figsize=(12,8))
sns.countplot(x='Q1',hue='Q2',data=multiple_resp)
plt.title("Age vs Gender Distribution")
plt.xlabel("Age Bins")
plt.ylabel("Frequency")
plt.show()


# ### Country Distribution
# 
# Most of the participants are from India and USA .

# In[ ]:


# Country Distribution
plt.figure(figsize=(12,10))
multiple_resp.Q3.value_counts(ascending=True).plot(kind='barh')
plt.show()


# ### Country vs Age Distribution
# 
# Majority are from 18-44 .Let me deep dive to see more clear picture of countires having more than 500 participants.

# In[ ]:


# Country vs Age Distribution

fig,ax = plt.subplots(figsize=(16,8))
ax1 = sns.countplot(x='Q3',hue='Q1',data=multiple_resp,ax=ax)
labels = ax1.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Country vs Age Distribution")
ax.set(xlabel='Country',ylabel='Age Frequency')
ax.legend(loc='right')
plt.show()


# In[ ]:


# Extract countries which have more than 500 participants

countries = multiple_resp.Q3.value_counts(ascending=False)
top10_countries = countries[countries>500].index.values.tolist()
top10_countries
multiple_resp_top10_countries = multiple_resp[(multiple_resp.Q3.isin(top10_countries))]


# ### Top 8 countries having more than 500 participants
# 
# Top8 countires vs Age distribution indicates india have higher in the range of 1200+ to 1300 between age range 21 to 34.
# For USA more density is at 200 frequency level for most of the age group.

# In[ ]:


# Top 8 countires and respective Age distribution

fig,ax = plt.subplots(figsize=(16,8))
ax1 = sns.countplot(x='Q3',hue='Q1',data=multiple_resp_top10_countries,ax=ax)
labels = ax1.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Country vs Age Distribution")
ax.set(xlabel='Country',ylabel='Age Frequency')
ax.legend(loc='right')
plt.show()


# ### Top 8 Countries vs Gender Distribution
# 
# For top 8 countires male gender are higher as compared to female gender in which India and USA are better in Female gender as compared to other countries.Please note this is just from a survery who intended to participate however there may be possibility that most of them not have participated(possible).

# In[ ]:


# Top 8 countires and respective Gender distribution

fig,ax = plt.subplots(figsize=(16,8))
ax1 = sns.countplot(x='Q3',hue='Q2',data=multiple_resp_top10_countries,ax=ax)
labels = ax1.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Country vs Gender Distribution")
ax.set(xlabel='Country',ylabel='Gender Frequency')
ax.legend(loc='right')
plt.show()


# ### Education Background
# 
# Most of the participants have Master's degree,Bachelor's degree and Doctoral degree.

# In[ ]:


# Highest Level Of Education

multiple_resp.Q4.value_counts().plot(kind='bar')
plt.title("Education Background")
plt.show()


# ### Top 8 countires vs Education
# 
# Bachelors degree participants are very high in Inida as compared to USA and Master's Degree are almost same in both the countries.
# Doctoral degree holders are high in USA as compared with other countries.

# In[ ]:


# Country vs Education Distribution
fig,ax = plt.subplots(figsize=(16,8))
ax1 = sns.countplot(x='Q3',hue='Q4',data=multiple_resp_top10_countries,ax=ax)
labels = ax1.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Country vs Education Distribution")
ax.set(xlabel='Country',ylabel='Education Frequency')
ax.legend(loc='right')
plt.show()


# ### Employment Distribution
# 
# Emploment distribution shows that majority of Inida participants are Students and in each top8 counties student and Sofftware engineer participants are competative in number.Actual employeed as Data Scientists are almost same for india and USA.

# In[ ]:


# Current employment role
#multiple_resp.Q5.value_counts().plot(kind='bar')

fig,ax = plt.subplots(figsize=(16,8))
ax1 = sns.countplot(x='Q3',hue='Q5',data=multiple_resp_top10_countries,ax=ax)
labels = ax1.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Country vs Employment Distribution")
ax.set(xlabel='Country',ylabel='Employment Frequency')
ax.legend(loc='right')
plt.show()


# ### Employees Size/Company Size
# 
# In India and USA most of the participants are from MNCs(>10000 and 1000-9999 employees) and may be Startups (0-49 employees).

# In[ ]:


# Country vs Size of the company

fig,ax = plt.subplots(figsize=(16,8))
ax1 = sns.countplot(x='Q3',hue='Q6',data=multiple_resp_top10_countries,ax=ax)
labels = ax1.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Country vs Size of company Distribution")
ax.set(xlabel='Country',ylabel='Company Size Frequency')
ax.legend(loc='right')
plt.show()


# ### Data Science role based workload distribution
# 
# It is quite interest to see 20+ and 1-2 workloads are almost in the same range and at the other end 10-14 and 15-19 data science workloads probably this category may be combination of all companies.

# In[ ]:


#Approximately how many individuals are responsible for data science workloads at your place of business?

multiple_resp.Q7.value_counts().plot(kind='bar')
plt.title("Data Science manpower size")
plt.xlabel("Workload Size bins")
plt.ylabel("Frequency")
plt.show()


# 
# By combining Workload and Size of company ,20+ workloads are from MNCs(>10000 employees and 1000-9,999 employees) and workloads 1-2 size is from 0-49 employees(Startups).
# Yes workloads 10-14 and 15-19 and rest are combination of all size companies.

# In[ ]:


# Workload Size vs Size off company Distribution
fig,ax = plt.subplots(figsize=(16,8))
ax1 = sns.countplot(x='Q7',hue='Q6',data=multiple_resp_top10_countries,ax=ax)
labels = ax1.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Workload Size vs Size of company Distribution")
ax.set(xlabel='Workload',ylabel='Company Size Frequency')
ax.legend(loc='right')
plt.show()


# ### Actual implementation Status Distribution
# 
# MNCs with 20+ workloads have well Established ML models in Production and few are exploring,generating insights and planning to implement in production.For 1-2 workloads and other workloads are combination of implemented in production ,exploring generating insights or planning to implement.

# In[ ]:


# Workload vs Implementation Status
fig,ax = plt.subplots(figsize=(16,8))
ax1 = sns.countplot(x='Q7',hue='Q8',data=multiple_resp_top10_countries,ax=ax)
labels = ax1.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Workload Size vs Implementation Distribution")
ax.set(xlabel='Workload',ylabel='Implementation Frequency')
ax.legend(loc='upper right')
plt.show()


# ### Programming Languages regularly Used
# 
# 1. From Employees Size point of view , the programming language for Data Science appears to be Python,SQL and R in order .So majority of them are using regulary python followed by SQL and R.
# 
# 2. From Designation point of vie ,this is similar to Employee size where in Python and SQL are major usage and then R.Specially Data Scientists are using Python,SQL and R and Students as well.

# In[ ]:


Q18_parts = ['Q18_Part_1', 'Q18_Part_2',
       'Q18_Part_3', 'Q18_Part_4', 'Q18_Part_5', 'Q18_Part_6',
       'Q18_Part_7', 'Q18_Part_8', 'Q18_Part_9', 'Q18_Part_10',
       'Q18_Part_11', 'Q18_Part_12']

#[multiple_resp_top10_countries.rename({q:multiple_resp_top10_countries[q].unique().tolist()[1]},inplace=True) for q in Q18_parts]


# In[ ]:


#[multiple_resp_top10_countries.rename({q:multiple_resp_top10_countries[q].unique().tolist()[1]},inplace=True) for q in Q18_parts]
multiple_resp_top10_countries_copy=multiple_resp_top10_countries.copy()
for q in Q18_parts:
    multiple_resp_top10_countries_copy.rename(columns={q:multiple_resp_top10_countries_copy[q].unique().tolist()[1]},inplace=True)
    #print("Q:",q)
multiple_resp_top10_countries_copy.head()


# In[ ]:


# distribution of employee size vs Programming Language Usage
ax = multiple_resp_top10_countries_copy.groupby(['Q6'],as_index=True)['Python', 'R', 'SQL', 'C', 'C++',
       'Java', 'Javascript', 'TypeScript', 'Bash', 'MATLAB'].count().plot(kind='bar',figsize=(12,8))
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Employees Size vs Programming Languages daily usage")
ax.set(xlabel="Employees Size",ylabel="Frequency of Prg Lang Usage")
plt.show()


# In[ ]:


# distribution of designation vs Programming Language Usage
ax = multiple_resp_top10_countries_copy.groupby(['Q5'],as_index=True)['Python', 'R', 'SQL', 'C', 'C++',
       'Java', 'Javascript', 'TypeScript', 'Bash', 'MATLAB'].count().plot(kind='bar',figsize=(12,8))
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Designation vs Programming Languages daily usage")
ax.set(xlabel="Designation",ylabel="Frequency of Prg Lang Usage")
plt.show()


# ### Programming Languages Recommendation
# 
# Though Students recommend Python as the highest PL for enthusiast Data Scientsts . I look at Data Scientsts , Software Engineer and Data Analyst who recommended Python as prefered prefered language.
# 
# As a aspiring Data Scientist this really helps me to strengthen my Python skills as it is regulary used and recommended by Data Scientists who regulary use python.

# In[ ]:


multiple_resp_top10_countries_copy.groupby(['Q5'],as_index=True)['Q19'].count()


# In[ ]:


#What programming language would you recommend an aspiring data scientist to learn first?
fig,ax = plt.subplots(figsize=(12,8))
#ax = multiple_resp_top10_countries_copy.groupby(['Q5'])['Q19'].value_counts().plot(kind='bar',figsize=(12,8))
ax1 = sns.countplot(x='Q5',hue='Q19',data=multiple_resp_top10_countries_copy,ax=ax)
labels = ax1.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Designation vs Programming Languages Recommendation")
ax.set(xlabel="Designation",ylabel="Frequency of Prg Lang Recommendation")
plt.legend(loc='upper right')
plt.show()


# ### Machine Learning Algorithms regular Usage
# 
# Following are more regularly used ML algorithms among Data Scientists and Students
# 
# 1. Linear or Logistic Regression
# 2. Decision Trees or Random Forests
# 3. Gradient Boosting Machines (xgboost, lightgbm, etc)
# 4. Bayesian Approaches
# 5. Dense Neural Networks (MLPs, etc)'
# 6. Convolutional Neural Networks
# 7. Recurrent Neural Networks
# 

# In[ ]:


# Q24 Which of the following ML algorithms do you use on a regular basis? (Select all that apply)

Q24_parts = ['Q24_Part_1',
       'Q24_Part_2', 'Q24_Part_3', 'Q24_Part_4', 'Q24_Part_5',
       'Q24_Part_6', 'Q24_Part_7', 'Q24_Part_8', 'Q24_Part_9',
       'Q24_Part_10', 'Q24_Part_11', 'Q24_Part_12']
#multiple_resp_top10_countries_copy[Q24_parts].head()


# In[ ]:


for q in Q24_parts:
    multiple_resp_top10_countries_copy.rename(columns={q:multiple_resp_top10_countries_copy[q].unique()[1]},inplace=True)

#multiple_resp_top10_countries_copy.columns.values


# In[ ]:


# distribution of disgnation vs ML Algorithms
ml_algorithms = ['Linear or Logistic Regression',
                 'Decision Trees or Random Forests',
                 'Gradient Boosting Machines (xgboost, lightgbm, etc)',
                 'Bayesian Approaches',
                 'Evolutionary Approaches',
                 'Dense Neural Networks (MLPs, etc)',
                 'Convolutional Neural Networks',
                 'Generative Adversarial Networks',
                 'Recurrent Neural Networks',
                 'Transformer Networks (BERT, gpt-2, etc)']

ax = multiple_resp_top10_countries_copy.groupby(['Q5'],as_index=True).count()[ml_algorithms].plot(kind='bar',figsize=(14,10))
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Designation vs ML Algorithms")
ax.set(xlabel="Designation",ylabel="Frequency ML Algorithms")
plt.show()


# ### Automated Machine Learning Tools
# 
# Data Scientists and other designations use most of the listed ML tools on a regular basis.

# In[ ]:


# Q25 Which categories of ML tools do you use on a regular basis? (Select all that apply)

Q25_parts = ['Q25_Part_1', 'Q25_Part_2', 'Q25_Part_3',
       'Q25_Part_4', 'Q25_Part_5', 'Q25_Part_6', 'Q25_Part_7',
       'Q25_Part_8']
#multiple_resp_top10_countries_copy[Q25_parts].head()


# In[ ]:


for q in Q25_parts:
    multiple_resp_top10_countries_copy.rename(columns={q:multiple_resp_top10_countries_copy[q].unique()[1]},inplace=True)

#multiple_resp_top10_countries_copy.columns.values


# In[ ]:


# Distribution of Designation vs ML tools
ml_tools=['Automated data augmentation (e.g. imgaug, albumentations)',
       'Automated feature engineering/selection (e.g. tpot, boruta_py)',
       'Automated model selection (e.g. auto-sklearn, xcessiv)',
       'Automated model architecture searches (e.g. darts, enas)',
       'Automated hyperparameter tuning (e.g. hyperopt, ray.tune)',
       'Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)']

ax = multiple_resp_top10_countries_copy.groupby(['Q5'],as_index=True).count()[ml_tools].plot(kind='bar',figsize=(14,10))
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Designation vs ML Tools")
ax.set(xlabel="Designation",ylabel="Frequency ML Tools")
plt.show()


# ### Computer Vision methods
# 
# Data Scientists use all Computer vision methods and obviously higher is Transfer Learning method(VGG ,Inception etc) for reusability and faster training the DL models.

# In[ ]:


# 26 Which categories of computer vision methods do you use on a regular basis? (Select all that apply)

Q26_parts = ['Q26_Part_1', 'Q26_Part_2',
       'Q26_Part_3', 'Q26_Part_4', 'Q26_Part_5', 'Q26_Part_6',
       'Q26_Part_7']

#multiple_resp_top10_countries_copy[Q26_parts].head()


# In[ ]:


for q in Q26_parts:
    print(multiple_resp_top10_countries_copy[q].unique()[1])
    multiple_resp_top10_countries_copy.rename(columns={q:multiple_resp_top10_countries_copy[q].unique()[1]},inplace=True)

#multiple_resp_top10_countries_copy.columns.values


# In[ ]:


computer_vision_methods = ['General purpose image/video tools (PIL, cv2, skimage, etc)',
       'Image segmentation methods (U-Net, Mask R-CNN, etc)',
       'Object detection methods (YOLOv3, RetinaNet, etc)',
       'Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)',
       'Generative Networks (GAN, VAE, etc)']

ax = multiple_resp_top10_countries_copy.groupby(['Q5'],as_index=True).count()[computer_vision_methods].plot(kind='bar',figsize=(14,10))
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Designation vs Computer vision Methods")
ax.set(xlabel="Designation",ylabel="Frequency Computer Vision Methods")
plt.show()


# ### Natural Language Processing Methods
# 
# Data Scientist use all of the listed NLP methods Word embeddings/vectors ,Encoder-decorder models,Contextualized embeddings and Transformer language models and there is a similarity usage between Software Engineer and Students.

# In[ ]:


# Q27 Which of the following natural language processing (NLP) methods do you use on a regular basis? (Select all that apply)

Q27_parts = ['Q27_Part_1', 'Q27_Part_2', 'Q27_Part_3',
       'Q27_Part_4', 'Q27_Part_5', 'Q27_Part_6']
#multiple_resp_top10_countries_copy[Q27_parts].head()


# In[ ]:


for q in Q27_parts:
    print(multiple_resp_top10_countries_copy[q].unique()[1])
    multiple_resp_top10_countries_copy.rename(columns={q:multiple_resp_top10_countries_copy[q].unique()[1]},inplace=True)
    
#multiple_resp_top10_countries_copy.columns.values


# In[ ]:


nlp_methods=['Word embeddings/vectors (GLoVe, fastText, word2vec)',
       'Encoder-decorder models (seq2seq, vanilla transformers)',
       'Contextualized embeddings (ELMo, CoVe)',
       'Transformer language models (GPT-2, BERT, XLnet, etc)']

ax = multiple_resp_top10_countries_copy.groupby(['Q5'],as_index=True).count()[nlp_methods].plot(kind='bar',figsize=(14,10))
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Designation vs NLP Methods")
ax.set(xlabel="Designation",ylabel="Frequency NLP Methods")
plt.show()


# ### Machine Learning Frameworks
# 
# Data Scientists obviously use skikit learn and Tensorflow,Keras,Randomforest,xgboost are used regularly.And even Students and Software Engineers as well.
# 

# In[ ]:


# Q28 Which of the following machine learning frameworks do you use on a regular basis? (Select all that apply)

Q28_parts = ['Q28_Part_1', 'Q28_Part_2',
       'Q28_Part_3', 'Q28_Part_4', 'Q28_Part_5', 'Q28_Part_6',
       'Q28_Part_7', 'Q28_Part_8', 'Q28_Part_9', 'Q28_Part_10',
       'Q28_Part_11', 'Q28_Part_12']

#multiple_resp_top10_countries_copy[Q28_parts].head()


# In[ ]:


for q in Q28_parts:
    print(multiple_resp_top10_countries_copy[q].unique()[1].strip())
    multiple_resp_top10_countries_copy.rename(columns={q:multiple_resp_top10_countries_copy[q].unique()[1].strip()},inplace=True)

#multiple_resp_top10_countries_copy.columns.values


# In[ ]:


ml_frameworks=['Scikit-learn', 'TensorFlow', 'Keras',
       'RandomForest', 'Xgboost', 'PyTorch', 'Caret', 'LightGBM',
       'Spark MLib', 'Fast.ai']

ax = multiple_resp_top10_countries_copy.groupby(['Q5'],as_index=True).count()[ml_frameworks].plot(kind='bar',figsize=(14,10))
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Designation vs ML Frameworks")
ax.set(xlabel="Designation",ylabel="Frequency ML Frameworks")
plt.show()


# ### Cloud Computing Platforms
# 
# Majority of Data Scientists and Softare Engineers use AWS ,GCP and Azure.Since these platforms are charged per usage most of the job seekers do not opt.Therefore Students and Not Employeed are zero .In this case probably Students and Not Employeed opt for training.

# In[ ]:


# Q29 Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply)

Q29_parts = ['Q29_Part_1', 'Q29_Part_2', 'Q29_Part_3', 'Q29_Part_4',
       'Q29_Part_5', 'Q29_Part_6', 'Q29_Part_7', 'Q29_Part_8',
       'Q29_Part_9', 'Q29_Part_10', 'Q29_Part_11', 'Q29_Part_12']

#multiple_resp_top10_countries_copy[Q29_parts].head()


# In[ ]:


for q in Q29_parts:
    print(multiple_resp_top10_countries_copy[q].unique()[1].strip())
    multiple_resp_top10_countries_copy.rename(columns={q:multiple_resp_top10_countries_copy[q].unique()[1].strip()},inplace=True)

#multiple_resp_top10_countries_copy.columns.values


# In[ ]:


cloud_platforms=['Google Cloud Platform (GCP)', 'Amazon Web Services (AWS)',
       'Microsoft Azure', 'IBM Cloud', 'Alibaba Cloud',
       'Salesforce Cloud', 'Oracle Cloud', 'SAP Cloud', 'VMware Cloud',
       'Red Hat Cloud']

ax = multiple_resp_top10_countries_copy.groupby(['Q5'],as_index=True).count()[cloud_platforms].plot(kind='bar',figsize=(14,10))
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Designation vs Cloud Computing Platforms")
ax.set(xlabel="Designation",ylabel="Frequency of Cloud Platforms")
plt.show()


# ### Machine Learning Products
# 
# Most of these ML products are used regulary across designations.Data Scientists and Software Engineers are more frequent users.Again Students and Not Employeed do not opt these as these are charged per usage.
# 

# In[ ]:


# Q32 Which of the following machine learning products do you use on a regular basis? (Select all that apply)

Q32_parts = ['Q32_Part_1', 'Q32_Part_2', 'Q32_Part_3',
       'Q32_Part_4', 'Q32_Part_5', 'Q32_Part_6', 'Q32_Part_7',
       'Q32_Part_8', 'Q32_Part_9', 'Q32_Part_10', 'Q32_Part_11',
       'Q32_Part_12']

#multiple_resp_top10_countries_copy[Q32_parts].head()
for q in Q32_parts:
    print(multiple_resp_top10_countries_copy[q].unique()[1])
    multiple_resp_top10_countries_copy.rename(columns={q:multiple_resp_top10_countries_copy[q].unique()[1]},inplace=True)

#multiple_resp_top10_countries_copy.columns.values


# In[ ]:


ml_products=['SAS', 'Cloudera',
       'Azure Machine Learning Studio',
       'Google Cloud Machine Learning Engine', 'Google Cloud Vision',
       'Google Cloud Speech-to-Text', 'Google Cloud Natural Language',
       'RapidMiner', 'Google Cloud Translation', 'Amazon SageMaker']

ax = multiple_resp_top10_countries_copy.groupby(['Q5'],as_index=True).count()[ml_products].plot(kind='bar',figsize=(14,10))
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Designation vs ML Products")
ax.set(xlabel="Designation",ylabel="Frequency of ML Products")
plt.show()


# ### Relational Database Products
# 
# By comparing the ML Products with Relational DB products, These products are packages of ML products.

# In[ ]:


# Q34 Which of the following relational database products do you use on a regular basis? (Select all that apply)

Q34_parts = ['Q34_Part_1',
       'Q34_Part_2', 'Q34_Part_3', 'Q34_Part_4', 'Q34_Part_5',
       'Q34_Part_6', 'Q34_Part_7', 'Q34_Part_8', 'Q34_Part_9',
       'Q34_Part_10', 'Q34_Part_11', 'Q34_Part_12']

for q in Q34_parts:
    print(multiple_resp_top10_countries_copy[q].unique()[1])
    multiple_resp_top10_countries_copy.rename(columns={q:multiple_resp_top10_countries_copy[q].unique()[1]},inplace=True)
    
#multiple_resp_top10_countries_copy.columns.values


# In[ ]:


rl_db_products=['MySQL',
       'PostgresSQL', 'SQLite', 'Microsoft SQL Server', 'Oracle Database',
       'Microsoft Access', 'AWS Relational Database Service',
       'AWS DynamoDB', 'Azure SQL Database', 'Google Cloud SQL']

ax = multiple_resp_top10_countries_copy.groupby(['Q5'],as_index=True).count()[rl_db_products].plot(kind='bar',figsize=(14,10))
labels = ax.get_xticklabels()
ax.set_xticklabels(labels,rotation=90)
ax.set_title("Designation vs RL DB Products")
ax.set(xlabel="Designation",ylabel="Frequency of RL DB Products")
plt.show()


# This concludes my analysis.

# In[ ]:




