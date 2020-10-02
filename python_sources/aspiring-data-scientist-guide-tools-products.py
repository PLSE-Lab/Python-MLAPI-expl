#!/usr/bin/env python
# coding: utf-8

# **Kaggle ML & DS Survey
# **

# **This is the third time Kaggle Community attached Machine Learning and Data Science Survey.** 
# 
# Affter reading the questions I decided to explorer basically dataset but to focus on **technologies** which Kaggle's users use. 
# As an aspiring **Data Scientist** I want to get knowledge how to become well prepared, perfectly learned future Enginner in Data and Machine Learning. What is more I want to share with Kaggle this outputs to connect all of the beginner, because explorind data is amazingly siphisticated and addictive. 

# ![Data-Scientists.jpg](attachment:Data-Scientists.jpg)

# In[ ]:


from __future__ import division
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objects as go
import plotly.tools as tls


# In[ ]:


data = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', encoding = "utf-8")


# In[ ]:


data = data.drop(['Q2_OTHER_TEXT', 'Q5_OTHER_TEXT', 'Q9_OTHER_TEXT', 'Q13_OTHER_TEXT', 'Q14_Part_1_TEXT', 'Q14_Part_2_TEXT', 'Q14_Part_3_TEXT', 'Q14_Part_4_TEXT', 
           'Q14_Part_5_TEXT', 'Q14_OTHER_TEXT', 'Q17_OTHER_TEXT', 'Q18_OTHER_TEXT', 
          'Q19_OTHER_TEXT', 'Q20_OTHER_TEXT', 'Q21_OTHER_TEXT', 'Q24_OTHER_TEXT',
          'Q25_OTHER_TEXT', 'Q26_OTHER_TEXT', 'Q27_OTHER_TEXT', 'Q28_OTHER_TEXT', 'Q29_OTHER_TEXT',
          'Q30_OTHER_TEXT', 'Q31_OTHER_TEXT', 'Q32_OTHER_TEXT', 'Q33_OTHER_TEXT', 
          'Q34_OTHER_TEXT'], axis=1)


# 1. General analysis of Kaggle Survey 2019

# 1.1. Age distribution between Kaggle users

# In[ ]:


plt.figure(figsize=(15,8))
vis1 = sns.countplot(data['Q1'].iloc[1:].sort_values(ascending=True), palette='summer')
plt.xlabel('Age', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=0, size=13)
plt.title('Age distribution', fontsize=15)

for p in vis1.patches:
    vis1.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# - As we can see the biggest number of Kaggle users we can see in range 18 - 39
# - There are also users after 40
# - This is amazing that Kaggle community embrace people in every age - it means that we can learn many things from probalby people who are more experienced than us
# - Fo aspiring Data Scientists Kaggle Community can become amazing public forum to gain knowledge!

# 1.2. Gender distribution

# In[ ]:


plt.figure(figsize=(15,8))
vis2 = sns.countplot(data['Q2'].iloc[1:].sort_values(ascending=True), palette='summer')
plt.xlabel('Gender', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=0, size=13)
plt.title('Gender distribution', fontsize=15)

for p in vis2.patches:
    vis2.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# In[ ]:


male_count = len(data[data['Q2'] == 'Male'])
female_count = len(data[data['Q2'] == 'Female'])

print('Percentage of female: {:.2f} %' .format(female_count/len(data['Q2'].iloc[1:])*100))
print('Percentage of male: {:.2f} %' .format(male_count/len(data['Q2'].iloc[1:])*100))


# - More than 80 % of people here are Males
# - Number of Females is much more lower - only about 16% of Kaggle Community

# 1.3. Country distribution

# In[ ]:


plt.figure(figsize=(15,8))
data1 = pd.DataFrame(data.iloc[1:]['Q3'].value_counts().sort_values(ascending=False)).reset_index().head(20)
vis3 = sns.barplot(data1['index'], data1.Q3, palette='summer')
plt.xlabel('Top countries', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=80)
plt.title('Country distribution', fontsize=15)

for p in vis3.patches:
    vis3.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# - The biggest community on Kaggle create India and USA
# - There are also people here from Brazil, Asian and West Countries. 
# - The smallest community is from Europe
# - I am from Poland and since now I have never met here someone from my country :)

# 1.4. Education of Kaggle users

# In[ ]:


data2 = pd.DataFrame(data.iloc[1:]['Q4'].value_counts().sort_values(ascending=False)).reset_index().head(25)
data2.head()
plt.figure(figsize=(15,8))
vis4 = sns.barplot(y=data2['index'], x=data2.Q4, palette='summer')
plt.xlabel('Count', fontsize=15)
plt.ylabel('', fontsize=15)
plt.xticks(rotation=0, fontsize=13)
plt.yticks(rotation=0, fontsize=15)
plt.title('Education distribution', fontsize=20)

# for p in vis4.patches:
#     vis4.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
#                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# - Majority of people here are with Master Degree
# - On the second place we can see Bachelors
# - Amazing is that also Doctoral and Professional degrees are among us

# 1.5. Kaggle's users' jobs

# In[ ]:


data3 = pd.DataFrame(data.iloc[1:]['Q5'].value_counts().sort_values(ascending=False)).reset_index().head(25)
plt.figure(figsize=(15,8))
sns.barplot(y=data3['index'], x=data3.Q5, palette='summer')
plt.xlabel('Count', fontsize=15)
plt.ylabel('', fontsize=15)
plt.xticks(rotation=0, fontsize=13)
plt.yticks(rotation=0, fontsize=15)
plt.title('Job of Kaggle users', fontsize=20)


# - First place is for Data Scientists
# - Second is for Students - so for people who are interested in analysis or just like me - are aspiring Data Scientists

# In[ ]:


data4 = pd.DataFrame(data.loc[1:, 'Q16_Part_1': 'Q16_Part_12']).reset_index()
data4.drop('index', axis=1, inplace=True)
data4.head()
a =pd.DataFrame(pd.value_counts(data4.values.flatten())).reset_index()
a.columns = ['A', 'B']


# 2. Programming basics for aspiring Data Scientists

# I decided to focus this Notebook on later questions about programming languages, methods, tools etc to create** KAGGLE'S GUIDE FOR ASPIRING DATA SCIENTIST**.
# After that everyone who wants to change job, get first job with Data will be now sure what are the best things to learn and master!

# 2.1. Development Environments

# In[ ]:


plt.figure(figsize=(15,8))
vis5 = sns.barplot(data=a, x='A', y='B', palette='summer')
plt.xlabel('Development environments', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=80)
plt.title('The most popular development environments', fontsize=15)

for p in vis5.patches:
    vis5.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. Jupyter
# 2. Visual Studio
# 3. RStudio
# 4. Pycharm
# 
# The advantage has of course Jupyter!

# 2.2. Notebook products

# In[ ]:


data5 = pd.DataFrame(data.loc[1:, 'Q17_Part_1': 'Q17_Part_12']).reset_index()
data5.drop('index', axis=1, inplace=True)
data55 =pd.DataFrame(pd.value_counts(data5.values.flatten())).reset_index()
data55.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis6 = sns.barplot(data=data55, x='A', y='B', palette='summer')
plt.xlabel('Notebook products', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=80)
plt.title('The most popular notebook products', fontsize=15)

for p in vis6.patches:
    vis6.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. None
# 2. Kaggle kernels
# 3. Google Colab

# 2.3. Programming language

# In[ ]:


data6 = pd.DataFrame(data.loc[1:, 'Q18_Part_1': 'Q18_Part_12']).reset_index()
data6.drop('index', axis=1, inplace=True)
data66 =pd.DataFrame(pd.value_counts(data6.values.flatten())).reset_index()
data66.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis7 = sns.barplot(data=data66, x='A', y='B', palette='summer')
plt.xlabel('Programming languages', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=50)
plt.title('The most popular programming languages', fontsize=15)

for p in vis7.patches:
    vis7.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. Python
# 2. SQL
# 3. R
# 4. Java
# 
# The most popular languages are of course Python and SQL. 

# * * 2.4. Data visualisation methods, tools

# In[ ]:


data8 = pd.DataFrame(data.loc[1:, 'Q20_Part_1': 'Q20_Part_12']).reset_index()
data8.drop('index', axis=1, inplace=True)
data8.head()

data88 =pd.DataFrame(pd.value_counts(data8.values.flatten())).reset_index()
data88.columns = ['A', 'B']
data88.head(5)

plt.figure(figsize=(15,8))
vis8 = sns.barplot(data=data88, x='A', y='B', palette='summer')
plt.xlabel('Data visualisation methods/tools', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=50)
plt.title('The most popular data visualisation methods/tools', fontsize=15)

for p in vis8.patches:
    vis8.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. Matplotlib
# 2. Seaborn
# 3. Ggplot
# 4. Plotly

# 3. Machine Learning
# 
# In the next field I focused on machine learning methods, products, algorithms.

# 3.1. Machine Learning algorithms

# In[ ]:


data9 = pd.DataFrame(data.loc[1:, 'Q24_Part_1': 'Q24_Part_12']).reset_index()
data9.drop('index', axis=1, inplace=True)

data99 =pd.DataFrame(pd.value_counts(data9.values.flatten())).reset_index()
data99.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis9 = sns.barplot(data=data99, x='A', y='B', palette='summer')
plt.xlabel('Algorithms', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=50)
plt.title('Machine learning algorithms', fontsize=15)

for p in vis9.patches:
    vis9.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. Linear and Logistic Regressions
# 2. Decision Trees and Random Forests
# 3. Gradient Boosting
# 4. Neural networks
# 5. Bayesian 

# 3.2. Machine Learning Tools

# In[ ]:


data10 = pd.DataFrame(data.loc[1:, 'Q25_Part_1': 'Q25_Part_7']).reset_index()
data10.drop('index', axis=1, inplace=True)

data100 =pd.DataFrame(pd.value_counts(data10.values.flatten())).reset_index()
data100.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis10 = sns.barplot(data=data100, x='A', y='B', palette='summer')
plt.xlabel('ML tools', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=80)
plt.title('The most popular machine learning tools', fontsize=15)

for p in vis10.patches:
    vis10.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. None
# 2. Automated model selection - sklearn
# 3. Imguag, albumentation

# 3.3. Computer Vision Methods

# In[ ]:


data11 = pd.DataFrame(data.loc[1:, 'Q26_Part_1': 'Q26_Part_7']).reset_index()
data11.drop('index', axis=1, inplace=True)

data111 =pd.DataFrame(pd.value_counts(data11.values.flatten())).reset_index()
data111.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis11 = sns.barplot(data=data111, x='A', y='B', palette='summer')
plt.xlabel('Computer vision methods', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=80)
plt.title('The most popular computer vision methods', fontsize=15)

for p in vis11.patches:
    vis11.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. Image classification - VGC, Inception
# 2. General purpose tools
# 3. Image segmentation - U-Net, Mask

# 3.4. Natural Language Processing Methods

# In[ ]:


data12 = pd.DataFrame(data.loc[1:, 'Q27_Part_1': 'Q27_Part_6']).reset_index()
data12.drop('index', axis=1, inplace=True)

data122 =pd.DataFrame(pd.value_counts(data12.values.flatten())).reset_index()
data122.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis12 = sns.barplot(data=data122, x='A', y='B', palette='summer')
plt.xlabel('NLP methods', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=80)
plt.title('The most popular NLP methods', fontsize=15)

for p in vis12.patches:
    vis12.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. Word embeddings - GLoVe, fastText
# 2. Encoder-decodedr models - seq2seq, vanilla transformers
# 3. Transformer language methods - GPT-2, BERT

# 3.5 Machine learning frameworks

# In[ ]:


data13 = pd.DataFrame(data.loc[1:, 'Q28_Part_1': 'Q28_Part_12']).reset_index()
data13.drop('index', axis=1, inplace=True)

data133 =pd.DataFrame(pd.value_counts(data13.values.flatten())).reset_index()
data133.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis13 = sns.barplot(data=data133, x='A', y='B', palette='summer')
plt.xlabel('ML frameworks', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=50)
plt.title('The most popular ML frameworks', fontsize=15)

for p in vis13.patches:
    vis13.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. Scikit-Learn
# 2. Tensorflow
# 3. Keras
# 4. RandomoForest

# 3.6. Machine learning products

# In[ ]:


data17 = pd.DataFrame(data.loc[1:, 'Q32_Part_1': 'Q32_Part_12']).reset_index()
data17.drop('index', axis=1, inplace=True)

data177 =pd.DataFrame(pd.value_counts(data17.values.flatten())).reset_index()
data177.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis17 = sns.barplot(data=data177, x='A', y='B', palette='summer')
plt.xlabel('Machine learning products', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=50)
plt.title('The most popular machine learning products', fontsize=15)

for p in vis17.patches:
    vis17.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. Google cloud machine learning engine
# 2. Azure Machine learning studio
# 3. Amazon SageMaker
# 4. SAS

# 4. Cloud platforms

# Every data scientist has to have knowledge about cloud platforms

# 4.1. Cloud platforms

# In[ ]:


data14 = pd.DataFrame(data.loc[1:, 'Q29_Part_1': 'Q29_Part_12']).reset_index()
data14.drop('index', axis=1, inplace=True)

data144 =pd.DataFrame(pd.value_counts(data14.values.flatten())).reset_index()
data144.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis14 = sns.barplot(data=data144, x='A', y='B', palette='summer')
plt.xlabel('Cloud platforms', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=50)
plt.title('The most popular cloud platforms', fontsize=15)

for p in vis14.patches:
    vis14.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. AWS
# 2. Google Cloud
# 3. Microsoft Azure

# 4.2. Specified cloud platforms

# In[ ]:


data15 = pd.DataFrame(data.loc[1:, 'Q30_Part_1': 'Q30_Part_12']).reset_index()
data15.drop('index', axis=1, inplace=True)

data155 =pd.DataFrame(pd.value_counts(data15.values.flatten())).reset_index()
data155.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis15 = sns.barplot(data=data155, x='A', y='B', palette='summer')
plt.xlabel('Cloud platforms', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=70)
plt.title('The most popular specific cloud platforms', fontsize=15)

for p in vis15.patches:
    vis15.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. None
# 2. AWS Elastic Search
# 3. Google Compute Engine
# 4. AWS Lambda

# 5. Big Data Tools

# 5.1. Specific Big Data Products

# In[ ]:


data16 = pd.DataFrame(data.loc[1:, 'Q31_Part_1': 'Q31_Part_12']).reset_index()
data16.drop('index', axis=1, inplace=True)

data166 =pd.DataFrame(pd.value_counts(data16.values.flatten())).reset_index()
data166.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis16 = sns.barplot(data=data166, x='A', y='B', palette='summer')
plt.xlabel('Big data products', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=50)
plt.title('The most popular big data products', fontsize=15)

for p in vis16.patches:
    vis16.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. Google Big Query
# 2. Databricks
# 3. AWS Redshift
# 4. Google Cloud Dataflow

# 6. Databases

# 6.1. Relational databases products

# In[ ]:


data18 = pd.DataFrame(data.loc[1:, 'Q34_Part_1': 'Q34_Part_12']).reset_index()
data18.drop('index', axis=1, inplace=True)

data188 =pd.DataFrame(pd.value_counts(data18.values.flatten())).reset_index()
data188.columns = ['A', 'B']


# In[ ]:


plt.figure(figsize=(15,8))
vis18 = sns.barplot(data=data188, x='A', y='B', palette='summer')
plt.xlabel('Databases', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.xticks(rotation=50)
plt.title('The most popular databases', fontsize=15)

for p in vis18.patches:
    vis18.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')


# 1. MySQL
# 2. PostgresSQL
# 3. Microsoft SQL Server
# 4. SQLite

# 7. Summary

# ![juniorzy-w-It.png](attachment:juniorzy-w-It.png)

# **What aspiring Data Scientist should learn and master?**
# 
# **BASICS: evelopment environments, visualisations, programming language**
# 
# Development environments:
# 1. Jupyter - Notebook, Lab
# 2. Visual Studio
# 3. RStudio
# 4. Pycharm
# The most popular is Jupyter and it can be chosen as the first one.
# 
# Notebook products:
# 1. Kaggle kernels
# 2. Google Colab
# 3. Binder, JupyterHUB
# Probably the best option to create own portfolio is having Kaggle profile
# 
# Programming language:
# 1. Python
# 2. SQL
# 3. R
# 4. Java
# Python should be the first choose to starts adventure as data scientist.
# 
# Data visualisation methods, tools
# 1. Matplotlib
# 2. Seaborn
# 3. Ggplot
# 4. Plotly
# Here are the best visualisation tools to create charts.
# 
# **MACHINE LEARNING**
# 
# Machine Learning algorithms:
# 1. Linear and Logistic Regressions
# 2. Decision Trees and Random Forests
# 3. Gradient Boosting
# 4. Neural networks
# 5. Bayesian 
# The most popular and every data siecntist has to know.
# 
# Machine learning tools:
# 1. Automated model selection - sklearn
# 2. Imguag, albumentation
# 
# Computer vision methods:
# 1. Image classification - VGC, Inception
# 2. General purpose tools
# 3. Image segmentation - U-Net, Mask
# 
# Natural Language Processing Methods:
# 1. Word embeddings - GLoVe, fastText
# 2. Encoder-decodedr models - seq2seq, vanilla transformers
# 3. Transformer language methods - GPT-2, BERT
# 
# Machine Learning frameworks:
# 1. Scikit-Learn
# 2. Tensorflow
# 3. Keras
# 4. RandomoForest
# Scikit learn and tensorflow are the most popular and we have to know them.
# 
# Machine learning products:
# 1. Google cloud machine learning engine
# 2. Azure Machine learning studio
# 3. Amazon SageMaker
# 4. SAS
# 
# **CLOUD PLATFORMS**
# 
# Cloud platforms:
# 1. AWS
# 2. Google Cloud
# 3. Microsoft Azure
# 
# Specific cloud platform products:
# 1. None
# 2. AWS Elastic Search
# 3. Google Compute Engine
# 4. AWS Lambda
# 
# **BIG DATA
# DATABASES**
# 
# Big Data products:
# 1. Google Big Query
# 2. Databricks
# 3. AWS Redshift
# 4. Google Cloud Dataflow
# 
# **BIG DATA
# DATABASES**
# 
# Databases:
# 1. MySQL
# 2. PostgresSQL
# 3. Microsoft SQL Server
# 4. SQLite
