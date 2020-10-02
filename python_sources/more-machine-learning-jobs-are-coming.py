#!/usr/bin/env python
# coding: utf-8

# **"*Sexiest job of the 21st century*" by Harvard Business Review**
# <img src="https://www.kaggle.com/static/images/site-logo.png" width="300px">
# **Introduction**  
# 
# As we are aware of the fact that Kaggle has published their second machine learning and data science survey (2018) data set, it is very much fun to analyse the data set and explore  some interesting story from the data set.
# Also I would like to thank Kaggle for this type of survey which is extremely useful to understand the data science/machine learning community accross the  world.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings
warnings.filterwarnings("ignore")

import os
base_dir = '../input'
fileName = 'multipleChoiceResponses.csv'
filePath = os.path.join(base_dir,fileName)

mcr = pd.read_csv(filePath)


# **More machine learning jobs are coming**  
# Machine Learning is a buzz word in 21st century. But according to the following analysis it is clear that only few are using machine learning into their business. So more machine learning jobs are coming as industries are gradually setting up their machine learning infrastructure.

# In[ ]:


labels = 'We are exploring ML methods \n (and may one day put a model into production)','No (we do not use ML methods)','We recently started using ML methods \n (i.e., models in production for less than 2 years)','I do not know','We have well established ML methods \n (i.e., models in production for more than 2 years)','We use ML methods for generating insights \n (but do not put working models into production)'
sizes = [4688,4411,3790,2893,2782,2105]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','red', 'green']
explode = (0.2,0,0,0,0,0)  # explode 1st slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140,rotatelabels = False)
plt.title("Does your current employer incorporate machine learning methods into their business?\n\n")
plt.show()


# **We need more  females in the data science domain**  
# Ther are many problems in medical science and other fields in the industries where female data scientist are extremely important because  they can solve certain type of problems very effectively. So it's our responsibility so that more females join in the data science domain.

# In[ ]:


ax=mcr.Q1[1:].value_counts().plot(kind='bar',title='What is your gender?\n',figsize=(10,5)) #returns matplotlib object
plt.xticks(rotation=30) #use matplotlib functions
ax.set_ylabel("Number of respondents")
plt.show()


# **Young generation is leading the Data Science community**  
# People who are in the age range 20-30 (in years) are the primary human resources for this booming industry. Also there are lots of experienced people who are constantly supporting this young generation to become a more successful data scientist.

# In[ ]:


ax=mcr.Q2[1:].value_counts().plot(kind='bar',title='What is your age (# years)?\n',figsize=(10,5)) #returns matplotlib object
plt.xticks(rotation=30) #use matplotlib functions
ax.set_xlabel("Age range (years)")
ax.set_ylabel("Number of respondents")
plt.show()


# **USA and India are the top two countries**  
# More number of data scientists are coming from USA and India followed by China and other countries.

# In[ ]:


ax=mcr.Q3[1:].value_counts().plot(kind='bar',figsize=(16,8),title='In which country do you currently reside?\n')
ax.set_ylabel("Number of respondents")
plt.show()


# **Highest level of education:** Most of them have a Master's degree  
# Master's degree followed by Bachelor's degree people are leading the data science/machine learning industry.

# In[ ]:


ax=mcr.Q4[1:].value_counts().plot(kind='pie',figsize=(6,6),autopct='%1.1f%%',title='What is the highest level of formal education that you have attained or plan to attain within the next 2 years?\n')
ax.set_ylabel(" ")
plt.show()


# **Engineers are dominating the industry**  
# Data Science/Machine Learning community are full of engineers followed by mathematician/statistician.

# In[ ]:


labels = 'Computer science (software engineering, etc.)','Engineering (non-computer focused)','Mathematics or statistics','A business discipline (accounting, economics, finance, etc.)','Physics or astronomy','Information technology, networking, or system administration','Medical or life sciences (biology, chemistry, medicine, etc.)','Other'
sizes = [9430, 3705, 2950, 1791,1110,1029,871,2061]
explode = (0.2,0,0,0,0,0,0,0)  # explode 1st slice
 
plt.pie(sizes, explode=explode,autopct='%1.1f%%', shadow=True, startangle=140) #labels=labels,rotatelabels = True
#plt.axis('equal')
plt.title('Which best describes your undergraduate major?\n')
plt.legend(labels=labels,loc='right',bbox_to_anchor=(1, 1, 1, -1))
plt.show()


# **Data Science/Machine Learning is a very new and attractive  field of research**  
# Most of the people have experience between 0-5 years. So there are lot of opportunities to explore in the field of data science and machine learning.

# In[ ]:


ax=mcr.Q8[1:].value_counts().plot(kind='bar',title='How many years of experience do you have in your current role?\n',figsize=(10,5)) #returns matplotlib object
plt.xticks(rotation=330) #use matplotlib functions
ax.set_ylabel("Number of respondents")
ax.set_xlabel("Experience (in years)")
plt.show()


# **Data Scientist/Machine Learning Scientis are earning a lot on an average**  
# Though many people are not interested to disclose their compensation, it is said that Data Scientist/Machine Learning Scientis are earning a lot on an average.

# In[ ]:


ax=mcr.Q9[1:].value_counts().plot(kind='barh',title='What is your current yearly compensation (approximate $USD)?\n',figsize=(10,5)) #returns matplotlib object
plt.xticks(rotation=330) #use matplotlib functions
ax.set_ylabel("$USD")
ax.set_xlabel("Number of respondents")
plt.show()


# **Important role at work**  
# Analyze and understand data to influence product or business decisions followed by Build prototypes to explore applying machine learning to new areas are the most two important roles at work.

# In[ ]:


df1 = mcr.loc[1:, 'Q11_Part_1':'Q11_Part_7']  #column slicing
df1 = df1.melt(var_name='Question',value_name='important_role')
df1 = df1.dropna()
df1=df1.groupby(['Question','important_role']).size().reset_index(name="Number of respondents")
df1.set_index('important_role',drop=True,inplace=True)
ax=df1.sort_values(by="Number of respondents",ascending=False).plot(kind='barh',figsize=(5,5),                                                  title="Select any activities that make up an important part of your role at work\n\n")

plt.show()


# **Primary tool used at work/school**  
# Local or hosted development environments like RStudio, JupyterLab, etc. followed by basic statistical softwarelike Microsoft Excel, Google Sheets, etc. are the most used tool within the data science community.

# In[ ]:


ax=mcr.Q12_MULTIPLE_CHOICE[1:].value_counts().plot(kind='barh',title='What is the primary tool that you use at work or school to analyze data? \n',figsize=(10,5)) #returns matplotlib object
plt.xticks(rotation=330) #use matplotlib functions
ax.set_xlabel("Number of respondents")
plt.show()


# **Most used IDE**  
# Jupyter notebook and RStudio are the most used IDE in the field of data science and machine learning.

# In[ ]:


from matplotlib import cm #colormap

df1 = mcr.loc[1:, 'Q13_Part_1':'Q13_Part_15']  #column slicing
df1 = df1.melt(var_name='Question',value_name='IDE')
df1 = df1.dropna()
df1=df1.groupby(['Question','IDE']).size().reset_index(name='Number of respondents')
df1.set_index("IDE",drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='barh',figsize=(5,5),                                                  title="Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years?\n\n",                                                  colormap=cm.get_cmap('Spectral'))

plt.show()


# **Most used hosted notebook**  
# Kaggle Kernels followed by JupyterHub and Google Colab are the most used hosted notebooks in the field of data science and machine learning.

# In[ ]:


from matplotlib import colors

df1 = mcr.loc[1:, 'Q14_Part_1':'Q14_Part_11']  #column slicing
df1 = df1.melt(var_name='Question',value_name='Hosted notebooks')
df1 = df1.dropna()

df1=df1.groupby(['Question','Hosted notebooks']).size().reset_index(name='Number of respondents')
df1.set_index("Hosted notebooks",drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(5,5),                                                  title="Which of the following hosted notebooks have you used at work or school in the last 5 years?\n\n",                                                  colors = 'lightgreen')

plt.xticks(rotation=75)
plt.show()


# **Most used cloud computing services**  
# AWS followed by Google Cloud Platform are the most used cloud computing services in the field of data science and machine learning.

# In[ ]:


df1 = mcr.loc[1:, 'Q15_Part_1':'Q15_Part_7']  #column slicing
df1 = df1.melt(var_name='Question',value_name='Cloud computing services')
df1 = df1.dropna()

df1=df1.groupby(['Question','Cloud computing services']).size().reset_index(name='Number of respondents')
df1.set_index('Cloud computing services',drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(5,5),                                                  title="Which of the following cloud computing services have you used at work or school in the last 5 years?\n\n",                                                  colors = 'brown')

plt.xticks(rotation=75)
plt.show()


# **Most used programming languages**  
# Python followed by SQL and R are the most used programming languages in the field of data science and machine learning.

# In[ ]:


df1 = mcr.loc[1:, 'Q16_Part_1':'Q16_Part_18']  #column slicing
df1 = df1.melt(var_name='Question',value_name='Programming languages')
df1 = df1.dropna()

df1=df1.groupby(['Question','Programming languages']).size().reset_index(name='Number of respondents')
df1.set_index('Programming languages',drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(8,6),                                                  title="What programming languages do you use on a regular basis?\n\n",                                                  color='darkcyan')

plt.xticks(rotation=75)
plt.show()


# In[ ]:


ax=mcr.Q17[1:].value_counts().plot(kind='bar',figsize=(8,6),title="What specific programming language do you use most often? \n\n")
plt.xticks(rotation=75)
ax.set_ylabel("Number of respondents")
plt.show()


# **Most recommended programming language to an aspiring data scientist**  
# Python is the most recommended programming language to an aspiring data scientist.

# In[ ]:


ax=mcr.Q18[1:].value_counts().plot(kind='bar',figsize=(8,6),title="What programming language would you recommend an aspiring data scientist to learn first? \n\n")
plt.xticks(rotation=75)
ax.set_ylabel("Number of respondents")
plt.show()


# **Most used machine learning frameworks**  
# Scikit-Learn followed by TensorFlow and Keras are the most used machine learning frameworks.

# In[ ]:


df1 = mcr.loc[1:, 'Q19_Part_1':'Q19_Part_19']  #column slicing
df1 = df1.melt(var_name='Question',value_name='ML frameworks')
df1 = df1.dropna()

df1=df1.groupby(['Question','ML frameworks']).size().reset_index(name='Number of respondents')
df1.set_index('ML frameworks',drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(8,6),                                                  title="What machine learning frameworks have you used in the past 5 years? \n\n",                                                  color='cyan')

plt.xticks(rotation=75)
plt.show()


# In[ ]:


ax=mcr.Q20[1:].value_counts().plot(kind='bar',figsize=(8,6),title="Which ML library have you used the most?  \n\n")
plt.xticks(rotation=75)
ax.set_ylabel("Number of respondents")
plt.show()


# **Most used data visualization libraries**  
# Matplotlib followed by Seaborn and ggplot2 are the most used data visualization libraries in the field of data science and machine learning.

# In[ ]:


df1 = mcr.loc[1:, 'Q21_Part_1':'Q21_Part_13']  #column slicing
df1 = df1.melt(var_name='Question',value_name='Visualization libraries')
df1 = df1.dropna()

df1=df1.groupby(['Question','Visualization libraries']).size().reset_index(name='Number of respondents')
df1.set_index('Visualization libraries',drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(8,6),                                                  title="What data visualization libraries or tools have you used in the past 5 years?\n\n",                                                  color='darkcyan')

plt.xticks(rotation=75)
plt.show()


# In[ ]:


ax=mcr.Q22[1:].value_counts().plot(kind='bar',figsize=(8,6),title="Which specific data visualization library or tool have you used the most?\n\n")
plt.xticks(rotation=75)
ax.set_ylabel("Number of respondents")
plt.show()


# **Coding is extremely  important**  
# Coding is one of the most important aspect to perform a data science/machine learning related projects.

# In[ ]:


ax=mcr.Q23[1:].value_counts().plot(kind='bar',figsize=(8,6),title="Approximately what percent of your time at work or school is spent actively coding?\n\n")
plt.xticks(rotation=70)
ax.set_ylabel("Number of respondents")
plt.show()


# **Machine learning is now one of the hottest research area**  
# Most of the people started using machine learning methods from last 4-5 years. So people are gradually using more machine learning methods in their work.

# In[ ]:


ax=mcr.Q25[1:].value_counts().plot(kind='barh',figsize=(8,6),title="For how many years have you used machine learning methods (at work or in school)?\n\n")
plt.xticks(rotation=60)
ax.set_xlabel("Number of respondents")
plt.show()


# **Most used cloud computing products**  
# AWS EC2 followed by Google Compute Engine are the most used cloud computing products in the field of data science and machine learning.

# In[ ]:


df1 = mcr.loc[1:, 'Q27_Part_1':'Q27_Part_20']  #column slicing
df1 = df1.melt(var_name='Question',value_name='Cloud computing products')
df1 = df1.dropna()

df1=df1.groupby(['Question','Cloud computing products']).size().reset_index(name='Number of respondents')
df1.set_index('Cloud computing products',drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='barh',figsize=(8,6),                                                  title="Which of the following cloud computing products have you used at work or school in the last 5 years?\n\n",                                                  color='darkmagenta')

plt.xticks(rotation=50)
plt.show()


# **Most used relational database products**  
# MySQL followed by PostgresSQL and SQLite are the most used relational database products in the field of data science and machine learning.

# In[ ]:


df1 = mcr.loc[1:, 'Q29_Part_1':'Q29_Part_28']  #column slicing
df1 = df1.melt(var_name='Question',value_name='Database products')
df1 = df1.dropna()

df1=df1.groupby(['Question','Database products']).size().reset_index(name='Number of respondents')
df1.set_index('Database products',drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(12,6),                                                  title="Which of the following relational database products have you used at work or school in the last 5 years? \n\n",                                                  color='darkmagenta')

plt.xticks(rotation=83)
plt.show()


# **Most used big data and analytics products**  
# Google BigQuery is the most used big data and analytics products in the field of data science and machine learning.

# In[ ]:


df1 = mcr.loc[1:, 'Q30_Part_1':'Q30_Part_25']  #column slicing
df1 = df1.melt(var_name='Question',value_name='Big data and analytics products')
df1 = df1.dropna()

df1=df1.groupby(['Question','Big data and analytics products']).size().reset_index(name='Number of respondents')
df1.set_index('Big data and analytics products',drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(12,6),                                                  title="Which of the following big data and analytics products have you used at work or school in the last 5 years?\n\n",                                                  color='darkmagenta')

plt.xticks(rotation=80)
plt.show()


# **Data that you interact most often**  
# Numerical data followed by text data and categorical data are the most used data types in the field of data science and machine learning.

# In[ ]:


df1 = mcr.loc[1:, 'Q31_Part_1':'Q31_Part_12']  #column slicing
df1 = df1.melt(var_name='Question',value_name='Types of data')
df1 = df1.dropna()

df1=df1.groupby(['Question','Types of data']).size().reset_index(name='Number of respondents')
df1.set_index('Types of data',drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(12,6),                                                  title="Which types of data do you currently interact with most often at work or school?\n\n",                                                  color='darkmagenta')

plt.xticks(rotation=60)
plt.show()


# In[ ]:


ax=mcr.Q32[1:].value_counts().plot(kind='bar',figsize=(8,6),title="What is the type of data that you currently interact with most often at work or school?\n\n")
plt.xticks(rotation=60)
ax.set_ylabel("Number of respondents")
plt.show()


# **Resources of public datasets**  
# kaggle, Google, and Github are the most used platforms to find public data sets.

# In[ ]:


df1 = mcr.loc[1:, 'Q33_Part_1':'Q33_Part_11']  #column slicing
df1 = df1.melt(var_name='Question',value_name='Datasets')
df1 = df1.dropna()

df1=df1.groupby(['Question','Datasets']).size().reset_index(name='Number of respondents')
df1.set_index('Datasets',drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='barh',figsize=(12,6),                                                  title="Where do you find public datasets? \n\n",                                                  color='darkmagenta')

plt.xticks(rotation=50)
plt.show()


# **Time taken by different phases of a typical data science projects**  
# Data cleaning and modeling takes most of the time in a typical data science projects.

# In[ ]:


df1 = mcr.loc[1:, 'Q34_Part_1':'Q34_Part_6']
df1=df1.dropna()

#df1=df1.reset_index()
df1=df1.rename(columns={'Q34_Part_1':'Gathering','Q34_Part_2':'Cleaning','Q34_Part_3':'Visualizing','Q34_Part_4':'Model','Q34_Part_5':'Putting the model into production','Q34_Part_6':'Finding'})
df1=df1.astype(float)  #V.V.I. step
ax=df1.plot(kind='box',figsize=(12,8),title="During a typical data science project at work or school, approximately what proportion of your time is devoted to the following?\n\n")
ax.set_ylabel("Percentage")
plt.xticks(rotation=60)
plt.show()


# **Online courses are more popular nowadays in the field of machine learning training.**  
# 

# In[ ]:


df1 = mcr.loc[1:, 'Q35_Part_1':'Q35_Part_6']
df1=df1.dropna()

#df1=df1.reset_index()
df1=df1.rename(columns={'Q35_Part_1':'Self-taught','Q35_Part_2':'Online courses (Coursera, Udemy, edX, etc.)','Q35_Part_3':'Work','Q35_Part_4':'University','Q35_Part_5':'Kaggle competitions','Q35_Part_6':'Other'})

df1=df1.astype(float)  #V.V.I. step
ax=df1.plot(kind='box',figsize=(12,8),title="What percentage of your current machine learning/data science training falls under each category?\n\n")
ax.set_ylabel("Percentage")
plt.xticks(rotation=60)
plt.show()


# **Most popular online platforms to explore data science**  
# Coursera is the most popular among all others.

# In[ ]:


df1 = mcr.loc[1:, 'Q36_Part_1':'Q36_Part_13']  #column slicing
df1 = df1.melt(var_name='Question',value_name='Online platforms')
df1 = df1.dropna()

df1=df1.groupby(['Question','Online platforms']).size().reset_index(name='Number of respondents')
df1.set_index('Online platforms',drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(12,6),                                                  title="On which online platforms have you begun or completed data science courses?\n\n",                                                  color='darkmagenta')

plt.xticks(rotation=60)
plt.show()


# In[ ]:


ax=mcr.Q37[1:].value_counts().plot(kind='bar',figsize=(8,6),title="On which online platform have you spent the most amount of time?\n\n")
plt.xticks(rotation=75)
ax.set_ylabel("Number of respondents")
plt.show()


# **Most popular media sources to explore data science**  
# Kaggle and Medium are  the most popular among all others.

# In[ ]:


df1 = mcr.loc[1:, 'Q38_Part_1':'Q38_Part_22']  #column slicing
df1 = df1.melt(var_name='Question',value_name='Favorite media sources')
df1 = df1.dropna()

df1=df1.groupby(['Question','Favorite media sources']).size().reset_index(name='Number of respondents')
df1.set_index('Favorite media sources',drop=True,inplace=True)
df1.sort_values(by='Number of respondents',ascending=False).plot(kind='bar',figsize=(12,6),                                                  title="Who/what are your favorite media sources that report on data science topics? \n\n",                                                  color='darkmagenta')

plt.xticks(rotation=70)
plt.show()


# **Online MOOCs platforms are better to learn data science/machine learning.**

# In[ ]:


ax=mcr.Q39_Part_1[1:].value_counts().plot(kind='bar',figsize=(8,6),title="How do you perceive the quality of online learning platforms and MOOCs as compared to the quality of the education provided by traditional brick and mortar institutions?\n\n")
plt.xticks(rotation=60)
ax.set_ylabel("Number of respondents")
plt.show()


# In[ ]:


ax=mcr.Q39_Part_2[1:].value_counts().plot(kind='bar',figsize=(8,6),title="How do you perceive the quality of In-person bootcamps as compared to the quality of the education provided by traditional brick and mortar institutions?\n\n")
plt.xticks(rotation=60)
ax.set_ylabel("Number of respondents")
plt.show()


# **Independent projects are more important than academic achievements**  
# Independent projects are one of most important aspect to get an internship or a job.

# In[ ]:


labels = 'Independent projects are much more \nimportant than academic achievements','Independent projects are slightly more \nimportant than academic achievements','Independent projects are equally \nimportant as academic achievements','No opinion; I do not know','Independent projects are slightly less important than academic achievements','Independent projects are much less important than academic achievements'
sizes = [4990,4473,4343,936,831,306]
explode = (0.2,0,0,0,0,0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%', shadow=True)
plt.title("Which better demonstrates expertise in data science: academic achievements or independent projects?\n\n")
#plt.savefig('output.png', dpi=300, bbox_inches='tight')
plt.show()

