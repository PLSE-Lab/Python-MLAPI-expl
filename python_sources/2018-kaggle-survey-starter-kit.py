#!/usr/bin/env python
# coding: utf-8

# # A starting point
# This notebook is intended to be a good starting point for your data exploration. As the competition states you are tasked to **tell a rich story about a subset of the data science and machine learning community** - it is up to you to find this story.  

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# print(os.listdir("../input"))
import matplotlib.pylab as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Survey Schema Overview

# In[ ]:


ss = pd.read_csv('../input/SurveySchema.csv')

# Reformat so that each row is a column.
ss_transpose = ss.T.rename(columns=ss.T.iloc[0]).drop('2018 Kaggle Machine Learning and Data Science Survey')

ss_transpose['# of Respondents:'] = pd.to_numeric(ss_transpose['# of Respondents:'])
plot1 = ss_transpose.sort_values('# of Respondents:')['# of Respondents:']     .drop('Time from Start to Finish (seconds)')     .plot(figsize=(15, 5), kind='bar', title='Numer of respondents by question')


# - **Most Answered Questions (Q1, Q2, Q3)**
# - **Least Answered Questions (Q37, Q30, Q29)**
# 
# Later questions are answered less, as might be expected.

# In[ ]:


# Most Answered questions printed
print('----- Most Answered Questions: -----')
print('Q1:', ss['Q1'][0])
print('Q2:', ss['Q2'][0])
print('Q3:', ss['Q3'][0])
# Most Answered questions printed
print('----- Least Answered Questions: -----')
print('Q37:', ss['Q37'][0])
print('Q30:', ss['Q30'][0])
print('Q29:', ss['Q29'][0])


# # Multiple Choice Responses

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
mc = pd.read_csv('../input/multipleChoiceResponses.csv')


# ## Demographics
# Information about the type of people who reponded.

# In[ ]:


plot2 = mc.drop(0).groupby('Q1').count()['Q2'].plot(figsize=(15, 5), kind='bar', title='Gender of Respondents', rot=0)


# In[ ]:


plot3 = mc.drop(0).groupby('Q2').count()['Q1'].plot(figsize=(15, 5), kind='bar', title='Age of Respondents', rot=0)


# In[ ]:


plot4 = mc.drop(0).groupby('Q4').count()['Q1'].plot(figsize=(15, 5), kind='bar', title='Education', rot=30)


# In[ ]:


plot5 = mc.drop(0).groupby('Q5').count()['Q1']     .sort_values(ascending=False)     .plot(figsize=(10, 10), kind='barh', title='Undergrad Education', rot=0)


# # Employment
# The most common type of employment for kaggle survey respondents was Student, followed by Data Scientist.

# In[ ]:


plot6 = mc.drop(0).groupby('Q6').count()['Q1']     .sort_values(ascending=False)     .plot(figsize=(10, 10), kind='barh', title='Job Title', rot=0)


# In[ ]:


plot7 = mc.drop(0).groupby('Q7').count()['Q1']     .sort_values(ascending=False)     .plot(figsize=(10, 10), kind='barh', title='Job Industry', rot=0)


# Check out the bump for 5-10 of experience. The first wave of "Data scientists"?

# In[ ]:


# Make index categorical for sorting
q8_index = pd.CategoricalIndex(mc.drop(0).groupby('Q8').count()['Q1'].index,
                   ['0-1', '1-2','2-3', '3-4','4-5', '5-10','10-15', '15-20',  '20-25', '25-30', '30 +'])
q8 = mc.drop(0).groupby('Q8').count()['Q1'] 
q8.index = q8_index
plot8 = q8.sort_index(ascending=True)     .plot(figsize=(15, 5), kind='bar', title='Years of Job Experience', rot=0)


# In[ ]:


plot9 = mc.drop(0).groupby('Q9').count()['Q1']     .sort_values(ascending=False)     .plot(figsize=(10, 10), kind='barh', title='Salary', rot=0)


# ## A lot of ML explorers, few using ML in production

# In[ ]:


plot10 = mc.drop(0).groupby('Q10').count()['Q1']     .sort_values(ascending=False)     .plot(figsize=(10, 10), kind='barh', title='Employer ML Adoption', rot=0)


# # What IDE group are you in?
# 
# I thought it would be fun to think of IDE useage in terms of groups. My hypothesis: respondents would cluster around certain types of IDES. The type of person who uses VIM may not dare touch PyCharm - similarly those that work mostly in Jupyter notebooks may not.
# 
# Using cluster analysis to put survey respondents into 3 groups! Looking at the similarity in IDE useage. What group do you fit into?
# 
# ![vim](https://cdn-images-1.medium.com/max/1200/1*BPkK5FHiS6rXsygxNoO2XA.jpeg)
# 
# The data prep is a little annoying because we have to reformat this question into a data structure easier to plot. I had to convert the columns with the IDE responses into binary values 0/1.

# In[ ]:


# Change the columns so that we have a binary 1/0 for the IDE and the column names are the IDE

ide_qs = mc[['Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5',
             'Q13_Part_6','Q13_Part_7','Q13_Part_8','Q13_Part_9','Q13_Part_10',
             'Q13_Part_11','Q13_Part_12','Q13_Part_13','Q13_Part_14','Q13_Part_15']].drop(0)

column_rename = {'Q13_Part_1': 'Jupyter/IPython',
                 'Q13_Part_2': 'RStudio',
                'Q13_Part_3': 'PyCharm',
                'Q13_Part_4': 'Visual Studio Code',
                'Q13_Part_5': 'nteract',
                'Q13_Part_6': 'Atom',
                'Q13_Part_7': 'MATLAB',
                'Q13_Part_8': 'Visual Studio',
                'Q13_Part_9': 'Notepad++',
                'Q13_Part_10': 'Sublime Text',
                'Q13_Part_11': 'Vim',
                'Q13_Part_12': 'IntelliJ',
                'Q13_Part_13': 'Spyder',
                'Q13_Part_14': 'None',
                'Q13_Part_15': 'Other',
                }
ide_qs_binary = ide_qs.rename(columns=column_rename).fillna(0).replace('[^\\d]',1, regex=True)

# Show what the data now looks like
ide_qs_binary.head()


# In[ ]:


# Make the clusters
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3, random_state=1).fit_predict(ide_qs_binary)
ide_qs_binary['cluster'] = y_pred


# ## IDE Cluster #1 - Jupyter lovers! (8060 people in this group)
# All the jupyter/python people get clumped into this cluster. It's a steep dropoff before the next popular IDE (RStudio).**

# In[ ]:


print(len(ide_qs_binary.loc[ide_qs_binary['cluster'] == 0]))

plot_ideg1 = ide_qs_binary.loc[ide_qs_binary['cluster'] == 0].describe()     .T[['mean','std']]     .sort_values('mean', ascending=False)     .drop('cluster')['mean']     .plot(kind='bar', figsize=(10,5), title='IDE Cluster #1 Jupyter Lovers!')


# ## IDE Cluster #2 - Jupyer Haters (8827 respondents in this group)

# In[ ]:


print(len(ide_qs_binary.loc[ide_qs_binary['cluster'] == 1]))

plot_ideg2 = ide_qs_binary.loc[ide_qs_binary['cluster'] == 1].describe()     .T[['mean','std']]     .sort_values('mean', ascending=False)     .drop('cluster')['mean']     .plot(kind='bar', figsize=(10,5), title='IDE Cluster #2 Jupyter Haters')


# ## Cluster #3 - Use some Jupyter but also need a text editor. (6972 respondents in this group)

# In[ ]:


print(len(ide_qs_binary.loc[ide_qs_binary['cluster'] == 2]))

plot_ideg3 = ide_qs_binary.loc[ide_qs_binary['cluster'] == 2].describe()     .T[['mean','std']]     .sort_values('mean', ascending=False)     .drop('cluster')['mean']     .plot(kind='bar', figsize=(10,5), title='IDE Cluser #3 (Jupyter + Traditional IDE)')


# In[ ]:




