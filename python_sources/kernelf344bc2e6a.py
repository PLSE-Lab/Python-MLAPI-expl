#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))
happiness_report = pd.read_csv('../input/world-happiness-report-2019.csv')
happiness_report = happiness_report.rename(columns = {'Country (region)':'Country','SD of Ladder':'SD_of_ladder',
                         'Positive affect':'Positive_affect','Negative affect':'Negative_affect','Social support':'Social_support','Log of GDP per capital':'log_of_GDP_per_capital', 'Healthy life expectancy':'Healthy_life_expectancy'
                         })


# In[ ]:


happiness_report.head(n=10)


# In[ ]:


happiness_report.describe()


# In[ ]:


happiness_report.info()


# In[ ]:


happiness_report.columns


# In[ ]:


happiness_report.info()


# In[ ]:


happiness_report.isnull().sum()


# In[ ]:


happiness_report=happiness_report.fillna(method = 'ffill')


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(happiness_report.corr(),ax=ax,annot=True,linewidth=0.05,fmt='.2f',cmap='magma')
plt.show()


# In[ ]:


plt.scatter(happiness_report['Positive_affect'],happiness_report.Negative_affect)
plt.title('Positive affect compare with Negative affect')
plt.xlabel('Positive_affect')
plt.ylabel('Negative_affect')
plt.show()


# In[ ]:


plt.scatter(happiness_report['Positive_affect'],happiness_report.Social_support)
plt.xlabel('Positive_affect')
plt.ylabel('Social_support')
plt.title('Positive_affect compare with Social_support')


# In[ ]:


plt.scatter(happiness_report['Positive_affect'],happiness_report.Freedom)
plt.xlabel('Positive_affect')
plt.ylabel('Freedom')
plt.title('Positive_affect compare with Freedom')


# In[ ]:


plt.scatter(happiness_report['Positive_affect'],happiness_report.Generosity)
plt.xlabel('Positive_affect')
plt.ylabel('Generosity_affect')
plt.title('Positive_affect compare with Generosity')


# In[ ]:


plt.scatter(happiness_report['Negative_affect'],happiness_report.Corruption)
plt.xlabel('Negative_affect')
plt.ylabel('Corruption')
plt.title('Negative_affect compare with Corruption')


# In[ ]:


list_of_columns = list(happiness_report.columns)
log = list_of_columns[-2]


# In[ ]:


plt.scatter(happiness_report['Negative_affect'],happiness_report[log])
plt.xlabel('Negative_affect')
plt.ylabel('Log of GDP per capita')
plt.title('Negative_affect compare with Log of GDP per capita')


# In[ ]:


freedom = happiness_report['Freedom']
features = happiness_report.drop('Freedom', axis=1)


# In[ ]:


for v in ['Positive_affect']:
    sns.regplot(happiness_report[v], freedom, marker ='+', color='red')


# In[ ]:


log_of_GDP_per_capita= happiness_report[log]
features = happiness_report.drop(log, axis=1)


# In[ ]:


for v in ['Negative_affect']:
    sns.regplot(happiness_report[v], log_of_GDP_per_capita, marker ='+', color ='red')


# In[ ]:


list_of_columns = list(happiness_report.columns)
Health = list_of_columns[-1]


# In[ ]:


log_of_GDP_per_capita= happiness_report[log]
features = happiness_report.drop(log, axis=1)
for v in [Health]:
    sns.regplot(happiness_report[v], log_of_GDP_per_capita, marker ='+', color ='red')


# In[ ]:


Ladder = happiness_report['Ladder']
features = happiness_report.drop(log, axis=1)
for v in ['Social_support']:
    sns.regplot(happiness_report[v], Ladder, marker ='+', color ='red')


# In[ ]:


happiness_report.tail(n=10)


# In[ ]:


Ladder = happiness_report['Ladder']
features = happiness_report.drop(log, axis=1)
for v in [Health]:
    sns.regplot(happiness_report[v], Ladder, marker ='+', color ='red')


# In[ ]:


Ladder = happiness_report['Ladder']
features = happiness_report.drop(log, axis=1)
for v in [log]:
    sns.regplot(happiness_report[v], Ladder, marker ='+', color ='red')


# In[ ]:




