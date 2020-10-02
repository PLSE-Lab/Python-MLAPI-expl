#!/usr/bin/env python
# coding: utf-8

# Extension of Notebook prepared by Ravi Chaubey. 
# 
# Reference: https://www.kaggle.com/ravichaubey1506/student-performance-correlation

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.head()


# In[ ]:


plt.figure(dpi=100)
plt.title('Correlation Analysis')
sns.heatmap(df.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')
plt.xticks(rotation=60)
plt.yticks(rotation = 60)
plt.show()


# Let us try to find the correlation between gender, race/ethnicity, parental level of education, lunch, test preparation course completion. 
# 
# I have a feeling that test preparation course, parental level of education and gender might show some positive correlation. 

# In[ ]:


encoder=OneHotEncoder(sparse=False)

df_encoded = pd.DataFrame(encoder.fit_transform(df[['gender']]))
df_encoded.columns = encoder.get_feature_names(['gender'])
df.drop(['gender'] ,axis=1, inplace=True)
df_new= pd.concat([df, df_encoded], axis=1)
df_new


# In[ ]:


plt.figure(dpi=100)
plt.title('Correlation Analysis')
sns.heatmap(df_new.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')
plt.xticks(rotation=60)
plt.yticks(rotation = 60)
plt.show()


# In[ ]:


df_new.groupby(["test preparation course"])["math score", "reading score", "writing score"].mean()


# In[ ]:


df_new.groupby(["parental level of education"])["math score", "reading score", "writing score"].mean()


# In[ ]:


df_new.groupby(["lunch"])["math score", "reading score", "writing score"].mean()


# In[ ]:


df_new.groupby(["race/ethnicity"])["math score", "reading score", "writing score"].mean()


# In[ ]:


df_new['test preparation course'].value_counts()


# In[ ]:


df_new['lunch'].value_counts()


# In[ ]:


df_new.groupby(["lunch", "test preparation course"])["math score", "reading score", "writing score"].mean()


# In[ ]:


df_encoded = pd.DataFrame(encoder.fit_transform(df_new[['lunch']]))
df_encoded.columns = encoder.get_feature_names(['lunch'])
df_new.drop(['lunch'] ,axis=1, inplace=True)
df_new = pd.concat([df_new, df_encoded], axis=1)
df_new


# In[ ]:


df_encoded = pd.DataFrame(encoder.fit_transform(df_new[['test preparation course']]))
df_encoded.columns = encoder.get_feature_names(['test preparation course'])
df_new.drop(['test preparation course'] ,axis=1, inplace=True)
df_new = pd.concat([df_new, df_encoded], axis=1)
df_new


# In[ ]:


plt.figure(dpi=100)
plt.title('Correlation Analysis')
sns.heatmap(df_new.corr(),annot=True,lw=1,linecolor='white',cmap='viridis')
plt.xticks(rotation=60)
plt.yticks(rotation = 60)
plt.show()


# Looking at this, I see a significant correlation between the scores, and test preparation course completion. What's strange is that the same can be seen for students who didn't get any free/reduced lunch plans. 
# 
# Who would have thought there can be a connection between the test scores of a student, and whether the student is offered free/reduced lunch plans? 
# 
# However, I wouldn't be quick to conclude that yet. There seems to be some connection between race/ethnicity, parental level of education of the student with the lunch plans as well. So those could be the main underlying reason between the difference caused due to lunch plans. 

# In[ ]:




