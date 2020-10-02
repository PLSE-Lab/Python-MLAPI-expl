#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(111)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.set_option('display.max_colwidth', None)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing, reading and formatting

# In[ ]:


import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


whole = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
whole.gender = np.where(whole.gender=='M','Male','Female')
whole.salary = whole.salary.fillna(0)
whole.sample(5)


# In[ ]:


whole.shape


# In[ ]:


def tag(x):
    if x<246666:
        return 'low'
    elif 246666<x<493332:
        return 'medium'
    elif x>493332:
        return 'high'
whole['salary_level'] = whole.salary.apply(tag)


# # Visualizations

# In[ ]:


gender_count = whole.groupby(['gender','status'])['sl_no'].count().reset_index()
px.bar(gender_count, 'gender', 'sl_no', title='Gender count w.r.t placement status',
      width=600, height=600, labels={'sl_no':'Count','gender':'Gender'},color='status',template='seaborn')


# ### Count of males is more in comparison to females in placed as well as not placed category, but there is drastic difference in Males/Females ratio in placed and not placed. 

# In[ ]:


salary = whole[whole.salary!=0].groupby(['specialisation','workex'])['salary'].mean().reset_index()
px.bar(salary, 'specialisation', 'salary', title='Salary for specialisations and workex',
      width=600, height=600,color='workex')


# ### Here, work experience seems to affect salary of the candidate. This might be due to the fact that experience adds to less pre-training costs and candidates are already familiar to the environment. 

# In[ ]:


px.bar(whole[whole.status=='Placed'], y='salary',color='gender')


# ### Even though Male count under placed category is more, Females seem to bag higher packages than Males

# In[ ]:


px.histogram(whole[whole.status=='Placed'],x='salary',nbins=10,color_discrete_sequence=['indianred'],opacity=0.8,
            title='Salary Distribution',marginal="box")


# ### Maximum salary packages range between 0.2M to 0.4M

# In[ ]:


status_ls = list(whole.status)*4
cert = ['SSC']*215 + ['HSC']*215 + ['DEGREE']*215 + ['MBA']*215
scores = list(whole.ssc_p) + list(whole.hsc_p) + list(whole.degree_p) + list(whole.mba_p)
cmp = pd.DataFrame({'Qualification':cert,'Percentage':scores,'Placement status':status_ls})


# In[ ]:


fig1 = px.violin(cmp[cmp['Placement status']=='Placed'], y='Percentage', color='Qualification', box=True, 
                     points='all', template='plotly_white',title='Placed')
fig1.show()
fig2 = px.violin(cmp[cmp['Placement status']=='Not Placed'], y='Percentage', color='Qualification', box=True, 
                 points='all', template='plotly_white',title='Not placed')
fig2.show()


# In[ ]:


px.violin(cmp, x='Qualification', y='Percentage', color='Placement status', box=True, points='all', template='plotly_white',
                 title='Placed vs Not placed score comparision')


# ### In general Placed students seem to have higher score at every level of education

# In[ ]:


stream = whole[whole.status=='Placed']['degree_t'].value_counts()
fig3=px.pie(stream, names = stream.index, values = stream.values,color_discrete_sequence=px.colors.qualitative.T10,
          title='Bachelors Specialization demanded by corporates')
fig3.update_traces(textposition='inside', textinfo='percent+label',textfont_size=15)
fig3.show()


# ### Corporates seem to be more interested in candidates from Commerce and management background followed by science and technology

# In[ ]:


stream = whole[whole.status=='Placed']['specialisation'].value_counts()
fig3=px.pie(stream, names = stream.index, values = stream.values,color_discrete_sequence=px.colors.qualitative.T10,
          title='MBA Specialization demanded by corporates',hole=0.5)
fig3.update_traces(textposition='inside', textinfo='percent+label',textfont_size=15)
fig3.show()


# ### Students from Marketing and Finance, MBA specialisation form 64% of total employed followed by Marketing and HR at 36%

# In[ ]:


fig4 = px.treemap(whole,path=('hsc_s','specialisation','status'),color_discrete_sequence=px.colors.qualitative.G10,
                 title='HSC stream->MBA specialisation->Placement status')
fig4.show()
fig5 = px.treemap(whole,path=('degree_t','specialisation','status'),color_discrete_sequence=px.colors.qualitative.D3,
                 title='Bachelor stream->MBA specialisation->Placement status')
fig5.show()


# ### Breakdown of placement statistics from the point of education

# In[ ]:


px.sunburst(whole,path=('specialisation','salary_level'),color_discrete_sequence=px.colors.qualitative.Set1,
                 title='MBA specialisation->Salary level')


# ### Of the two MBA specialisations, Mkt and Fin constitute for max medium range salary packages whereas Mkt and HR counts for max lower range salary packages. Interestingly, only Mkt and Fin had higher range packages candidate. 

# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


train = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
train.head()


# In[ ]:


train.status = np.where(train.status=='Placed',1,0)
train.gender = np.where(train.gender=='M',1,0)
train.ssc_b = np.where(train.ssc_b=='Central',1,0)
train.hsc_b = np.where(train.hsc_b=='Central',1,0)
train.workex = np.where(train.workex=='Yes',1,0)
train.salary = train.salary.fillna(0)
train.drop('sl_no',axis=1,inplace=True)
train = pd.get_dummies(train)
train.sample(5)


# In[ ]:


train.shape


# In[ ]:


X_cols = list(train.columns)
X_cols.remove('status')


# In[ ]:


X = train[X_cols]
y = train['status']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7, random_state=0)


# In[ ]:


lr = LogisticRegression()
lr.fit(X_train,y_train)


# In[ ]:


lr.score(X_test,y_test)


# ### Logistic Regression model overfits the training data therefore is not a good generalizer. One reason for overfitting might be due to skewness in training data, Lets check it out.

# In[ ]:


y_train.value_counts()


# ### Clearly 1 i.e Placed class samples are more than twice of 0 i.e Not Placed

# In[ ]:


np.unique(lr.predict(X_test), return_counts=True)

