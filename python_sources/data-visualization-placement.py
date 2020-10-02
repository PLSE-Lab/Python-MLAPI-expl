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


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('ggplot')


# In[ ]:


df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()


# In[ ]:


for col in df.columns.unique():
    print('\n', col ,'\n', df[col].unique())


# In[ ]:


df1 = df.copy()


# In[ ]:


df.isna().any()


# In[ ]:


df = df.fillna(0)


# In[ ]:


df.info()


# In[ ]:


fig, axs = plt.subplots(ncols=4,figsize=(20,5))
sns.countplot(df['gender'], ax = axs[0])
sns.countplot(df['ssc_b'], ax = axs[1], palette="vlag")
sns.countplot(df['hsc_b'], ax = axs[2], palette="rocket")
sns.countplot(df['hsc_s'], ax = axs[3], palette="deep")


# # Above charts show you the count of gender, SSC board, HSC board, HSC Stream of the candiadtes

# In[ ]:


fig, axs = plt.subplots(ncols=3,figsize=(20,5))
sns.countplot(df['workex'], ax = axs[0], palette="Paired")
sns.countplot(df['specialisation'], ax = axs[1], palette="muted")
sns.countplot(df['status'], ax = axs[2],palette="dark")


# # Above code shows you the count of candidates having work exprience, the specialization they have chosen, & the status if placed or not placed

# In[ ]:


df = df.drop(['sl_no'], axis = 1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


df['gender'] = le.fit_transform(df['gender'])
df['ssc_b'] = le.fit_transform(df['ssc_b'])
df['workex'] = le.fit_transform(df['workex'])
df['specialisation'] = le.fit_transform(df['specialisation'])
df['status'] = le.fit_transform(df['status'])
df['hsc_b'] = le.fit_transform(df['hsc_b'])
df['hsc_s'] = le.fit_transform(df['hsc_s'])
df['degree_t'] = le.fit_transform(df['degree_t'])


# In[ ]:


plt.figure(figsize=(15,10))
corr = df.corr()
sns.heatmap(corr, annot = True)


# # ***From the above correlation matric its clear that Placement & Salary has strong relationship with three factors 
# # 1)degree_p i.e. Degree % , 2)hsc_p, i.e. HSC % , 3)ssc_p i.e SSC % marks ***

# In[ ]:


plt.figure(figsize=(5,5))
sns.scatterplot(x='status', y = 'degree_p', hue ='gender', data = df1)


# # Above chart shows you the relation of degree % and the status of gender if placed or not placed

# In[ ]:


fig, axs = plt.subplots(ncols=5,figsize=(25,5))
sns.distplot(df1['degree_p'], ax= axs[0], color = 'g')
sns.distplot(df1['hsc_p'], ax= axs[1])
sns.distplot(df1['ssc_p'],  ax= axs[2], color = 'b')
sns.distplot(df1['etest_p'],  ax= axs[3], color = 'r')
sns.distplot(df1['mba_p'],  ax= axs[4], color = 'c')


# # Above graphs show you the distplot for Degree %, HSC%, SSC%, MBA% scored

# In[ ]:


plt.figure(figsize=(7,7))
plt.hist(df1['salary'], bins = 10)
plt.show()


# # Above shows you the histogram of the salary after placement.

# In[ ]:


fig, axs = plt.subplots(ncols=3,figsize=(20,5))
sns.scatterplot(x = 'degree_p',y='hsc_p',hue='status',data = df1, ax= axs[0])
sns.scatterplot(x = 'degree_p',y='hsc_p',hue='gender',data = df1, ax= axs[1], palette="muted")
sns.scatterplot(x = 'degree_p',y='hsc_p',hue='degree_t',data = df1, palette="dark", ax= axs[2])


# **Above chart shows you the relationship of different parameters e.g. % of HSC scored vs Degree % with the status wether placed or not place, or the gender, the specialization they chose.**

# In[ ]:


fig, axs = plt.subplots(ncols=3,figsize=(20,5))
sns.scatterplot(x = 'degree_p',y='hsc_p',hue='hsc_s',data = df1, ax= axs[0])
sns.scatterplot(x = 'degree_p',hue='specialisation',y='mba_p',data = df1, ax= axs[1], palette="muted")
sns.scatterplot(x = 'degree_p',hue='workex',y='salary',data = df1, palette="dark", ax= axs[2])


# In[ ]:


df = pd.DataFrame(df)


# In[ ]:


df_placed = df1[df1['status'] == 'Placed']


# **I have created dataframe only for Staus = Placed**

# In[ ]:


df2 = df_placed.groupby(['degree_t','degree_p','mba_p','specialisation','hsc_p', 'hsc_s','salary', 'workex']).sum().sort_values(by ='salary')
df2


# In[ ]:


df_np = df1[df1.status == 'Not Placed']


# # I have created dataframe for not placed candidates.Let's see what the data looks like
#  

# In[ ]:


df3 = df_np.groupby(['degree_t','degree_p','mba_p','specialisation','hsc_p', 'hsc_s','workex']).sum().sort_values(by ='degree_p')
df3

