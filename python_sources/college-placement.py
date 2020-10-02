#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


dataset = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


dataset.head(5)


# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.describe(include='O')


# In[ ]:


for data in [dataset]:
    data['status'] = data['status'].map({'Placed':1,'Not Placed':0}).astype(int)
    


# In[ ]:


dataset.groupby(['status','gender'])['gender','status'].count()
        


# In[ ]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")


g = sns.FacetGrid(data=dataset, row="gender", col="status", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "status", color="steelblue", bins=bins)


# In[ ]:


dataset.groupby(['ssc_b','status'])['status'].count()


# In[ ]:


sns.set(style="darkgrid")


g = sns.FacetGrid(data=dataset, row="ssc_b", col="status", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "status", color="steelblue", bins=bins)


# In[ ]:


dataset.groupby(['hsc_b','status'])['status'].count()


# In[ ]:


sns.set(style="darkgrid")


g = sns.FacetGrid(data=dataset, row="hsc_b", col="status", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "status", color="green", bins=bins)


# In[ ]:


# train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
dataset[['hsc_b','status']].groupby(['hsc_b']).mean().sort_values(by='status', ascending=False)


# In[ ]:


dataset.groupby(['degree_t','status'])[['status']].count()


# In[ ]:


sns.set(style="darkgrid")


g = sns.FacetGrid(data=dataset, row="degree_t", col="status", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "status", color="green", bins=bins)


# In[ ]:


dataset.groupby(['status','workex'])[['status']].count()


# In[ ]:


sns.set(style="darkgrid")


g = sns.FacetGrid(data=dataset, row="workex", col="status",hue='gender', margin_titles=True)
bins = np.linspace(0, 60, 13)
(g.map(plt.hist, "status",bins=bins).add_legend())


# In[ ]:


# ratio of no of female and male students
dataset[dataset['status']==1].groupby(['gender'])['gender'].count()


# In[ ]:


dataset[dataset['status']==1].groupby(['gender'])['gender'].count().plot.bar()


# In[ ]:


dataset[dataset['status']==1].groupby(['specialisation','gender'])[['specialisation']].count()


# In[ ]:


sns.set(style="darkgrid")

g = sns.FacetGrid(data=dataset, row="specialisation", col="status",hue='gender', margin_titles=True)
bins = np.linspace(0, 60, 13)
(g.map(plt.hist, "status",bins=bins).add_legend())


# In[ ]:


data = dataset[dataset['status']==1]
data[data['specialisation']=='Mkt&HR'].groupby('gender')[['status']].count().plot.bar()


# In[ ]:


dataset['salary'].dropna().plot()


# In[ ]:


dataset['salary'].dropna().plot.box()


# In[ ]:


dataset[['salary']].dropna().describe()


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data = dataset
data['gender'] = le.fit_transform(data['gender'])
data.tail(20)


# In[ ]:


# dataset['degree_p'].plot.hist()
sns.catplot(y="degree_p",x='status', hue="gender", kind="bar", data=dataset)


# In[ ]:


# dataset['degree_p'].plot.hist()
sns.catplot(y="degree_p",x='status', hue="specialisation", kind="bar", data=dataset);


# In[ ]:



sns.catplot(y="hsc_p",x='gender', hue="status", kind="bar", data=dataset);


# pearson correlation test

# In[ ]:


from scipy.stats import pearsonr
corr, _ = pearsonr(dataset['degree_p'], dataset['hsc_p'])
corr

