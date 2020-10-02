#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/all.csv',sep=',')
df.head(2)


# In[ ]:


state=df.groupby(['State']).sum()
#state=state.loc['Persons']

#list(state)
import matplotlib.pyplot as plt
import seaborn as sb
g=state['Persons'].sort_values(ascending=False)
g.plot(kind='bar')
#state.head(2)


# In[ ]:


list(state.index)
South=state.loc['Kerala']
South.head()


# In[ ]:





# In[ ]:


df.max()


# In[ ]:


state=state.iloc[:,1:]
South=state.loc[['TN','Karnataka','Kerala','Andhra']]
South.head()
South.max()


# In[ ]:


table=pd.crosstab(df['State'],df['Household.size..per.household.'])
table.head()
from scipy.stats import chi2_contingency
chi2 , p ,dof ,expected= chi2_contingency(table.values)
print('{0:.100f}'.format(dof))
expected


# In[ ]:


import sklearn
from sklearn.cluster import KMeans
from sklearn import *
from sklearn.metrics import *
import sklearn.metrics as sn
from sklearn.preprocessing import scale


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(6,6))


# In[ ]:


iris=datasets.load_iris()
x=scale(iris.data)
y=pd.DataFrame(iris.target)
variable_names=iris.feature_names


# In[ ]:


clustering = KMeans(n_clusters=3, random_state=5)
clustering.fit(x)


# In[ ]:


iris_df=pd.DataFrame(iris.data)
iris_df.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y.columns=['Targets']


# In[ ]:


color_theme=np.array(['darkgray','lightsalmon','powderblue'])
plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[iris.target],s=25)
plt.title('Classificatioon')
plt.subplot(1,2,2)

plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[clustering.labels_],s=25)
plt.title('K Means Classificatioon')


# In[ ]:


relabel=np.choose(clustering.labels_,[2,0,1]).astype(np.int64)
color_theme=np.array(['darkgray','lightsalmon','powderblue'])
plt.subplot(1,2,1)
plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[iris.target],s=50)
plt.title('Classificatioon')
plt.subplot(1,2,2)

plt.scatter(x=iris_df.Petal_Length,y=iris_df.Petal_Width,c=color_theme[relabel],s=50)
plt.title('K Means Classificatioon')


# In[ ]:


print(classification_report(y,relabel))


# In[ ]:


iris.target


# In[ ]:




