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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_excel("../input/Train.xlsx")


# In[ ]:


data.head()


# In[ ]:


dataframe_suspicious=data[data["Suspicious"]=="Yes"]


# In[ ]:


dataframe_not_suspicious=data[data["Suspicious"]=='No']


# In[ ]:


dataframe_indeterminate=data[data['Suspicious']=='indeterminate']


# In[ ]:


salespersons=data['SalesPersonID'].unique()


# In[ ]:


temp=pd.DataFrame()


# In[ ]:


temp['SalesPersonID']=salespersons


# In[ ]:


temp.head()


# In[ ]:


temp.shape


# In[ ]:


temp1=pd.DataFrame(dataframe_suspicious.groupby("SalesPersonID").sum()['TotalSalesValue'])


# In[ ]:


temp2=pd.DataFrame(dataframe_not_suspicious.groupby("SalesPersonID").sum()['TotalSalesValue'])


# In[ ]:


temp3=pd.DataFrame(dataframe_indeterminate.groupby("SalesPersonID").sum()['TotalSalesValue'])


# In[ ]:


temp1=temp1.reset_index()


# In[ ]:


temp2=temp2.reset_index()


# In[ ]:


temp3=temp3.reset_index()


# In[ ]:


iter1=pd.merge(temp,temp1,how='left')


# In[ ]:


temp2.dtypes


# In[ ]:


iter2=pd.merge(iter1,temp2,how='left',on='SalesPersonID')


# In[ ]:


iter3=pd.merge(iter2,temp3,how='left',on='SalesPersonID')


# In[ ]:


iter3.head()


# In[ ]:


iter3=iter3.fillna(0)


# In[ ]:


iter3.shape


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


le=LabelEncoder()


# In[ ]:


le.fit(iter3['SalesPersonID'])


# In[ ]:


iter3['SalesPersonID']=le.transform(iter3['SalesPersonID'])


# In[ ]:


iter3.head()


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeans = KMeans(n_clusters=3)


# In[ ]:


kmeans.fit(iter3)


# In[ ]:


print(kmeans.cluster_centers_)


# In[ ]:


len(kmeans.predict(iter3))


# In[ ]:


y_km = kmeans.predict(iter3)


# In[ ]:


value,counts=np.unique(y_km,return_counts=True)
print(np.asarray([value,counts]))


# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


model = TSNE(n_components=2, random_state=0)


# In[ ]:


tsne_data = model.fit_transform(iter3)


# In[ ]:


tsne_data = np.vstack((tsne_data.T, y_km)).T


# In[ ]:


tsne_data


# In[ ]:


tsne_data.shape


# In[ ]:


tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "clusterid"))


# In[ ]:


sns.FacetGrid(tsne_df, hue="clusterid", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.show()

