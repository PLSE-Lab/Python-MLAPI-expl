#!/usr/bin/env python
# coding: utf-8

# **Unsupervised Learning**

# Libraries

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


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)


# In[ ]:


df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv", index_col = 0)
df_test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv", index_col = 0)
df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df["Province/State"]=df["Province/State"].fillna(df["Country/Region"])
df.isnull().sum()


# In[ ]:


df.info


# In[ ]:


df.describe().T


# In[ ]:


df.head()


# In[ ]:


df = df[df["Date"] == max(df["Date"])]
df=df.reset_index(drop=True)
df=df.reset_index(drop=True)



# In[ ]:


df


# In[ ]:


df.hist(figsize = (10,10));


# In[ ]:


df.head()


# In[ ]:





# In[ ]:


df=df.drop(["Date","Province/State","Lat","Long"], axis=1)


# In[ ]:


df=df.groupby(["Country/Region"]).sum()
df=df.reset_index()
df


# In[ ]:


keys=df["Country/Region"]
keys


# In[ ]:


df.set_index(keys, drop=False,inplace=True)
df=df.drop(["Country/Region"], axis=1)
df


# In[ ]:


kmeans = KMeans(n_clusters = 4)


# In[ ]:


kmeans


# In[ ]:


k_fit = kmeans.fit(df)


# In[ ]:


k_fit.n_clusters


# In[ ]:


k_fit.cluster_centers_


# In[ ]:


k_fit.labels_


# In[ ]:


k_means = KMeans(n_clusters = 2).fit(df)


# In[ ]:


kumeler = k_means.labels_


# In[ ]:


kumeler


# In[ ]:


plt.scatter(df.iloc[:,0], df.iloc[:,1], c = kumeler, s = 50, cmap = "viridis");


# In[ ]:


merkezler = k_means.cluster_centers_


# In[ ]:


merkezler


# In[ ]:


plt.scatter(df.iloc[:,0], df.iloc[:,1], c = kumeler, s = 50, cmap = "viridis")
plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha=0.5);


# In[ ]:


ssd = []

K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters = k).fit(df)
    ssd.append(kmeans.inertia_)


# In[ ]:


plt.plot(K, ssd, "bx-")
plt.xlabel("Total K Distance Against Different K Values")
plt.title("Elbow Method for Optimum Cluster Number")


# In[ ]:


get_ipython().system('pip install yellowbrick')


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer


# In[ ]:


kmeans = KMeans()
visu = KElbowVisualizer(kmeans, k = (2,20))
visu.fit(df)
visu.poof()


# In[ ]:


kmeans = KMeans(n_clusters = 4).fit(df)
kmeans


# In[ ]:


df


# In[ ]:


kumeler = kmeans.labels_


# In[ ]:


pd.DataFrame({"Country/Region": df.index, "Kumeler": kumeler})


# In[ ]:


df["Kume_No"] = kumeler


# In[ ]:


df.head()


# In[ ]:


df.sort_values("ConfirmedCases",ascending=False)


# In[ ]:


df.groupby(["Kume_No"]).sum()


# In[ ]:


df[df["Kume_No"]==1]


# In[ ]:


df[df["Kume_No"]==2]


# In[ ]:


df[df["Kume_No"]==3]


# In[ ]:


df[df["Kume_No"]==0].head(90)


# In[ ]:





# In[ ]:




