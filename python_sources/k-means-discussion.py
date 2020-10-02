#!/usr/bin/env python
# coding: utf-8

# # Please upvote if you like this kernel for further encouragement
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
from sklearn.model_selection  import train_test_split
from sklearn.cluster import KMeans

from scipy.stats import zscore

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Read the datset

tech_supp_df = pd.read_csv("/kaggle/input/technical-customer-support-data/technical_support_data.csv")
tech_supp_df.dtypes


# **The dataset contains one record for each unique problem type. It has metrics for each type like count, average calls to resolve, average resolution time etc.**

# In[ ]:


#Shape of the dataset
tech_supp_df.shape


# In[ ]:


#Displaying the first five rows of dataset 
tech_supp_df.head()


# In[ ]:


#Plotiing the pairplot
techSuppAttr=tech_supp_df.iloc[:,1:]
techSuppScaled=techSuppAttr.apply(zscore)
sns.pairplot(techSuppScaled,diag_kind='kde')


# ## Group Data into similar clusters
# 
# Now, we will use K-Means clustering to group data based on their attribute. First, we need to determine the optimal number of groups. For that we conduct the knee test to see where the knee happens.

# In[ ]:


#Finding optimal no. of clusters
from scipy.spatial.distance import cdist
clusters=range(1,10)
meanDistortions=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(techSuppScaled)
    prediction=model.predict(techSuppScaled)
    meanDistortions.append(sum(np.min(cdist(techSuppScaled, model.cluster_centers_, 'euclidean'), axis=1)) / techSuppScaled.shape[0])


plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')


# Though the bend is not coming out clearly as there are many bends, let us look at 3 clusters and 5 clusters

# In[ ]:


# Let us first start with K = 3
final_model=KMeans(3)
final_model.fit(techSuppScaled)
prediction=final_model.predict(techSuppScaled)

#Append the prediction 
tech_supp_df["GROUP"] = prediction
techSuppScaled["GROUP"] = prediction
print("Groups Assigned : \n")
tech_supp_df.head()


# ### Analyze the distribution of the data among the two groups (K = 3). One of the most informative visual tool is boxplot.

# In[ ]:


techSuppClust = tech_supp_df.groupby(['GROUP'])
techSuppClust.mean()


# In[ ]:


techSuppScaled.boxplot(by='GROUP', layout = (2,4),figsize=(15,10))


# In[ ]:


#  Let us next try with K = 6, the next elbow point


# In[ ]:


# Let us first start with K = 6
final_model=KMeans(6)
final_model.fit(techSuppScaled)
prediction=final_model.predict(techSuppScaled)

#Append the prediction 
tech_supp_df["GROUP"] = prediction
techSuppScaled["GROUP"] = prediction
print("Groups Assigned : \n")
tech_supp_df.head()


# In[ ]:


techSuppClust = tech_supp_df.groupby(['GROUP'])
techSuppClust.mean()


# In[ ]:


techSuppScaled.boxplot(by='GROUP', layout = (2,4),figsize=(15,10))

