#!/usr/bin/env python
# coding: utf-8

# # Bank Customer Clustering

# ## K-Mode Clustering

# ### Problem Statement
# 
# The data is related with direct marketing campaigns of a Portuguese banking institution.
# Cluster customers on the basis of attributes.
# 
# Note: This python demonstration is for understanding the use of K-Modes clustering algorithm.
# 
# ### Data
# Only Categorical attributes of Bank Marketing Data Set(UCI Repository: <https://archive.ics.uci.edu/ml/datasets/bank+marketing>) are used for demonstration.

# **Attribute Information(Categorical):**
# 
# - age (numeric)
# - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
# - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
# - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
# - default: has credit in default? (categorical: 'no','yes','unknown')
# - housing: has housing loan? (categorical: 'no','yes','unknown')
# - loan: has personal loan? (categorical: 'no','yes','unknown')
# - contact: contact communication type (categorical: 'cellular','telephone') 
# - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
# - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
# - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# - UCI Repository: <https://archive.ics.uci.edu/ml/datasets/bank+marketing>

# In[ ]:


# supress warnings
import warnings
warnings.filterwarnings('ignore')

# Importing all required packages
import numpy as np
import pandas as pd

# Data viz lib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pyplot import xticks


# ## Data Reading and Understading

# In[ ]:


bank = pd.read_csv('../input/bankmarketing.csv')


# In[ ]:


bank.head()


# In[ ]:


bank.columns


# In[ ]:


# Importing Categorical Columns


# In[ ]:


bank_cust = bank[['age','job', 'marital', 'education', 'default', 'housing', 'loan','contact','month','day_of_week','poutcome']]


# In[ ]:


bank_cust.head()


# In[ ]:


# Converting age into categorical variable.


# In[ ]:


bank_cust['age_bin'] = pd.cut(bank_cust['age'], [0, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                              labels=['0-20', '20-30', '30-40', '40-50','50-60','60-70','70-80', '80-90','90-100'])
bank_cust  = bank_cust.drop('age',axis = 1)


# In[ ]:


bank_cust.head()


# ## Data Inspection

# In[ ]:


bank_cust.shape


# In[ ]:


bank_cust.describe()


# In[ ]:


bank_cust.info()


# ## Data Cleaning

# In[ ]:


# Checking Null values
bank_cust.isnull().sum()*100/bank_cust.shape[0]
# There are no NULL values in the dataset, hence it is clean.


# In[ ]:


# Data is clean.


# ### As it is just a demo for K-Modes we will skip EDA and jump straight to model building

# ## Model Building

# In[ ]:


# First we will keep a copy of data
bank_cust_copy = bank_cust.copy()


# ### Data Preparation

# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
bank_cust = bank_cust.apply(le.fit_transform)
bank_cust.head()


# In[ ]:


# Importing Libraries

from kmodes.kmodes import KModes


# ## Using K-Mode with "Cao" initialization

# In[ ]:


km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(bank_cust)


# In[ ]:


# Predicted Clusters
fitClusters_cao


# In[ ]:


clusterCentroidsDf = pd.DataFrame(km_cao.cluster_centroids_)
clusterCentroidsDf.columns = bank_cust.columns


# In[ ]:


# Mode of the clusters
clusterCentroidsDf


# ## Using K-Mode with "Huang" initialization

# In[ ]:


km_huang = KModes(n_clusters=2, init = "Huang", n_init = 1, verbose=1)
fitClusters_huang = km_huang.fit_predict(bank_cust)


# In[ ]:


# Predicted clusters
fitClusters_huang


# ## Choosing K by comparing Cost against each K

# In[ ]:


cost = []
for num_clusters in list(range(1,5)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(bank_cust)
    cost.append(kmode.cost_)


# In[ ]:


y = np.array([i for i in range(1,5,1)])
plt.plot(y,cost)


# In[ ]:


## Choosing K=2


# In[ ]:


km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(bank_cust)


# In[ ]:


fitClusters_cao


# ### Combining the predicted clusters with the original DF.

# In[ ]:


bank_cust = bank_cust_copy.reset_index()


# In[ ]:


clustersDf = pd.DataFrame(fitClusters_cao)
clustersDf.columns = ['cluster_predicted']
combinedDf = pd.concat([bank_cust, clustersDf], axis = 1).reset_index()
combinedDf = combinedDf.drop(['index', 'level_0'], axis = 1)


# In[ ]:


combinedDf.head()


# ### Cluster Identification

# In[ ]:


cluster_0 = combinedDf[combinedDf['cluster_predicted'] == 0]
cluster_1 = combinedDf[combinedDf['cluster_predicted'] == 1]


# In[ ]:


cluster_0.info()


# In[ ]:


cluster_1.info()


# In[ ]:


# Job


# In[ ]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combinedDf['job'],order=combinedDf['job'].value_counts().index,hue=combinedDf['cluster_predicted'])
plt.show()


# In[ ]:


# Marital


# In[ ]:


plt.subplots(figsize = (5,5))
sns.countplot(x=combinedDf['marital'],order=combinedDf['marital'].value_counts().index,hue=combinedDf['cluster_predicted'])
plt.show()


# In[ ]:


# Education


# In[ ]:


plt.subplots(figsize = (15,5))
sns.countplot(x=combinedDf['education'],order=combinedDf['education'].value_counts().index,hue=combinedDf['cluster_predicted'])
plt.show()


# In[ ]:


# Default


# In[ ]:


f, axs = plt.subplots(1,3,figsize = (15,5))
sns.countplot(x=combinedDf['default'],order=combinedDf['default'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[0])
sns.countplot(x=combinedDf['housing'],order=combinedDf['housing'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[1])
sns.countplot(x=combinedDf['loan'],order=combinedDf['loan'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[2])

plt.tight_layout()
plt.show()


# In[ ]:


f, axs = plt.subplots(1,2,figsize = (15,5))
sns.countplot(x=combinedDf['month'],order=combinedDf['month'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[0])
sns.countplot(x=combinedDf['day_of_week'],order=combinedDf['day_of_week'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[1])

plt.tight_layout()
plt.show()


# In[ ]:


f, axs = plt.subplots(1,2,figsize = (15,5))
sns.countplot(x=combinedDf['poutcome'],order=combinedDf['poutcome'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[0])
sns.countplot(x=combinedDf['age_bin'],order=combinedDf['age_bin'].value_counts().index,hue=combinedDf['cluster_predicted'],ax=axs[1])

plt.tight_layout()
plt.show()


# In[ ]:


# Above visualization can help in identification of clusters.

