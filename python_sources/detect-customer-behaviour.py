#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_excel('../input/overdue.xlsx',sheet_name='Data Fin')
data.head()


# In[ ]:


#removing garbage column
data.drop(['ForMonth'],axis=1,inplace=True)
data.head()


# In[ ]:


#checking data shape
data.shape


# In[ ]:


#checking for null values
data.isnull().sum()


# In[ ]:


#checking for duplicate rows
data.duplicated().value_counts()


# In[ ]:


#checking for duplicate Customer Code
data['Code'].duplicated().value_counts()


# In[ ]:


#checking data types of columns
data.info()


# In[ ]:


#summary statistics
data.describe()


# In[ ]:


list(data.columns)


# In[ ]:


#renaming columns 
data.rename(columns={'OD segment 17':'OD_segment_17','Inst. Size Jan 17':'Inst_Size_Jan_17','Inst. OD No. ':'Inst_OD_No','Inst. OD Jan 18':'Inst_OD_Jan_18','Inst. Outstanding Jan 18':'Inst_Outstanding_Jan_18'},inplace=True)


# In[ ]:


data.head()


# In[ ]:


data['OD_segment_17'].unique()


# In[ ]:


#Status
od_segment_17=data['OD_segment_17'].value_counts()
od_segment_17.plot(kind='bar')
print(od_segment_17)


# In[ ]:


data.head()


# In[ ]:


inst_size_jan_17=data['Inst_Size_Jan_17']
#inst_size_jan_17.plot(kind='kde')
plt.figure(figsize=(15,15))
fig,ax1=plt.subplots()
sns.distplot(inst_size_jan_17,color='red')
ax1.set_ylabel('Count')
plt.title('Distribution of Inst_Size_Jan_17')
plt.show()


# In[ ]:


plt.figure(figsize=(15,15))
data.hist()
plt.show()


# In[ ]:


#box whisker plot
plt.figure(figsize=(15,15))
data.iloc[:,8:].plot(kind='box')
plt.title('Box whisker plot')
plt.show()


# In[ ]:


#pair plot
sns.pairplot(data)


# In[ ]:


data.head()


# In[ ]:


data['Inst. OD No.'].plot(kind='hist')
data['Inst. OD No.'].describe()


# In[ ]:


feature=data.iloc[:,7:]
feature.head()


# In[ ]:


feature[data['OD_segment_17']=='Expire'].head()


# In[ ]:


feature['OD_segment_17'].replace({"Expire":0,"Running":1},inplace=True)
feature.head()


# In[ ]:


feature_matrix=feature.as_matrix()
feature_matrix


# In[ ]:


running=feature[feature['OD_segment_17']==1]
running.head()
print(running.shape)
#running_matrix=running_
running_matrix=running.iloc[:,2:4].as_matrix()
running_matrix
X=running_matrix


# In[ ]:


plt.scatter(X[:,0],X[:,1],color='red',marker='*')


# In[ ]:


X.shape


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=3).fit(X)
print("Clusters:",np.unique(clustering.labels_))
s=pd.Series(clustering.labels_)
print(s.value_counts())
print(len(s))
#print("Cluster label:",clustering.labels_)


# # Cluster Behavior Analysis

# # Cluster 0 : Very Risky Customer

# In[ ]:


running.iloc[:,:][clustering.labels_==0]['Inst. OD No.'].describe()


# In[ ]:


running.iloc[:,:][clustering.labels_==0].sample(20)


# # Cluster 1 : Risky Customer

# In[ ]:


running.iloc[:,:][clustering.labels_==1]['Inst. OD No.'].describe()


# In[ ]:


running.iloc[:,:][clustering.labels_==1].sample(20)


# # Cluster 2: Good Customer

# In[ ]:


running.iloc[:,:][clustering.labels_==2]['Inst. OD No.'].describe()


# In[ ]:


running.iloc[:,:][clustering.labels_==2].sample(20)


# In[ ]:


running_customer=running.copy()


# In[ ]:


running_customer['Cluster Label']=list(clustering.labels_)


# In[ ]:


running_customer


# In[ ]:


running_customer[running_customer['Cluster Label']==0].describe()


# In[ ]:


data.head()


# In[ ]:


customer_cluster=data[data['OD_segment_17']=='Running'].copy()


# In[ ]:


customer_cluster.shape


# In[ ]:


customer_cluster['Cluster Label']=list(clustering.labels_)


# In[ ]:


customer_cluster.head()


# In[ ]:


customer_cluster[customer_cluster['Cluster Label']==0].describe()


# In[ ]:


customer_cluster['Cluster Label'].replace({0:"Very risky",1:"Risky",2:"Good"},inplace=True)


# In[ ]:


customer_cluster.head()


# In[ ]:


customer_cluster['Cluster Label'].value_counts()


# In[ ]:


customer_cluster.to_csv('Customer_Cluster.csv')


# In[ ]:


very_risky=customer_cluster[customer_cluster['Cluster Label']=='Very risky']


# In[ ]:


very_risky.head()


# In[ ]:


very_risky.describe()


# In[ ]:


very_risky.to_csv('Very_Risky_Customer.csv')


# In[ ]:




