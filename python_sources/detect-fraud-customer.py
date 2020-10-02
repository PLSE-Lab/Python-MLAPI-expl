#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_excel('../input/kaggle_upload.xlsx',sheet_name='Data Fin')
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


#checking data types of columns
data.info()


# In[ ]:


#summary statistics
data.describe()


# In[ ]:


list(data.columns)


# In[ ]:


#renaming columns 
data.rename(columns={'OD segment 17':'OD_segment_17','Inst. Size Jan 17':'Inst_Size_Jan_17','Inst. OD No.':'Inst_OD_No','Inst. OD Jan 18':'Inst_OD_Jan_18','Inst. Outstanding Jan 18':'Inst_Outstanding_Jan_18'},inplace=True)
data.head()


# In[ ]:


data['OD_segment_17'].unique()


# In[ ]:


#Status
od_segment_17=data['OD_segment_17'].value_counts()
od_segment_17.plot(kind='bar')
print(od_segment_17)


# In[ ]:


#separating running customer from expired customer
running_customer=data[data['OD_segment_17']=='Running'].copy()
print(running_customer['OD_segment_17'].value_counts())
running_customer.head()


# In[ ]:


inst_size_jan_17=running_customer['Inst_Size_Jan_17']
#inst_size_jan_17.plot(kind='kde')
plt.figure(figsize=(15,15))
fig,ax1=plt.subplots()
sns.distplot(inst_size_jan_17,color='red')
ax1.set_ylabel('Count')
plt.title('Distribution of Inst_Size_Jan_17')
plt.show()


# In[ ]:


#Histogrm of all numeric feature
plt.figure(figsize=(20,20))
plt.figure(figsize=(15,15))
running_customer.hist()
plt.show()


# In[ ]:


#box whisker plot
plt.figure(figsize=(15,15))
running_customer.iloc[:,8:].plot(kind='box')
plt.title('Box whisker plot')
plt.show()


# In[ ]:


ax = sns.boxplot(data=running_customer, orient="v", palette="Set2")


# In[ ]:


ax = sns.violinplot(data=running_customer, orient="h", palette="Set2")


# In[ ]:


#pair plot
sns.pairplot(running_customer)


# In[ ]:


running_customer['Inst_OD_No'].plot(kind='hist')
running_customer['Inst_OD_No'].describe()


# In[ ]:


running_customer['Inst_OD_Jan_18'].plot(kind='hist')
running_customer['Inst_OD_Jan_18'].describe()


# In[ ]:


sns.jointplot(x="Inst_OD_No",y="Inst_OD_Jan_18",data=running_customer)


# In[ ]:


sns.jointplot(x="Inst_Size_Jan_17",y="Inst_OD_Jan_18",data=running_customer)


# In[ ]:


corr=running_customer.corr()
plt.figure()
ax = sns.heatmap(corr,annot=True)
plt.title('Correlation')
#print(corr)


# In[ ]:


running_customer.head()


# In[ ]:


feature=running_customer[['Inst_OD_No','Inst_OD_Jan_18']]
feature.head()


# In[ ]:


X=feature.as_matrix()
plt.scatter(X[:,0],X[:,1],color='red',marker='*')


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#scaler=StandardScaler()
scaler=MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)

from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=3).fit(X[:,0:1])
print("Clusters:",np.unique(clustering.labels_))
s=pd.Series(clustering.labels_)
print(s.value_counts())
print(len(s))
#print("Cluster label:",clustering.labels_)


# In[ ]:


from sklearn.cluster import KMeans
clustering = KMeans(n_clusters=3, random_state=0).fit(X[:,0:1])
print("Clusters:",np.unique(clustering.labels_))
s=pd.Series(clustering.labels_)
print(s.value_counts())
print(len(s))


# In[ ]:


from sklearn.metrics import silhouette_score
print(silhouette_score(X,clustering.labels_))


# In[ ]:


#hdbscan algorithm
"""import hdbscan
clustering = hdbscan.HDBSCAN(min_cluster_size=250, gen_min_span_tree=True)
clustering.fit(X)
print("Clusters:",np.unique(clustering.labels_))
s=pd.Series(clustering.labels_)
print(s.value_counts())
print(len(s))"""


# In[ ]:


running_customer['Cluster_Label']=list(clustering.labels_)
ax=sns.boxplot(x="Cluster_Label",y="Inst_OD_No",data=running_customer)


# In[ ]:


ax=sns.boxplot(x="Cluster_Label",y="Inst_OD_Jan_18",data=running_customer)


# In[ ]:


# Aforementioned box plot cluster 0 means very risky, 1 means risky and cluster 2 means good
#Cluster Analysis


# In[ ]:


#Good Customer
running_customer[running_customer['Cluster_Label']==0].describe()


# In[ ]:


running_customer[running_customer['Cluster_Label']==0].sample(20)


# In[ ]:


#Risky Customer, Cluster 1
running_customer[running_customer['Cluster_Label']==1].describe()


# In[ ]:


running_customer[running_customer['Cluster_Label']==1].sample(20)


# In[ ]:


#Very Risky Customer, Cluster 2
running_customer[running_customer['Cluster_Label']==2].describe()


# In[ ]:


running_customer[running_customer['Cluster_Label']==2].sample(20)


# In[ ]:


running_customer['Cluster_Label'].replace({0:'Good',1:'Risky',2:'Very Risky'},inplace=True)
running_customer.sample(20)


# In[ ]:


X


# In[ ]:


#Predict
clustering.predict(X[2:4,0:1])


# In[ ]:


running_customer.sort_values(by=['Code']).head()


# In[ ]:


#Predict for next month ie February 2018
#import pandas as pd
february=pd.read_excel('../input/February2018.xls')
february.head()


# In[ ]:


february.rename(columns={'OD segment 18':'OD_segment_18','Inst. Size Jan 17':'Inst_Size_Jan_17','Inst. OD No.':'Inst_OD_No','Inst. OD Feb 18':'Inst_OD_Feb_18','Inst. Outstanding Jan 18':'Inst_Outstanding_Jan_18'},inplace=True)


# In[ ]:


february.head()


# In[ ]:


#Status
od_segment_18=february['OD_segment_18'].value_counts()
od_segment_18.plot(kind='bar')
print(od_segment_18)


# In[ ]:


running_customer_feb=february[february['OD_segment_18']=='Running'].copy()
print(running_customer_feb['OD_segment_18'].value_counts())
running_customer_feb.head()


# In[ ]:


plt.figure(figsize=(20,20))
plt.figure(figsize=(15,15))
running_customer_feb.hist()
plt.show()


# In[ ]:


ax = sns.boxplot(data=running_customer_feb, orient="v", palette="Set2")


# In[ ]:


corr=running_customer_feb.corr()
plt.figure()
ax = sns.heatmap(corr,annot=True)
plt.title('Correlation')


# In[ ]:


feature_feb=running_customer_feb[['Inst_OD_No','Inst_OD_Feb_18']]
feature_feb.head()


# In[ ]:


X_feb=feature_feb.as_matrix()


# In[ ]:


X_feb=scaler.fit_transform(X_feb)


# In[ ]:


label=clustering.predict(X_feb[:,0:1])
print("Clusters:",np.unique(label))
s=pd.Series(label)
print(s.value_counts())
print(len(s))


# In[ ]:


running_customer_feb['Cluster_Label']=list(label)
running_customer_feb.head()


# In[ ]:


ax=sns.boxplot(x="Cluster_Label",y="Inst_OD_No",data=running_customer_feb)


# In[ ]:


ax=sns.boxplot(x="Cluster_Label",y="Inst_OD_Feb_18",data=running_customer_feb)


# In[ ]:


running_customer_feb['Cluster_Label'].replace({0:'Good',1:'Risky',2:'Very Risky'},inplace=True)
running_customer_feb.sample(20)


# In[ ]:


running_customer_feb[running_customer_feb['Cluster_Label']=='Good'].describe()


# In[ ]:


running_customer_feb[running_customer_feb['Cluster_Label']=='Risky'].describe()


# In[ ]:


running_customer_feb[running_customer_feb['Cluster_Label']=='Very Risky'].describe()


# In[ ]:


running_customer.head()


# In[ ]:


running_customer_feb.head()


# In[ ]:


print(running_customer_feb.shape)
print(running_customer.shape)


# In[ ]:


running_customer_jan=running_customer.drop(['Showroom'],axis=1)
running_customer_jan.head()


# In[ ]:


jan_matrix=running_customer_jan.as_matrix()
feb_matrix=running_customer_feb.as_matrix()


# In[ ]:


dataset=np.vstack([jan_matrix,feb_matrix])
dataset.shape


# In[ ]:


list(running_customer_jan.columns)


# In[ ]:


dataset=pd.DataFrame(dataset,columns=['Code','Customer Name','Part','Region','Area','Territory','OD_segment','Inst_Size','Inst_OD_No','Inst_OD_Amount','Inst_OutStanding','Cluster_Label'])


# In[ ]:


dataset.head()


# In[ ]:


dataset.describe()

