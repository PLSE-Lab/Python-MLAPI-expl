#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''

Context
Computer Network Traffic Data - A ~500K CSV with summary of some real network traffic data from the past. The dataset has ~21K rows and covers 10 local workstation IPs over a three month period. Half of these local IPs were compromised at some point during this period and became members of various botnets.

Content
Each row consists of four columns:

date: yyyy-mm-dd (from 2006-07-01 through 2006-09-30)
l_ipn: local IP (coded as an integer from 0-9)
r_asn: remote ASN (an integer which identifies the remote ISP)
f: flows (count of connnections for that day)
Reports of "odd" activity or suspicions about a machine's behavior triggered investigations on the following days (although the machine might have been compromised earlier)

Date : IP 08-24 : 1 09-04 : 5 09-18 : 4 09-26 : 3 6


'''


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


import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore


# In[ ]:


df = pd.read_csv('../input/cs448b_ipasn.csv')


# In[ ]:


df.head()


# In[ ]:


dfOrig = df.copy()
#df = dfOrig.copy()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:





# In[ ]:





# In[ ]:


#vivewing given examples 


# 

# 

# In[ ]:





# In[ ]:


df[(df.l_ipn == 1) & (df.date == '2006-08-24')]


# In[ ]:


df[(df.l_ipn == 5) & (df.date == '2006-09-04')]

 


# In[ ]:


df[(df.l_ipn == 4) & (df.date == '2006-09-18')]


# In[ ]:


df[(df.l_ipn == 3) & (df.date == '2006-09-26')]


# In[ ]:





# #EDA - Data Analysis

# In[ ]:


#removing f == 1
#df = df[df.f > 1]


# In[ ]:


#len(df)/len(dfOrig)


# In[ ]:


df.l_ipn.value_counts()


# In[ ]:


# 0 is he most active user, 3 is the least


# In[ ]:


for ip in set(df.l_ipn):
    fNormed = df.loc[(df.l_ipn == ip),'f']
    plt.boxplot(fNormed,len(fNormed) * [0],".")
    plt.title('IP:' + str(ip))
    plt.show()


# In[ ]:


for ip in set(df.l_ipn):
    df[df.l_ipn == ip].f.hist(bins = 100)
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.title(('IP: %d') % ip)
    plt.show()


# In[ ]:





# In[ ]:


# instead use log scale since anomaly detection (have skewness of large values for "normal activity")

for ip in set(df.l_ipn):
    df[df.l_ipn == ip].f.hist(log=True,bins =200)
    plt.title(('IP: %d') % ip)
    plt.show()


# In[ ]:


#normalize flows per IP since different ratios and scales 


# In[ ]:


#sort by IP address
df.sort_values(inplace=True, by=['l_ipn'])


# In[ ]:


from sklearn.preprocessing import robust_scale
# Scale features using statistics that are robust to outliers.
# using robust_scale instead of RobustScaler since want to Standardize a dataset along any axis


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[ ]:





# In[ ]:


#accessing columns

#sample[sample['l_ipn'] == 0].fNorm = 2 # not good. returns a copy of sample.l_ipn

#use iloc or loc instead 
# .loc[criterion,selection]
#use df.iloc[1, df.columns.get_loc('s')] = 'B'    or use df.loc[df.index[1], 's'] = 'B'

#sample[sample.iloc[:,sample.columns.get_loc('l_ipn')] == 0]
#sample[sample.loc[:,'l_ipn'] == 0]


# In[ ]:





# In[ ]:


#normalize traffic for each IP 

#scaler = robust_scale()
scaler = StandardScaler()


# In[ ]:


for ip in set(df.l_ipn):
    df.loc[(df.l_ipn == ip),'fNorm'] = scaler.fit_transform(df.loc[(df.l_ipn == ip),'f'].values.reshape(-1, 1)) # reshaped since it's scaling a single feature only 
    df.loc[(df.l_ipn == ip),'fMean'] = scaler.mean_
    df.loc[(df.l_ipn == ip),'fVar'] = scaler.var_


# In[ ]:


for ip in set(df.l_ipn):
    fNormed = df.loc[(df.l_ipn == ip),'fNorm']
    plt.plot(fNormed,len(fNormed) * [0],".")
    fMean = df.loc[(df.l_ipn == ip),'fMean'].iloc[0]# only need the first value as they are all the same in this column for this ip
    plt.plot(fMean,0,'ro')
    plt.title('IP:' + str(ip))
    plt.show()


# In[ ]:


# it is clear that there are anomalies in the amount of traffic flow 


# In[ ]:


# todo: trying a scaler for skewed data


# In[ ]:





# In[ ]:


#analyzing ASN


# In[ ]:


#is asn unique per user?


# In[ ]:


listOfAsnsPerUser = [[]] * len(df.l_ipn)


# In[ ]:


numAsnsPerIp = 0
for ip in set(df.l_ipn):
    numAsnsPerIp += len(set(df.loc[(df.l_ipn == ip),'r_asn']))


# In[ ]:


numAsnsPerIp


# In[ ]:


len(set(df.loc[:,'r_asn']))


# In[ ]:


#number of  unique of asns per ip != number of unqiue asns for total dataset
#therefore, asns are not unique per IP


# In[ ]:


#using asns as categorical variable 


# In[ ]:


dfDummy = df.copy()


# In[ ]:


dfDummy = pd.get_dummies(df,columns=['r_asn'],drop_first=True)


# In[ ]:


dfDummy.head()


# In[ ]:


#takes too long 
# dfDummy.drop(labels =['date','fNorm','fMean','fVar','l_ipn'],axis=1).corr() 


# In[ ]:


dfCorrAsnFlow = dfDummy.drop(labels =['date','fNorm','fMean','fVar','l_ipn'],axis=1)


# In[ ]:





# In[ ]:


# todo: look for anomalies in users using suddently a different ASN and have a high traffic flow 


# In[ ]:





# In[ ]:


# time sampling


# In[ ]:


df.head()


# In[ ]:


df.date = pd.to_datetime(df.date,errors='coerce')


# In[ ]:


len(df.date) == len(df.date.dropna()) 


# In[ ]:


# all dates were valid 


# In[ ]:





# In[ ]:


df.info()


# In[ ]:


#df.date.hist(bins = 100)


# In[ ]:


#fig = plt.figure(figsize = (15,20))
#ax = fig.gca()
#df.date.hist(bins = 50, ax = ax)


# In[ ]:


df = df.sort_values('date', ascending=True)
plt.figure(figsize=(15,20))
plt.plot(df['date'], df['f'])
plt.xticks(rotation='vertical')


# In[ ]:


# now per ip 


# In[ ]:


for ip in set(df.l_ipn):
    plt.figure(figsize=(10,15))
    plt.xticks(rotation='vertical')
    plt.title(('IP: %d') % ip)
    plt.plot(df[df.l_ipn == ip]['date'], df[df.l_ipn == ip]['f'])
    plt.show()


# In[ ]:





# In[ ]:


#using IP = 4 as poc 


# In[ ]:


dataset = df.copy()


# In[ ]:


dataset = pd.get_dummies(dataset,columns=['r_asn'],drop_first=True)


# In[ ]:


dataset.head()


# In[ ]:


dataset = dataset.drop(labels =['date','fNorm','fMean','fVar'],axis=1)


# In[ ]:


dataset.head()


# In[ ]:





# In[ ]:





# # classification models

# In[ ]:


#using IP == 4 as POC


# In[ ]:


dataset = dataset[dataset.l_ipn == 4].drop(['l_ipn'],axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:





# In[ ]:


dataset.loc[:,'f'] = scaler.fit_transform(dataset.f.values.reshape(-1, 1))


# In[ ]:


dataset


# In[ ]:





# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(dataset)


# In[ ]:


kmeans.labels_


# In[ ]:





# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(n_components=2)


# In[ ]:


centroids = kmeans.cluster_centers_


# In[ ]:


centroids


# In[ ]:


centroids2d = pd.DataFrame(pca.fit_transform(centroids))


# In[ ]:


centroids2d


# In[ ]:


xPca = centroids2d.loc[:,0]
yPca = centroids2d.loc[:,1]


# In[ ]:


xPca


# In[ ]:


yPca


# In[ ]:


plt.scatter(xPca,yPca)


# In[ ]:





# In[ ]:


# will create a single dataset without any anomalies for IP = 4 


# In[ ]:


plt.figure(figsize=(10,15))
plt.xticks(rotation='vertical')
plt.yscale('log')
plt.title(('IP: %d') % 4)
plt.plot(range(len(dataset)), dataset['f'])
plt.show()


# In[ ]:


len(dataset[(dataset.f < 10**-.5)])


# In[ ]:


len(dataset)


# In[ ]:


#seems as those two days were anomalious


# In[ ]:


negativeClass = dataset[(dataset.f >= 10**-.5)]


# In[ ]:


negativeClass


# In[ ]:


positiveClass = dataset.head(2)


# In[ ]:


positiveClass


# In[ ]:


#removing test classes from dataset
dataset = dataset.drop(dataset.index[[0,1]])


# In[ ]:


len(dataset)


# In[ ]:


dataset = dataset[(dataset.f < 10**-.5)]


# In[ ]:


len(dataset)


# In[ ]:


kmeans.predict(positiveClass)


# In[ ]:


kmeans.predict(negativeClass)


# In[ ]:


posRes = kmeans.transform(positiveClass)


# In[ ]:


negRes = kmeans.transform(negativeClass)


# In[ ]:


negRes[0]


# In[ ]:


from numpy import linalg


# In[ ]:


centroids


# In[ ]:


dist = numpy.linalg.norm(a-b)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# todo checkout auto encoders


# In[ ]:


#todo checkout one class SVM


# In[ ]:





# In[ ]:


#create classifier and split data into train test and validation 


# In[ ]:




