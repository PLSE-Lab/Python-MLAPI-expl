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


ls ../input


# In[ ]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
import seaborn as sns


# In[ ]:


cat ../input/creditcard.csv | head -n 2


# In[ ]:


data = pd.read_csv('../input/creditcard.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe().Amount


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


#eda


# In[ ]:


#check for any missing values
data.isnull().values.any()


# In[ ]:


data.hist(figsize=(20,20))
plt.show()


# In[ ]:


# analyze the two classses


# In[ ]:


classDist = data['Class'].value_counts()


# In[ ]:





# In[ ]:


plt.title("Class Distribution")
classDist.plot(kind = 'bar',log=True, rot=0)
plt.xlabel("Class")
plt.ylabel("Frequency")
print("Fraud Class: " + str(classDist[1]/(classDist[0]+classDist[1])) + str('%'))


# In[ ]:


fraud = data[data['Class']==1]
normal = data[data['Class']==0]


# In[ ]:


fraud.shape


# In[ ]:


normal.shape


# In[ ]:


fraud.describe()


# In[ ]:


normal.describe()


# In[ ]:





# In[ ]:


#How do the amounts compare in the two classes?


# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount')
plt.ylabel('Num of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[ ]:


#using same x axis scale for both classes' log scaled histograms

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
f.suptitle('Amount per transaction')
bins=50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount')
plt.ylabel('Num of Transactions')
plt.xlim((0, 25000))
plt.yscale('log')
plt.show();


# In[ ]:


fraud.Amount.hist(log=True)


# In[ ]:


normal.Amount.hist(log=True)


# In[ ]:





# In[ ]:


normal.boxplot(column='Amount')


# In[ ]:


normal.Amount.plot(kind='box', logy =True)


# In[ ]:


fraud.boxplot(column='Amount')


# In[ ]:


fraud.Amount.plot(kind='box', logy =True)


# In[ ]:





# In[ ]:


# Do fraudulent transactions occur more during certain time ?


# In[ ]:


plt.hist(fraud.Time,bins=25);


# In[ ]:


plt.hist(normal.Time,bins=25);


# In[ ]:


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.show()


# In[ ]:





# In[ ]:


# check correlations


# In[ ]:


correlation_matrix = data.corr()

fig = plt.figure(figsize=(12,9))

sns.heatmap(correlation_matrix,square = True)

plt.show()


# In[ ]:


nonPcaFeature = data [['Time','Amount','Class']]


# In[ ]:


correlation_matrix = nonPcaFeature.corr()

fig = plt.figure()

sns.heatmap(correlation_matrix,square = True)

plt.show()


# In[ ]:


# we see there is no correlation between class, amount, and time
# we see there is no correlation between the V1 to V28 PCA components 
# we see there is correlation between class and some of the V components


# In[ ]:





# In[ ]:


# checking correlation with different time segments during the day


# In[ ]:


timeBins =   int(fraud.Time.max()/30000 +1 )
timeBins


# In[ ]:


timeRanges = pd.cut(data.Time,timeBins,labels=range(0,timeBins))


# In[ ]:


timeRanges.value_counts()


# In[ ]:


#dataOrig = data.copy()
#data = dataOrig.copy()


# In[ ]:


len(timeRanges.values)


# In[ ]:


data['TimeRanges']=timeRanges.values


# In[ ]:


data.head()


# In[ ]:


data.TimeRanges.value_counts()


# In[ ]:


data = pd.get_dummies(data,columns=['TimeRanges'],drop_first = True)


# In[ ]:


data.head()


# In[ ]:


correlation_matrix = data[['Amount','Class','TimeRanges_1','TimeRanges_2','TimeRanges_3','TimeRanges_4','TimeRanges_5']].corr()
fig = plt.figure(figsize=(7,7))
sns.heatmap(correlation_matrix,square = True)
plt.show()


# In[ ]:


# not much correlation between the time ranges with the class or amount


# In[ ]:





# In[ ]:


#models 


# In[ ]:


sample = data.sample(frac = 0.1,random_state=42)
sample.shape


# In[ ]:


sampleFraud = sample[sample['Class']==1]
sampleValid = sample[sample['Class']==0]
sampleRatio = len(sampleFraud)/float(len(sampleValid))
print(sampleRatio)


# In[ ]:


#good sampling!


# In[ ]:


y = sample.Class


# In[ ]:


data.head()


# In[ ]:


X = sample.copy().drop(columns=['Class','TimeRanges_1','TimeRanges_2','TimeRanges_3','TimeRanges_4', 'TimeRanges_5'])


# In[ ]:





# In[ ]:


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report,accuracy_score


# In[ ]:


def ClassificationResults(yPred,yActual):
    
    yPred[yPred == 1] = 0
    yPred[yPred == -1] = 1
    nErrors = (yPred != yActual).sum()
    # Run Classification Metrics
    print("Accuracy Score :")
    print(accuracy_score(yActual,yPred))
    print("Classification Report :")
    print(classification_report(yActual,yPred))


# In[ ]:


# LocalOutlierFactor


# In[ ]:


cl_LOF = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination=sampleRatio) #default settings except for contaminatin
y_pred = cl_LOF.fit_predict(X)
scores_prediction = cl_LOF.negative_outlier_factor_
ClassificationResults(y_pred,y)


# In[ ]:





# In[ ]:


# svm


# In[ ]:


clf_SVM = OneClassSVM(kernel='rbf', degree=3, gamma=0.1,nu=0.05)
clf_SVM.fit(X)
y_pred = clf_SVM.predict(X)
ClassificationResults(y_pred,y)


# In[ ]:





# In[ ]:


#islolation forest


# In[ ]:


clf_IF = IsolationForest(n_estimators=100, max_samples=len(X), contamination=sampleRatio, verbose=0)
clf_IF.fit(X)
scores_prediction = clf_IF.decision_function(X)
y_pred = clf_IF.predict(X)
ClassificationResults(y_pred,y)


# In[ ]:


'''
Accuracy:
isloation forest has the highest accuracy score (.99779) compared to SVM (.6655) and LocalOutlierFactor (.9968)

Precision & Recall:
isloation forest has the highest recall rate (.33) compared to SVM (.28 ) and LocalOutlierFactor (.02)
'''


# In[ ]:




