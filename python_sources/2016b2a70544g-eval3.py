#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift,Birch,SpectralClustering,FeatureAgglomeration
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')


# In[ ]:


dataset.head()


# In[ ]:


len(dataset)


# In[ ]:


#dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'],errors='coerce')


# In[ ]:


temp = dataset.keys()
co = dataset.copy()
for x in temp:
    if (dataset[x].dtypes=='object'):
        co = pd.concat([co,pd.get_dummies(co[x], prefix=x)],axis=1)
        co.drop([x],axis=1, inplace=True)


# In[ ]:


co.columns


# In[ ]:


#data = co[['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','Satisfied','gender_Male','Married_Yes','Children_Yes','TVConnection_Cable','TVConnection_DTH','TVConnection_No','Channel6_Yes','Channel1_Yes','Channel2_Yes','Channel3_Yes','Channel4_Yes','Channel5_Yes','Internet_Yes','HighSpeed_Yes','AddedServices_Yes','Subscription_Annually','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Bank transfer','PaymentMethod_Cash','PaymentMethod_Credit card','PaymentMethod_Net Banking']].copy()
#data = co[['Satisfied','gender_Male','Married_Yes','Children_Yes','Channel1_Yes','Channel2_Yes','Channel4_Yes','Channel6_Yes','AddedServices_Yes','Subscription_Annually','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Bank transfer','PaymentMethod_Cash','PaymentMethod_Credit card','PaymentMethod_Net Banking']].copy()
#data = co[['Satisfied','gender_Male','Married_Yes','Children_Yes','Channel6_Yes','Channel1_Yes','Channel2_Yes','Channel3_Yes','Channel4_Yes','Channel5_Yes','Internet_Yes','HighSpeed_Yes','AddedServices_Yes','Subscription_Annually','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Bank transfer','PaymentMethod_Cash','PaymentMethod_Credit card','PaymentMethod_Net Banking']].copy()
#data = co[['SeniorCitizen','gender_Male','Married_Yes','Children_Yes','TVConnection_No','Channel6_Yes','Channel1_Yes','Channel2_Yes','Channel3_Yes','Channel4_Yes','Channel5_Yes','Internet_Yes','HighSpeed_Yes','AddedServices_Yes','Subscription_Annually','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Bank transfer','PaymentMethod_Cash','PaymentMethod_Credit card','PaymentMethod_Net Banking','Satisfied']].copy()
data = co[['SeniorCitizen', 'Satisfied','gender_Male',  'Married_Yes',  'Children_Yes', 'TVConnection_Cable','TVConnection_DTH',  'Channel1_Yes','Channel2_Yes','Channel3_Yes','Channel4_Yes','Channel5_Yes','Channel6_Yes','HighSpeed_Yes','AddedServices_Yes','Subscription_Monthly', 'PaymentMethod_Bank transfer', 'PaymentMethod_Cash', 'PaymentMethod_Credit card',  'PaymentMethod_Net Banking']]


# In[ ]:


def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


# In[ ]:


dataset=pd.get_dummies(dataset, prefix=None, prefix_sep='_', dummy_na=False, columns=['gender','SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod'], sparse=False, drop_first=False, dtype=None)
dataset['TotalCharges']=dataset['TotalCharges'].apply(lambda x:x.split(' ')[0]).replace(to_replace=[''], value=0).apply(lambda x:float(x))
features=['TVConnection_No', 'Channel1_Yes','Channel3_Yes','Channel4_Yes', 'Channel6_Yes', 'Internet_Yes', 'HighSpeed_Yes', 'Subscription_Annually', 'Subscription_Biannually',  'Subscription_Monthly', 'PaymentMethod_Bank transfer', 'PaymentMethod_Cash', 'PaymentMethod_Credit card', 'PaymentMethod_Net Banking']


# In[ ]:


"""parameters = {'threshold' : [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
             'branching_factor' : [10,20,30,40,50,60,70,80,90],
             'n_clusters':[2]}
clf = GridSearchCV(Birch(compute_labels = True, copy = True ), parameters, cv=5, n_jobs = 4, verbose = 2, scoring = make_scorer(auc) )"""

X=dataset[features]
y=dataset['Satisfied']
br=Birch(threshold=0.5, branching_factor=50, n_clusters=2, compute_labels=True, copy=True)
#br = Birch(branching_factor=40, compute_labels=True, copy=True, n_clusters=2,threshold=0.4)

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=13)
br.fit(X,y)
print(roc_auc_score(y_val,br.predict(X_val)))


# In[ ]:


"""clf.fit(X,y)
print(clf.best_estimator_)
print(clf.best_score_)"""


# In[ ]:


"""
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=500,
      n_clusters=4, n_init=10, n_jobs=-1, precompute_distances=True,
      random_state=42, tol=0.0001, verbose=0)
0.5516474874627272
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=100,
      n_clusters=4, n_init=10, n_jobs=-1, precompute_distances=True,
      random_state=42, tol=0.0001, verbose=0)
0.5516474874627272

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=50,
      n_clusters=12, n_init=10, n_jobs=-1, precompute_distances=True,
      random_state=42, tol=0.0001, verbose=0)
0.5298465015088709

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=10,
      n_clusters=2, n_init=15, n_jobs=-1, precompute_distances=True,
      random_state=42, tol=0.01, verbose=0)
0.5603438391073038

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=10,
      n_clusters=2, n_init=15, n_jobs=-1, precompute_distances=True,
      random_state=42, tol=0.01, verbose=0)
0.5603438391073038

MeanShift(bandwidth=None, bin_seeding=False, cluster_all=True, min_bin_freq=1,
         n_jobs=None, seeds=None)
0.5
Birch(branching_factor=70, compute_labels=True, copy=True, n_clusters=5,
     threshold=0.8)
0.5157966535173778

Birch(branching_factor=30, compute_labels=True, copy=True, n_clusters=6,
     threshold=0.8)
0.6062101760142043

Birch(branching_factor=30, compute_labels=True, copy=True, n_clusters=6,
     threshold=0.8)
0.6062101760142043

Birch(branching_factor=30, compute_labels=True, copy=True, n_clusters=6,
     threshold=0.8)
0.6874305912172263
Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=7,
     threshold=0.9)
0.7662240423552872




None
Birch(branching_factor=90, compute_labels=True, copy=True, n_clusters=2,
     threshold=0.8)
0.6281094529837322

senior
Birch(branching_factor=20, compute_labels=True, copy=True, n_clusters=2,
     threshold=0) 
0.6374092968535263

married
Birch(branching_factor=20, compute_labels=True, copy=True, n_clusters=2,
     threshold=2.0)
0.5814676846002385

tv connection cable
Birch(branching_factor=40, compute_labels=True, copy=True, n_clusters=2,
     threshold=0)
0.6369487587736951

tv connection dth
Birch(branching_factor=10, compute_labels=True, copy=True, n_clusters=2,
     threshold=2.0)
0.6385703721458587



'TVConnection_No',
Birch(branching_factor=70, compute_labels=True, copy=True, n_clusters=2,
     threshold=0.4)
0.6315781078588159

'Channel1_Yes'

Birch(branching_factor=10, compute_labels=True, copy=True, n_clusters=2,
     threshold=3)
0.5


'Channel2_Yes'
Birch(branching_factor=20, compute_labels=True, copy=True, n_clusters=2,
     threshold=2.0)
0.5758264094723197

'Channel3_Yes',
Birch(branching_factor=10, compute_labels=True, copy=True, n_clusters=2,
     threshold=2.0)
0.6132015278094272

'Channel4_Yes',
Birch(branching_factor=10, compute_labels=True, copy=True, n_clusters=2,
     threshold=3)
0.5

'Channel5_Yes',
Birch(branching_factor=10, compute_labels=True, copy=True, n_clusters=2,
     threshold=2.0)
0.6184788523531571

'Channel6_Yes',
Birch(branching_factor=70, compute_labels=True, copy=True, n_clusters=2,
     threshold=0)
0.5667487382887971

'Internet_Yes',
Birch(branching_factor=30, compute_labels=True, copy=True, n_clusters=2,
     threshold=0.4)
0.6356839924673252


'HighSpeed_Yes'
Birch(branching_factor=10, compute_labels=True, copy=True, n_clusters=2,
     threshold=2.0)
0.6270778320017213


'Subscription_Annually',
Birch(branching_factor=10, compute_labels=True, copy=True, n_clusters=2,
     threshold=2.0)
0.5600747940971453

'Subscription_Biannually',
Birch(branching_factor=10, compute_labels=True, copy=True, n_clusters=2,
     threshold=3)
0.5


'Subscription_Monthly',
Birch(branching_factor=10, compute_labels=True, copy=True, n_clusters=2,
     threshold=1.0)
0.5972953251423624


'PaymentMethod_Bank transfer',
Birch(branching_factor=30, compute_labels=True, copy=True, n_clusters=2,
     threshold=0)
0.6346941766618829

'PaymentMethod_Cash',
Birch(branching_factor=20, compute_labels=True, copy=True, n_clusters=2,
     threshold=0.4)
0.6361412911283537


'PaymentMethod_Credit card',
Birch(branching_factor=40, compute_labels=True, copy=True, n_clusters=2,
     threshold=0.4)
0.6382168286406807

,'PaymentMethod_Net Banking'
Birch(branching_factor=10, compute_labels=True, copy=True, n_clusters=2,
     threshold=0)
0.6351061276976602
"""


# In[ ]:


test = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')


# In[ ]:


test_data=pd.get_dummies(test, prefix=None, prefix_sep='_', dummy_na=False, columns=['gender','SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod'], sparse=False, drop_first=False, dtype=None)
test_data['TotalCharges']=test_data['TotalCharges'].apply(lambda x:x.split(' ')[0]).replace(to_replace=[''], value=0).apply(lambda x:float(x))
X_test=test_data[features]
#X_test=preprocessing.scale(X_test)


# In[ ]:


predicted=br.fit_predict(X_test)
predicted=predicted-1
predicted=predicted*-1


# In[ ]:


test_data['Satisfied']=np.array(predicted)
ans=test_data[['custId','Satisfied']]
ans=ans.astype(int)
ans.to_csv('output.csv',index=False)


# In[ ]:




