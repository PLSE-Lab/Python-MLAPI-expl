#!/usr/bin/env python
# coding: utf-8

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

## we will import the libraries as and when required.


# In[ ]:


import os
print("Current Directory:%s"%os.getcwd())
print("List Directories")
os.listdir('../input/kddcup99/kddcup.data')


# In[ ]:


##  print first 5 lines of file  using python
for each_index,each_line in enumerate(open('../input/kddcup99/kddcup.data/kddcup.data')):
    if each_index < 5:
        print(each_line.strip())


# In[ ]:


## directly using the unix command line .
get_ipython().system("head -5 '../input/kddcup99/kddcup.data/kddcup.data'")


# In[ ]:


## got the list of column from KDD website. Its not provided here

columns_list =["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


# In[ ]:


kdd_raw =pd.read_csv('../input/kddcup99/kddcup.data/kddcup.data',names=columns_list,header=None)


# In[ ]:


kdd_raw.shape


# In[ ]:


kdd_raw.head(3)


# In[ ]:


kdd_raw.service.value_counts()


# In[ ]:


##  we will work only on http request.
kdd_http =kdd_raw[kdd_raw["service"]=='http']


# In[ ]:


kdd_http.service.value_counts()


# In[ ]:


kdd_http.label.value_counts()
## other than normal we have some anomolies. which we need to build a model to idenitify non-normal ones.


# In[ ]:


kdd_http.info()
## we have few object data types. also since we want to go with tree models than the distance models. we are fine with only label encoding. Dont need any one hot encoding.
##  tree models work fine with categorical variables , also no need of normalization.


# In[ ]:


## using select_dtypes we were able to get the list of non-numeric columns which might be cateogorical/
kdd_http.select_dtypes(exclude=np.number).columns


# In[ ]:


## so there are only fixed number of columns, lets do label encoding.
for each_col in list(kdd_http.select_dtypes(exclude=np.number)):
    print(each_col)
    print('*'*20)
    print(kdd_http[each_col].value_counts())


# In[ ]:


from sklearn.preprocessing  import LabelEncoder


# In[ ]:


for each_col in list(kdd_http.select_dtypes(exclude=np.number)):
    label_encoder = LabelEncoder()
    label_encoder.fit(kdd_http[each_col])
    kdd_http[each_col]=label_encoder.transform(kdd_http[each_col])


# In[ ]:


kdd_http.head(5)


# In[ ]:


## We dont have any non-numeric columns
kdd_http.select_dtypes(exclude=np.number).columns


# In[ ]:


### lets shuffle our Data frame before we do train,test and validation split.
## we will use np.random.permutation and iloc combination for the same.
## we will shuffle for 3 times.

for each_shuffle in range(3):
    kdd_http=kdd_http.iloc[np.random.permutation(len(kdd_http))]


# In[ ]:


kdd_http.head(5)


# In[ ]:


kdd_http.shape


# ## We will do a Train, Test and validation with 60,20,20 using np.split

# In[ ]:


kdd_http.head(2)


# In[ ]:


train,test,validation=np.split(kdd_http.sample(frac=1),[int(len(kdd_http)*0.6),int(len(kdd_http)*0.8)])


# In[ ]:


print("shape of Train:%s"%str(train.shape))
train.head(2)


# In[ ]:


print("shape of Test:%s"%str(test.shape))
test.head(2)


# In[ ]:


print("shape of validation:%s"%str(validation.shape))
validation.head(2)


# In[ ]:


#### Doing a  X and Y split in each of the Train , test and validation
X_train=train.loc[:,train.columns!='label']
y_train=train.loc[:,train.columns=='label']
X_test=test.loc[:,test.columns!='label']
y_test=test.loc[:,test.columns=='label']
X_validation=validation.loc[:,validation.columns!='label']
y_validation=validation.loc[:,validation.columns=='label']


# In[ ]:


print("X and Y Shape for Train is  %s and %s"%(str(X_train.shape),str(y_train.shape)))
print("X and Y Shape for test is  %s and %s"%(str(X_test.shape),str(y_test.shape)))
print("X and Y Shape for validation is  %s and %s"%(str(X_validation.shape),str(y_validation.shape)))


# ## Model Building using isolation Forest

# In[ ]:


from sklearn.ensemble  import IsolationForest
isolation_forest=IsolationForest(n_estimators=100,max_samples=256,contamination=0.1,random_state=123)


# In[ ]:


isolation_forest.fit(X_train)


# In[ ]:


## decission function gives the Average Anomoly score .lets calculate for the x_validation .

anomaly_scores=isolation_forest.decision_function(X_validation)


# In[ ]:


## lts plot the Anomoly score . so we can see anomolies.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=[20,20])
plt.hist(anomaly_scores,bins=100)
plt.xlabel("Average Path Lengths")
plt.ylabel("Number of Data Points")

## we can see Anomolies at Average Path length below  <  -0.2 . since there might be outliers too, we will take less than -0.19


# ###  Model evaluation  - using ROC curver - F1 score curve

# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


label_encoder.classes_


# In[ ]:


list(label_encoder.classes_).index('normal.')


# In[ ]:


anomalies=anomaly_scores > -0.19
matches=y_validation==4
auc=roc_auc_score(anomalies,matches)
print(auc)
print("AUC : {:.2%}".format(auc))


# In[ ]:


## good score on validation. Lets test that on testdataset


# In[ ]:


anomaly_scores_test=isolation_forest.decision_function(X_test)


# In[ ]:


plt.figure(figsize=[20,20])
plt.hist(anomaly_scores_test,bins=100)
plt.xlabel("Average Path Lengths")
plt.ylabel("Number of Data Points")


# In[ ]:


test_anomalies=anomaly_scores_test > -0.19
matches_test=y_test==4
auc_test=roc_auc_score(test_anomalies,matches_test)
print(auc_test)
print("AUC of Test : {:.2%}".format(auc_test))


# In[ ]:


### Overall The Model is performing good on both validation and test data.


# In[ ]:




