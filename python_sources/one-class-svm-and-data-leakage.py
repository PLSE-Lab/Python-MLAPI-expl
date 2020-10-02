#!/usr/bin/env python
# coding: utf-8

# I will discuss the following:
# 1. When One Class SVM(OC-SVM) is useful, 
# 1. A quick implementation of the model, and 
# 1. Investigation into whether there is any data leakage when random sampling is done.

# ### When to use OC-SVM

# OC-SVM, or one class learning in general, is suitable when your **dataset is imbalanced**, because you only need the majority class to train your data, and you don't need to worry how the outliers are distributed/modelled to create a useful decision boundary. This is useful when it is hard to create outliers. In many practical scenarios, it is hard and expensive to create anomalies [1], certainly the case for fraud detection which happens rarely. Also, if you have a new type of fraud that has never been encountered before, it would be easy for OC-SVM to detect it, whereas a classfication model may struggle to do so [1].

# ### Quick implementation of OC-SVM

# In[ ]:


import pandas as pd # for data analytics
import numpy as np # for numerical computation
from sklearn.metrics import precision_recall_fscore_support, classification_report,confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn import utils  
from sklearn import svm
from sklearn.model_selection import train_test_split
from fastai.structured import *
from fastai.column_data import *
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df = pd.read_csv('../input/creditcard.csv')
df.loc[df['Class'] == 1, "Class"] = -1
df.loc[df['Class'] == 0, "Class"] = 1


# In[ ]:


#getting random set of nonfraud data to train on
non_fraud = df[df['Class']==1]
df_train, val = train_test_split(non_fraud, test_size=0.20, random_state=42)
fraud = df[df['Class']==-1]


# #### Here I use fastai to work with my dataset. It's the easiest way for me to do some basic processing on structured data.

# In[ ]:


#fastai
df, _, nas, mapper = proc_df(df_train, 'Class', do_scale=True)
df_val, _, nas, mapper = proc_df(val, 'Class', mapper=mapper, na_dict=nas, do_scale=True)
df_fraud, _, nas, mapper = proc_df(fraud, 'Class', mapper=mapper, na_dict=nas, do_scale=True)


# In[ ]:


model = svm.OneClassSVM(kernel='rbf', nu=0.0005,gamma=0.007)
model.fit(df)


# In[ ]:


#Creating a test set that contains both fraud and non fraud
y_val = val['Class']
y_fraud = fraud['Class']
y_testval = pd.concat([y_val, y_fraud])
y_testval = np.array(y_testval)
df_testval = pd.concat([df_val, df_fraud])


# In[ ]:


#predicting on test set, which consists of both fraud and non-fraud
pred_testval = model.predict(df_testval)


# In[ ]:


print(classification_report(y_testval, pred_testval))


# In[ ]:


prec, rec, f2, _ = precision_recall_fscore_support(y_testval, pred_testval, beta=2, 
                                                   pos_label=-1, average='binary')
print(f'precision is {prec}, recall is {rec} and F2 score is {f2}')


# In[ ]:


roc = roc_auc_score(y_testval, pred_testval)
print(f'ROC score is {roc}')


# ### Exploring if data is leaked
# Since we are dealing with a time series, there is the concern that the results will not hold in production. A likely culprit would be that the model is benefiting from looking back and forward in time to create its decision boundary [2]. By doing random split using sklearn, the model may have learnt things that it will not be able to use when the model goes live.
# 
# To make sure we do not commit that crime and to see if there is indeed 'data leakage', we replicate the steps above, only difference being we train the model using first X rows, and test on the subsequent rows, so the model will not benefit from crystal balling into the future.
# 
# One key thing to note is that for OC-SVM, the training set should ideally be all non-fraud cases, which was done earlier. However to allow proper split between training and testing sets, we make a compromise and account for the 'training_error' in the modelling.

# In[ ]:


df = pd.read_csv('../input/creditcard.csv')
df.loc[df['Class'] == 1, "Class"] = -1
df.loc[df['Class'] == 0, "Class"] = 1


# Where I differ from above section: Instead of a random split, I split by time

# In[ ]:


#227845 because that is 80% of dataset, just like training amount earlier
df_train = df.iloc[:227845,:]
val = df.iloc[227845:,:]
print(df_train.shape, val.shape)


# In[ ]:


#fastai
df, _, nas, mapper = proc_df(df_train, 'Class', do_scale=True)
df_val, _, _, _ = proc_df(val, 'Class', mapper=mapper, na_dict=nas, do_scale=True)


# To set the training error, I check how many outliers are there in my train set. It is still very small(0.1%), so it should not affect the model adversely. In sklearn, nu represents the expected training error.

# In[ ]:


nu = df_train[df_train['Class']==-1].shape[0]/df_train.shape[0]
nu


# In[ ]:


model = svm.OneClassSVM(kernel='rbf', nu=nu,gamma=0.007)
model.fit(df)


# In[ ]:


y_val = val['Class']
pred_val = model.predict(df_val)


# In[ ]:


print(classification_report(y_val, pred_val))


# In[ ]:


prec, rec, f2, _ = precision_recall_fscore_support(y_val, pred_val, beta=2, 
                                                   pos_label=-1, average='binary')
print(f'precision is {prec}, recall is {rec} and F2 score is {f2}')


# In[ ]:


roc = roc_auc_score(y_val, pred_val)
print(f'ROC score is {roc}')


# So we can see that all the metrics have gone down, which is very likely due to data leakage.

# **References:**
# 
# [1] A Survey of Outlier Detection Methodologies. http://eprints.whiterose.ac.uk/767/1/hodgevj4.pdf 
# 
# [2] How and why to create a good valdiation set. https://www.fast.ai/2017/11/13/validation-sets/
