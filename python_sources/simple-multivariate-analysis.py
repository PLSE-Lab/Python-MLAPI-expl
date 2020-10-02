#!/usr/bin/env python
# coding: utf-8

# Multivariate Analysis - An Unsupervised Approach

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Importing the libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
plt.style.use('seaborn-whitegrid')


# In[ ]:


# Read in the csv file
df = pd.read_csv('../input/creditcard.csv')


# In[ ]:


# Sort the Values by Class. 
# Results in all Negative (No Fraud Samples) up top and the Positive (Fraud Samples) after it
# Reset the index

df.sort_values(by='Class',inplace=True)
df.reset_index(inplace=True)



# In[ ]:


# Store the indices to build train, validation and test sets
negative_indices = df[df['Class'] == 0].index.tolist()
positive_indices = df[df['Class'] == 1].index.tolist()

print(min(negative_indices),max(negative_indices))
print(min(positive_indices),max(positive_indices))


# In[ ]:


# Choosing only negative samples for training Data Set,
# Choosing 284000 out of 284315 rows. Will use the remaining rows for validation and test sets
training_indices = np.random.choice(negative_indices,284000,replace=False)
remaining_negative_indices = [x for x in negative_indices if x not in training_indices]

# Choose 100 out of the 315 remaining negative rows for validation set
val_negative_indices = np.random.choice(remaining_negative_indices,100,replace=False).tolist()
# Choose 100 out of the 491 positive samples to be a part of the validation set
val_positive_indices = np.random.choice(positive_indices,100,replace=False).tolist()

# Use the remaining to build a Test Set
test_negative_indices = [x for x in remaining_negative_indices if x not in val_negative_indices]
test_positive_indices = [x for x in positive_indices if x not in val_positive_indices]


# In[ ]:


# Build the list of interesting columns to use as features
collist = df.columns.tolist()
collist.remove('index')
collist.remove('Class')
collist.remove('Time')
print(collist)


# In[ ]:


# Build the Data Sets - Training, Validation and Testing 

train_df = df.ix[training_indices,:]
val_df = df.ix[val_negative_indices+val_positive_indices,:]
test_df = df.ix[test_negative_indices+test_positive_indices,:]

print(len(train_df),len(val_df),len(test_df))


# In[ ]:


# Standardizing our DataSets

from sklearn.preprocessing import  StandardScaler

scaler  = StandardScaler().fit(train_df[collist])

X_train_std = scaler.transform(train_df[collist])
X_val_std = scaler.transform(val_df[collist])
X_test_std = scaler.transform(test_df[collist])


# In[ ]:


# Compute the Per Column Mean and the Covariance Matrix

sigma = np.cov(X_train_std.T)
mean = np.mean(X_train_std,axis=0)


# In[ ]:


# Compute the Probability of each validation sample using the multivariate PDF
from scipy.stats import multivariate_normal

val_pred_probs = multivariate_normal.pdf(X_val_std,mean=mean,cov=sigma)


# In[ ]:


# Look at the Average Proabability and Std in probability of the negative and the positive samples
print("Negative Samples", val_pred_probs[:100].mean(),val_pred_probs[:100].std())
print("Positive Samples",val_pred_probs[100:].mean(),val_pred_probs[100:].std())


# In[ ]:


# Lets plot and see how the probabilties for the Negative and the Positive Samples are distributes

plt.hist(val_pred_probs[:100],alpha=0.4);
plt.hist(val_pred_probs[100:],alpha=0.7);

# Looks like the Positive Samples are all within a narrow range of Probability


# In[ ]:


# if we set an episilon of 1e-16 then all samples with probability less than this epsilon are fraudalent (hypothesis)
preds = (val_pred_probs < 1e-16).astype('int')
labels = val_df.Class.values
print(confusion_matrix(labels,preds))
print(classification_report(labels,preds))


# In[ ]:


# Lets try the same with our Test Set 

test_pred_probs = multivariate_normal.pdf(X_test_std,mean=mean,cov=sigma)
preds = (test_pred_probs < 1e-16).astype('int')
labels = test_df.Class.values

print(confusion_matrix(labels,preds))
print(classification_report(labels,preds))


# In[ ]:




