#!/usr/bin/env python
# coding: utf-8

# How does one exactly detect credit card fraud? Let's examine a couple of ways in which we can. Two methods I can think of immediately are:
# 
# * Change point detection (i.e. looking at statistically significant changes in credit card expenditures)
# * Supervised learning, creating a classifier (logistic regression, decision trees, etc) to predict whether or not a transaction will be fraudulent. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Let's load in and get to know the data a bit. 

# In[ ]:


# load in data
credit = pd.read_csv('../input/creditcard.csv')

# useful printouts
print (credit.head(10))
print ("============================")
print (credit.dtypes)


# The thing most people would be most immediately interested in are the amounts. Let's look at cumulative transaction amounts across time. 

# In[ ]:


# split data set into fraudulent and non-fraudulent sets 
fraud = credit[credit['Class'] == 1]
good = credit[credit['Class'] == 0]


# In[ ]:


plt.figure()

fig, ([ax1, ax2, ax3]) = plt.subplots(3, 1, sharex=True)

ax1.plot(fraud['Time'], np.cumsum(fraud['Amount']))
ax1.set_title('Cumulative Fraudulent Transaction Amount')
ax1.set_xlabel('Time')

ax2.plot(good['Time'], np.cumsum(good['Amount']))
ax2.set_title('Cumulative Non-Fraudulent Transaction Amount')

ax3.plot(credit['Time'], np.cumsum(credit['Amount']))
ax3.set_title('Cumulative Transaction Amount')

plt.tight_layout()


# They have the same basic trend, but the jumps in the graph of the fraudulent transactions are more profound (only due to the fact there are only 490 such observations). The non-fraudulent transactions as well as transactions for the entire set look about the same. 

# In[ ]:


# grab ratio of fraudulent transactions to good transactions
percentage = len(fraud)/float(len(good))
# undersample by this percentage, set seed to 1 for reproducibility
np.random.seed(1)
# subsample of good transactions will be some percentage of all the good transactions
good_trans = good.take(np.random.permutation(len(good))[:round(percentage*len(good))])
# fraudulent transactions are fraudulent
fraud_trans = fraud
# combine into a new frame, resetting the index
cred_data = pd.concat([good_trans, fraud_trans], ignore_index= True)


# In[ ]:


target = 'Class'
features = cred_data.columns[1:30]


# In[ ]:


from sklearn.model_selection import train_test_split

train_and_val, test = train_test_split(cred_data, test_size = 0.1)
train, val = train_test_split(train_and_val, test_size = 0.1)


# # Train a logistic regression model

# In[ ]:


from sklearn.linear_model import LogisticRegression

logit = LogisticRegression()
logit.fit(train[features], train[target]);


# In[ ]:


# import tools for evaluating performance of classifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


# In[ ]:


prediction = logit.predict(test[features])
actual = test[target]
# accuracy 
acc = accuracy_score(actual, prediction)
# precision
prec = precision_score(actual, prediction)
# recall
rec = recall_score(actual, prediction)
# F1 score
f1 = f1_score(actual, prediction)

print ("The accuracy is: %0.2f." %acc)
print ("The precision is: %0.2f." %prec)
print ("The recall is: %0.2f." %rec)
print ("The F1 score is: %0.2f." %f1)


# In[ ]:


def plot_f1_scores(train_data, validation_data, features, target, reg_params):
    f1_scores = []
    for c in reg_params:
        logit = LogisticRegression(C = c, penalty = 'l2')
        logit.fit(train_data[features], train_data[target])
        predicts = logit.predict(validation_data[features])
        f1 = f1_score(validation_data[target], predicts)
        f1_scores.append(f1)
        
    plt.plot(reg_params, f1_scores)
    plt.xlabel('Regularization Parameter')
    plt.ylabel('F1 Score')
    plt.tight_layout()


# In[ ]:


plot_f1_scores(train, val, features, target, np.arange(1,11))


# In[ ]:




