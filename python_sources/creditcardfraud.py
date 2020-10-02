#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


card = pd.read_csv("../input/creditcardfraud/creditcard.csv")
card.columns


# Check for missing values in the columns

# In[ ]:


card.isnull().any()


# Identifying which transactions are fraud with **Fraud = 1** and **Non-Fraud = 0**

# In[ ]:


card['Class'].value_counts(normalize = True)


# There were about .1727% cases of fraud in the credit card transactions

# PCA scatterplot

# In[ ]:


v_cols = card.drop(columns=['Time', 'Amount', 'Class'])
sns.boxplot(data=v_cols, palette="Set3")


# Fraud amount during a specific time

# In[ ]:


fraud = card.loc[card['Class'] == 1]
no_fraud = card.loc[card['Class'] == 0]


# Scatterplot of the fraudulent transaction amount over the 2 day period

# In[ ]:


sns.scatterplot(x="Time", y="Amount", data=fraud)


# In[ ]:


print("The most fraud done in the transaction over the past 2 days was {}".format(fraud.Amount.max()))
print("The average fraud done in the transaction over the past 2 days was {}".format(fraud.Amount.mean()))


# Scatterplot of the non-fraudulent transaction amount over the 2 day period

# In[ ]:


sns.scatterplot(x="Time", y="Amount", data=no_fraud)


# In[ ]:


print("The most non-fraud done in the transaction over the past 2 days was {}".format(no_fraud.Amount.max()))
print("The average non-fraud done in the transaction over the past 2 days was {}".format(no_fraud.Amount.mean()))


# In[ ]:


X_var=card.drop(['Class'], axis=1)
y_var=card["Class"]
print(X_var.shape)
print(y_var.shape)
X=X_var.values
y=y_var.values


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.70, test_size=0.30, random_state=1)
card_model = DecisionTreeClassifier()
card_model.fit(X_train, y_train)
card_preds = card_model.predict(X_test)
print(mean_absolute_error(y_test, card_preds))


# Confusion Matrix

# In[ ]:


con_mat = confusion_matrix(y_test, card_preds)
con_mat


# In[ ]:


tp = con_mat[0][0]
fp = con_mat[0][1]
tn = con_mat[1][1]
fn = con_mat[1][0]
precision = (tp)/(tp+fp)
accuracy = (tp+tn)/(tp+tn+fp+fn)
sensitivity = (tp)/(tp+fn)
specificity = (tn)/(tn+fp)
recall_score = (tp)/(tp+fp)


# Precision - Predicting "yes" and being correct
# 
# Accuracy - Being correct with any prediction
# 
# Sensitivity - Predicting "yes" and being correct with the "yes" predictions (true positive rate)
# 
# Specificity - Predicting "no" and being correct with the "no" predictions (true negative rate)
# 
# Recall Score - Being correct with all of the positive predictions

# In[ ]:


print("Precision:", precision)
print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Recall Score:", recall_score)


# The specificity is the least accurate measurement in this classifier. Some of the "no" predictions may actually be false. The model appears to work well since a majority of scores are around 99%.

# Conclusion: It is .1727% likely that fraud will happen in this particular bank
