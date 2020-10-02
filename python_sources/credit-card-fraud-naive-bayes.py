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


# In this data-set we have 30 features related to credit cards, most of them are hidden except for time and amount. Apart from features "time" and "amount" we have 28 other features, named from V1 to V28.
# 
# Based on these features we have to classify the cases into two parts, the class with genuine transactions (marked with class "0" in the data-set) and the class with fraudlent transactions (marked with class "1" in the data-set).
# 
# Modelling here is done using Gaussian Naive Bayes.
# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Creating a pie_chart for the number of class_0 values and number of class_1 values.

# In[ ]:


data_frame = pd.read_csv('../input/creditcard.csv')

fig, ax = plt.subplots(1,1)
ax.pie(data_frame.Class.value_counts(),explode=(0,0.1), autopct='%1.1f%%', labels = ['Genuine', 'Fraud'], colors=['y','r'])
plt.axis = 'equal'


# From pie-chart, it is observed that the data-set is highly unbalanced, cases with class_0 dominates the data-set. Genuine transaction accounts for 99.8% of the total transactions, while fraudlent transaction accounts only for 0.2%.
# 
# If we train our model on this data-set, the results would be highly biased, skewing heavily towards the class having greater count.

# The original data_frame is split into two parts data_frame_1 and data_frame_2. data_frame_1 contains cases with class values = 0, and data_frame_2 containes cases with class values = 1. 
# Frequency of count_0 in data_frame_1 and count_1 in data_frame_2 is plotted against features V1 to V28. 

# In[ ]:


data_frame_1 = data_frame[data_frame['Class'] == 0]
data_frame_2 = data_frame[data_frame['Class'] == 1]
print("Cases with genuine transaction >> ", len(data_frame_1))
print("Cases with fraud transaction   >> ", len(data_frame_2)) 


# In[ ]:


for i in range(1,29):
    sns.distplot(data_frame_1.iloc[:,i])
    sns.distplot(data_frame_2.iloc[:,i], color='r')
    plt.show()


# It can be observed that, frequency distribution for both the classes ("0" and "1") against features V8, V13, V15, V20, V21, V22, V23, V24, V25, V26, V27 and V28 are approximately similar. These features would most certainly not help out for the purpose of differentiating between class 0 and 1. So, we drop these features from our data_frames so as to make our model less complex.

# In[ ]:


data_frame_1 = data_frame_1.drop(columns=["V8","V13","V15","V20","V21","V22","V23","V24","V25","V26","V27","V28"])
data_frame_2 = data_frame_2.drop(columns=["V8","V13","V15","V20","V21","V22","V23","V24","V25","V26","V27","V28"])
print(data_frame_1.head())
print(data_frame_2.head())


# In[ ]:


plot = sns.distplot(data_frame_1["Amount"],color='b', kde=False )
plot.set_yscale('log')
plt.xlabel("Amount for Genuine cases")
plt.show()


# In[ ]:


plot = sns.distplot(data_frame_2["Amount"],color='r', kde=False )
plot.set_yscale('log')
plt.xlabel("Amount for Fraudlent cases")
plt.show()


# As the data-set is highly unbalanced, dominated with the class values of 0, it is required to oversample the minority class in the training data-set. This oversampling is done with the help SMOTE algorithm. 
# 
# But before oversampling the minority class in the training data-set, we segregate a test data-set, which contains 50 cases of class 1 and 2000 cases of class 0. This data-set would be used to measure various performance parameters for Naive Bayes Classification model.
# 
# (Note: The training data-set doesn't contain any of the entries from test data-set).

# In[ ]:


data_frame_final_test_1 = data_frame_2.head(50)
data_frame_final_test_0 = data_frame_1.head(2500)
print("Fraud cases in final test-set",len(data_frame_final_test_1))
print("Genuine cases in final test-set", len(data_frame_final_test_0))


# In[ ]:


data_frame_train_class_1 = data_frame_2.tail(len(data_frame_2)-50)
data_frame_train_class_0 = data_frame_1.tail(len(data_frame_1)-2500)
print("Fraud cases in training data-set", len(data_frame_train_class_1))
print("Genine cases in training data-set",len(data_frame_train_class_0))


# In[ ]:


training_data_frame = pd.concat([data_frame_train_class_1, data_frame_train_class_0])
testing_data_frame = pd.concat([data_frame_final_test_1, data_frame_final_test_0])

data_x_test = testing_data_frame.drop(columns=['Class'])
test_x = data_x_test.values
test_y = testing_data_frame['Class'].values

data_x = training_data_frame.drop(columns=['Class'])
x = data_x.values
y = training_data_frame['Class'].values
count_0 = 0
count_1 = 0
for i in range(len(y)):
    if y[i] == 0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1
print('count_0 in training data-set', count_0)
print('count_1 in training data-set', count_1)


# After segregating a training and testing data-set, the minority class in training data-set is oversampled using  kMeansSMOTE.

# In[ ]:


from kmeans_smote import KMeansSMOTE
sm = KMeansSMOTE(imbalance_ratio_threshold=float('Inf'), kmeans_args={'n_clusters':1})
x_res, y_res = sm.fit_resample(x, y)
count_0 = 0
count_1 = 0
for i in range(len(y_res)):
    if y_res[i] == 0:
        count_0 = count_0 + 1
    else:
        count_1 = count_1 + 1
print('count_0 in training data-set after SMOTE', count_0)
print('count_1 in training data-set after SMOTE', count_1)


# Naive-Bayes classification model.

# In[ ]:


from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(x_res, y_res)


# In[ ]:


y_pred = GNB.predict(test_x)
y_true = test_y
y_pred_prob = GNB.predict_proba(test_x)
print("Predictions for the test-set")
print(y_pred)
print("True values for the test-set")
print(y_true)
print("\n")
print("Prediction probability for the test-set")
print(y_pred_prob)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_1 = confusion_matrix(y_true, y_pred)
print("Confusion Matrix for Gaussian NB")
print("")
print(confusion_1)


# Confusion matrix for the test-data-set showS that out of 50 fraud cases 46 were truly predicted.  

# In[ ]:


from sklearn.metrics import roc_auc_score,f1_score,precision_score, accuracy_score, recall_score
roc_auc_score_1 = roc_auc_score(y_true, y_pred)
accuracy_score_1 = accuracy_score(y_true, y_pred)
precision_score_1 = precision_score(y_true, y_pred)
recall_score_1 = recall_score(y_true, y_pred)
f1_score_1 = f1_score(y_true, y_pred)

print("roc_auc_score", "%.3f" %roc_auc_score_1)
print("accuracy_score", "%.3f" %accuracy_score_1)
print("precision_score", "%.3f" %precision_score_1)
print("recall_score", "%.3f" %recall_score_1)
print("f1_score", "%.3f" %f1_score_1)

