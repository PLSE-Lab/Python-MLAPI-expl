#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection using Logistic Regression, Naive Bayes and Neural Nets

# **Start by reading data**

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


# **Start by importing required packages**

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB


# **Other required libraries are imported now**

# In[ ]:


import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score,confusion_matrix, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/creditcardfraud/creditcard.csv")


# **Let us read and get some info from the data we read**

# In[ ]:


df.describe()


# **Dataset Inferences**
# * The data is presented with Time, Amount, Class and a series of columns with naming that ranges from V1 to V28
# * Due to confidentiality issues, the actual names of V1-V28 is not provided by the source
# * V1-V28 are principal components obtained via PCA
# * This means V1 through V28 are important in determining whether a transaction is fraud or not and none of them can be neglected
# * 'Time' and 'Amount' columns are not transformed with PCA
# * Feature 'Class' is the target column, have value 1 for a fraud transaction and 0 otherwise.
# 
# 

# In[ ]:


print('The total number of transactions in dataset : ', len(df))
print('The total number of columns : ',len(list(df)))
print('The dimension of data : ', df.shape)
print('The target column is : ', list(df)[30])
print('Total number of unique values in target column is : ', len(df['Class'].unique()))
print('The unique values in Class column : ', df.Class.unique())


# **More on Class column**

# In[ ]:


print('Total number of zeroes (non-fraud transactions) : ', df['Class'].value_counts()[0])
print('Total number of ones (fraud transactions) : ', df['Class'].value_counts()[1])
print('Percentage of non-fraud transactions : ', 100*(df['Class'].value_counts()[0])/ len(df))
print('Percentage of fraud transactions : ', 100*(df['Class'].value_counts()[1])/ len(df))


# Plotting the number fraud transactions & non-fraud transactions

# In[ ]:


sns.countplot('Class', data=df, palette=None)
plt.title("Target Column frequency distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")


# **Class column inference
# **
# *   The target column is heavily imbalanced
# *     Percentage of fraud transactions over total transactions is just 0.17%
# *     Building a model with this target column will definitely lead to overfitting issue
# *     Accuracy of such a model(irrespective of algorithm) will be > 99%
# 
# **Feature Engineering requirements
# **
# *     'Class' column is heavily biased. So,it is not advised to proceed without doing something for the bias
# *     'Time' and 'Amount' columns are not transformed. So, it is required to transform them to match with the other values(V1 - V28)
# 
# 
# 
# 

# **Plot Time vs Amount to identify if there is any relationship between transaction amount over time**

# In[ ]:


df.plot(x='Time', y='Amount', style='-')
plt.title("Transaction Amount vs Time")
plt.xlabel("Time")
plt.ylabel("Amount")


# 
# *     The above graph clearly illustrates there is absolutely no relationship between transaction amount over time
# *     This means the transaction time column can be eliminated from the original data frame before further analysis
# 
# 

# **Deleting 'Time' column from original dataframe**

# In[ ]:


df = df.drop(['Time'],axis=1)


# **Scale the 'Amount' column before further analysis, name it as a new column and drop the 'Amount' column**

# In[ ]:


df['Normalized_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(['Amount'],axis=1)
df.head()


# **Change the index of Normalized_Amount and insert the same in the beginning to have a better look of data frame**

# In[ ]:


Normalized_Amount = df['Normalized_Amount']
df=df.drop(['Normalized_Amount'],axis=1)
df.insert(0, 'Normalized_Amount', Normalized_Amount)
df.head()


# ## Analysis neglecting class imbalance problem
# 
# 
# *     Here the class imbalance is completely neglected
# *     Different algorithms are applied
# *     Accuracy is measured in all cases and the values will be obviously very high, indicating overfitting problems
# 

# Well before diving into processing, let us see if there is/are any missing values in the dataframe.
# 

# In[ ]:


flag = df.isnull().sum().any()

if (flag == True):
    df.isnull().sum()
    print("There are null values in the dataframe")
    
else :
    print("There are no null values and dataframe is clear for further analysis")


# In[ ]:


# ignore all future warnings
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)


# ## Logistic Regression using Imbalanced data

# In[ ]:


X = df.drop(['Class'], axis = 1) 
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 2)
clf = LogisticRegression().fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on imbalanced training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on imbalanced test set: {:.2f}'.format(clf.score(X_test, y_test)))


# ### Confusion Matrix with imbalanced data

# In[ ]:


lr = LogisticRegression().fit(X_train, y_train)
lr_predicted = lr.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)

print('Logistic regression classifier (default settings)\n', confusion)


# **The FN and FP in Credit Card Fraud Detection**
# 
# *     False negatives (FN) in this case would be those transactions which are actually fraud transacations, but classified as non-fraud type.
# *     False positives (FP) in this case would be those transactions which are actually non-fraud type, but classified as fraud transactions.
# *     Considering the above two, the first case is very sensitive where it may classify a fraud transaction as a legally valid one
# *     This needs to be sorted out
# 
# **Revisiting Accuracy, Precision and Recall**
# * Accuracy = (TP+TN)/Total Transactions
# *  Precision = TP/(TP+FP)
# *    Recall = TP/(TP+FN)
# 
# 
# 
# 

# In[ ]:


print("Logistic Regression Evaluation Parameters with imbalanced data")
print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, lr_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, lr_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, lr_predicted)))


# **Precision-Recall Trade Off
# **
# *     The False Negatives is a serious threat in this case as it could classify fraud transactions as non-fraud type
# *     From the equations in the above cell, it is pretty obvious that it is required to increase the Recall value. Means, a high recall accounts for minimal False Negatives
# *     This inturn means that the detection problem is a recall oriented problem
# *     The above value of Recall is much lower than Precision. It is required to find a mechanism to increase the value of Recall
# 
# 

# **Let us now try two other algorithms also on this imbalanced data**

# ## Neural Net in imbalanced data

# In[ ]:


scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0,
                   random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)

print('Accuracy of NN classifier on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'.format(clf.score(X_test_scaled, y_test)))


# ## Naive Bayes in imbalanced data

# In[ ]:


nbclf = GaussianNB().fit(X_train, y_train)
print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(nbclf.score(X_train, y_train)))
print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(nbclf.score(X_test, y_test)))


# **Inferences obtained by dealing directly with imbalanced data**
# 
# * No algorithm could perform better, as the accuracies are very high
# * Clear case of overfitting due to imbalanced data
# * Some sort of data manipulation is very much required

# ## Dealing with class imbalance
# 
# 
# 

# In[ ]:


#Recalling the amount of unique values in 'Class' column
print('Total number of zeroes (non-fraud transactions) : ', df['Class'].value_counts()[0])
print('Total number of ones (fraud transactions) : ', df['Class'].value_counts()[1])


# ### For training and test data, two approaches are mentioned below
# #### Method 1
# 
# *     Randomnly select equal number of non-fraud transactions from original data set
# *     So, in this case it is required to select 492 non-fraud transactions
# *     Create a new data frame with the new set of 492 non-fraud and 492 fraud transactions
# *     Build a model with the new data frame of 984 transactions (492 + 492)
# *     This method ensures a 50-50 split of both classes of targets (fraud & non-fraud)
# 
# #### Method 2 - Random Over-Sampling
# 
# *     Duplicate the 492 fraud transactions to make it equal to 284315 non-fraud transactions
# *     Create a new data frame with the new set of 284315 non-fraud and ~ 284315 fraud transactions
# *     Build a model with the new data frame of 568630 transactions (284315 + 284315)
# *     This method also ensures a 50-50 split of both classes of targets (fraud & non-fraud)
# 
# **Both methods have its own pros and cons**
# 
# *     Both method ensures there is a target label of 50-50 split, so that the class imbalance problem is avoided
# *     Model building will be fair as target label is equally split
# *     In Method 1, only 492 fair transactions out of total 284315 fair transactions is considered for model building. This amounts to only 0.17%. This doesnt guarantee an accurate model.
# *     In Method 2, all of the fair transactions are considered - and is a good thing. But the fraud transactions are duplicated 577 times (284315/492 = 577). Still I believe this is a fair approach.
# 
# 

# ## Method 1 Analysis
# 
# **Initial step**
# 
# *     Randomnly select 492 non-fraud transactions
# *     Combine 492 non-fraud + 492 fraud to create a new dataframe
# 
# Split fraud & non-fraud transactions as two seperate dataframes
# 

# In[ ]:


non_fraud_transactions_df = df[df['Class'] == 0]
fraud_transactions_df = df[df['Class']==1]


# In[ ]:


print('The dimension of fraud transactions dataframe is : ', fraud_transactions_df.shape)
print('The dimension of non-fraud transactions dataframe is : ', non_fraud_transactions_df.shape)


# **Get a randomn sample of 492 transactions from non fraud transactions dataframe**

# In[ ]:


sample_492_non_fraud_transactions_df = non_fraud_transactions_df.sample(n=492)
print('The dimension of sample non-fraud transactions df is : ', sample_492_non_fraud_transactions_df.shape)


# **Combine sample non-fraud transactions df with 492 fraud transactions df**
# * Also shuffle and reset the index

# In[ ]:


method_1_df = pd.concat([sample_492_non_fraud_transactions_df, fraud_transactions_df])
method_1_df = method_1_df.sample(frac=1).reset_index(drop=True)


# In[ ]:


method_1_df.head()


# In[ ]:


print('The dimension of dataframe for Method 1 is : ',method_1_df.shape )


# **method_1_df is the data frame with equally distributed target column values for further analysis**
# 

# Let's now verify this with a frequency distribution plot of method_1_df

# In[ ]:


sns.countplot('Class', data=method_1_df, palette=None)
plt.title("Method 1 Frequency distribution plot")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[ ]:


print('Percentage of non-fraud transactions in method_1_df : ',  100*(method_1_df['Class'].value_counts()[0])/ len(method_1_df))
print('Percentage of fraud transactions in method_1_df : ',  100*(method_1_df['Class'].value_counts()[1])/ len(method_1_df))


# **Split the method_1_df into inputs and target labels for further analysis (X & y split)**

# In[ ]:


X = method_1_df.drop(['Class'], axis = 1) 
y = method_1_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)


# In[ ]:


print("Number transactions in training dataset for Method 1: ", len(X_train))
print("Number transactions in testing dataset  for Method 1: ", len(X_test))
print("Total number of transactions  for Method 1 : ", len(X_train)+len(X_test))


# ### Applying Logistic Regression in Method 1 - (492 + 492)

# In[ ]:


clf = LogisticRegression().fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# **Confusion Matrix**

# In[ ]:


lr_predicted = clf.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)
print('Logistic regression classifier (default settings)\n', confusion)


# **Precision & Recall Scores**

# In[ ]:


print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, lr_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, lr_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, lr_predicted)))


# **Precision-Recall Tradeoff in under-sampled data**
# 
#    * Clearly the Recall value have increased way better than that of unbalanced data
# 
# 

# **Using GridSearch to find best parameters for Logistic Regression**

# In[ ]:


logistic_parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid = GridSearchCV(LogisticRegression(), logistic_parameters)
grid.fit(X_train, y_train)
best_log_reg = grid.best_estimator_


# In[ ]:


logistic_score = cross_val_score(best_log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', logistic_score.mean())


# **Confusion Matrix after tuning in with best parameters**

# In[ ]:


lr_predicted = grid.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)
print('Logistic regression classifier with Cross-validation (default settings)\n', confusion)


# In[ ]:


print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, lr_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, lr_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, lr_predicted)))


#  ## Naive Bayes in Method 1

# In[ ]:


nbclf = GaussianNB().fit(X_train, y_train)
print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(nbclf.score(X_train, y_train)))
print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(nbclf.score(X_test, y_test)))


# ## Neural Net in Method 1

# In[ ]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0,
                   random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)

print('Accuracy of NN classifier on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'.format(clf.score(X_test_scaled, y_test)))


# ## Method 2 Analysis
# 
# *     Here the 492 fraud transactions will be duplicated to reach and become equal to 284315 non-fraud transactions
# *     The analysis done in Method 1 will be repeated for this set of data
# 

# Revisiting some codes

# In[ ]:


print('The dimension of fraud transactions dataframe is : ', fraud_transactions_df.shape)
print('The dimension of non-fraud transactions dataframe is : ', non_fraud_transactions_df.shape)


# 
# *     non_fraud_transactions_df can be kept intact as no change is required in Method 2 as well
# *     492 entries in fraud_transactions_df will be replicated to reach 284315
# * First step is to understand how many times bigger is non_fraud_transactions_df compared to fraud_transactions_df. This can be obtained by simply dividing total number of rows of non fraud transactions with that of fraud transactions.
# 

# In[ ]:


len(non_fraud_transactions_df) / len(fraud_transactions_df)


# * This means, the 492 transactions of fraud_transactions_df needs to be duplicated 577 times to make the number of rows equal to that of non_fraud_transactions_df
# 

# In[ ]:


upsampled_df = pd.concat([fraud_transactions_df] * 577, ignore_index=True)


# In[ ]:


print('The dimension of upsampled fraud transactions dataframe is : ', upsampled_df.shape)
print('The dimension of non-fraud transactions dataframe is : ', non_fraud_transactions_df.shape)


# In[ ]:


print('Difference in number of rows between two dataframes after upsampling is : ', len(non_fraud_transactions_df) - len(upsampled_df))


# * The difference is quite negligible compared to total number of transacctions and so the new data frame can be used for analysis

# In[ ]:


upsampled_df.describe()


# **Combine the two dataframes and shuffle the rows **

# In[ ]:


method_2_df = pd.concat([upsampled_df, non_fraud_transactions_df])
method_2_df = method_2_df.sample(frac=1).reset_index(drop=True)
method_2_df.describe()


# In[ ]:


print('The dimension of method_2_df is :', method_2_df.shape)


# **Frequency distribution of Class column values in method_2_df**

# In[ ]:


sns.countplot('Class', data=method_2_df, palette=None)
plt.title("Method 2 Frequency distribution plot")
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[ ]:


print('Percentage of non-fraud transactions in method_2_df : ',  100*(method_2_df['Class'].value_counts()[0])/ len(method_2_df))
print('Percentage of fraud transactions in method_2_df : ',  100*(method_2_df['Class'].value_counts()[1])/ len(method_2_df))


# **Split the method_1_df into inputs and target labels for further analysis (X & y split)**

# In[ ]:


X = method_2_df.drop(['Class'], axis = 1) 
y = method_2_df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)


# In[ ]:


print("Number transactions in training dataset for Method 2: ", len(X_train))
print("Number transactions in testing dataset  for Method 2: ", len(X_test))
print("Total number of transactions  for Method 2 : ", len(X_train)+len(X_test))


# ## Logistic Regression in Method 2

# In[ ]:


clf = LogisticRegression().fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


# **Confusion Matrix**

# In[ ]:


lr_predicted = clf.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)
print('Logistic regression classifier (default settings)\n', confusion)


# **Precision & Recall Scores**

# In[ ]:


print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, lr_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, lr_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, lr_predicted)))


# **Using GridSearch to find best parameters for Logistic Regression**

# In[ ]:


logistic_parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid = GridSearchCV(LogisticRegression(), logistic_parameters)
grid.fit(X_train, y_train)
best_log_reg = grid.best_estimator_
logistic_score = cross_val_score(best_log_reg, X_train, y_train, cv=5)
print('Logistic Regression Cross Validation Score: ', logistic_score.mean())


# In[ ]:


lr_predicted = grid.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)
print('Logistic regression classifier with Cross-validation (default settings)\n', confusion)


# In[ ]:


print('Accuracy: {:.2f}'.format(accuracy_score(y_test, lr_predicted)))
print('Precision: {:.2f}'.format(precision_score(y_test, lr_predicted)))
print('Recall: {:.2f}'.format(recall_score(y_test, lr_predicted)))
print('F1: {:.2f}'.format(f1_score(y_test, lr_predicted)))


# ## Naive Bayes in Method 2

# In[ ]:


nbclf = GaussianNB().fit(X_train, y_train)
print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(nbclf.score(X_train, y_train)))
print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(nbclf.score(X_test, y_test)))


# ## Neural Net in Method 2

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes = [100, 100], alpha = 5.0,
                   random_state = 0, solver='lbfgs').fit(X_train_scaled, y_train)

print('Accuracy of NN classifier on training set: {:.2f}'.format(clf.score(X_train_scaled, y_train)))
print('Accuracy of NN classifier on test set: {:.2f}'.format(clf.score(X_test_scaled, y_test)))


# ## Summary
# * This is a very basic Kernel which gives idea about how to deal with imbalanced data sets.
# * Of all the three classification algorithms used, Logistic Regression is found to be better when dealing with imbalanced data.
# * The Kernel can be improved by bringing in more classification algorithms, more hyper tuning parameters to existing algorithms etc.**
# 

# In[ ]:




