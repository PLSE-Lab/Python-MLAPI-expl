#!/usr/bin/env python
# coding: utf-8

# # Financial Fraud Detection
# 
# With the rise of digital financing fraud, there is a need to be able to be able to better identify fraudulent transactions. The goal of this analysis is to better understand the dataset we have, preprocess it and create models for predictions.
# 
# **General Steps:**
# - **Step 1**: Load prerequisites and required modules
# - **Step 2**: Import dataset
# - **Step 3**: Perform exploratory data analysis
# - **Step 4**: Preprocess data for predictive modelling
# - **Step 5**: Analyse the models obtained and decide on the best one

# ### Prerequisites

# Let's first load the necessary libraries

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


get_ipython().system('pip install xgboost')
get_ipython().system('pip install imbalanced-learn')


# ### Importing the libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


df = pd.read_csv('../input/paysim1/PS_20174392719_1491204439457_log.csv')
df.head()


# In[ ]:


df.info()


# ## Feature Descriptions:
# 
# **step**: Maps a unit of time in the real world. In this case 1 step is 1 hour of time.
# 
# **type**: CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER
# 
# **amount**: amount of the transaction in local currency
# 
# **nameOrig**: customer who started the transaction
# 
# **oldbalanceOrg**: initial balance before the transaction
# 
# **newbalanceOrig**: customer's balance after the transaction.
# 
# **nameDest**: recipient ID of the transaction.
# 
# **oldbalanceDest**: initial recipient balance before the transaction.
# 
# **newbalanceDest**: recipient's balance after the transaction.
# 
# **isFraud**: identifies a fraudulent transaction (1) and non fraudulent (0)
# 
# **isFlaggedFraud**: flags illegal attempts to transfer more than 200.000 in a single transaction.

# In[ ]:


frauds = df[df.isFraud == 1]
non_frauds = df[df.isFraud == 0]

frauds['balanceChange'] = frauds['newbalanceOrig'] - frauds['oldbalanceOrg']
non_frauds['balanceChange'] = non_frauds['newbalanceOrig'] - non_frauds['oldbalanceOrg']

frauds_mean_balancechange = frauds['balanceChange'].mean()
nfrauds_mean_balancechange = non_frauds['balanceChange'].mean()

width = 0.8
fig, ax = plt.subplots(1,1, figsize = (10, 6))
ax.bar(0.5, nfrauds_mean_balancechange, width, label='Avg. Non Fraud Account Balance Change', align='center')
ax.bar(1.5, frauds_mean_balancechange, width, label='Avg. Fraud Account Balance Change')
fig.legend(loc='best')
plt.axis([0, 2, -1500000, 200000])


# #### We can already see that fraudulent transactions have behavior that deviates greatly from the norm! Specifically, the average bank account balance change when fraud occurs is very large and negative
# 
# ## This invites the question: How often does a fraudulent transaction lead to a positive account balance change?

# In[ ]:


num_positive_frauds = len(frauds[frauds['balanceChange'] > 0])
num_positive_frauds


# ### None of the fradulent transactions result in a positive account balance change! 
# 
# This makes sense if we consider that criminals would not be in the business of giving money away.
# 
# ### Now, let's drop all the transactions that resulted in a positive account balance change, so that we are left with only transactions where we would expect fraud to happen 

# In[ ]:


paysim_negative = pd.concat([frauds[frauds['balanceChange'] <= 0], non_frauds[non_frauds['balanceChange'] <= 0]])


# ### We can figure out what percentage of these non-positive balance change transactions turn out to be fradulent

# In[ ]:


num_frauds = len(paysim_negative[paysim_negative['isFraud'] == 1])
num_frauds / len(paysim_negative)


# ### Frauds account for only 0.17% of suspected transactions! This reveals that our dataset is highly imbalanced between frauds and non-frauds
# 
# #### In fact, the data has so few frauds that a stupid classifier (i.e. always predicting not fraud) would get ~99.8% accuracy on our selected data! Because of this, accuracy will not be an important metric to consider while building our model
# 
# 
# ### Let's consider now what features are correlated with the fraudulent transactions we do have

# In[ ]:


paysim_negative.corr()


# #### We can group the most relevant features by the name of the client who receives the funds

# In[ ]:


paysim_byClient = paysim_negative[['nameDest', 'oldbalanceOrg', 'oldbalanceDest', 'balanceChange']].groupby(['nameDest']).mean()
frauds_byClient = paysim_negative[['nameDest', 'isFraud']].groupby(['nameDest']).sum()
clientData = pd.concat([paysim_byClient, frauds_byClient], axis=1)
clientData['numTrans'] = paysim_negative[['nameDest', 'isFraud']].groupby(['nameDest']).count()['isFraud']
clientData = clientData.sort_values(by='isFraud', ascending=False)
clientData.head(20)


# ### We can look further into each feature
# 
# 
# #### First, we can plot number of frauds versus the original account balance of victims

# In[ ]:


from scipy import stats

oldBalance = clientData[['oldbalanceOrg', 'isFraud']].groupby(['oldbalanceOrg']).sum()
kde = stats.gaussian_kde(oldBalance.index)
xx = np.linspace(0, 9, 1000)
plt.plot(xx, kde(xx))
oldBalance[1:].hist(bins=10)


# #### We can do a similar thing with the resulting acccount balance of victims AFTER the fraud occurs

# In[ ]:


newBalanceOrig = paysim_negative[['newbalanceOrig', 'isFraud']].groupby(['newbalanceOrig']).sum()
newBalanceOrig[1:].hist(bins=10)


# #### A plot of the account balance of the fraudsters BEFORE the fradulent transactions

# In[ ]:


oldBalanceDest = paysim_negative[['oldbalanceDest', 'isFraud']].groupby(['oldbalanceDest']).sum()
oldBalanceDest[1:].hist(bins=10)


# #### We can see that most people who commit fraud have less than a dollar in their account

# In[ ]:


newBalanceDest = paysim_negative[['newbalanceDest', 'isFraud']].groupby(['newbalanceDest']).sum()
newBalanceDest[1:].hist(bins=10)


# #### After fraud, account balances creep up by a few dollars
# 
# 
# ### So most fraudulent transactions leave victim's with only dollars in their accounts and go to recipients with low account balances

# In[ ]:


numTrans = clientData[['numTrans', 'isFraud']].groupby(['numTrans']).sum()
kde = stats.gaussian_kde(numTrans.index)
xx = np.linspace(0, max(numTrans.isFraud), 1000)
plt.plot(xx, kde(xx))
numTrans.hist(bins=10)


# #### Also, fraud is more spread out versus the number of transactions feature, but with a majority of fraud committed by clients who perform a relatively small number of transactions.
# 
# We can also see how transactions are distributed across various payment types.

# In[ ]:


ax = df.type.value_counts().plot.bar(figsize = (12, 6))
ax.set_title("Distribution of transactions for different payment types", fontsize = 20)
plt.xticks(rotation = 45)


# Now that we have a better understanding of our dataset, let's move onto preprocessing the data and creating predictive models.

# ### Data Preprocessing

# Let's start off by dropping unnecessary features. 'step' refers to a timestep but there is no way of tying multiple transactions together to form a sequence of related payments. So it doesn't serve any purpose in our models. 'isFlaggedFraud' predominantly features zeroes with around 15 rows with a one. 'nameOrig' and 'nameDest' are identifiers which shouldn't be a part of models.

# In[ ]:


df.drop(columns = ['step', 'isFlaggedFraud', 'nameOrig', 'nameDest'], inplace = True)
df.isFraud.value_counts()


# Let's split-up the features and the label.

# In[ ]:


X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].copy()
y = df['isFraud'].copy()
X.type = X.type.astype('category')


# Now, let's encode the categorical variable 'type' using one-hot encoding. In order to prevent data leakage, we will be doing this transformation based on only the training dataset.

# In[ ]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

col_transform = make_column_transformer((['type'], OneHotEncoder(sparse = False, drop = 'first')),
                                        remainder = 'passthrough')
X = pd.DataFrame(col_transform.fit_transform(X))


# # Predictive Modelling
# 
# For a problem with highly imbalanced classes, there are three approaches one can take.
# - **No sampling**: Train models as it is without disturbing the balance of the dataset
# - **Undersampling**: Randomly select observations from the bigger class such that it ends up having the same number of observations as the smaller class.
# - **Oversampling**: Systematically create new observations for the smaller class such that it ends up having the same number of observations as the bigger class.
# 
# For all these approaches, we are using Random Forests and XGBoost as they are known to perform well on imbalanced classes. Just for the undersampling case, we are also training an artifical neural network (as it is feasible and practical in this case). 
# 
# In order to evaluate the models, we are using 5-fold cross-validation with the evaluation metric being the area under the precision-recall curve.
# 
# 
# ## Type I - No sampling technique
# 
# ### Random Forests

# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

cv_pr_aucs = []
recalls = []
precisions = []
thresholds_list = []
skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print("Started training the model for the current fold at", datetime.now())
    rf_model = RandomForestClassifier(random_state = 42)
    rf_model.fit(X_train, y_train)
    y_pred = list(rf_model.predict_proba(X_test)[:, 1])
    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)
    recalls.append(recall)
    precisions.append(precision)
    thresholds_list.append(thresholds)
    cv_pr_aucs.append(auc(recall, precision))
    print("Completed training the model for the current fold at", datetime.now(), "\n")


# Let's plot the feature importances since this happens to be the best model we've obtained.

# In[ ]:


rf_model = RandomForestClassifier(random_state = 42)
rf_model.fit(X, y)

fig, axis = plt.subplots(figsize = (10, 6))
feature_list = ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
axis.bar(feature_list, (rf_model.feature_importances_ * 100))
plt.xticks(rotation = 45)
axis.set_title('Feature Importance from Random Forests', fontsize = 17)
axis.set_xlabel('Feature', fontsize = 17)
axis.set_ylabel('Importance (%)', fontsize = 17)


# It's quite intuitive that the old and new balances seem to be the most important features looking at the EDA above.
# 
# Now, let's plot the precision-recall curve and calculate the area under it.

# In[ ]:


print("Area under PR curve:", np.mean(cv_pr_aucs))

precision = np.mean(precisions, axis = 0)
recall = np.mean(recalls, axis = 0)
threshold = np.mean(thresholds_list, axis = 0)

fig = plt.figure(figsize = (10, 6))
plt.plot(recall, precision, marker = '.')
plt.plot([0, 1], [sum(y == 1)/len(y), sum(y == 1)/len(y)], linestyle = '--')
ax = plt.gca()
ax.axvline(x = recall[int(np.where(threshold == 0.5)[0])], color = 'r', linestyle = '--', ymin = 0.05, ymax = 1)
ax.annotate('50% Threshold', (recall[int(np.where(threshold == 0.5)[0])] - 0.085, recall[int(np.where(threshold == 0.5)[0])] - 0.3), fontsize = 15)
ax.set_title("Precision-Recall Curve", fontsize = 17)
ax.set_xlabel('Recall', fontsize = 17)
ax.set_ylabel('Precision', fontsize = 17)


# ### XGBoost

# In[ ]:


from xgboost import XGBClassifier

cv_pr_aucs = []
recalls = []
precisions = []
skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print("Started training the model for the current fold at", datetime.now())
    xgb_model = XGBClassifier(random_state = 42)
    xgb_model.fit(X_train, y_train)
    y_pred = list(xgb_model.predict_proba(X_test)[:, 1])
    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)
    recalls.append(recall)
    precisions.append(precision)
    cv_pr_aucs.append(auc(recall, precision))
    print("Completed training the model for the current fold at", datetime.now(), "\n")
    
print("Area under PR curve:", np.mean(cv_pr_aucs))


# Given that XGBoost leads to a different set of thresholds for every fold of the cross-validation, we are unable to plot the curve.

# ## Type II - Undersampling
# 
# ### Random Forests

# In this case, since it's practically feasible given the relatively smaller dataset owing to undersampling, we would be using GridSearchCV to try out different hyperparameter combinations.

# In[ ]:


from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [10, 50, 100],
              'criterion': ['gini', 'entropy'],
              'max_depth': [None, 3, 5]
             }


# In[ ]:


from sklearn.utils import resample

cv_pr_aucs = []
recalls = []
precisions = []
thresholds_list = []
skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]    
    
    X_frauds_df = X_train[y_train == 1]
    y_frauds_df = y_train[y_train == 1]
    
    X_not_frauds_df = X_train[y_train == 0]
    y_not_frauds_df = y_train[y_train == 0]
    
    X_not_frauds_df, y_not_frauds_df = resample(X_not_frauds_df, y_not_frauds_df, replace = False, n_samples = len(X_frauds_df), random_state = 42)
    
    X_train = pd.concat([X_frauds_df, X_not_frauds_df])
    y_train = pd.concat([y_frauds_df, y_not_frauds_df])
    
    X_train = X_train.reset_index().drop(columns = ['index'])
    y_train = y_train.reset_index().drop(columns = ['index'])
    
    print("Started training the model for the current fold at", datetime.now())
    rf_model = RandomForestClassifier(random_state = 42)
    rf_model = GridSearchCV(rf_model, parameters, cv = 5, scoring = 'average_precision')
    rf_model.fit(X_train, y_train)
    y_pred = list(rf_model.predict_proba(X_test)[:, 1])
    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)
    recalls.append(recall)
    precisions.append(precision)
    thresholds_list.append(thresholds)
    cv_pr_aucs.append(auc(recall, precision))
    print("Completed training the model for the current fold at", datetime.now(), "\n")


# In[ ]:


print("Area under PR curve:", np.mean(cv_pr_aucs))


# ### XGBoost

# In[ ]:


parameters = {'n_estimators': [10, 50, 100],
              'criterion': ['gini', 'entropy'],
              'max_depth': [2, 3, 6]
             }


# In[ ]:


cv_pr_aucs = []
recalls = []
precisions = []
skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]    
    
    X_frauds_df = X_train[y_train == 1]
    y_frauds_df = y_train[y_train == 1]
    
    X_not_frauds_df = X_train[y_train == 0]
    y_not_frauds_df = y_train[y_train == 0]
    
    X_not_frauds_df, y_not_frauds_df = resample(X_not_frauds_df, y_not_frauds_df, replace = False, n_samples = len(X_frauds_df), random_state = 42)
    
    X_train = pd.concat([X_frauds_df, X_not_frauds_df])
    y_train = pd.concat([y_frauds_df, y_not_frauds_df])
    
    X_train = X_train.reset_index().drop(columns = ['index'])
    y_train = y_train.reset_index().drop(columns = ['index'])
    
    print("Started training the model for the current fold at", datetime.now())
    xgb_model = XGBClassifier(random_state = 42)
    xgb_model = GridSearchCV(xgb_model, parameters, cv = 5, scoring = 'average_precision')
    xgb_model.fit(X_train, y_train)
    y_pred = list(xgb_model.predict_proba(X_test)[:, 1])
    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)
    recalls.append(recall)
    precisions.append(precision)
    cv_pr_aucs.append(auc(recall, precision))
    print("Completed training the model for the current fold at", datetime.now(), "\n")


# In[ ]:


print("Area under PR curve:", np.mean(cv_pr_aucs))


# ### Neural Network

# We are running a neural network only for the undersampling case as the training time for the other two cases (oversampling and no sampling) are extremely high (more than 8 hours) and the results are far worse than some of the other less complex models.
# 
# Also, we are not using cross-validation to evaluate the model as it's extremely time-consuming. Instead, we are using the train-test set approach.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout


# The network we are creating consists of an input layer with 9 nodes, the first hidden layer with 12 nodes and RELU activation functions, the second hidden layer with 8 nodes and RELU activation functions and finally, the output node with a sigmoid activation function to get the prediction probability.
# 
# We are also standardizing the features as it improves the performance and accuracy of neural networks.

# In[ ]:


model = Sequential()
model.add(Dense(12, input_dim = 9, activation = 'relu'))
model.add(Dropout(0.3, seed = 42))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, stratify = y, random_state = 42)

X_frauds_df = X_train[y_train == 1]
y_frauds_df = y_train[y_train == 1]

X_not_frauds_df = X_train[y_train == 0]
y_not_frauds_df = y_train[y_train == 0]

X_not_frauds_df, y_not_frauds_df = resample(X_not_frauds_df, y_not_frauds_df, replace = False, n_samples = len(X_frauds_df), random_state = 42)

X_train = pd.concat([X_frauds_df, X_not_frauds_df])
y_train = pd.concat([y_frauds_df, y_not_frauds_df])

X_train = X_train.reset_index().drop(columns = ['index'])
y_train = y_train.reset_index().drop(columns = ['index'])

std_scaler = StandardScaler()
X_train = pd.DataFrame(std_scaler.fit_transform(X_train))


# In[ ]:


model.fit(X_train, y_train, epochs = 150, batch_size = 10)


# In[ ]:


losses = model.history.history['loss']
accs = model.history.history['acc']


# In[ ]:


fig = plt.figure(figsize = (10, 6))
plt.plot(losses)
ax = plt.gca()
ax.set_title("Loss (Cross-Entropy) vs Epochs", fontsize = 17)
ax.set_xlabel('Epoch', fontsize = 17)
ax.set_ylabel('Loss', fontsize = 17)


# In[ ]:


fig = plt.figure(figsize = (10, 6))
plt.plot(np.array(accs)*100)
ax = plt.gca()
ax.set_title("Accuracy vs Epochs", fontsize = 17)
ax.set_xlabel('Epoch', fontsize = 17)
ax.set_ylabel('Accuracy (%)', fontsize = 17)


# In[ ]:


y_pred = model.predict_proba(std_scaler.transform(X_test))[:, 0]
precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)
print("Area under PR curve: ", auc(recall, precision))


# In[ ]:


fig = plt.figure(figsize = (10, 6))
plt.plot(recall, precision, marker = '.')
plt.plot([0, 1], [sum(y == 1)/len(y), sum(y == 1)/len(y)], linestyle = '--')
ax = plt.gca()
ax.axvline(x = recall[int(np.where((thresholds > 0.49) & (thresholds < 0.51))[0][0])], color = 'r', linestyle = '--', ymin = 0.05, ymax = 0.96)
ax.annotate('50% Threshold', (recall[int(np.where((thresholds > 0.49) & (thresholds < 0.51))[0][0])] - 0.15, recall[int(np.where((thresholds > 0.49) & (thresholds < 0.51))[0][0])] - 0.3), fontsize = 15)
ax.set_title("Precision-Recall Curve", fontsize = 17)
ax.set_xlabel('Recall', fontsize = 17)
ax.set_ylabel('Precision', fontsize = 17)


# ## Type III - Oversampling
# 
# ### Random Forests

# In[ ]:


from imblearn.over_sampling import SMOTE
from datetime import datetime

cv_pr_aucs = []
recalls = []
precisions = []
thresholds_list = []
skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]   
    smote = SMOTE(random_state = 42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train)
    
    print("Started training the model for the current fold at", datetime.now())
    rf_model = RandomForestClassifier(random_state = 42)
    rf_model.fit(X_train, y_train)
    y_pred = list(rf_model.predict_proba(X_test)[:, 1])
    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)
    recalls.append(recall)
    precisions.append(precision)
    thresholds_list.append(thresholds)
    cv_pr_aucs.append(auc(recall, precision))
    print("Completed training the model for the current fold at", datetime.now(), "\n")


# In[ ]:


print("Area under PR curve:", np.mean(cv_pr_aucs))

precision = np.mean(precisions, axis = 0)
recall = np.mean(recalls, axis = 0)
threshold = np.mean(thresholds_list, axis = 0)

fig = plt.figure(figsize = (10, 6))
plt.plot(recall, precision, marker = '.')
plt.plot([0, 1], [sum(y == 1)/len(y), sum(y == 1)/len(y)], linestyle = '--')
ax = plt.gca()
ax.axvline(x = recall[int(np.where(threshold == 0.5)[0])], color = 'r', linestyle = '--', ymin = 0.05, ymax = 1)
ax.annotate('50% Threshold', (recall[int(np.where(threshold == 0.5)[0])] - 0.085, recall[int(np.where(threshold == 0.5)[0])] - 0.3), fontsize = 15)
ax.set_title("Precision-Recall Curve", fontsize = 17)
ax.set_xlabel('Recall', fontsize = 17)
ax.set_ylabel('Precision', fontsize = 17)


# ### XGBoost

# In[ ]:


cv_pr_aucs = []
recalls = []
precisions = []
thresholds_list = []
skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]   
    smote = SMOTE(random_state = 42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    X_train = pd.DataFrame(X_train)
    y_train = pd.Series(y_train)
    
    print("Started training the model for the current fold at", datetime.now())
    xgb_model = XGBClassifier(random_state = 42)
    xgb_model.fit(X_train, y_train)
    y_pred = list(xgb_model.predict_proba(X_test)[:, 1])
    precision, recall, thresholds = precision_recall_curve(y_test.tolist(), y_pred)
    recalls.append(recall)
    precisions.append(precision)
    thresholds_list.append(thresholds)
    cv_pr_aucs.append(auc(recall, precision))
    print("Completed training the model for the current fold at", datetime.now(), "\n")


# In[ ]:


print("Area under PR curve:", np.mean(cv_pr_aucs))


# ## Conclusion

# Thus, we have analyzed the dataset and obtained a fairly accurate predictive model using Random Forests without any sampling, closely followed by Random Forests with oversampling. In order to reduce our chances of performing a Type II error (predicting a transaction as NOT FRAUD when it actually is one), we can move our threshold to less that 50% to increase Recall at the expense of Precision. This needs to be studied on a case-by-case basis.
