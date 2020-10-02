#!/usr/bin/env python
# coding: utf-8

# ## Financial Fraud Detection-XGBoost

# My ML learning attempt. This kernel is based on https://www.kaggle.com/arjunjoshua/predicting-fraud-in-financial-payment-services

# <a id='top'></a>
# #### Outline: 
# #### 1. <a href='#import'>Import relevant libraries and data</a>
# 11. <a href='#col_corr'>Import data and correct the spelling of original column headers for consistency</a>
# 
# #### 2. <a href='#eda'>Feature Selection</a>
# 21. <a href='#eda_feature'>Find the relevance of each feature in identifying the fraudulent transactions</a>
# 22. <a href='#eda_conclusions'>Feature Selection- Conclusions</a>
# 
# #### 3. <a href='#dataclean'>Data Cleaning</a>
# 
# #### 4. <a href='#ML_wo_iORf'>Machine Learning (without any imputation or feature engineering)</a>
# 41. <a href='#ML1_features_imp'>What are the important features for the ML model?</a>
# 
# #### 5 <a href='#imputation'>Imputation of Latent Missing Values</a>
# 
# #### 6. <a href='#ML_after_imputation'>Machine Learning (after imputation, but not feature engineering)</a>
# 61. <a href='#ML2_features_imp'>What are the important features for the ML model?</a>
# 
# #### 7. <a href='#feature'>Feature Engineering</a>
# 
# #### 8. <a href='#ML'>Machine Learning (after imputation and feature engineering)</a>
# 81. <a href='#ML3_features_imp'>What are the important features for the ML model?</a>
# 

# <a id='import'></a>
# ### 1. Import relevant libraries and data

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


# sets the backend of matplotlib to the inline backend
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance


# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# <a href='#top'>back to top</a>

# <a id = 'col_corr'> </a>
# #### 1.1 Import data and correct the spelling of original column headers for consistency

# In[ ]:


#raw_csv_data = pd.read_csv('PS_20174392719_1491204439457_log.csv')
#ip_data = raw_csv_data.copy()
ip_data = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')


# In[ ]:


ip_data.head()


# In[ ]:


ip_data = ip_data.rename(columns = {'oldbalanceOrg': 'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', 
                                    'oldbalanceDest':'oldBalanceDest', 'newbalanceDest' : 'newBalanceDest'})


# In[ ]:


# check for missing values.
ip_data.isnull().values.any()


# In[ ]:


ip_data.info()


# It turns out there are no obvious missing values but as we will see above, this does not rule out proxies by a numerical value like 0

# <a href='#top'>back to top</a>

# <a id='eda'></a>
# ### 2. Feature Selection

# In this section and until section 4, we wrangle with the data exclusively using Dataframe methods. This is the most succinct way to gain insights into the dataset.

# <a id = 'eda_feature'> </a>
# #### 2.1 Find the relevance of each feature in identifying the fraudulent transactions

# In[ ]:


from collections import Counter

step_list = list(ip_data.loc[ip_data.isFraud == 1].step.values)

step_counted_list = Counter(step_list)
step_counted_list.most_common(40)


# In[ ]:


type_list = list(ip_data.loc[ip_data.isFraud == 1].type.values)

type_counted_list = Counter(type_list)
type_counted_list.most_common(20)


# In[ ]:


amount_list = list(ip_data.loc[ip_data.isFraud == 1].amount.values)

amount_counted_list = Counter(amount_list)
amount_counted_list.most_common(20)


# In[ ]:


nameOrig_list = list(ip_data.loc[ip_data.isFraud == 1].nameOrig.values)

nameOrig_counted_list = Counter(nameOrig_list)
nameOrig_counted_list.most_common(20)


# In[ ]:


oldBalanceOrig_list = list(ip_data.loc[ip_data.isFraud == 1].oldBalanceOrig.values)

oldBalanceOrig_counted_list = Counter(oldBalanceOrig_list)
oldBalanceOrig_counted_list.most_common(20)


# In[ ]:


newBalanceOrig_list = list(ip_data.loc[ip_data.isFraud == 1].newBalanceOrig.values)

newBalanceOrig_counted_list = Counter(newBalanceOrig_list)
newBalanceOrig_counted_list.most_common(20)


# In[ ]:


nameDest_list = list(ip_data.loc[ip_data.isFraud == 1].nameDest.values)

nameDest_counted_list = Counter(nameDest_list)
nameDest_counted_list.most_common(20)


# In[ ]:


oldBalanceDest_list = list(ip_data.loc[ip_data.isFraud == 1].oldBalanceDest.values)

oldBalanceDest_counted_list = Counter(oldBalanceDest_list)
oldBalanceDest_counted_list.most_common(20)


# In[ ]:


newBalanceDest_list = list(ip_data.loc[ip_data.isFraud == 1].newBalanceDest.values)

newBalanceDest_counted_list = Counter(newBalanceDest_list)
newBalanceDest_counted_list.most_common(20)


# In[ ]:


isFlaggedFraud_list = list(ip_data.loc[ip_data.isFraud == 1].isFlaggedFraud.values)

isFlaggedFraud_counted_list = Counter(isFlaggedFraud_list)
isFlaggedFraud_counted_list.most_common(20)


# <a href='#top'>back to top</a>

# <a id = 'eda_conclusions'> </a>
# #### 2.2 Feature Selection - Conclusions
# By analysing the dataset, we come to the following conclusions about the features:
# 
# #### Features:                         
# <b>1. step </b>   :         Include this feature. The fraudulent transactions distributed in many 'step' values.
# 
# <b>2. type </b>	    :         Include this feature. The fraudulent transaction happened only in 'CASH_OUT' and 'TRANSFER' transaction types. So we will include only the records with type as 'CASH_OUT' and 'TRANSFER.'
# 
# <b>3. amount</b>:         Include this feature. Though it won't explain all fraudulent transactions, amount as 10000000.0 and 0.0 denotes a high chance of fraud. 
# 
# <b>4. nameOrig	</b>:         Drop this feature. There is no useful information from this column.
# 
# <b>5. oldbalanceOrig </b>:   Include this feature. You could see that in almost all fraudulent transactions, 'oldbalanceOrig' and 'amount' has the same value. This is a strong indicator of a fraudulent transaction. 
# 
# <b>6. newbalanceOrig </b>:   Include this feature. For most of the fraudulent transactions, 'newbalanceOrig' = 0 (this fact supports our finding in #5)
# 
# <b>7. nameDest	</b>:         Drop this feature. There is no useful information from this column.
# 
# <b>8. oldbalanceDest </b>:   Include this feature. Value of 'oldbalanceDest' is zero for nearly half of the fraudulent transaction.
# 
# <b>9. newbalanceDest </b>:   Include this feature. Value of 'oldbalanceDest' is zero for more than half of the fraudulent transaction. We will include this feature in our model. 
# 
# <b>10. isFlaggedFraud </b>:  Drop this feature. Only 16 transactions flagged correctly. We can drop this feature.
# 

# <a href='#top'>back to top</a>

# <a id = 'dataclean'> </a>
# ### 3. Data Cleaning

# In[ ]:


X = ip_data.loc[(ip_data.type == 'CASH_OUT') | (ip_data.type == 'TRANSFER')]
randomState = 5
np.random.seed(randomState)


# In[ ]:


Y = X['isFraud']
del X['isFraud']


del X['nameDest'] 
del X['nameOrig']
del X['isFlaggedFraud']


# In[ ]:


X.head()


# In[ ]:


#Binary-encoding of labelled data in 'type'
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1

X.type = X.type.astype(int)


# In[ ]:


X_fraud = X.loc[Y==1]
X_nonFraud = X.loc[Y==0]


# <a href='#top'>back to top</a>

# <a id = 'ML_wo_iORf'> </a>
# ### 4. Machine Learning (without any imputation or feature engineering)

# In[ ]:


print('skew = {}'.format(len(X_fraud)/float(len(X))))


# *Selection of metric*: 
# Since the data is highly skewed, I use the area under the precision-recall curve (AUPRC) rather than the conventional area under the receiver operating characteristic (AUROC). This is because the AUPRC is more sensitive to differences between algorithms and their parameter settings rather than the AUROC (see <a href='http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf'>Davis and Goadrich, 2006</a>).

# *Selection of ML algorithm*: A first approach to deal with imbalanced data is to balance it by discarding the majority class before applying an ML algorithm. The disadvantage of  undersampling is that a model trained in this way will not perform well on real-world skewed test data since almost all the information was discarded. A better approach might be to oversample the minority class, say by the synthetic minority oversampling technique (SMOTE) contained in the 'imblearn' library. Motivated by this, I tried a variety of anomaly-detection and supervised learning approaches. I find, however, that the best result is obtained on the original dataset by using a ML algorithm based on ensembles of decision trees that intrinsically performs well on imbalanced data. Such algorithms not only allow for constructing a model that can cope with the missing values in our data, but they naturally allow for speedup via parallel-processing. Among these algorithms, the extreme gradient-boosted (XGBoost) algorithm used below slightly outperforms random-forest. Finally, XGBoost, like several other ML algorithms, allows for weighting the positive class more compared to the negative class --- a setting that also allows to account for the skew in the data.

# In[ ]:


#Split the data into training and test sets in a 80:20 ratio
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = randomState)

weights = (Y==0).sum()/(1.0 *  (Y==1).sum())
xgb_classifier1 = XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4)
xgb_prediction1 = xgb_classifier1.fit(X_train, Y_train).predict_proba(X_test)

print('AUPRC = {}'.format(average_precision_score(Y_test, xgb_prediction1[:,1])))


# <a id='ML1_features_imp'> </a>
# #### 4.1. What are the important features for the ML model?

# In[ ]:


fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(xgb_classifier1, height = 1, color = colours, grid = False,                      show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);


# <a href='#top'>back to top</a>

# <a id='imputation'> </a>
# ### 5 Imputation of Latent Missing Values
# 
# The data has several transactions with zero balance in the destination account both before and after a non-zero amount is transacted. The fraciton of such transactions, where zero likely denotes a missing value, is much larger in fraudelent (50%) compared to genuine transactions (0.06%)

# In[ ]:


#Xfraud = X.loc[Y == 1]
#XnonFraud = X.loc[Y == 0]
print('\nThe fraction of fraudulent transactions with \'oldBalanceDest\' = \'newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: {}'.format(len(X_fraud.loc[(X_fraud.oldBalanceDest == 0) & (X_fraud.newBalanceDest == 0) & (X_fraud.amount)]) / (1.0 * len(X_fraud))))

print('\nThe fraction of genuine transactions with \'oldBalanceDest\' = newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: {}'.format(len(X_nonFraud.loc[(X_nonFraud.oldBalanceDest == 0) & (X_nonFraud.newBalanceDest == 0) & (X_nonFraud.amount)]) / (1.0 * len(X_nonFraud))))


# Since the destination account balances being zero is a strong indicator of fraud, we do not impute the account balance (before the transaction is made) with a statistic or from a distribution with a subsequent adjustment for the amount transacted. Doing so would mask this indicator of fraud and make fraudulent transactions appear genuine. Instead, below we replace the value of 0 with -1 which will be more useful to a suitable machine-learning (ML) algorithm detecting fraud.

# In[ ]:


X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0),       ['oldBalanceDest', 'newBalanceDest']] = - 1


# The data also has several transactions with zero balances in the originating account both before and after a non-zero amount is transacted. In this case, the fraction of such transactions is much smaller in fraudulent (0.3%) compared to genuine transactions (47%). Once again, from similar reasoning as above, instead of imputing a numerical value we replace the value of 0 with a null value.

# In[ ]:


X.loc[(X.oldBalanceOrig == 0) & (X.newBalanceOrig == 0) & (X.amount != 0),       ['oldBalanceOrig', 'newBalanceOrig']] = np.nan


# <a href='#top'>back to top</a>

# <a id = 'ML_after_imputation'> </a>
# ### 6. Machine Learning (after imputation, but not feature engineering)

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = randomState)

weights = (Y==0).sum()/(1.0 *  (Y==1).sum())
xgb_classifier2 = XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4)
xgb_prediction2 = xgb_classifier2.fit(X_train, Y_train).predict_proba(X_test)

print('AUPRC = {}'.format(average_precision_score(Y_test, xgb_prediction2[:,1])))


# <a id='ML2_features_imp'> </a>
# #### 6.1. What are the important features for the ML model?

# In[ ]:


fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(xgb_classifier2, height = 1, color = colours, grid = False,                      show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);


# <a href='#top'>back to top</a>

# <a id='feature'></a>
# ### 7. Feature-engineering

# Motivated by the possibility of zero-balances serving to differentiate between
# fraudulent and genuine transactions, we take the data-imputation of section <a href='#imputation'>3.1</a> a
# step further and create 2 new features (columns) recording errors in the 
# originating and
# destination accounts for each transaction. These new features turn out to be 
# important in obtaining the best performance from the ML algorithm that we will
# finally use.
# 
# Note : from tests, I could find that the model can achieve 99.85% accuracy only with this feature engineering. It doesn't require any impuatation. 

# In[ ]:


X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig
X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest


# In[ ]:


X


# <a href='#top'>back to top</a>

# <a id = 'ML'> </a>
# ### 8. Machine Learning (after imputation and feature engineering)

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = randomState)

weights = (Y==0).sum()/(1.0 *  (Y==1).sum())
xgb_classifier3 = XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4)
xgb_prediction3 = xgb_classifier3.fit(X_train, Y_train).predict_proba(X_test)

print('AUPRC = {}'.format(average_precision_score(Y_test, xgb_prediction3[:,1])))


# <a href='#top'>back to top</a>

# <a id='ML3_features_imp'> </a>
# #### 8.1. What are the important features for the ML model?
# The figure below shows that the new feature errorBalanceOrig that we created is the most relevant feature for the model. The features are ordered based on the number of samples affected by splits on those features.

# In[ ]:


fig = plt.figure(figsize = (14, 9))
ax = fig.add_subplot(111)

colours = plt.cm.Set1(np.linspace(0, 1, 9))

ax = plot_importance(xgb_classifier3, height = 1, color = colours, grid = False,                      show_values = False, importance_type = 'cover', ax = ax);
for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        
ax.set_xlabel('importance score', size = 16);
ax.set_ylabel('features', size = 16);
ax.set_yticklabels(ax.get_yticklabels(), size = 12);
ax.set_title('Ordering of features by importance to the model learnt', size = 20);


# <a href='#top'>back to top</a>
