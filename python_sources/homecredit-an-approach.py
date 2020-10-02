#!/usr/bin/env python
# coding: utf-8

# # HomeCredit Default Risk

# ![](https://www.homecredit.co.id/HCID/media/images/HCID_logo.jpg)

# Due to the large amount of data, I've imported all the tables into my local SQL Server instance and joined them into a view according to the data schema provided. That's not obviously possible here at Kaggle, so I've just used a single table to show you, how I would proceed in treating an imbalance dataset with a hefty amount of attribures. Thus, do not expect the model accuracy to be anywhere beyond random - this is only a methodical approache.
# 
# What you'll essentialy see here is an object encoding followed by PCA reduction.

# ### Steps to solve the problem

# 1. Load libraries
# 2. Load data
# 3. EDA
# 4. Data preparation
#     * Anomaly detection
#     * Outlier detection
#     * Null handling
# 5. Object encoding
#     * Option 1 - Dummy variables
#     * Option 2 - OneHotEncoding
#     * Standartization
#     * PCA
# 6. Model Baseline
# 7. Model Evaluation
#     * Confusion Matrix
#     * F-1 Score
#     * Precision and Recall
#     * Precision-Recall Curve
#     * ROC Curve
#     * AUROC Score

# ## LOAD LIBRARIES

# In[ ]:


# Basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Common tools
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter
'''
# Advanced visualization
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
'''
# Model
from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import VotingClassifier

# Configure Defaults
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_colwidth', -1)


# ## LOAD DATA 

# I'll load just very basic files here. We're not going to do data merge as mentioned above.

# In[ ]:


# List all files
print(os.listdir('../input/'))


# In[ ]:


train = pd.read_csv('../input/application_train.csv')


# In[ ]:


train.shape


# In[ ]:


test = pd.read_csv('../input/application_test.csv')


# In[ ]:


test.shape


# In[ ]:


sample_submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


sample_submission.shape


# ### Target distribution 

# In[ ]:


sns.countplot(x='TARGET', data=train)
print(train.TARGET.sum()/train.TARGET.count())


# The data-set is cleary imbalanced. We'll have that in mind.

# ## Basic EDA 

# The number of features is medium high, so we'll do a general glance of the data and will not dig deep into invidual features.

# ### General overview

# In[ ]:


train.info()


# In[ ]:


train.head()


# In[ ]:


# Number of categories within a categorical feature
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# There's possibly 41 numerical categories and 16 categorial features which could be treat as nominal. Given the size of the dataset, we'll probably employ OHE encoding, followed by PCA reduction.

# In[ ]:


# Missing values
n = train.isnull().sum() / len(train)
n.sort_values(ascending=False).head(10)


# There's a lot of missing values within numerical features reaching 70%. We can drop them or use some model which can handle Nulls like XGBoost.

# In[ ]:


# Number of features exceeding 1/6 of missing values.
sum(i>0.1667 for i in n)


# Almost half of the dataset contains missing values exceeding 1/6 of a feature. Numerical and object categories could be imputed with a default value. Float features < 16% imputed with a mean.

# ## DATA WRANGLING

# ### Anomaly detection

# In[ ]:


# Age
(train['DAYS_BIRTH'] / -365).describe()


# This feature looks good. Loans are provided to people within an age range 20-70 years.

# #### Days employed

# In[ ]:


(train['DAYS_EMPLOYED'] / 365).describe()


# Maximal employed time over 1,000 years doesn't look valid. High std confirms there's a huge dispercy in data.

# In[ ]:


train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');


# The value is quite frequently used for some reason, possible as a kind of a default value.

# In[ ]:


# Out of curiosity...
anom = train[train['DAYS_EMPLOYED'] == 365243]
non_anom = train[train['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))


#  We will feature engineer the column now here before we'll do outlier removal and dataset merge.

# In[ ]:


# Create new feature
train['DAYS_EMPLOYED_BOOL'] = train['DAYS_EMPLOYED'] == 365243
train['DAYS_EMPLOYED'].replace({365243:np.nan}, inplace=True)


# In[ ]:


# Check orginal feature
train['DAYS_EMPLOYED'].plot.hist()


# ### Outlier removal  

# Money-related features would be are best guess, but let's iterate over the whole dataset as due to the Tukey method.

# In[ ]:


def detect_outliers(df,n,features):
    
    outlier_indices = []
    
    for col in features:
        Q1 = df[col].quantile(0.02)
        Q3 = df[col].quantile(0.98)
        IQR = Q3 - Q1
        
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR )].index
        outlier_indices.extend(outliers)
        
    # Select observations with more than n outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   


# In[ ]:


# Return only float64 features
numerical_feature_mask = train.dtypes==float
numerical_cols = train.columns[numerical_feature_mask].tolist()


# In[ ]:


# Detect outliers
Outliers_to_drop = detect_outliers(train,2,numerical_cols)


# In[ ]:


# Number of outliers to drop
len(Outliers_to_drop)


# That's an acceptable number.

# In[ ]:


# Remove outliers
train.drop(Outliers_to_drop, inplace=True)


# ### Concat data 

# Create a single dataframe for conviniet feature engineering and futher handling.

# In[ ]:


# Save Id for the submission at the very end.
Id = test['SK_ID_CURR']


# In[ ]:


# Get marker
split = len(train)


# In[ ]:


# Merge into one dataset
data =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# In[ ]:


# We don't need the Id anymore now.
data.drop('SK_ID_CURR', axis=1, inplace=True)


# In[ ]:


data.shape


# ### Handle nulls 

# Alternatively, we can calculate std and randomely generate a number within, which would allow us to set a higher threshold and potentially spare some features.

# In[ ]:


# Remove mostly sparse features
for f in data:
   if data[f].isnull().sum() / data.shape[0] >= 0.5: del data[f] # Or do a boolean flag here


# In[ ]:


# Check for TARGET data type
data['TARGET'].dtype


# In[ ]:


# Select columns due to theirs data type
float_col = data.select_dtypes('float').drop(['TARGET'], axis=1)
int_col = data.select_dtypes('int')
object_col = data.select_dtypes('object')


# In[ ]:


# Remove and impute numerical features
for f in float_col:
   if data[f].isnull().sum() / data.shape[0] > 0.1667: del data[f] # Remove 1/6+ of NANs
   else: data[f] = data[f].fillna(data[f].mean()) # Impute others with a mean value


# In[ ]:


# Impute default value into a numerical category
for i in int_col:
   data[i] = data[i].fillna(-1)


# In[ ]:


# Impute object type with a default
for o in object_col:
   data[o] = data[o].fillna('Unknown')


# In[ ]:


# Check
data.isnull().sum().sort_values(ascending=False).head(5)


# Only the target in test dataset should be left with NANs.

# ### Obejct encoding

# One way here would be to use LabelEncoder followed by OHE or to use DictVectorizer which both supports spare matrix output for high model training performance. Another possibility is to use dummy variables which is probably the most easist way of object encoding that I'll prefere here as the number of features isn't yet that huge.

# In[ ]:


data = pd.get_dummies(data, prefix_sep='_', drop_first=True) # Drop originall feature to avoid multi-collinearity


# ### Object encoding (the other way)

# In[ ]:


'''# Categorical mask
categorical_feature_mask = train.dtypes==object
# Get categorical columns
categorical_cols = train.columns[categorical_feature_mask].tolist()'''


# In[ ]:


'''# Instantiate LE
le = LabelEncoder()'''


# In[ ]:


'''# Apply LE
train[categorical_cols] = train[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))'''


# In[ ]:


'''# Check
train[categorical_cols].head(10)'''


# In[ ]:


'''# Instantiate OHE
ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse=False ) #Can be enabled True for higher preformance'''


# In[ ]:


'''# Apply OHE
train_ohe = ohe.fit_transform(train) #an numpy array'''


# ### Standardize data 

# In order to perform PCA, we need to standardize our data.

# In[ ]:


#Split data
train_c = data[:split]
test_c = data[split:].drop(['TARGET'], axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split

# Get variables for a model
x = train_c.drop(["TARGET"], axis=1)
y = train_c["TARGET"]

#Do train data splitting
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.22, random_state=101)


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train) # Fit on training set only.

# Apply transform to both the training set and the test set.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(.95)
pca.fit(X_train)


# In[ ]:


X_train = pca.transform(X_train)
X_test = pca.transform(X_test)


# ## Model 

# ### Baseline

# I've choise following algorithm because it is fast.

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver = 'lbfgs').fit(X_train, y_train)
pred = lr.predict(X_test)
acc = lr.score(X_test, y_test)

print("Accuracy: ", acc)


# ## Model Evaluation  

# ### Confusion Matrix 

# In[ ]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

predicts = cross_val_predict(lr, X_train, y_train, cv=3)
confusion_matrix(y_train, predicts)


# In the first row, 218718 clients were correctly predicted as not having a payment difficulties (true negatives) and 17 were wrongly classified as not having a payment difficulty (false negative).
# In the second row 19192 clients were wrongly classified as having a payment difficulty (false possitive) and 17 of them correcly classified as having a payment difficulty (true negative).

# ### Precision and Recall 

# In[ ]:


from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_train, predicts))
print("Recall:",recall_score(y_train, predicts))


# The precission tells a probability with a client will be classified correctly. The recall tells us that it predicted a payment difficulty of 73 % of the clients who actually had the payment difficulty.

# ### F-1 Score

# In[ ]:


from sklearn.metrics import f1_score

f1_score(y_train, predicts)


# The F-score is computed as the harmonic mean of both precision and recall, thus high F1 score is possible only if both precesion and recall are high.

# ### Precision Recall Curve

# In[ ]:


from sklearn.metrics import precision_recall_curve

y_scores = lr.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train, y_scores)

def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()


# Based on rapid recall curve fall is possible to set recall/precision trade-off before.

# ### ROC Curve 

# In[ ]:


from sklearn.metrics import roc_curve

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()


# This is probably the most important measure which is shared among most model-predicting technologies.
# The more is the blue curve leaning towards upper-left corned (right angle), the better a model is in predicting actual results. 

# ### ROC Score

# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


y_scores = lr.predict_proba(X_train)
y_scores = y_scores[:,1]


# In[ ]:


auroc = roc_auc_score(y_train, y_scores)
print("ROC-AUC Score:", auroc)

