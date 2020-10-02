#!/usr/bin/env python
# coding: utf-8

# ## Objective
# ##### to create a model for banks and automize the fraud detection among credit card users.
# All the columns in the dataset are output of PCA applied on original dataset. Hence all columns are numerical and standardized in given form. We also have no missing values in the dataset.

# ## Import libraries, read data and perform descriptive statistics

# In[ ]:


# import tensorflow library to use GPU into this code for faster processing
import tensorflow


# In[ ]:


# import the required libraries

import zipfile  #read the csv file from zip format without extracting it in drive, we save space
import numpy as np  #linear algebra computations and transformations
import pandas as pd  #read the dataframe and dataframe operations
import matplotlib.pyplot as plt  #visualization of data
import seaborn as sns  #visualization of data
import re  #support for regular expressions

# pd.set_option('display.max_columns', 500)  #set the default option to show all columns when we want to show data

import warnings
warnings.filterwarnings(action='ignore')

from scipy import stats


# In[ ]:


# read contents of zip file
zf = zipfile.ZipFile('creditcard.csv.zip')

# read the data from csv file into pandas dataframe
cc_fraud = pd.read_csv(zf.open('creditcard.csv'))


# ### Descriptive Analytics

# In[ ]:


# create a copy of original dataframe so as to avoid reading from drive again
df = cc_fraud.copy()
print(df.shape)
df.head()


# We have 2.84L records with 30 possible predictors and 1 predicted variable.

# In[ ]:


# We perform descriptive statistics to check the mean and std of each variable. 
df.describe()


# ### Check for class imbalance

# In[ ]:


print(df.Class.value_counts())
print(df.Class.value_counts(normalize=True))


# Predicted variable contains 284315 records of negative class (i.e. no fraud recorded) and 492 positive records (i.e. fraud recorded). There is high skewness with class distribution seen to be 99.82%:0.17%

# ## Exploratory Data Analysis

# #### Visualize the pair plot of given data

# In[ ]:


sns.pairplot(df)
plt.show()


# ##### Normality check of Time

# In[ ]:


print(f'skewness in Time column: {df.Time.skew():.2f}')
plt.subplots(figsize=(8,6))
# plt.subplot(121)
sns.distplot(df.Time)
# plt.subplot(122)
# stats.probplot(df.Time, plot=plt)
plt.xlabel('Time elapsed in seconds', fontsize=12)
plt.show()


# Bimodal indicates that there are lesser transactions at certain time of day.

# In[ ]:


sns.boxplot(df.Time)
plt.show()


# we can see that there are no outliers in Time feature, hence no treatment needed on Time column

# ##### hour wise transaction

# In[ ]:


bins = [0,25000,50000,75000,100000,125000,150000,175000]
time_bin = pd.cut(df.Time, bins, right=False)
df['Time_Bins'] = time_bin
time_table = pd.crosstab(index=df.Time_Bins, columns=df.Class)

df['Hour'] = df['Time'].apply(lambda x: int(np.ceil(float(x)/3600) % 24))+1
hour_table = df.pivot_table(values='Amount',index='Hour',columns='Class',aggfunc='count', margins=True)


# In[ ]:


time_table.head()
hour_table.sort_values(by=[1], ascending=False).head(10)


# In[ ]:


hour_table.plot(kind='bar', stacked=True, figsize=(8,6))
plt.xticks(np.arange(0,25), rotation=0)
plt.xlabel('Hour of the day', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)
plt.show()


# transactions are lesser at early hours of the day, and increases during the day

# In[ ]:


max_amount_0 = df[df.Class==0].groupby(by='Hour', observed=True).max()['Amount']
min_amount_0 = df[df.Class==0].groupby(by='Hour', observed=True).min()['Amount']
count_0 = df.Hour[df.Class==0].value_counts().sort_index()


# In[ ]:


max_amount_1 = df[df.Class==1].groupby(by='Hour', observed=True).max()['Amount']
min_amount_1 = df[df.Class==1].groupby(by='Hour', observed=True).min()['Amount']
count_1 = df.Hour[df.Class==1].value_counts().sort_index()


# In[ ]:


df_time = pd.DataFrame({('0','min_amount_0'):min_amount_0,
                        ('0','max_amount_0'):max_amount_0,
                        ('0','count_0'):count_0,
                        ('1','min_amount_1'):min_amount_1,
                        ('1','max_amount_1'):max_amount_1,
                        ('1','count_1'):count_1}, index=count_0.index)


# In[ ]:


df_time.sort_values(by=('1','count_1'), ascending=False)


# In[ ]:


df.drop(['Time_Bins','Hour'], axis=1, inplace=True)


# ##### Normality check of Amount

# In[ ]:


plt.subplots(figsize=(15,10))
plt.subplot(221)
sns.distplot(df.Amount)
plt.title('Distribution plot of Amount')
plt.subplot(222)
stats.probplot(df.Amount, plot=plt)

plt.subplot(223)
sns.distplot(np.log1p(df.Amount))
plt.title('Distribution plot of Amount after Log Transformation')
plt.subplot(224)
stats.probplot(np.log1p(df.Amount), plot=plt)

plt.show()


# ##### Outlier detection in Amount

# In[ ]:


np.quantile(a=df.Amount, q=[0.25,0.5,0.75])

LW = max(5.6 - (77.165-5.6), 0)
print('LW: ',LW)

UW = 77.165+(77.165-5.6)
print('UW: ',UW)


# In[ ]:


df1 = df[df.Amount <= 148]
df1.Class.value_counts()


# In[ ]:


df2 = df1[df1.Amount <= 55]


# In[ ]:


plt.subplots(figsize=(15,8))
plt.subplot(131)
sns.boxplot(df.Amount, orient='vertical')
plt.title('BoxPlot of original Amount Data')
plt.subplot(132)
sns.boxplot(df1.Amount, orient='vertical')
plt.title('BoxPlot of Amount Data after outlier handling')
plt.subplot(133)
sns.boxplot(df2.Amount, orient='vertical')
plt.title('BoxPlot after handling the ourliers in Data')
plt.show()


# We can see many outliers in Amount column, but we cannot remove them as higher values can mean that a fraud has happened. Also if we try to keep only fraud records and cap them then we change the meaning of transaction and disturb the pattern observed.

# We create a new dataset df1 with log transformation applied on Amount column

# In[ ]:


df1 = df.copy()
df1.Amount = np.log1p(df1.Amount)

df2 = df1[df1.Amount <= 8]


# In[ ]:


df2.Class.value_counts()


# In[ ]:


plt.subplots(figsize=(12,8))
plt.subplot(131)
sns.boxplot(df.Amount, orient='vertical')
plt.title('Given Amount Column')
plt.subplot(132)
sns.boxplot(df1.Amount, orient='vertical')
plt.title('Amount after Log Transformation')
plt.subplot(133)
sns.boxplot(df2.Amount, orient='vertical')
plt.title('Amount after outlier handling')
plt.show()


# In[ ]:


correlation_matrix = df.corr()
fig = plt.figure(figsize=(15,9))
sns.heatmap(correlation_matrix, vmax=0.8, square = True)
plt.show()


# Using correlation matrix we can understand that since the features given are from PCA, so these are uncorrelated and only correlation can be observed with Time and Amount columns which were not transformed with PCA earlier

# Since the data is highly imbalanced our first choice would be Random Forest. RF will not overfit the data

# In[ ]:


# import necessary modules
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score  #comparison metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor  #feature selection
from statsmodels.tools.tools import add_constant #feature selection computing VIF


# In[ ]:


ss = StandardScaler()
df2.Time = ss.fit_transform(np.array(df2.Time).reshape(-1,1))


# In[ ]:


X = add_constant(df2)
# X.drop(['Amount'], axis=1, inplace=True)
# X = pd.get_dummies(X)
X.drop('const', axis=1, inplace=True)
a = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)


# In[ ]:


b = pd.DataFrame(a, columns=['VIF'])


# In[ ]:


b.sort_values(by='VIF', ascending=False)


# In[ ]:


X = df2.drop(['Class'], axis=1)
y = df2.Class


# In[ ]:


X_train = X.sample(frac=0.8, random_state=10)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=10)


# In[ ]:


y_train.value_counts(normalize=True)


# In[ ]:


y_test.value_counts(normalize=True)


# ##### Random Forest Classifier

# Since it is skewed data we need to grow the tree to sufficient depth so as to learn the patterns in minority class records. More number of trees will help to reduce the overfit resulting from full/higher depth of trees.

# In[ ]:


get_ipython().run_cell_magic('time', '', "rf = RandomForestClassifier(max_depth=15, n_estimators=100, oob_score=True, class_weight='balanced_subsample')\nrf.fit(X_train, y_train)")


# In[ ]:


y_pred_rf = rf.predict(X_test)
y_pred_train = rf.predict(X_train)


# In[ ]:


print('Train Confusion matrix:\n', confusion_matrix(y_train, y_pred_train))
print('Test Confusion matrix:\n', confusion_matrix(y_test, y_pred_rf))
print('Classification Report:\n', classification_report(y_test, y_pred_rf))
print(f'\nROC Score: {roc_auc_score(y_test, y_pred_rf):.4f}')


# In[ ]:


print(pd.crosstab(y, y_pred_rf, rownames=['Actual'], colnames=['Predicted'], margins=True))


#     Random Forest
#     All data with Amount log transformed: ROC=0.9192, Recall=0.92, (413,79)
#     original Amount: ROC=0.9167, Recall=0.92, (410,82)
#     max_depth=10, n_estimators=50, ROC=0.9177, Recall=0.92, (411,81)
#     n_estimators=100, ROC=0.9197, Recall=0.91, (412,80)
#     ROC=0.9042, Recall=0.90, (76,18)

# To improve results we need to gather more data, mostly of minority class. Data generation and feature engineering usually has higher payoff in terms of time invested and improved performance. 
# 
# Oversampling, Undersampling, smote cannot be useful here as the positive class is only 0.17% and taking it up to 30-50% data means either adding lot of noise, redundant data or losing lots of information from negative class. This way the model will be forced to learn incorrect patterns in the data and the model cannot generalize.
# 
# We do not have scope for either hence we move towards the model hyperparameter tuning

# ##### Hyperparameter Tuning of RF

# In[ ]:


max_depth = [int(x) for x in np.linspace(5,20,4)]
# max_features = ['auto', 'sqrt']
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]

random_grid = {'max_depth': max_depth}


# In[ ]:


get_ipython().run_cell_magic('time', '', "rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')\nrandom_rf = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=10, cv=5, n_jobs=-1, scoring='roc_auc')")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'random_rf.fit(X,y)')


# In[ ]:


print(f'grid best params: {random_rf.best_params_}')
print(f'grid best: {random_rf.best_score_}')


# ##### XGBoost Classifier

# In[ ]:


get_ipython().run_cell_magic('time', '', "param = {'max_depth':[13,15,18]\n         }\nxgb = XGBClassifier(subsample=0.7, colsample_bytree=0.8)\ngrid_xgb = GridSearchCV(estimator=xgb, param_grid=param, scoring='roc_auc', n_jobs=4, cv=5)\ngrid_xgb.fit(X,y)")


# In[ ]:


print(f'grid best params: {grid_xgb.best_params_}')
print(f'grid best: {grid_xgb.best_score_}')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'xgb=XGBClassifier(max_depth=18, subsample=0.7, scale_pos_weight=1, colsample_bytree=0.8)\nxgb.fit(X_train,y_train)\ny_pred_xgb = xgb.predict(X_test)\ny_pred_train = xgb.predict(X_train)')


# In[ ]:


print('Train Confusion matrix:\n', confusion_matrix(y_train, y_pred_train))
print('Test Confusion matrix:\n', confusion_matrix(y_test, y_pred_xgb))
print('Classification Report:\n', classification_report(y_test, y_pred_xgb))
print(f'\nROC Score: {roc_auc_score(y_test, y_pred_xgb):.4f}')


#     default parameters: ROC=0.5, FN=492
#     after gridsearchcv: ROC=0.9329, Recall=0.93, max_depth=5, colsample_bytree=0.8, scale_pos_weight = 1, subsample=0.5 (426,66)
#     ROC=0.9360, Recall=0.94, max_depth=5, colsample_bytree=0.8, subsample=0.7, (429,63)
#     ROC=0.9553, Recall=0.96, max_depth=15, (448,44)
#     ROC=0.9096, Recall=0.91, max_depth=18, (77,17)

# ##### Precision Recall Curve and ROC Curve

# In[ ]:


rf_p,rf_r,rf_t = precision_recall_curve(y_test,y_pred_rf)
rf_fpr,rf_tpr,rf_thr = roc_curve(y_test,y_pred_rf)

xgb_p,xgb_r,xgb_t = precision_recall_curve(y_test,y_pred_xgb)
xgb_fpr,xgb_tpr,xgb_thr = roc_curve(y_test,y_pred_xgb)


# In[ ]:


plt.figure(figsize=(18,8))
plt.subplot(121)
plt.title('ROC Curve', fontsize=16)
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred_rf)))
plt.plot(xgb_fpr, xgb_tpr, label='XGBoost Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred_xgb)))
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([-0.01, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
            arrowprops=dict(facecolor='#6E726D', shrink=0.05))
plt.legend(loc='lower right', fontsize=12)

plt.subplot(122)
plt.title('Precision Recall Curve', fontsize=16)
plt.plot(rf_r, rf_p, label='Random Forest Classifier Score: {:.4f}'.format(average_precision_score(y_test, y_pred_rf)))
plt.plot(xgb_r, xgb_p, label='XGBoost Classifier Score: {:.4f}'.format(average_precision_score(y_test, y_pred_xgb)))
plt.axis([0, 1.01, 0, 1.01])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.legend(fontsize=12)

plt.show()


# Cost sensitive learning - We assign costs of FN and FP misclassifications and try to reduce this cost
