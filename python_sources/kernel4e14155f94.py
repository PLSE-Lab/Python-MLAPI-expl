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
#import data 

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_submission = pd.read_csv ('../input/sample_submission.csv')





# In[ ]:


df_train.head(5)


# In[ ]:


df_test.head(5)


# In[ ]:


#import some libraries
import gc
import os
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')


# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


pd.options.display.max_columns = 202
df_train.info(), df_test.info()


# Both the test and train set contain 200.000 cases. Columns:
# 
# - 1 ID column 
# - 1 target (train set) 
# - 200 features
# 

# In[ ]:


#There are no missing values in train and test set
df_train.isnull().sum().sum(), df_test.isnull().sum().sum()


# In[ ]:


df_train.describe()


# In[ ]:


df_train.skew(axis = 0, skipna = True) 


# In[ ]:


df_test.skew(axis = 0, skipna = True) 


# Skewness seems to be within -1 and + 1 for all columns, except target. 

# Great way to show skewness in one graph, thanks https://www.kaggle.com/gpreda/santander-eda-and-prediction#Model

# In[ ]:


columns_train = df_train.columns.values[2:202]
columns_test=df_test.columns.values[1:201]
plt.figure(figsize=(16,10))
plt.title("Distribution of skewness per column in train and test set")
sns.distplot(df_train[columns_train].skew(axis=0),color="red", kde=True,bins=120, label='df_train')
sns.distplot(df_test[columns_test].skew(axis=0),color="blue", kde=True,bins=120, label='df_test')
plt.legend()
plt.show()


# #Nice function to get a lot of info about the dataset
# import pandas_profiling as pp
# eda = pp.ProfileReport(df_train)
# display(eda)

# In[ ]:


df_train.kurtosis(axis = 0, skipna = True) 


# In[ ]:


df_test.kurtosis(axis = 0, skipna = True) 


# Kurtosis also seems to be within the -1, +1 range, except for target

# In[ ]:


columns_train = df_train.columns.values[2:202]
columns_test=df_test.columns.values[1:201]
plt.figure(figsize=(16,10))
plt.title("Distribution of kurtosis per column in train and test set")
sns.distplot(df_train[columns_train].kurtosis(axis=0),color="red", kde=True,bins=120, label='df_train')
sns.distplot(df_test[columns_test].kurtosis(axis=0),color="blue", kde=True,bins=120, label='df_test')
plt.legend()
plt.show()


# In[ ]:


#check correlations
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


features = df_train.columns.values[2:202]
corr_matrix = df_train[features].corr().abs()


sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack()
       .sort_values(ascending=False))


# In[ ]:


features = df_test.columns.values[1:201]
corr_matrix = df_test[features].corr().abs()


sol2 = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack()
       .sort_values(ascending=False))


# In[ ]:


sol.head(5)


# In[ ]:


sol.tail()


# In[ ]:


sol2.head(5)


# In[ ]:


sol2.tail(5)


# Correlation between the features is (very) low. Look at distribution of target variable

# In[ ]:


sns.countplot(df_train['target'])


# In[ ]:


df_train['target'].value_counts(normalize=True) * 100


# The classes of the target are very inbalanced. A model will always have at least 89,51% accuracy because that is what will happen if you always predict '0'. I will try different approaches to deal with these inbalanced classes like up-sampling the minority class or down-sampling the majority class. But first add some features. Since 'negative' things (overdraft, debts etc.) with regards to banks are seldom good, I will recode these into '0'. All positive values will become '1'. 

# In[ ]:


#make copy of dataframes 
df_copy_train = df_train.copy()
df_copy_test = df_test.copy()


# In[ ]:


#merge datasets for faster feature engineering 
df_combi = df_copy_train.append(df_copy_test,ignore_index=True)
df_combi.shape
df_combi.tail(5)


# In[ ]:


#drop ID_code and target
for columns in ['ID_code', 'target']:
    df_combi.drop(columns, axis=1, inplace=True)    


# In[ ]:


#recode all negative values to '0'
cols = df_combi.columns # list of all columns

df_combi[cols] = df_combi[cols].where(df_combi[cols] >= 0, 0)


# In[ ]:


#recode all positive values to 1
df_combi[cols] = df_combi[cols].where(df_combi[cols] < 0.0001, 1)


# In[ ]:


#add suffix to new boolean columns 
df_combi = df_combi.add_suffix('_bool')


# In[ ]:


#create one total score per row
df_combi['Sum'] = df_combi.sum(1)


# In[ ]:


# seaborn histogram
sns.distplot(df_combi['Sum'], hist=True, kde=False, 
             bins=int(180/5), color = 'blue',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Histogram of Sum')
plt.xlabel('Sum')
plt.ylabel('Number of records')


# In[ ]:


df_combi.tail(5)


# In[ ]:


## split back in train and test set
df_1= df_combi.iloc[0:200000] # first 200000 rows of the combi dataframe
df_2=df_combi.iloc[200000:400000]


# In[ ]:


df_1.shape, df_2.shape


# In[ ]:


#merge new columns with df_train and df_test 

df_train=pd.concat([df_train,df_1],axis=1).ffill()


# In[ ]:


df_train.head(3)


# In[ ]:


#reset index df_2 to match index of df_test (0 to 199999)
df_2.index = range(len(df_2))


# In[ ]:


df_test=pd.concat([df_test,df_2],axis=1).ffill()


# In[ ]:


df_test.head(4)


# In[ ]:


df_train['target'].corr(df_train['Sum'])


# Correlation between new feature Sum and target is still very low but higher than the highest correlation found between original features 

# In[ ]:


#run logistic regression first without standardization on training set 
features = df_train.columns.values[2:404]

X, y = df_train[features], df_train['target']



# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

lr=LogisticRegression()

#train model using training set 
lr.fit(X_train,y_train)


# In[ ]:


#make predictions on the test set 
y_pred = lr.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))


# In[ ]:


print( np.unique(y_pred))


# In[ ]:


uniqueValues, occurCount = np.unique(y_pred, return_counts=True)
 
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# * The accuracy of the model is 0.91 which is only slightly better than the 0.89 it would have if the model would always predict that the target is '0'. It predicts '1'correctly in 2.7% of the cases where in reality it is 10.04%. The AUC score is 0.63 which means that the model has a 63% chance to distinguish between the positive class and negative class, not very good. 

# In[ ]:


#define function for ROC curve 

from sklearn.metrics import roc_curve  

def plot_roc_curve(fpr, tpr):  
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# In[ ]:


#Plot ROC curve
#predict probabilities
y_pred = lr.predict_proba(X_test)  

#keep probabilities of positive class only 
y_pred = y_pred[:, 1]

#compute auc (again)
auc = roc_auc_score(y_test, y_pred)  

#get the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)  

#plot curve 
plot_roc_curve(fpr, tpr)  







# In[ ]:


# There are 401 features. What is best number? 
from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
# The "accuracy" scoring is proportional to the number of correct classifications
rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='roc_auc')
rfecv.fit(X, y)

print("Optimal number of features: %d" % rfecv.n_features_)
print('Selected features: %s' % list(X.columns[rfecv.support_]))

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (roc_auc)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[ ]:


#build model with standardized features
#standardize features
from sklearn.preprocessing import StandardScaler; 
X_train_std, X_test_std, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.3, 
                                                    random_state=42)


# In[ ]:


#Logistic Regression on standardized dataset
lr_std=LogisticRegression()
#train model using training set 
lr_std.fit(X_train_std,y_train)


# In[ ]:


#make predictions on the test set 
y_pred_std = lr_std.predict(X_test_std)
print(accuracy_score(y_test,y_pred_std))
print(roc_auc_score(y_test,y_pred_std))


# In[ ]:


uniqueValues, occurCount = np.unique(y_pred_std, return_counts=True)
 
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)


# In[ ]:


#print ROC curve 

#predict probabilities
y_pred_std = lr_std.predict_proba(X_test_std) 

# keep probabilities for the positive outcome only
y_pred_std = y_pred_std[:, 1]
#compute auc (again)
auc = roc_auc_score(y_test, y_pred_std)  

#get the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_std)  

#plot curve 
plot_roc_curve(fpr, tpr)  


# Standardizing the features did not improve the model. Maybe min-max scaler

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

X_train_min, X_test_min, y_train, y_test = train_test_split(MinMaxScaler().fit_transform(X), y, test_size=0.3, 
                                                    random_state=42)


# In[ ]:


#Logistic Regression on on min/max transformed dataset
lr_min=LogisticRegression()
#train model using training set 
lr_min.fit(X_train_min,y_train)


# In[ ]:


#make predictions on the test set 
y_pred_min = lr_min.predict(X_test_min)
print(accuracy_score(y_test,y_pred_min))
print(roc_auc_score(y_test,y_pred_min))


# In[ ]:


uniqueValues, occurCount = np.unique(y_pred_min, return_counts=True)
 
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)


# Min/max transformation didn't change anything either. Final attempt with normalizing the features

# In[ ]:


from sklearn.preprocessing import Normalizer

X_train_nor, X_test_nor, y_train, y_test = train_test_split(Normalizer().fit_transform(X), y, test_size=0.3, 
                                                    random_state=42)


# In[ ]:


#Logistic Regression on normalized dataset
lr_nor=LogisticRegression()
#train model using training set 
lr_nor.fit(X_train_nor,y_train)


# In[ ]:


#make predictions on the test set 
y_pred_nor = lr_nor.predict(X_test_nor)
print(accuracy_score(y_test,y_pred_nor))
print(roc_auc_score(y_test,y_pred_nor))


# In[ ]:


uniqueValues, occurCount = np.unique(y_pred_nor, return_counts=True)
 
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)


# Normalizing the features made the performance of the model even worse. It now only predicts '1' in 1,9% of the cases. Try upsampling the minority class next...

# In[ ]:


df_train['target'].value_counts()


# In[ ]:


from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df_train[df_train.target==0]
df_minority = df_train[df_train.target==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=179902,    # to match majority class
                                 random_state=123) 
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 


# In[ ]:


# Display new class counts
df_upsampled.target.value_counts()


# In[ ]:


#run logistic regression on upsampled training set (without feature transformation) 
features = df_upsampled.columns.values[2:202]

X, y = df_upsampled[features], df_upsampled['target']


# In[ ]:


#split in training and test set
X_train_up, X_test_up, y_train_up, y_test_up = train_test_split(X, y, test_size=0.3)

print (X_train_up.shape, y_train_up.shape)
print (X_test_up.shape, y_test_up.shape)


# In[ ]:


#Logistic Regression on upsampled dataset
lr_up=LogisticRegression()
#train model using training set 
lr_up.fit(X_train_up,y_train_up)


# In[ ]:


#make predictions on the test set 
y_pred_up = lr_up.predict(X_test_up)
print(accuracy_score(y_test_up,y_pred_up))
print(roc_auc_score(y_test_up,y_pred_up))


# In[ ]:


uniqueValues, occurCount = np.unique(y_pred_up, return_counts=True)
 
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)


# In[ ]:


#print ROC curve 

#predict probabilities
y_pred_up = lr_up.predict_proba(X_test) 

# keep probabilities for the positive outcome only
y_pred_up = y_pred_up[:, 1]
#compute auc (again)
auc = roc_auc_score(y_test, y_pred_up)  

#get the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_up)  

#plot curve 
plot_roc_curve(fpr, tpr)


# Up-sampling improved the AUC-score but took very long. Maybe down-sampling the majority class gives the same result. 

# In[ ]:


from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df_train[df_train.target==0]
df_minority = df_train[df_train.target==1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=True,     # sample with replacement
                                 n_samples=20098,    # to match minority class
                                 random_state=123) 
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])


# In[ ]:


df_downsampled.target.value_counts()


# In[ ]:


#run logistic regression on downsampled training set (without feature transformation) 
features = df_downsampled.columns.values[2:404]

X, y = df_downsampled[features], df_downsampled['target']


# In[ ]:


#split in training and test set
X_train_down, X_test_down, y_train_down, y_test_down = train_test_split(X, y, test_size=0.3)

print (X_train_down.shape, y_train_down.shape)
print (X_test_down.shape, y_test_down.shape)


# In[ ]:


#Logistic Regression on downsampled dataset
lr_down=LogisticRegression()
#train model using training set 
lr_down.fit(X_train_down,y_train_down)


# In[ ]:


#make predictions on the test set 
y_pred_down = lr_down.predict(X_test_down)
print(accuracy_score(y_test_down,y_pred_down))
print(roc_auc_score(y_test_down,y_pred_down))


# In[ ]:


uniqueValues, occurCount = np.unique(y_pred_down, return_counts=True)
 
print("Unique Values : " , uniqueValues)
print("Occurrence Count : ", occurCount)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test_down, y_pred_down)
print(confusion_matrix)


# In[ ]:


#print ROC curve 

#predict probabilities
y_pred_down = lr_down.predict_proba(X_test_down) 

# keep probabilities for the positive outcome only
y_pred_down = y_pred_down[:, 1]
#compute auc (again)
auc = roc_auc_score(y_test_down, y_pred_down)  

#get the ROC curve
fpr, tpr, thresholds = roc_curve(y_test_down, y_pred_down)  

#plot curve 
plot_roc_curve(fpr, tpr)


# Downsampled dataset gives almost the same AUC- and accuracy score as the upsampled one (but is much faster..). 

# In[ ]:


#make predictions on the original, non standardized, test set 
y_pred_down = lr_down.predict(X_test)
print(accuracy_score(y_test,y_pred_down))
print(roc_auc_score(y_test,y_pred_down))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_down)
print(confusion_matrix)


# The AUC score for the down-sampled dataset is almost the same as for the up-sampled one (but is a lot faster). The model now predicts '1' correct in 7.77% of the cases (in reality 10,04%)

# Now, let's build a more sophisticated Random Forest model. First with the original dataset, then with the downsampled one. 

# In[ ]:


#split in training and test set
from sklearn.model_selection import train_test_split

features = df_train.columns.values[2:202]

X, y = df_train[features], df_train['target']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y, test_size=0.3)

print (X_train_clf.shape, y_train_clf.shape)
print (X_test_clf.shape, y_test_clf.shape)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_clf= RandomForestClassifier(n_jobs=-1, n_estimators = 80, 
                                  max_depth = 10, min_samples_leaf = 10, min_samples_split = 10)


# In[ ]:


#fit model and make predictions 
model_clf.fit(X_train_clf,y_train_clf)


# In[ ]:


#make predictions on the test set 
y_pred_clf = model_clf.predict(X_test_clf)
print(accuracy_score(y_test_clf,y_pred_clf))
print(roc_auc_score(y_test_clf,y_pred_clf))


# In[ ]:


#make predictions on the original, non standardized, test set 
y_pred_clf = model_clf.predict(X_test)
print(accuracy_score(y_test,y_pred_clf))
print(roc_auc_score(y_test,y_pred_clf))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_clf)
print(confusion_matrix)


# In[ ]:


#print ROC curve 

#predict probabilities
y_pred_clf = model_clf.predict_proba(X_test) 

# keep probabilities for the positive outcome only
y_pred_clf = y_pred_clf[:, 1]
#compute auc (again)
auc = roc_auc_score(y_test, y_pred_clf)  

#get the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_clf)  

#plot curve 
plot_roc_curve(fpr, tpr)


# Back to square 1..RF Classifier only predicts '0'. 

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


params_to_test = {
    'n_estimators':[4,40,80],
    'max_depth':[5,8,15],
}

#here you can put any parameter you want at every run, like random_state or verbosity
rf_model = RandomForestClassifier(random_state=42)
#here you specify the CV parameters, number of folds, numberof cores to use...
grid_search = GridSearchCV(rf_model, param_grid=params_to_test, cv=10,scoring='roc_auc', n_jobs=-1)

grid_search.fit(X_train_clf, y_train_clf)

best_params = grid_search.best_params_ 


# In[ ]:


best_params


# In[ ]:


rf_best_model = RandomForestClassifier(max_depth=15, n_estimators=80, n_jobs=-1, 
                                       min_samples_leaf = 10, min_samples_split = 10)


# In[ ]:


rf_best_model.fit(X_train_clf,y_train_clf)


# In[ ]:


#make predictions on the test set 
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

y_pred_rf = rf_best_model.predict(X_test_clf)
print(accuracy_score(y_test_clf,y_pred_rf))
print(roc_auc_score(y_test_clf,y_pred_rf))

