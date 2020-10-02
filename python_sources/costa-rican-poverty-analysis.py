#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt


# In[ ]:


# To remove the limit on the number of rows displayed by pandas
pd.set_option("display.max_rows", None)

# Read csv files in pandas dataframe
testDf = pd.read_csv('../input/test.csv')
trainDf = pd.read_csv('../input/train.csv')
print("Training dataset basic information")
print("Rows: {}".format(len(trainDf)))
print("Columns: {}".format(len(trainDf.columns)))
trainDf.head()


# In[ ]:


print("Test dataset basic information")
print("Rows: {}".format(len(testDf)))
print("Columns: {}".format(len(testDf.columns)))
testDf.head()


# In[ ]:


# Add null Target column to test
testDf['Target'] = np.nan
data = trainDf.append(testDf, ignore_index = True)


# #### Exploratory Data Analysis (EDA)
# 1. Find missing values
# 2. Find outliers
# 3. Find incosistent values
# 4. Remove correlated features
# 5. Feature engineering
# 6. Feature scaling
# 7. Impute missing values

# ### 1. Find and fix missing feature values

# In[ ]:


# 1. Find missing values in training and test dataset
def findColumnsWithNan(df):
    cols = df.columns[df.isna().any()]
    print("Number of columns with Nan: {}".format(len(cols)))
    print("Column names: {}".format(cols))
    print("-" * 80)
    for col in cols:
        print("Column: [{}] missing {} values.".format(col, len(df[df[col].isna() == True])))

print("Analysis of training dataset...")
findColumnsWithNan(trainDf)


# In[ ]:


print()
print("Analysis of test dataset...")
findColumnsWithNan(testDf)


# #### 1.1. Fix missing values of v2a1 
# It means Monthly rent payment. To find what Nan means, v2a1 is compared with 'tipovivi' feature which gives information whether the house is rented/completely paid off etc. <br>
# 
# tipovivi1 =1 own and fully paid house <br>
# tipovivi2 =1 own,  paying in installments" <br>
# tipovivi3 =1 rented <br>
# tipovivi4 =1 precarious <br>
# tipovivi5 =1 other(assigned,  borrowed)" <br>
# 
# 'v2a1' is replaced with 0, wherever tipovivi1=1, and all other missing values are left which will be imputed later. This means that if the house is fully owned by household then they don't pay any rent.

# In[ ]:


data.loc[(data['tipovivi1'] == 1) & (data['v2a1'].isna()), 'v2a1'] = 0
print("Missing values after replacing: {}".format(len(data.loc[data['v2a1'].isna()])))


# #### 1.2. Fix missing values of v18q1
# It means number of tablets household owns. After careful analysis of household members, it can concluded that NaN means household does not own a tablet. We replace NaN with 0.

# In[ ]:


data.loc[data['v18q1'].isna(), 'v18q1'] = 0
print("Missing values after replacing: {}".format(len(data.loc[data['v18q1'].isna()])))


# #### 1.3 Fix missing values of rez_esc
# It means years behind in school. From the discussions on Kaggle, it can be concluded that this value is defined only for people whose age is between 7 and 19. So the missing values can be updated to 0 using this criteria. Age of an individual is in the column appropriately named 'age'.

# In[ ]:


data.loc[(data['age'] < 7) & (data['rez_esc'].isna()), 'rez_esc'] = 0
data.loc[(data['age'] > 19) & (data['rez_esc'].isna()), 'rez_esc'] = 0
print("Missing values after replacing: {}".format(len(data.loc[data['rez_esc'].isna()])))


# #### 1.4 Fix missing values of meaneduc
# It means average years of education for adults (18+). This implies that if the age of an individual is less than 18 and the value is NaN, then we can replace it with 0. Other NaN are left to be imputed.

# In[ ]:


data.loc[data['age'] < 19 & data['meaneduc'].isna(), 'meaneduc'] = 0
print("Missing values after replacing: {}".format(len(data.loc[data['meaneduc'].isna()])))


# #### 1.5 Fix missing values of SQBmeaned
# It means square of the mean years of education of adults (>=18) in the household. It is highly correlated with feature 'age' and there is no real need of it. Hence, this feature is dropped from the dataset.

# In[ ]:


data.drop('SQBmeaned', inplace=True, axis=1)
print("Total number of columns left: {}".format(len(data.columns)))


# ### 2. Find outliers
# Inter-quartile range(IQR) is used to identify outliers in the dataset. IQR is the difference between the 75th and 25th percentile of the data. It is measure of dispersion along the lines of standard deviation. During this analysis, features were found which had incosistent values like integer and boolean string (yes/no) mixed together. These needs to be removed and is the main focus of the next step.

# In[ ]:


for cols in data.columns[1:]:
    if cols in ['idhogar', 'dependency', 'edjefe', 'edjefa']:
        continue
    percentile75 = np.percentile(data[cols].fillna(0), 75)
    percentile25 = np.percentile(data[cols].fillna(0), 25)
    threshold = (percentile75 - percentile25) * 1.5
    lower, upper = (percentile25 - threshold), (percentile75 + threshold)
    # identify outliers
    outliers = data.loc[(data[cols] < lower) & (data[cols] > upper)]
    if len(outliers) > 0:
        print('Feature: {}. Identified outliers: {}'.format(cols, len(outliers)))


# ### 3. Find incosistent values
# When finding outliers, following three features ('dependency', 'edjefe', 'edjefa') were found to have incosistent values. To take care of this, 'yes' is replaced with 1 and 'no' is replaced with 0. Also, to make sure each value in the feature are of the same data type, features are converted to float.

# In[ ]:


for col in ['dependency', 'edjefe', 'edjefa']:
    data.loc[data[col] == 'yes', col] = 1.0
    data.loc[data[col] == 'no', col] = 0.0
    data[col] = pd.to_numeric(data[col])


# ### 4. Remove correlated features
# Highly correlated feature pairs are redundant, one of the pairs is selected to be removed.

# In[ ]:


corrMat = data.corr()
plt.figure(figsize=(30, 10))
sns.heatmap(corrMat.iloc[:10, :10])


# In[ ]:


def featuresToDrop(corrMatrix):
    """
    To remove correlated features, used this gem of a code from here:
    https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features
    """
    # Select upper triangle of correlation matrix
    upper = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.95
    return [column for column in upper.columns if any(upper[column] > 0.95)]

toDrop = featuresToDrop(corrMat)
data.drop(toDrop, inplace=True, axis=1)
print("Correlated features which are dropped: {}".format(toDrop))


# ### 5. Feature engineering
# Definition: "Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work."
# 
# 5.1 Add aggregated features (min, max, std, sum)
# 5.2 Add features per household
# 5.3 Explore features

# In[ ]:


features = list(data.drop(columns = ['Id', 'idhogar', 'Target']).columns)
aggDf = data.drop(columns='Target').groupby('idhogar').agg(['min', 'max', 'sum', 'count', 'std'])
# Rename the columns
new_col = []
for c in aggDf.columns.levels[0]:
    for stat in aggDf.columns.levels[1]:
        new_col.append('{}-{}'.format(c, stat))
        
aggDf.columns = new_col
toDrop = featuresToDrop(aggDf.corr())
aggDf.drop(toDrop, inplace=True, axis=1)
data = data.merge(aggDf, on='idhogar', how ='left')
print('Training feature shape: ', data.shape)


# In[ ]:


data['phones-per-capita'] = data['qmobilephone'] / data['tamviv']
data['tablets-per-capita'] = data['v18q1'] / data['tamviv']
data['rooms-per-capita'] = data['rooms'] / data['tamviv']
data['rent-per-capita'] = data['v2a1'] / data['tamviv']


# ### 6. Feature imputing and scaling
# Scaling means to transform data in such a way that they fit within a range say 0-100 or 0-1. We will be using min-max scaler to transform the feature to be in the range 0-1.

# In[ ]:


# Labels for training
trainTarget = np.array(list(data[data['Target'].notnull()]['Target'].astype(np.uint8)))
submission = data.loc[data['Target'].isnull(), 'Id'].to_frame()

# Extract the training data
trainData = data[data['Target'].notnull()].drop(columns = ['Id', 'idhogar', 'Target'])
testData = data[data['Target'].isnull()].drop(columns = ['Id', 'idhogar', 'Target'])

# Impute training and test data
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
trainData = imputer.fit_transform(trainData)
testData = imputer.transform(testData)

# Scale training and test data
scaler = MinMaxScaler()
trainData = scaler.fit_transform(trainData)
testData = scaler.transform(testData)


# ### 7. Machine Learning model

# In[ ]:


model = lightgbm.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)
kfold = 5
kf = StratifiedKFold(n_splits=kfold, shuffle=True)

predicts_result = []
for idx, (train_index, test_index) in enumerate(kf.split(trainData, trainTarget)):
    print("Fold: {}".format(idx))
    X_train, X_val = trainData[train_index], trainData[test_index]
    y_train, y_val = trainTarget[train_index], trainTarget[test_index]
    model.fit(X_train, y_train, verbose=100)
    predicts_result.append(model.predict(testData))
submission['Target'] = np.array(predicts_result).mean(axis=0).round().astype(int)
submission.to_csv('submission.csv', index=False)
print("Completed!")

