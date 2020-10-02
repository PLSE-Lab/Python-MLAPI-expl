#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import chi2_contingency
from scipy.stats import chisquare
import scipy.stats as ss

import seaborn as sns

import math

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df = pd.read_csv('../input/mercedes-benz-greener-manufacturing/train.csv.zip')
print(df)
print(df.dtypes)

# select the float columns
df_float = df.select_dtypes(include=[np.float])
print(df_float.columns)
# select int columns
df_int = df.select_dtypes(include=[np.int])
print(df_int.columns)
# select object columns
df_int = df.select_dtypes(include=[object])
print(df_int.columns)


# -  Check for inconsistencies in the data
#  - 1) Check for Duplicates rows
#  - 2) Check for missing data
#  - 3) Check for outliers in 'y'
#  - 4) Check for categorical columns with low variance
#  - 5) Check for columns with duplicate values

# In[ ]:


df[df.duplicated(['ID'], keep=False)]


# In[ ]:


df.isnull().sum().sum()


# In[ ]:


boxplot = df.boxplot(column=['y'])


# In[ ]:


# For numerical values
columnLowStd = []
for col in df.columns:
    
    if df[col].dtype=='int64':
        
        if df[col].std() < 0.1:
            columnLowStd.append(col)
            
print(columnLowStd)


# In[ ]:


# Check frequency of variables in categorical columns
df["X0"].describe()


# In[ ]:


df["X1"].describe()


# In[ ]:


df["X2"].describe()


# In[ ]:


df["X3"].describe()


# In[ ]:


df["X4"].describe()


# In[ ]:


df["X5"].describe()


# In[ ]:


df["X6"].describe()


# In[ ]:


df["X8"].describe()


# In[ ]:


DuplicateColumns = []
    
for col1 in range(df.shape[1]):
       
     
    for col2 in range(col1 + 1, df.shape[1]):            
                  
        if df.iloc[:,col1].equals(df.iloc[:,col2]):
            
            DuplicateColumns.append(df.columns.values[col2])

print(DuplicateColumns)


# # Look for mathematical correlations between categorical values and also mixed columns
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
# - Look for categorical features correlated with each other (CramerV)
# - Look for features correlated with ID (CorrelationRatio)
# - Look for features correlated with y  (CorrelatioRatio)

# In[ ]:


data = df.copy()
data = data.drop(columns = ['ID'])
data = data.drop(columns = ["y"])

column_to_drop = DuplicateColumns + columnLowStd
data = data.drop(columns=column_to_drop)

data = data.applymap(str)

# Cramers V for categorical correlations
def cramers_v(x, y):
    x = np.array(x)
    y = np.array(y)
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


cramersv = pd.DataFrame(index=data.columns,columns=data.columns)
columns = data.columns

for i in range(0,len(columns)):
    for j in range(0,len(columns)):
        #print(data[columns[i]].tolist())
        u = cramers_v(data[columns[i]].tolist(),data[columns[j]].tolist())
        cramersv.loc[columns[i],columns[j]] = u
        
cramersv.fillna(value=np.nan,inplace=True)


# In[ ]:


plt.figure(figsize=(50,50))
sns.heatmap(cramersv)
plt.show()


# If we observe closely, cells X0,X1,X2 have a lot of correlations with the integer columns. Lets see individually

# In[ ]:


upper = cramersv.where(np.triu(np.ones(cramersv.shape),k=1).astype(np.bool))

print("X0",cramersv[cramersv['X0']>0.95]['X0'])
print("X1",cramersv[cramersv['X1']>0.95]['X1'])
print("X2",cramersv[cramersv['X2']>0.95]['X2'])
print("X3",cramersv[cramersv['X3']>0.9]['X3'])
print("X4",cramersv[cramersv['X4']>0.9]['X4'])
print("X5",cramersv[cramersv['X5']>0.9]['X5'])
print("X6",cramersv[cramersv['X6']>0.9]['X6'])
print("X8",cramersv[cramersv['X8']>0.9]['X8'])


# It can be assumed that the binary features are just encodings of categorical columns or the features are just strongly correlated or dependant on each other. In the notebook "My frustrated approach"(https://www.kaggle.com/robertoruiz/my-frustrated-approach) by Cro - Magnon where the suggestion was variables X0 to X8 are processes or configurations and the integer binary columns are car models. It could be that variables X0 and X8 are custom features of the cars and the remaining binary columns show whether the car has the feature or not. So we can deduce our results in two ways -:
# - 1) Which car takes the most inspection time (Binary column) or
# - 2) Which sub feature in a feature columns takes the most time(Categorical columns)
# 

# In[ ]:


# Lets check for ID
data = df.copy()
data = data.drop(columns = ["y"])

# From low Std and equal column values
data = data.drop(columns=column_to_drop)
data = data.applymap(str)

column_to_drop = DuplicateColumns + columnLowStd

data['ID'] = data['ID'].astype(int)


# For correlation between categorical and numerical values
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    #print(fcat)
    #print(type(fcat[0]))
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    measurements = np.array(measurements)
    
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        #print(cat_measures)
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
        
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

corrRatio = pd.DataFrame(index=['ID'],columns=data.columns)
columns = data.columns

for j in range(0,len(columns)):
    #print(df[columns[j]].tolist())
    u = correlation_ratio(data[columns[j]].tolist(),data['ID'].tolist())
    corrRatio.loc[:,columns[j]] = u
    
corrRatio.fillna(value=np.nan,inplace=True)


# In[ ]:


plt.figure(figsize=(70,20))
sns.heatmap(corrRatio)
plt.show()


# - Strong correlation with ID and X5. Lets check with boxplot

# In[ ]:


plt.figure(figsize=(20,20))
boxplot = sns.boxplot(x="X5",y='ID',data=df)
plt.show()


# - It could mean the testing for feature X5 in an ordered fashion. Now let's check for y

# In[ ]:


data = df.drop(columns=['ID'])

# From low Std and equal column values
data = data.drop(columns=column_to_drop)

data = data.applymap(str)
data['y'] = data['y'].astype(float)

# For correlation between categorical and numerical values
def correlation_ratio(categories, measurements):
    
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    measurements = np.array(measurements)
    
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
        
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta

corrRatio = pd.DataFrame(index=['y'],columns=data.columns)
columns = data.columns

for j in range(0,len(columns)):
    u = correlation_ratio(data[columns[j]].tolist(),data['y'].tolist())
    corrRatio.loc[:,columns[j]] = u
    
corrRatio.fillna(value=np.nan,inplace=True)


# In[ ]:


plt.figure(figsize=(70,20))
sns.heatmap(corrRatio)
plt.show()


# In[ ]:


upper = corrRatio.where(np.triu(np.ones(corrRatio.shape),k=1).astype(np.bool))
# Find index of feature columns with correlation smaller than 0.1
index_with_low_coor = [column for column in upper.columns if any(upper[column] < 0.1)]
print(index_with_low_coor)


# In[ ]:


# Between numerical values ID and y
print(df['ID'].corr(df['y']))


# - Among the categorical variables,we can see from the above a high correlation with X0 and its dependant features. This shows a strong correlation between X0 and its dependant features. Hence, we can assume that the dependant features would be enough to show impact of X0
# - Now lets implement the features in a Random Forest Regressor
# 

# In[ ]:


# Lets first train and test within the training set

# Dropping X4 and X3 due to low frequency of values
# X2('aj') and X5('t') contain variables in the test set which are not present in the training set. X5 is highly correlated with the ID
# which can be a suitable way to represent it in the model.Similar problem with X0('av')

columns_to_drop = columnLowStd + DuplicateColumns + ['X4','X2','X5','X0','X3'] + index_with_low_coor

data = df.copy()

# Remove only the extreme outlier
data = data[data['y']<250]

data = data.drop(columns=columns_to_drop)

X = data
X = X.drop(columns='y')

y = data['y']
y = y.values
y = y.reshape((len(y), 1))

# split into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=32)

columns  = X_train.columns

# Label encoding values
for c in columns:
    if X_train[c].dtype == 'object':
        le = LabelEncoder() 
        X_train[c] = le.fit_transform(X_train[c])
        X_test[c] = le.transform(X_test[c])
        
param_grid={
        'n_estimators': range(200,500,100), 
        'min_samples_leaf': range(10,20,10), 
        'min_samples_split': range(10,20,10),  
        'max_depth': range(40,60,10)       
    }
cv = KFold(n_splits=5, shuffle=True, random_state=40)
gridSearch = GridSearchCV(estimator=RandomForestRegressor(), scoring='r2',cv=cv,param_grid=param_grid)
result = gridSearch.fit(X_train, y_train)
print("Best Score " + str(result.best_score_)+" with paramter "+ str(result.best_params_))


# In[ ]:


# Taking values as per grid score
model = RandomForestRegressor(n_estimators=result.best_params_['n_estimators'],
                              min_samples_split=result.best_params_['min_samples_split'],
                              min_samples_leaf=result.best_params_['min_samples_leaf'], 
                              max_depth=result.best_params_['max_depth'])

model.fit(X_train, y_train)

pred = model.predict(X_test)
print('r2_score:',r2_score(pred,y_test))


# In[ ]:


plt.figure(figsize=(20,20))
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(40).plot(kind='barh')
print(feat_importances.nsmallest(90).index.to_list())

Low_importance_columns = feat_importances.nsmallest(90).index.to_list()

# Remove columns with least importance and train again


# Eliminate columns with low feature importance to check if score improves

# In[ ]:


columns_to_drop = columnLowStd + DuplicateColumns + ['X4','X2','X5','X0','X3'] + index_with_low_coor+ Low_importance_columns

data = df.copy()

# Remove only the extreme outlier
data = data[data['y']<250]

data = data.drop(columns=columns_to_drop)

X = data
X = X.drop(columns='y')

y = data['y']
y = y.values
y = y.reshape((len(y), 1))

# split into train and test sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=32)

columns  = X_train.columns

# Label encoding values
for c in columns:
    if X_train[c].dtype == 'object':
        le = LabelEncoder() 
        X_train[c] = le.fit_transform(X_train[c])
        X_test[c] = le.transform(X_test[c])
        
param_grid={
        'n_estimators': range(200,500,100), 
        'min_samples_leaf': range(10,20,10), 
        'min_samples_split': range(10,20,10),  
        'max_depth': range(40,60,10)       
    }
cv = KFold(n_splits=5, shuffle=True, random_state=40)
gridSearch = GridSearchCV(estimator=RandomForestRegressor(), scoring='r2',cv=cv,param_grid=param_grid)
result = gridSearch.fit(X_train, y_train)
print("Best Score " + str(result.best_score_)+" with paramter "+ str(result.best_params_))


# In[ ]:


# Taking values as per grid score
model = RandomForestRegressor(n_estimators=result.best_params_['n_estimators'],
                              min_samples_split=result.best_params_['min_samples_split'],
                              min_samples_leaf=result.best_params_['min_samples_leaf'], 
                              max_depth=result.best_params_['max_depth'])

model.fit(X_train, y_train)

pred = model.predict(X_test)
print('r2_score:',r2_score(pred,y_test))


# In[ ]:


plt.figure(figsize=(20,20))
feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(40).plot(kind='barh')


# - A high r2 score score during grid search did not correspond to a similar value in the test set. Lets plot the training curve and check for overfitting
# - https://www.dataquest.io/blog/learning-curves-machine-learning/

# In[ ]:


data = df.sample(frac =1,random_state=32)

columns_to_drop = columnLowStd + DuplicateColumns + ['X4','X2','X5','X0','X3'] + index_with_low_coor+ Low_importance_columns

# Remove only the extreme outlier
data = data[data['y']<250]

data = data.drop(columns=columns_to_drop)

X = data
X = X.drop(columns='y')

y = data['y']
y = y.values
y = y.reshape((len(y), 1))

X_train = X
y_train = y

# Label Encoding values
columns = X_train.columns
for c in columns:
        
    if X_train[c].dtype == 'object':
        le = LabelEncoder() 
        X_train[c] = le.fit_transform(X_train[c])
       
train_sizes = [100, 500, 2000, 3000, 3366]

train_sizes, train_scores, validation_scores = learning_curve(
estimator = RandomForestRegressor(n_estimators=result.best_params_['n_estimators'],
                              min_samples_split=result.best_params_['min_samples_split'],
                              min_samples_leaf=result.best_params_['min_samples_leaf'], 
                              max_depth=result.best_params_['max_depth']),
X = X_train,y = y_train, train_sizes = train_sizes, cv = 5)

train_scores_mean = train_scores.mean(axis = 1)
validation_scores_mean = validation_scores.mean(axis =1)

plt.figure(figsize=(20,20))
plt.plot(train_sizes, train_scores_mean, label = 'Training error')
plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
plt.ylabel('R2', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for a Random Forest Regression model', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(0,1)
plt.show()


# Above graph shows that the validation error decreases with increasing training size. The mean of the validation error seems to be close to 0.5 but decreases with increasing sample size. It could be due to the freqency of the categorical variables or any eliminations of columns done above. Let's try on the test set.

# In[ ]:


columns_to_drop = columnLowStd + DuplicateColumns + ['X4','X2','X5','X0','X3'] + index_with_low_coor+ Low_importance_columns

data = df.copy()

# Remove only the extreme outlier
data = data[data['y']<250]

data = data.drop(columns=columns_to_drop)

X = data
X = X.drop(columns='y')

y = data['y']
y = y.values
y = y.reshape((len(y), 1))

X_train = X
y_train = y

df_test = pd.read_csv('../input/mercedes-benz-greener-manufacturing/test.csv.zip')
data = df_test.copy()
data = data.drop(columns=columns_to_drop)

X_test = data

columns  = X_train.columns

for c in columns:
    if X_train[c].dtype == 'object':
        le = LabelEncoder() 
        X_train[c] = le.fit_transform(X_train[c])
        X_test[c] = le.transform(X_test[c])
        

model = RandomForestRegressor(n_estimators=result.best_params_['n_estimators'],
                              min_samples_split=result.best_params_['min_samples_split'],
                              min_samples_leaf=result.best_params_['min_samples_leaf'], 
                              max_depth=result.best_params_['max_depth'])
model.fit(X_train, y_train)

pred = model.predict(X_test)


# In[ ]:


output = pd.DataFrame()

output['ID'] = X_test['ID']
output['y']  = pred

output.to_csv("submissionsRandomForest.csv",index=False)


# - Lets look at boxplot of target variable, variable with highest importance and it's associated categorical variable

# In[ ]:


plt.figure(figsize=(20,20))
boxplot = sns.boxplot(x='X0',y='y',data=df,hue="X314")
plt.show()


# Rather than finding out which feature is important, another way of looking at it is, which sub feature is important. So from feature importance, we know X314 is important but at the same time we know that X314 is highly correlated with X0. So we can deduce the following
# 
# - 1) Car or feature X314 plays a major role on time spent on test bench
# - 2) Or car or feature not having subfeature y could play a major role on the time spent on the test bench

# In[ ]:


# Let me know if further improvement is needed. It's my first notebook in Kaggle. 
# I would like to receive suggestions if there is anything I missed or made a mistake

