import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


PATH="../input/"
os.listdir(PATH)

train_df = pd.read_csv(PATH+"train.csv")
test_df = pd.read_csv(PATH+"test.csv")




print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))
print("Test:  rows:{} cols:{}".format(test_df.shape[0], test_df.shape[1]))


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return np.transpose(pd.concat([total, percent], axis=1, keys=['Total', 'Percent']))
    
    
missing_data(train_df)
missing_data(test_df)

train_df.info()
train_df.describe()


categorical_columns = ['waterfront', 'view', 'condition', 'grade']
i = 0
plt.figure()
fig, ax = plt.subplots(2,2,figsize=(16,10))
for col in categorical_columns:
    i += 1
    plt.subplot(2,2,i)
    sns.boxplot(x=train_df[col],y=train_df['price'])
    plt.xlabel(col, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();

numerical_columns = ['bedrooms', 'bathrooms', 'floors']
i = 0
plt.figure()
fig, ax = plt.subplots(1,3,figsize=(18,4))
for col in numerical_columns:
    i += 1
    plt.subplot(1,3,i)
    sns.boxplot(x=train_df[col],y=train_df['price'])
    plt.xlabel(col, fontsize=8)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=8)
plt.show();


area_columns = ['sqft_living','sqft_lot','sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15']

i = 0
plt.figure()
fig, ax = plt.subplots(3,2,figsize=(16,15))
for col in area_columns:
    i += 1
    plt.subplot(3,2,i)
    plt.scatter(x=train_df[col],y=train_df['price'],c='magenta', alpha=0.2)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('price', fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();

geo_columns = ['lat','long']

i = 0
plt.figure()
fig, ax = plt.subplots(1,2,figsize=(16,6))
for col in geo_columns:
    i += 1
    plt.subplot(1,2,i)
    plt.scatter(x=train_df[col],y=train_df['price'], c=train_df['zipcode'], alpha=0.2)
    plt.xlabel(col, fontsize=10)
    plt.ylabel('price', fontsize=10)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=10)
plt.show();

geo_columns = ['zipcode']
plt.figure()
fig, ax = plt.subplots(1,1,figsize=(18,4))
plt.subplot(1,1,1)
sns.boxplot(x=train_df['zipcode'],y=train_df['price'])
plt.xlabel(col, fontsize=8)
locs, labels = plt.xticks()
plt.tick_params(axis='x', labelsize=8, rotation=90)
plt.show();


features = ['bedrooms','bathrooms','floors',
            'waterfront','view','condition','grade',
            'sqft_living','sqft_lot','sqft_above','sqft_basement','sqft_living15','sqft_lot15',
            'yr_built','yr_renovated',
            'lat', 'long','zipcode', 
            'price']

mask = np.zeros_like(train_df[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(train_df[features].corr(),linewidths=0.25,vmax=1.0,square=True,cmap="Blues", 
            linecolor='w',annot=True,mask=mask,cbar_kws={"shrink": .75});
            
#We are using 80-20 split for train-test
VALID_SIZE = 0.2
#We also use random state for reproducibility
RANDOM_STATE = 2019

train, valid = train_test_split(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )

predictors = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15']
target = 'price'

train_X = train[predictors]
train_Y = train[target].values
valid_X = valid[predictors]
valid_Y = valid[target].values


RFC_METRIC = 'mse'  #metric used for RandomForrestClassifier
NUM_ESTIMATORS = 100 #number of estimators used for RandomForrestClassifier
NO_JOBS = 4 #number of parallel jobs used for RandomForrestClassifier


model = RandomForestRegressor(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)

model.fit(train_X, train_Y)

preds = model.predict(valid_X)

def plot_feature_importance():
    tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': model.feature_importances_})
    tmp = tmp.sort_values(by='Feature importance',ascending=False)
    plt.figure(figsize = (7,4))
    plt.title('Features importance',fontsize=14)
    s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()  
    
plot_feature_importance()
print("RF Model score: ", model.score(train_X, train_Y))


def rmse(preds, y):
    return np.sqrt(mean_squared_error(preds, y))
print("Root mean squared error (valid set):",round(rmse(preds, valid_Y),2))


test_X = test_df[predictors] 
test_preds = model.predict(test_X)

submission = pd.read_csv(PATH+"sample_submission.csv")
submission['price'] = test_preds
submission.to_csv('submission.csv', index=False)

    