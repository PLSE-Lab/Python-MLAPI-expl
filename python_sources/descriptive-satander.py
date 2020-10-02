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


# In[ ]:


#sample = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


#sample.head()


# In[ ]:


#len(sample)


# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train.shape


# In[ ]:


train.head()


# In[ ]:


# train columns
train.columns


# In[ ]:


# type of variables
train.dtypes


# In[ ]:


# summary statistics of all variables
train.describe()


# In[ ]:


# count of missing values
train.isnull().sum()


# In[ ]:


#  target count plot
import seaborn as sns
sns.countplot(train['target'])

 # dist plot
sns.distplot(train['target'], bins=50, kde=False)


# In[ ]:


# KDE plot
sns.kdeplot(train['target'])


# In[ ]:


# dist plot
sns.distplot(train['target'], bins=10, kde=True)


# In[ ]:


# log-transform the target
train['log_target'] = np.log(train.target)


# In[ ]:


# plot the new column
sns.distplot(train['log_target'], bins=10, kde=False)


# In[ ]:


# drop the new column
train.drop('log_target',axis=1,inplace=True)


# In[ ]:


train.columns


# In[ ]:


# pairplots
sns.pairplot(train[['target','48df886f9','0deb4b6a8']])


# In[ ]:


# Heat map
sns.heatmap(
    train.loc[:, ['target','48df886f9','0deb4b6a8','34b15f335','a8cb14b00']].corr(),
    annot=True
)


# In[ ]:


train.drop('ID',axis=1,inplace=True)


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


# outcome
y = train.target


# In[ ]:


# predictors
X = train.drop('target',axis=1)


# In[ ]:


X.shape


# In[ ]:


# split to training and testing datasets
from sklearn.model_selection import train_test_split
train_X,  test_X, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


train_X.shape


# In[ ]:


test_X.shape


# In[ ]:


# # Feature Scaling
# standardize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#train_X = sc.fit_transform(train_X)
#test_X = sc.transform(test_X)
# Normalize the data
#from sklearn.preprocessing import Normalizer
#sc = Normalizer()
#train_X = sc.fit_transform(train_X)
#test_X = sc.transform(test_X)


# In[ ]:


train_X.shape


# In[ ]:


train_df = pd.DataFrame(train_X,columns=X.columns)


# In[ ]:


train_df.columns


# In[ ]:


# pairplots
import seaborn as sns
sns.pairplot(train_df[['48df886f9','0deb4b6a8','34b15f335','a8cb14b00']])


# In[ ]:


# Heat map
#sns.heatmap(
#    train_df.loc[:, train_df.columns].corr(),
#    annot=True
#)


# In[ ]:


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
#train_X = pca.fit_transform(train_X)
#test_X = pca.transform(test_X)
#explained_variance = pca.explained_variance_ratio_


# In[ ]:


#train1 = pd.DataFrame(train_X)


# In[ ]:


#train1.shape


# In[ ]:


# Heat map
#import seaborn as sns
#sns.heatmap(
#    train1.loc[:, train1.columns].corr(),
#    annot=True
#)


# In[ ]:


#train_y1 = pd.DataFrame(train_y)


# In[ ]:


#train_y1.head()


# In[ ]:



#train2 = pd.concat([train_y1,train1],join='outer',axis=1)


# In[ ]:


#train2.head()


# In[ ]:


#train2.columns = ['target','a','b','c','d','e']


# In[ ]:


#train2.head()


# In[ ]:


# Heat map
import seaborn as sns
#sns.heatmap(
#    train2.loc[:, train2.columns].corr(),
#    annot=True
#)


# In[ ]:


# feature importance
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor()
regressor.fit(train_X, train_y)
names = train_X
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), regressor.feature_importances_), names), 
             reverse=True))


# In[ ]:


# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
#kpca = KernelPCA(n_components = 5, kernel = 'rbf')
#train_X = kpca.fit_transform(train_X)
#test_X = kpca.transform(test_X)


# In[ ]:


#train1 = pd.DataFrame(train_X)


# In[ ]:


# Heat map
import seaborn as sns
#sns.heatmap(
#    train1.loc[:, train1.columns].corr(),
#    annot=True
#)


# In[ ]:


#train2 = pd.concat([train_y1,train1],join='outer',axis=1)


# In[ ]:


#train2.columns = ['target','a','b','c','d','e']


# In[ ]:


#train2.head()


# In[ ]:


# Heat map
import seaborn as sns
#sns.heatmap(
#    train2.loc[:, train2.columns].corr(),
#    annot=True
#)


# In[ ]:


# Import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, Dropout


# In[ ]:


# fit the model
# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(output_dim = 128,  activation = 'relu', input_dim = 4991))

# Adding the output layer
regressor.add(Dense(1))
inp = Input(shape=(4991,))
hidden_1 = Dense(128, activation='relu')(inp)
dropout_1 = Dropout(0.2)(hidden_1)
hidden_2 = Dense(128, activation='relu')(dropout_1)
dropout_2 = Dropout(0.2)(hidden_2)
hidden_3 = Dense(128, activation='relu')(dropout_2)
dropout_3 = Dropout(0.2)(hidden_3)
hidden_4 = Dense(128, activation='relu')(dropout_3)
dropout_4 = Dropout(0.2)(hidden_4)
hidden_5 = Dense(128, activation='relu')(dropout_4)
dropout_5 = Dropout(0.2)(hidden_5)
out = Dense(1)(dropout_5)

regressor = Model(inputs=inp, outputs=out)


# In[ ]:


# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mse', metrics=['mae'])


# In[ ]:


# Fitting the ANN to the Training set
regressor.fit(train_X, train_y, epochs = 100, batch_size = 32, verbose=1, validation_split=0.1)


# In[ ]:


# make predictions - Neural Network
from sklearn.metrics import mean_absolute_error
preds = regressor.predict(test_X)
print("mean_absolute_error : " + str(mean_absolute_error(preds, test_y)))


# In[ ]:


# fit a model - GBoost
from sklearn.ensemble import GradientBoostingRegressor
regressor_gb = GradientBoostingRegressor()
regressor_gb.fit(train_X,train_y)


# In[ ]:


# make predictions - GBoost
preds = regressor_gb.predict(test_X)
from sklearn.metrics import mean_absolute_error
print("mean_absolute_error : " + str(mean_absolute_error(preds, test_y)))


# In[ ]:


# Cross-validation - GBoost
from sklearn.model_selection import cross_val_score, cross_val_predict
preds = cross_val_predict(regressor_gb, test_X, test_y, cv=10)
#scores 


# In[ ]:


#average error
print("mean_absolute_error : " + str(mean_absolute_error(preds, test_y)))


# In[ ]:


# fit a model - XGBoost
from xgboost import XGBRegressor
regressor_xgb = XGBRegressor()
regressor_xgb.fit(train_X,train_y)


# In[ ]:


# make predictions - XGBoost
from sklearn.metrics import mean_absolute_error
preds = regressor_xgb.predict(test_X)
print("mean_absolute_error : " + str(mean_absolute_error(preds, test_y)))


# In[ ]:


# Cross-validation - XGBoost
from sklearn.model_selection import cross_val_score, cross_val_predict
preds = cross_val_predict(regressor_xgb, test_X, test_y, cv=10)
#scores 


# In[ ]:


#average error
print("mean_absolute_error : " + str(mean_absolute_error(preds, test_y)))


# In[ ]:


# fit a Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor()
regressor_rf.fit(train_X,train_y)


# In[ ]:


# make predictions - Random Forets
from sklearn.metrics import mean_absolute_error
preds = regressor_rf.predict(test_X)
print("mean_absolute_error : " + str(mean_absolute_error(preds, test_y)))


# In[ ]:


# Cross-validation scores - Random forests
from sklearn.model_selection import cross_val_score, cross_val_predict
#scores_rf = cross_val_predict(regressor_rf, train_X, train_y, cv=10)
#scores_rf = cross_val_score(regressor_rf, X, y, cv=10)
#scores 


# In[ ]:


# cross-validation average scores
#scores_rf.mean()


# In[ ]:


# Cross-validation predictions - Random forests
from sklearn.model_selection import cross_val_score, cross_val_predict
#scores_rf = cross_val_predict(regressor_rf, train_X, train_y, cv=10)
#scores_rf_pred = cross_val_predict(regressor_rf, X, y, cv=10)
#scores 


# In[ ]:


#average error
#print("mean_absolute_error : " + str(mean_absolute_error(scores_rf_pred, y)))


# In[ ]:


# import the test dataset
test = pd.read_csv("../input/test.csv")


# In[ ]:


test.shape


# In[ ]:


test.columns


# In[ ]:


test1 = test.drop('ID',axis=1)


# In[ ]:


test1.columns


# In[ ]:


# # Feature Scaling
# standardize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#train_X = sc.fit_transform(train_X)
#test_new = sc.transform(test1)
# Normalize the data
#from sklearn.preprocessing import Normalizer
#sc = Normalizer()
#train_X = sc.fit_transform(train_X)
#test_new = sc.transform(test1)


# In[ ]:


from sklearn.decomposition import PCA
#pca = PCA(n_components = 5)
#test_new = pca.fit_transform(test_new)
#test_new1 = pca.transform(test_new)


# In[ ]:


# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
#kpca = KernelPCA(n_components = 10, kernel = 'rbf')
#test_new = kpca.fit_transform(test_new)


# In[ ]:


# make predictions for DNN
#regressor.fit(train_X,train_y)
#preds = regressor.predict(test_new)


# In[ ]:


#preds1 = pd.DataFrame(preds,columns=['preds'])


# In[ ]:


# make predictions - XGBoost
#preds = regressor_xgb.predict(test_new)


# In[ ]:


# Prepare submission file - XGBoost
#gb4 = pd.DataFrame({'ID':test.ID,'target':preds})
# prepare the csv file
#gb4.to_csv('gb4.csv',index=False)


# In[ ]:


# make predictions - Random Forest
preds = regressor_rf.predict(test1)


# In[ ]:


# Prepare submission file - Random Forest
rf2 = pd.DataFrame({'ID':test.ID,'target':preds})
# prepare the csv file
rf2.to_csv('rf2.csv',index=False)


# In[ ]:


# Prepare submission file - DNN
#dnn = pd.DataFrame({'ID':test.ID,'target':preds1.preds})
# prepare the csv file
#dnn.to_csv('dnn1.csv',index=False)

