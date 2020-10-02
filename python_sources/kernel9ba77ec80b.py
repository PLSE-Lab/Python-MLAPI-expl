#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


# In[ ]:


print(train.dtypes)


# In[ ]:


y = train['SalePrice']
fulldata = pd.concat(objs=[train.drop(columns=['SalePrice']), test], axis=0)


# In[ ]:


fulldata2 =  fulldata.copy()


# In[ ]:


plt.figure(figsize=(12,10))
cor = fulldata.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[ ]:


train.shape
#1460


# In[ ]:


sns.set_style("whitegrid")
missing = fulldata.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar()


# In[ ]:


fulldata.head()
fulldata.shape
#2919


# In[ ]:


import missingno as msno
msno.matrix(df=fulldata, figsize=(20,14), color=(0.5,0,0))


# In[ ]:


features_missing_1=[]
for i in np.arange(fulldata.shape[1]):
    n = fulldata.iloc[:,i].isnull().sum() 
    if n > 0:
        print(list(fulldata.columns.values)[i] + ': ' + str(n) + ' nans')
        if n==1 or n==2 : features_missing_1.append(fulldata.columns.values[i])
        #'LotFrontage','MasVnrArea','GarageYrBlt'


# In[ ]:


train['MSZoning'].value_counts()


# In[ ]:


features_missing_1
features_missing_1.append('MSZoning')


# In[ ]:


def cat_imputation(column, value):
    fulldata.loc[fulldata[column].isnull(),column] = value

#feature = ["Fence","PoolQC","Alley","MiscFeature","GarageFinish","GarageQual","GarageCond","GarageType"]
#feature_mode = ["Functional","SaleType"]


# In[ ]:


print(fulldata['LotFrontage'].corr(fulldata['LotArea']))
print(fulldata['LotFrontage'].corr(np.sqrt(fulldata['LotArea'])))
fulldata['SqrtLotArea']=np.sqrt(fulldata['LotArea'])


# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')
sns.pairplot(fulldata[['LotFrontage','SqrtLotArea']].dropna())


# In[ ]:


cond = fulldata['LotFrontage'].isnull()
fulldata.LotFrontage[cond]=fulldata.SqrtLotArea[cond]


# In[ ]:


del fulldata['SqrtLotArea']


# In[ ]:


fulldata['Alley'].value_counts()


# In[ ]:


feature_non = []
feature_non.append('Alley')


# In[ ]:


fulldata[['MasVnrType','MasVnrArea']][fulldata['MasVnrType'].isnull()==True]


# In[ ]:


fulldata['MasVnrType'].value_counts()


# In[ ]:


feature_non.append('MasVnrType')


# In[ ]:


cat_imputation('MasVnrArea', 0.0)


# In[ ]:


basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']
fulldata[basement_cols][fulldata['BsmtQual'].isnull()==True]


# In[ ]:


for cols in basement_cols:
    if 'FinSF'not in cols:
        feature_non.append(cols)


# In[ ]:


fulldata['Electrical'].value_counts()


# In[ ]:


cat_imputation('Electrical','SBrkr')


# In[ ]:


fulldata['FireplaceQu'].value_counts()


# In[ ]:


fulldata['Fireplaces'][fulldata
                       ['FireplaceQu'].isnull()==True].describe()


# In[ ]:


feature_non.append('FireplaceQu')


# In[ ]:




garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']
fulldata[garage_cols][fulldata['GarageType'].isnull()==True]


# In[ ]:


for cols in garage_cols:
    if fulldata[cols].dtype==np.object:
        feature_non.append(cols)
    else:
        cat_imputation(cols, 0)


# In[ ]:


print(fulldata['PoolQC'].value_counts())
print(fulldata['Fence'].value_counts())
print(fulldata['MiscFeature'].value_counts())


# In[ ]:


feature_non.append('PoolQC')
feature_non.append('Fence')
feature_non.append('MiscFeature')


# In[ ]:


feature_non


# In[ ]:


for cols in feature_non:
    if fulldata[cols].dtype==np.object:
        cat_imputation(cols,'None')
    else:
        cat_imputation(cols, 0)


# In[ ]:


for cols in features_missing_1:
        cat_imputation(cols,fulldata[cols].mode()[0])


# In[ ]:


import missingno as msno
msno.matrix(df=fulldata, figsize=(20,14), color=(0.5,0,0))


# In[ ]:


target_log = np.log(y+1)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.distplot(y, bins=50)
plt.title('Original Data')
plt.xlabel('Sale Price')

plt.subplot(1,2,2)
sns.distplot(target_log, bins=50)
plt.title('Natural Log of Data')
plt.xlabel('Natural Log of Sale Price')
plt.tight_layout()


# In[ ]:


sns.boxplot(y = target_log)
plt.ylabel('SalePrice (Log)')
plt.title('Price');


# In[ ]:


from sklearn.model_selection import train_test_split # import 'train_test_split'
from sklearn.ensemble import RandomForestRegressor # import RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer, mean_squared_error # import metrics from sklearn
from time import time


# In[ ]:


catigorical_features = fulldata.select_dtypes(exclude=[np.number])
numerical_features = fulldata.select_dtypes(include=[np.number])
one_hot = pd.get_dummies(catigorical_features)


# In[ ]:


full_new = pd.concat([numerical_features, one_hot],axis=1,ignore_index=False, sort=False) 
#full_new = pd.concat(numerical_features,one_hot)
print(full_new.shape)


# In[ ]:


train_df = full_new[:1460] 
test_df = full_new[1460:]


# In[ ]:


del train_df['Id']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_df, target_log, random_state=42)


# In[ ]:


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


# In[ ]:


col = X_train.columns


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_df, target_log, random_state=42)


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


# In[ ]:


predictors = fulldata2.copy()


# In[ ]:


train_NN = train_df


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
import keras


# In[ ]:


train_NN['SalePrice']= target_log


# In[ ]:


train_NN.head()


# In[ ]:


col_train = list(train_NN.columns)
col_train_bis = list(train_NN.columns)

col_train_bis.remove('SalePrice')

mat_train = np.matrix(train_NN)
#mat_test  = np.matrix(test)
#mat_new = np.matrix(train.drop('SalePrice',axis = 1))
mat_y = np.array(train_NN.SalePrice).reshape((1460,1))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)

prepro = MinMaxScaler()
prepro.fit(mat_train)

#prepro_test = MinMaxScaler()
#prepro_test.fit(mat_new)

train = pd.DataFrame(prepro.transform(mat_train),columns = col_train)
#test  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)


# In[ ]:


train.head()


# In[ ]:



COLUMNS = col_train
FEATURES = col_train_bis
LABEL = "SalePrice"
training_set = train_NN[COLUMNS]
prediction_set = target_log

# Train and Test 
x_train, x_test, y_train, y_test = train_test_split(train_NN, prediction_set, test_size=0.33, random_state=42)
y_train = pd.DataFrame(y_train, columns = [LABEL])
#training_set = pd.DataFrame(x_train, columns = FEATURES).merge(y_train, left_index = True, right_index = True)
#training_set.head()
# Training for submission
#training_sub = training_set


# In[ ]:


x_train.shape


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor

seed = 7
np.random.seed(seed)

# Model
model = Sequential()
model.add(Dense(400, input_dim=303, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(50, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())
#SGD(lr=0.01, clipnorm=1.)


# In[ ]:


history=model.fit(np.array(x_train), np.array(y_train), epochs=300, batch_size=10)
#978/978 [==============================] - 0s 373us/step - loss: 0.0551
#Epoch 199/200
#978/978 [==============================] - 0s 372us/step - loss: 0.0518
#Epoch 200/200
#978/978 [==============================] - 0s 360us/step - loss: 0.0548


# In[ ]:


from matplotlib import pyplot
pyplot.plot(history.history['loss'])
pyplot.show()


# In[ ]:


model.evaluate(np.array(x_test), np.array(y_test))
#0.05052204940205293
#0.04101903091041812


# In[ ]:


print('Predict submission', datetime.now(),)
submission = pd.read_csv("../input/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(model.predict(test_df)))


# In[ ]:



submission.to_csv("House_price_submission.csv", index=False)
print('Save submission', datetime.now(),)


# In[ ]:




