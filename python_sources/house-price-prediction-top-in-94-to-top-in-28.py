#!/usr/bin/env python
# coding: utf-8

# ## House Price Prediction
# 
# This is a very interesting competition provided by Kaggle. I believe that all the entry level Machine Learning enthusiasts should definitely get their hands-on to this competition as well as the [Titanic Survivor Prediction competition ](https://www.kaggle.com/c/titanic). One can also check [my notebook](https://www.kaggle.com/darshanjain29/titanic-survival-from-top-70-to-top-7-on-lb) on the same.
# 
# Coming back to this competition, as it says that the problem is predicting house price i.e. a continuous value. Hence it is a regression problem and thus all your basic regression knowledge will be tested here.
# 
# Now, from a beginner's perspective how do you start with this? Well, I will show very easy steps with which you can easily jump from **top 95%** on leaderboard to **top 28%**. So, let's get started.
# 
# 1. Let's read about the data that is given. So, in the train.csv there are 1460 rows and 81 columns whereas in test.csv there are 1459 and 80 columns.
# 2. After reading the file, we start data preprocessing and feature engineering. Before moving ahead we are combining both the dataframes so that changes can be done in both the dfs together.
# 3. So, with the help of heatmap we can check number of nulls in each feature. So, based on the column name and after reading the description in the file data_description.txt of that column, we can decide if we want to replace nulls by mean(), mode(), median() or 'NA' or something else.
# 4. Now, we have to replace all the object/string values to numerical type using one hot encoding. While doing this many new columns will be created. So, we also run a code to drop all duplicate columns if any
# 5. Now I am separating the train and test data from all_data and appending SalePrice column to the train data and the training models starts from here.
# 6. While training the model, we are using Kfold cross validation for the better root mean squared log error as it is the evaluation criteria on the leaderboard.
# 7. Training with SVM algorithm: rmsle, r2_score = 0.42655965911720095, -0.21383191346231967 respectively
# 8. Training with Decision Tree: rmsle, r2_score = 0.20087638073362202, 0.7569636210379455 respectively
# 9. Training with Random Forest Regressor model: rmsle, r2_score = 0.17295960323425913, 0.8220478028104417 respectively
# 10. Training with ridge model: rmsle, r2_score = 0.1562332068843027, 0.8350220277268043 respectively
# 11. Training with Elastic Net: rmsle, r2_score = 0.15524191588871572, 0.8355319656725282 respectively
# 12. Training with Adaboost regressor model: rmsle, r2_score = 0.20880462846912984, 0.7886447451714611 respectively
# 13. Training with Bagging Classifier model: rmsle, r2_score = 0.14719806924931875, 0.8539938038399392 respectively
# 14. Training with XGboost model: rmsle, r2_score = 0.13372606548796895, 0.8742884283414025 respectively
# 15. Training with Gradient Boosting Regressor model: rmsle, r2_score = 0.14187482285610134, 0.8865067659268699 respectively
# 16. Now you can do a submission and check your score on the leaderboard.
# 17. After trying each model, try using the same one with hyper parameter tuning and you will be able to small improvement in rmsle and r2_score as well as your position on the leaderboard
# 18. At the end we have check the feature importance count and removed all the features with importance count less than 0. 
# 
# I hope this helps. Please comment below if you haven't understood anything from the above steps.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Plotting started code requied
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
import seaborn as sns


# In[ ]:


train_df  = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train_df.info()
#train_df.head(10)


# In[ ]:


test_df  = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
test_df.info()
#train_df.head(10)


# In[ ]:


plt.figure(figsize = (15,5))

plt.title("Heatmap to see the nulls")
sns.heatmap(train_df.isnull(), yticklabels = False, cbar = False)
plt.xlabel("Features")
plt.ylabel("Nulls as white lines")


# In[ ]:


all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
x_saleprice = train_df["SalePrice"]
all_data.drop(["SalePrice"], axis = 1, inplace= True)
all_data.shape


# In[ ]:


all_data.PoolQC.value_counts()


# In[ ]:


x = all_data.isnull().sum().sort_values(ascending = False)
print (x[x>0])


# In[ ]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("NA")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("NA")
all_data["Alley"] = all_data["Alley"].fillna("NA")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("NA")
all_data["Fence"] = all_data["Fence"].fillna("NA")
all_data["GarageQual"] = all_data["GarageQual"].fillna("NA")
all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna("NA")
all_data["GarageCond"] = all_data["GarageCond"].fillna("NA")
all_data["GarageFinish"] = all_data["GarageFinish"].fillna("NA")
all_data["GarageType"] = all_data["GarageType"].fillna("NA")
all_data["BsmtCond"] = all_data["BsmtCond"].fillna("NA")
all_data["BsmtExposure"] = all_data["BsmtExposure"].fillna("NA")
all_data["BsmtQual"] = all_data["BsmtQual"].fillna("NA")
all_data["BsmtFinType2"] = all_data["BsmtFinType2"].fillna("NA")
all_data["BsmtFinType1"] = all_data["BsmtFinType1"].fillna("NA")
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data["LotFrontage"] = all_data["LotFrontage"].fillna(all_data["LotFrontage"].median())
all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data["MSZoning"].mode()[0])
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data["Functional"] = all_data["Functional"].fillna(all_data["Functional"].mode()[0])
all_data["Utilities"] = all_data["Utilities"].fillna(all_data["Utilities"].mode()[0])
all_data["BsmtHalfBath"] = all_data["BsmtHalfBath"].fillna(all_data["BsmtHalfBath"].mode()[0])
all_data["BsmtFullBath"] = all_data["BsmtFullBath"].fillna(all_data["BsmtFullBath"].mode()[0])
all_data["BsmtFinSF2"] = all_data["BsmtFinSF2"].fillna(all_data["BsmtFinSF2"].mode()[0])

all_data["BsmtFinSF1"] = all_data["BsmtFinSF1"].fillna(all_data["BsmtFinSF1"].mode()[0])
all_data["GarageArea"] = all_data["GarageArea"].fillna(all_data["GarageArea"].mode()[0])
all_data["Exterior1st"] = all_data["Exterior1st"].fillna(all_data["Exterior1st"].mode()[0])
all_data["BsmtUnfSF"] = all_data["BsmtUnfSF"].fillna(all_data["BsmtUnfSF"].mode()[0])
all_data["TotalBsmtSF"] = all_data["TotalBsmtSF"].fillna(all_data["TotalBsmtSF"].mode()[0])
all_data["GarageCars"] = all_data["GarageCars"].fillna(all_data["GarageCars"].mode()[0])
all_data["Exterior2nd"] = all_data["Exterior2nd"].fillna(all_data["Exterior2nd"].mode()[0])
all_data["KitchenQual"] = all_data["KitchenQual"].fillna(all_data["KitchenQual"].mode()[0])
all_data["SaleType"] = all_data["SaleType"].fillna(all_data["SaleType"].mode()[0])
all_data["Electrical"] = all_data["Electrical"].fillna(all_data["Electrical"].mode()[0])


# In[ ]:


#all_data["GarageCars"].value_counts()

#print (all_data["MSZoning"].mode())


# In[ ]:



def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    i = 0
    for each in cols:
        #print (each)
        dummies = pd.get_dummies(df[each], prefix=each, drop_first= True)
        if i == 0: 
            print (dummies)
            i = i + 1
        df = pd.concat([df, dummies], axis=1)
    return df


# In[ ]:


all_data.shape


# In[ ]:



#One hot encoding done
all_data = one_hot(all_data, objList) 
all_data.shape


# In[ ]:



#Dropping duplicates columns if any
all_data = all_data.loc[:,~all_data.columns.duplicated()]
all_data.shape


# In[ ]:



#Dropping the original columns that has data type object 
all_data.drop(objList, axis=1, inplace=True)
all_data.shape


# In[ ]:


#data=pd.concat([train_df, test_df],axis=0)
objList = all_data.select_dtypes(include = "object").columns

print (objList)


# In[ ]:


'''
#Label Encoding for object to numeric conversion - Option 1
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in objList:
    all_data[feat] = le.fit_transform(all_data[feat].astype(str))

print (all_data.shape)
'''


# In[ ]:


train_df = all_data.iloc[:1460,:]  
test_df = all_data.iloc[1460 :,:]  


# In[ ]:


train_df["SalePrice"] = x_saleprice


# In[ ]:


train_df.info()


# In[ ]:


X_train = train_df.drop(["SalePrice"], axis = 1)
Y_train = train_df["SalePrice"]
X_test = test_df


# In[ ]:


X_train.shape 


# In[ ]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_log_error

n_folds = 5

cv = KFold(n_splits = 5, shuffle=True, random_state=42).get_n_splits(X_train.values)

def test_model(model):   
    msle = make_scorer(mean_squared_log_error)
    rmsle = np.sqrt(cross_val_score(model, X_train, Y_train, cv=cv, scoring = msle))
    score_rmsle = [rmsle.mean()]
    return score_rmsle

def test_model_r2(model):
    r2 = make_scorer(r2_score)
    r2_error = cross_val_score(model, X_train, Y_train, cv=cv, scoring = r2)
    score_r2 = [r2_error.mean()]
    return score_r2


# In[ ]:


'''
#1. SVM algorithm
from sklearn import svm
reg_svm = svm.SVR()
rmsle_svm = test_model(reg_svm)
print (rmsle_svm, test_model_r2(reg_svm))
'''


# In[ ]:


'''
#2. Decision Tree 
from sklearn.tree import DecisionTreeRegressor
reg_dtR = DecisionTreeRegressor(max_depth=5, random_state=51)
rmsle_dtR = test_model(reg_dtR)
print (rmsle_dtR, test_model_r2(reg_dtR))
'''


# In[ ]:


'''
#3. Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

reg_rFR = RandomForestRegressor(max_depth=5, random_state=51)
rmsle_rFR = test_model(reg_rFR)
print (rmsle_rFR, test_model_r2(reg_rFR))
'''


# In[ ]:


'''
#4. Ridge Model
from sklearn.linear_model import Ridge

reg_ridge = Ridge(alpha=10, copy_X=True, fit_intercept=True, max_iter=None, normalize=False,  random_state=None, solver='auto', tol=0.001)

rmsle_ridge = test_model(reg_ridge)

print (rmsle_ridge, test_model_r2(reg_ridge))
'''


# In[ ]:


'''
#5. Elastic Net
from sklearn.linear_model import ElasticNet

reg_elastic_net = ElasticNet(alpha=0.1, copy_X=True, fit_intercept=True, l1_ratio=0.9,
           max_iter=100, normalize=False, positive=False, precompute=False,
           random_state=None, selection='cyclic', tol=0.0001, warm_start=False)

rmsle_elastic_net = test_model(reg_elastic_net)

print (rmsle_elastic_net, test_model_r2(reg_elastic_net))
'''


# In[ ]:


'''
#6. Adaboost regressor
from sklearn.ensemble import AdaBoostRegressor
reg_aBR = AdaBoostRegressor(random_state=51, n_estimators=1000)
rmsle_aBR = test_model(reg_aBR)
print (rmsle_aBR, test_model_r2(reg_aBR))
'''


# In[ ]:


'''
#7. BaggingClassifier
from sklearn.ensemble import BaggingRegressor
reg_bgr = BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False,
                 max_features=1.0, max_samples=1.0, n_estimators=100,
                 n_jobs=None, oob_score=False, random_state=51, verbose=0,
                 warm_start=False)
rmsle_bgr = test_model(reg_bgr)
print (rmsle_bgr, test_model_r2(reg_bgr))
'''


# In[ ]:


'''
#8. XGboost
import xgboost as xgb
from xgboost import plot_importance

reg_xgb = xgb.XGBRegressor()
rmsle_xgb = test_model(reg_xgb)
print (rmsle_xgb, test_model_r2(reg_xgb))
'''


# In[ ]:



#9. GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
reg_gbr = GradientBoostingRegressor(alpha=0.9, ccp_alpha=0.0, criterion='friedman_mse',
                          init=None, learning_rate=0.05, loss='ls', max_depth=3,
                          max_features='sqrt', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=9, min_samples_split=8,
                          min_weight_fraction_leaf=0.0, n_estimators=1250,
                          n_iter_no_change=None, presort='deprecated',
                          random_state=None, subsample=0.8, tol=0.0001,
                          validation_fraction=0.1, verbose=0, warm_start=False)

rmsle_ggr = test_model(reg_gbr)
print (rmsle_ggr, test_model_r2(reg_gbr))
#[0.1321644225864123] [0.8880632316361895]
#[0.1333303699875545] [0.8813635273996498]


# In[ ]:


#Train the model with any of the above declared models
reg_gbr.fit(X_train, Y_train)
Y_pred  = reg_gbr.predict(test_df) 


# In[ ]:


pred=pd.DataFrame(Y_pred)
sub_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)
print("Your submission was successfully saved!")


# In[ ]:


from matplotlib import pyplot
pyplot.bar(range(len(reg_gbr.feature_importances_)), reg_gbr.feature_importances_)
pyplot.show()


# In[ ]:


#To view the descending values of the features i.e. top 10 %

#new_df = ([train_df.columns, clf_ggr.feature_importances_])
featureImp = []

for feat, importance in zip(train_df.columns, reg_gbr.feature_importances_):  
    temp = [feat, importance*100]
    featureImp.append(temp)

fT_df = pd.DataFrame(featureImp, columns = ['Feature', 'Importance'])
print (fT_df.sort_values('Importance', ascending = False))


# In[ ]:


#Hyper parameter tuning 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {'min_samples_split':[2,4,6,8,10,20,40,60,100], 
              'min_samples_leaf':[1,3,5,7,9, 15, 20, 25, 30, 40, 50],
              'subsample':[0.7,0.75,0.8,0.85,0.9,0.95,1],
              'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001, 0.2], 
              'n_estimators':[10, 30, 50, 100,250,500,750,1000,1250,1500,1750],
              'max_features' : ['sqrt']
             }

asdf = GradientBoostingRegressor()

#clf = GridSearchCV(asdf, param_grid=param_grid, scoring='r2', n_jobs=-1)
clf = RandomizedSearchCV(asdf, param_grid, scoring='r2', n_jobs=-1)
 
clf.fit(X_train, Y_train)

print(clf.best_estimator_)

