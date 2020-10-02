#!/usr/bin/env python
# coding: utf-8

# One of my first kernal for House price competition and I have drawn inspiration from *Adam Massachi* from[ dataquest blog](https://www.dataquest.io/blog/kaggle-getting-started/ ) 
# <br>**Next steps:** 
# ****Feature engineering ****
#     1. Experiment more features 
#     2. Experiment with missing value treatment
# ****Add more regression techniques ****
#     1. Follow other kernels 
# Update Log: <br>
# <br>14Feb: Condition1,Condition2 seemed important but R2 is reduced to .85 from .88 after encoding these two. Updated  garage area filter from 1200 to 1150 and R-sq has improved to .89. Score has improved to .138 to .137
# <br>15 Feb: Changing all cat, non cat vars using get_dummies and  treat missing values with mean for all. score improved .8991
# <br>16 Feb: Updating Lable for Garage car capacity graph
# <br>20 Feb: Updating outliers for GrLivArea < 4000  or replacing it with mean of 1510.722834 results in dropping the score to .8920. Not doing it 
# <br>27 Feb: Plan to add one linear regresssion technique from private kernel and compare the result
# <br>24 Mar: clean up plan, add new techniques
# <br>27 Mar: Adding XGB example from sample Dan Baker tutorial 
# <br>28 Mar: Re arranging few sections to restructure this Kernel. 
# <br>08 Apr: Tweak certain pieces 
# <br>21 Apr: Fixed target var error and added example for hyper paramter tuning  from [Kenji]( https://www.kaggle.com/kenji19840210/basic-prediction-with-xgboost-xgbregressor)  

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# list all directory 
import os
print(os.listdir("../input"))


# In[ ]:


#load train and test 
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


#check sample data 
train.head(10)


# In[ ]:


#get training data set dimensions
print("train data shape", train.shape)
#get testing data set dimensions
print("test data shape", test.shape)


# In[ ]:


#list of columns 
print("\ncolumn in training data set\n\n",train.columns.values)
print("\ncolumn in testing data set\n\n",test.columns.values)


# In[ ]:


print("extra columns found in training dataset", set(train.columns.values)-set(test.columns.values))


# In[ ]:


#check  summary of dependent variable
train.SalePrice.describe()


# In[ ]:



import matplotlib.pyplot as plt 
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize']=(10,6)
#check skewness of sale price 
print("Skewness : ", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()
#distribution is a bit skewed to the left 


# In[ ]:


#change sale price to natural log 
print("Skewness after log: ", np.log(train.SalePrice).skew())
plt.hist(np.log(train.SalePrice), color='blue')
plt.show()
target= np.log(train.SalePrice)
#natural log transformation of target var changes it to normal distribution 


# In[ ]:


#find numeric features 
num_features = train.select_dtypes(include=[np.number])
#check data types of these 
num_features.dtypes


# In[ ]:


#check correlation of numeric variables 
corr = num_features.corr()
#top 5 highly correlated vars 
print(corr['SalePrice'].sort_values(ascending=False)[:5],'\n')
#bottom 5 highly correlated vars 
print(corr['SalePrice'].sort_values(ascending=False)[-5:],'\n')


# In[ ]:


#check unique values of feature OverallQual
train.OverallQual.unique()


# In[ ]:


#check first overall quality variable with SalePirce
qual_pivot = train.pivot_table(index='OverallQual', 
                               values='SalePrice', 
                               aggfunc=np.mean)
display(qual_pivot)


# In[ ]:


#create pivot for overall quality 
qual_pivot.plot(kind='bar', color='green')
plt.xlabel('Overall Quality')
plt.ylabel('Mean Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


#create pivot for Gr Living area 
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade(ground) living area square feet')
plt.show()


# In[ ]:


plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


# In[ ]:


train = train[train['GarageArea'] < 1150]
plt.scatter(x=train['GarageArea'], y =np.log(train.SalePrice))
plt.xlim(-200,1600) #adjusting to same scale 
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()


# In[ ]:


nuls = pd.DataFrame(train.isnull().sum().sort_values(ascending =False)[:25])
nuls.columns = ['Null Count']
nuls.index.name = 'Feature'
nuls


# In[ ]:


print ("Unique values are:", train.MiscFeature.unique())


# In[ ]:


catgr  = train.select_dtypes(exclude=[np.number])
catgr.describe()


# In[ ]:


print('originals')
print(train.Street.value_counts(),"\n")


# In[ ]:


train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)


# In[ ]:


print("Encoded:")
print(train.enc_street.value_counts())


# In[ ]:


# One more variable, Garage car capacity
train.GarageCars.value_counts().plot(kind='bar', color='green')
plt.xlabel('Garage Car Capacity')
plt.ylabel('Counts')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


condition_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='skyblue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
#encoding steps
def encode_condition(x) : 
    return 1 if x =='Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode_condition)
test['enc_condition'] = test.SaleCondition.apply(encode_condition)


# In[ ]:


condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='gray')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


#update missing values 
train = train.fillna(train.mean())
test = test.fillna(test.mean())


# In[ ]:


#interpolate missing values 
dt = train.select_dtypes(include=[np.number]).interpolate().dropna()
#check if all cols have zero null values 
sum(dt.isnull().sum()!=0)


# In[ ]:


#change y to natural log 
y = np.log(train.SalePrice)
#drop original dependent var and id 
X = dt.drop(['Id','SalePrice'], axis=1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)


# In[ ]:


#Hyper paramteter tuning  example 
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
#Hyper parameter tuning example 
gbm = xgb.XGBRegressor()
reg_cv = GridSearchCV(gbm, {"colsample_bytree":[1.0],"min_child_weight":[1.0,1.2]
                            ,'max_depth': [3,4,6], 'n_estimators': [500,1000]}, verbose=1)
reg_cv.fit(X_train,y_train)
reg_cv.best_params_


# In[ ]:


###########
gbm = xgb.XGBRegressor(**reg_cv.best_params_)
gbm.fit(X_train,y_train)
##############
submit= pd.DataFrame()
submit['Id'] = test.Id
test_features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
preds = gbm.predict(test_features)
final_preds = np.exp(preds)
print('Original preds :\t', preds[:5])
print('Final preds :\t', final_preds[:5])
submit['SalePrice'] = final_preds
#final submission  
submit.to_csv('xgb_hyper_param_subm.csv', index=False)
print('XGB submission using hyper param tuning code  created')


# In[ ]:


#1. linear regression 
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
#r square 
print("R-Square : " ,model.score(X_test,y_test))
#rmse 
preds = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE: ', mean_squared_error(y_test, preds))


# In[ ]:


#Adding simple XGB output and test 
import warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
#
def xgb_regressor(learn_rate):
    #instance of XGB regressor 
    xgbmodel = XGBRegressor(n_estimators=1000, learning_rate=learn_rate)
    xgbmodel.fit(X_train, y_train, verbose=False)
    # make predictions
    predictions = xgbmodel.predict(X_test)
    from sklearn.metrics import mean_absolute_error
    print('{:^20}'.format('Learning Rate:')+ '{:^5}'.format(str(learn_rate)) +'{:^5}'.format("\tMAE: ")+'{:<20}'.format(str(mean_absolute_error( y_test,predictions))) +'{:^5}'.format("\tRMSE: ")+'{:<20}'.format(str(mean_squared_error( y_test,predictions)) +'{:^5}'.format("\tR^2: ")+'{:<20}'.format(xgbmodel.score(X_test,y_test)))  )
    
xgb_regressor(0.04) #experimented with .03 -.09, .04 looks best 

#using best learning rate xgb_regressor(0.04) and updating same code for submission 
def xgb_regressor_updated(learn_rate):
    #instance of XGB regressor 
    print('***********Final run with best learning rate*************')
    xgbmodel = XGBRegressor(n_estimators=1000, learning_rate=learn_rate)
    xgbmodel.fit(X_train, y_train, verbose=False)
    # make predictions
    predictions = xgbmodel.predict(X_test)
    from sklearn.metrics import mean_absolute_error
    print('{:^20}'.format('Learning Rate:')+ '{:^5}'.format(str(learn_rate)) +'{:^5}'.format("\tMAE: ")+'{:<20}'.format(str(mean_absolute_error( y_test,predictions))) +'{:^5}'.format("\tRMSE: ")+'{:<20}'.format(str(mean_squared_error( y_test,predictions)) +'{:^5}'.format("\tR^2: ")+'{:<20}'.format(xgbmodel.score(X_test,y_test)))  )
    #test 
    submit= pd.DataFrame()
    submit['Id'] = test.Id
    test_features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
    preds = xgbmodel.predict(test_features)
    final_preds = np.exp(preds)
    print('Original preds :\t', preds[:5])
    print('Final preds :\t', final_preds[:5])
    submit['SalePrice'] = final_preds
    #final submission  
    submit.to_csv('xgb_submit.csv', index=False)
    print('XGB submission file created')

#test and create xgb submission 
xgb_regressor_updated(0.04)


# In[ ]:


plt.scatter(preds, y_test, alpha=.75, color='g')
plt.xlabel('predicted price')
plt.ylabel('actual sale price ')
plt.title('Linear regression ')
plt.show()


# In[ ]:


#Regularization 
for i in range (-3, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)
    plt.scatter(preds_ridge, y_test, alpha=.75, color='g')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(ridge_model.score(X_test, y_test),
                                               mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()


# In[ ]:


submit= pd.DataFrame()
submit['Id'] = test.Id
#select features 
test_features = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
preds = model.predict(test_features)
#unlog/exp the prediction  
final_preds = np.exp(preds)
print('Original preds :\n', preds[:5])
print('Final preds :\n', final_preds[:5])
submit['SalePrice'] = final_preds
#final submission  
submit.to_csv('test_submit.csv', index=False)


# In[ ]:


#variables not used 
print("Vars not used : \n", set(test.columns.values)-set(X.columns.values))

