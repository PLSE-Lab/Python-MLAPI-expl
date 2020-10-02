#!/usr/bin/env python
# coding: utf-8

# # Regression: Predict Allstate Claims Severity

# ## Fire up packages

# In[ ]:


import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas
from sklearn.cross_validation import train_test_split
import numpy
import xgboost as xgb


# ## Load Training Data

# In[ ]:


train=pandas.read_csv("../input/train.csv")
train.head()


# In[ ]:


del train['id']


# ## Briefly explore the response variable

# **As we have a high dimensional dataset, feature engineering will be quite time consuming if I check the co-variate one by one. Therefore, I first focus on exploring the response variable. PCA might also be a good choice to reduce dimension if necessary.**

# In[ ]:


training,testing = train_test_split(train,test_size=0.2,random_state=42)
print(training.shape)
print(testing.shape)


# In[ ]:


Response= training['loss']


# ### 1. See the mean, sd and median of response variable

# In[ ]:


print ('Mean of Response Variable'+' '+'is'+' '+ str(numpy.mean(Response)))
print ('Median of Response Variable'+' '+'is'+' '+ str(numpy.median(Response)))
print ('Standard Deviation of Response Variable'+' '+'is'+' '+ str(numpy.std(Response)))


# ### 2:Check whether the response variable is normally distributed

# In[ ]:


import statsmodels.api as sm
fig=sm.qqplot(Response)


# **We can see that the normality of response variable is quite bad. The response variable seems to be exponentially distributed. Therefore, we transform it into the logarithm form and re-plot the qq-plot.**

# In[ ]:


training=training.reset_index(drop=True)
testing = testing.reset_index(drop=True)
training['logloss']=numpy.log(training['loss'])
fig2=sm.qqplot(training['logloss'])


# **The noramlity of the logthrim response variable is not bad. So we set it as the new lable.**

# ## Convert categorical variables and process real test data

# In[ ]:


features = training.columns
cat_feature=list(features[0:116])
test=pandas.read_csv("../input/test.csv")
for each in cat_feature:
    training[each]=pandas.factorize(training[each], sort=True)[0]
    testing[each]=pandas.factorize(testing[each],sort=True)[0]
    test[each]=pandas.factorize(test[each],sort=True)[0]


# ## Preliminary model selection: try five different regressors at first

# In[ ]:


from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
Predictors= training.ix[:,0:130]
Predictors_test= testing.ix[:,0:130]


# In[ ]:


Regressors = [LinearRegression(),Lasso(),DecisionTreeRegressor()
              #,RandomForestRegressor(n_estimator=200),
              #GradientBoostingRegressor(learning_rate=0.3,criterion='mae')
             ]
MAE=[]
Model_Name=[]
for reg in Regressors:
    Model=reg.fit(Predictors,training['logloss'])
    Prediction= numpy.exp(Model.predict(Predictors_test))
    eva = mae(testing['loss'],Prediction)
    MAE.append(eva)
    Name=reg.__class__.__name__
    Model_Name.append(Name)
    print('Accuracy of'+ ' '+Name+' '+'is'+' '+str(eva))


# **I also tried Random Forest Regressor locally. With the number of estimators set to 200, it produced a prediction with MAE of 1212.42750158. However, the running time is incredibly long and boring. For Gradient Boosting Regressor, the accuracy is 1183.90923254, with the learning rate setting to 0.3.**

# In[ ]:


MAE.append(1212.42750158)
MAE.append(1183.90923254)
Model_Name.append('RandomForestRegressor')
Model_Name.append('GradientBoostingRegressor')


# In[ ]:


Index = [1,2,3,4,5]
plt.bar(Index,MAE)
plt.xticks(Index, Model_Name,rotation=45)
plt.ylabel('MAE')
plt.xlabel('Model')
plt.title('MAE of Models')


# **We can see that rf and gb models are better than the other three models. In the next step, I will try to implement Extreme Gradient Boosting model and compare the performance between it and Gradient Boosting model.**

# ## Try Extreme Boosting Model

# **We first, have to transform the dataset into the ideal form in order to make XGboost running**

# In[ ]:


training_array = numpy.array(Predictors)
testing_array = numpy.array(Predictors_test)


# **Then I tune the parameters of model. My role of thumb is : finding the balance point between number of rounds and the learning rate.**

# In[ ]:


dtrain = xgb.DMatrix(training_array, label=training['logloss'])
dtest = xgb.DMatrix(testing_array)
xgb_params = {
    'seed':0,
    'colsample_bytree': 0.7,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 6,
    'min_child_weight': 1,
    'eval_metric': 'mae',
}
xgb_model=xgb.train(xgb_params, dtrain,750,verbose_eval=50)
xgb_pred=numpy.exp(xgb_model.predict(dtest))
print('Accuracy of XGboost model is'+' '+str(mae(testing['loss'],xgb_pred)))


# **The XGboost model performs way better than the Gradient boosting model and random forest, in terms of both accuracy and running time.**

# ## Retrain the model and generate the output file

# In[ ]:


features=Predictors.columns
train['logloss']=numpy.log(train['loss'])
for each in cat_feature:
    train[each]=pandas.factorize(train[each], sort=True)[0]
del test['id']


# In[ ]:


train_array=numpy.array(train[features])
train_d=xgb.DMatrix(train_array,label=train['logloss'])
test_array=numpy.array(test)
test_d=xgb.DMatrix(test_array)


# In[ ]:


Final_model=xgb.train(xgb_params, train_d,750,verbose_eval=50)
Prediction_Final=numpy.exp(Final_model.predict(test_d))


# In[ ]:


submission = pandas.read_csv('../input/sample_submission.csv')
submission.iloc[:, 1] = Prediction_Final
submission.to_csv('sub_xgb.csv', index=None)


# ## Question from a Newbie

# **I am still a newbie of Machine learning, so I always have questions. If someone can offer help, I will be really grateful.**
# 
# **Question is: How can we decide the initial parameter of a model and tune it with a specific rule? The decision is made based on what? Experience? Volume of dataset ?  Can some cool Kaggler give me a rule of thumb on parameter initializing and tuning?**
# 
# # Thanks in advance!

# In[ ]:




