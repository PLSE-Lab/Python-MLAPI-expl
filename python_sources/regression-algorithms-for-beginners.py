#!/usr/bin/env python
# coding: utf-8

# # Regression algorithms for predicting housing prices on Iowa data.
# 
# ## Contents
#  [Decision Tree](#decisiontree)
#  
#  [Random Forest prediction with select features](#randomforest)
#  
#  [Random Forest prediction with all features](#randomforestparams)
# 
# [XGBoost](#XGBoost)
#  
#  [XGBoost with parameters](#xgboostparams)  
#  
#  [Gradient Boosting](#gradientboosting)  
#  
#  [Gradient Boosting with params](#gradientboostingparams) 
#  
#  [Partial Dependence based on gradient boosting regressor](#pd) 
#  
#  [Decomposition with Principal Component Analysis and gradient boosting](#pca) 
#  
#  [Piplelines with Gradient boosting](#piplelines) 
#  
#  [LightGBM](#lightgbm)
#  
#  [Sequential model with 3 dense layers](#sequential)

# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence,partial_dependence
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA 
from sklearn.preprocessing import Imputer
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
import numpy as np
import math
import lightgbm as lgb
import tensorflow as tf 


# <a id="headin"></a>

# <h3 id="decisiontree">Decision Tree</h3>

# In[ ]:


test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


# In[ ]:


X = train.drop(['SalePrice'],axis=1)
y = train.SalePrice
#print (X.SalePrice)
val_X = test[list(test)]


# In[ ]:



one_hot_pred = pd.get_dummies(X)
one_hot_encoded_test_predictors = pd.get_dummies(val_X)
final_train, final_test = one_hot_pred.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)


# In[ ]:


my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(final_train)
imputed_X_test = my_imputer.transform(final_test)


# In[ ]:


new_dt_model=DecisionTreeRegressor()

new_dt_model.fit(imputed_X_train,y)

dt_predictions=new_dt_model.predict(imputed_X_test)


# In[ ]:


my_submission=pd.DataFrame({'ID': test.Id, 'SalePrice': dt_predictions})

my_submission.to_csv('dy_submission.csv',index=False)


# **Score: 0.20626**

# <h3 id="randomforest">Random Forest prediction with select features</h3>

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
import pandas as pd

train = pd.read_csv('../input/train.csv')
train_y=train.SalePrice
predictor_cols=['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X=train[predictor_cols]
forest_model=RandomForestRegressor()
forest_model.fit(train_X,train_y)
test= pd.read_csv('../input/test.csv')
test_X=test[predictor_cols]
predicted_prices=forest_model.predict(test_X)
#print(predicted_prices)


# In[ ]:


my_submission=pd.DataFrame({'ID': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv',index=False)


#  **Score: 0.19192**

# <h3 id="randomforestparams">Random Forest prediction with all features</h3>

# In[ ]:


X = train.drop(['SalePrice'],axis=1)
y = train.SalePrice
#print (X.SalePrice)
val_X = test[list(test)]


# In[ ]:


one_hot_pred = pd.get_dummies(X)
one_hot_encoded_test_predictors = pd.get_dummies(val_X)
final_train, final_test = one_hot_pred.align(one_hot_encoded_test_predictors, join = 'left',axis = 1)


# In[ ]:


my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(final_train)
imputed_X_test_plus = my_imputer.transform(final_test)


# In[ ]:


imp_model = RandomForestRegressor()
imp_model.fit(imputed_X_train_plus,y)
imp_model_prices = imp_model.predict(imputed_X_test_plus)


# In[ ]:


my_submission=pd.DataFrame({'ID': test.Id, 'SalePrice': imp_model_prices})

my_submission.to_csv('imp_submission.csv',index=False)


# **Score:0.15124**

# <h3 id="XGBoost">XGBoost</h3>

# In[ ]:


xgb_model = XGBRegressor()
xgb_model.fit(imputed_X_train_plus,y)
xgb_pred_prices = xgb_model.predict(imputed_X_test_plus)


# In[ ]:


my_submission=pd.DataFrame({'ID': test.Id, 'SalePrice': xgb_pred_prices})

my_submission.to_csv('xgboost_submission.csv',index=False)


# **Score:0.13520**

# <h3 id="xgboostparams">XGBoost with  parameters</h3>

# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(imputed_X_train_plus,
                                                    y, test_size=0.25)
xgb_plus_model = XGBRegressor(n_estimators=1000, learning_rate=0.5)


# In[ ]:


xgb_plus_model.fit(train_X, train_y, early_stopping_rounds=10,
                    eval_set=[(test_X, test_y)], verbose=False)
xgb_plus_prices = xgb_plus_model.predict(imputed_X_test_plus)


# In[ ]:


my_submission=pd.DataFrame({'ID': test.Id, 'SalePrice': xgb_plus_prices})

my_submission.to_csv('xgboost_plus_submission.csv',index=False)


# **Score:0.15490**

# <h3 id="gradientboosting">Gradient Boosting</h3>

# In[ ]:


gb_model = GradientBoostingRegressor()
gb_model.fit(imputed_X_train_plus,y)
gb_pred_prices = gb_model.predict(imputed_X_test_plus)


# In[ ]:


my_submission=pd.DataFrame({'ID': test.Id, 'SalePrice': gb_pred_prices})
my_submission.to_csv('gboost_submission.csv',index=False)


# **Score: 0.13471**

# <h3 id="gradientboostingparams">Gradient Boosting with params</h3>

# In[ ]:


gb_plus_model = GradientBoostingRegressor(n_estimators=10000, learning_rate=0.01)
gb_plus_model.fit(imputed_X_train_plus,y)
gb_plus_pred_prices = gb_plus_model.predict(imputed_X_test_plus)
print(gb_plus_pred_prices)


# In[ ]:


my_submission = pd.DataFrame({'ID': test.Id, 'SalePrice': gb_plus_pred_prices})
my_submission.to_csv('gboost_plus_submission.csv',index=False)


# **Score: 0.13457**

# ##Partial Dependence based on gradient boosting regressor<a name="pd"></a>

# In[ ]:


fig ,axs = plot_partial_dependence(gb_plus_model,
                                      features=[0,1,2,3,4,5,6,7,8],
                                      X=imputed_X_train_plus,
                                      feature_names=['GarageType','CentralAir','1stFlrSF','2stFlrSF','Alley','Street','MSSubClass','LotArea','RoofMatl'],
                                      grid_resolution=100)

plt.subplots_adjust(top=2.5,right=2.5)


# <h3 id="pca">Decomposition with Principal Component Analysis and gradient boosting</h3>

# In[ ]:


pca = PCA().fit(imputed_X_train_plus)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,10,1)


# In[ ]:


sklearn_pca = PCA(n_components=14)
print(sklearn_pca)


# In[ ]:


X_sklearn = sklearn_pca.fit_transform(imputed_X_train_plus)
print(X_sklearn.shape)

X_test_sklearn=sklearn_pca.transform(imputed_X_test_plus)
print(X_test_sklearn.shape)


# In[ ]:


pca_gb_plus_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01)
pca_gb_plus_model.fit(X_sklearn,y)
pca_gb_plus_pred_prices = pca_gb_plus_model.predict(X_test_sklearn)
print(pca_gb_plus_pred_prices)


# In[ ]:


my_submission = pd.DataFrame({'ID': test.Id, 'SalePrice': pca_gb_plus_pred_prices})

my_submission.to_csv('pca_gb_plus_submission.csv',index=False)


# <h3 id="pipelines">Piplelines with Gradient boosting</h3>

# In[ ]:


my_pipeline= make_pipeline(Imputer(),GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01))

print(cross_val_score(my_pipeline, imputed_X_train_plus, y, scoring = "neg_mean_absolute_error"))
my_pipeline.fit(imputed_X_train_plus,y)
pipe_line_predictions=my_pipeline.predict(imputed_X_test_plus)


# In[ ]:


my_submission = pd.DataFrame({'ID': test.Id, 'SalePrice': pipe_line_predictions})

my_submission.to_csv('pipe_line_predictions.csv',index=False)


# <h3 id="lightgbm">LightGBM</h3>

# In[ ]:


lgb_dataSet=lgb.Dataset(imputed_X_train_plus, label=y)
params={
    'learning_rate':0.1,
    'boosting_type':'gbdt',
    'objective':'regression',
    'sub_features':0.5,
    'num_leaves':10000,
    'min_data':100,
    'max_depth':1000,
}


# In[ ]:


lgb_model=lgb.train(params,lgb_dataSet, num_boost_round=10000)
#print(lgb_model.best_iteration)
lgb_pred_prices= lgb_model.predict(imputed_X_test_plus, num_iteration= 10000)


# In[ ]:


my_submission = pd.DataFrame({'ID': test.Id, 'SalePrice': lgb_pred_prices})

my_submission.to_csv('lgb_pred.csv',index=False)


# <h3 id="sequential">Sequential model with 3 dense layers</h3>

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from keras.utils import to_categorical
from keras import optimizers
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf


# In[ ]:


test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


# In[ ]:


train = train.dropna(axis=1)


# In[ ]:


X = train.copy()
X = X.drop("SalePrice",axis=1)


# In[ ]:


y = train["SalePrice"]


# In[ ]:





# In[ ]:


X.shape


# In[ ]:


X = pd.get_dummies(X)
test_X = pd.get_dummies(test)
# X, test_X = X.align(test_X,join='left',axis=1)
test_X.reindex(columns = X.columns, fill_value=0)


# In[ ]:


# ohe  = OneHotEncoder(categories = "auto")
# X = ohe.fit(X)
# test_X = ohe.transform(test_X)


# In[ ]:


mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean)/std
test_X = (test_X - mean)/std


# In[ ]:


# X = X.dropna(axis=1)


# In[ ]:


X.shape


# In[ ]:


# print(X["Price"].shape)
X,t_X,y,t_y = train_test_split(X,y,random_state=42, test_size=0.2)


# In[ ]:


print(test_X.shape)


# In[ ]:


model = Sequential([keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(X.shape[1],)),
                    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)])


# In[ ]:


model.weights


# In[ ]:


sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.mean_squared_error,
              optimizer="ADAM",
              metrics=['mae'])


# In[ ]:


y = np.array(y) 


# In[ ]:





# In[ ]:


X.shape


# In[ ]:


model.fit(X, y,
          batch_size=256,
          epochs=5,
          validation_split = 0.3)


# In[ ]:


model.summary()


# In[ ]:


test_X.shape


# In[ ]:


model.evaluate(t_X,t_y)


# In[ ]:


test_X = test_X.dropna(axis=1)


# In[ ]:


test_X.shape


# In[ ]:


for i in test_X.loc[0].isna():
    if i == True:
        print("Yes")


# In[ ]:


model.predict(test_X)


# In[ ]:




