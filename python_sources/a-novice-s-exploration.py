#!/usr/bin/env python
# coding: utf-8

# **A Novice's exploration**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import operator
from operator import itemgetter
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


print(train_df.shape, test_df.shape)


# In[ ]:


print(train_df["ID"].mean(), test_df["ID"].mean(), train_df["ID"].sum(), test_df["ID"].sum())


# In[ ]:


def missing_values(df):
    cols_nan_count = {}
    for col in df.columns:
        nan_count = df[col].isnull().sum()
        if nan_count > 0 :
            cols_nan_count[col] = nan_count
    if len(cols_nan_count.items()) == 0:
        print("No Missing Values")
    else:
        print(cols_nan_count)
    return cols_nan_count


# In[ ]:


missing_values(train_df)


# In[ ]:


missing_values(test_df)


# In[ ]:


train_df.head()


# In[ ]:


idv_cols = list(train_df.columns)
idv_cols.remove(u"ID")
idv_cols.remove(u"y")
binary_cols = idv_cols[8:]


# In[ ]:


binary_cols_mean = {}
for col in binary_cols:
    try:
        binary_cols_mean[col] = train_df[col].mean()
    except TypeError:
        print(col,"is not an integer column")


# In[ ]:


sorted_binary_cols_mean = sorted(binary_cols_mean.items(), key = operator.itemgetter(1))


# In[ ]:


#Selecting middle funnel - leaving out variables having 1's less than 5% or greater than 95%
threshold = 0.01
shortlisted_binary_cols = [a for a,b in sorted_binary_cols_mean if (b>=threshold and b<=(1-threshold))]


# In[ ]:


shortlisted_binary_cols_ttest = {}


# In[ ]:


for col in shortlisted_binary_cols:
    temp = train_df[["y",col]]
    y_1 = list(temp.ix[temp[col]==1,"y"])
    y_0 = list(temp.ix[temp[col]==0,"y"])
    p_value = stats.ttest_ind(a= y_1, b= y_0, equal_var=False).pvalue
    shortlisted_binary_cols_ttest[col] = p_value


# In[ ]:


sorted_shortlisted_binary_cols = sorted(shortlisted_binary_cols_ttest.items(), key = operator.itemgetter(1))


# In[ ]:


#Selecting variables which are significant for 95% Confidence Interval
threshold = 0.3
final_binary_cols = [a for a,b in sorted_shortlisted_binary_cols if b<=threshold]


# In[ ]:


len(final_binary_cols)


# In[ ]:


#Picking some rare variant binary variables which are highly significant
#Selecting middle funnel - leaving out variables having 1's less than 5% or greater than 95%
threshold = 0.01
leftout_binary_cols = [a for a,b in sorted_binary_cols_mean if (b>=0.003 and b<(threshold))]


# In[ ]:


len(leftout_binary_cols)


# In[ ]:


leftout_binary_cols_ttest = {}


# In[ ]:


for col in leftout_binary_cols:
    temp = train_df[["y",col]]
    y_1 = list(temp.ix[temp[col]==1,"y"])
    y_0 = list(temp.ix[temp[col]==0,"y"])
    p_value = stats.ttest_ind(a= y_1, b= y_0, equal_var=False).pvalue
    leftout_binary_cols_ttest[col] = p_value


# In[ ]:


sorted_leftout_binary_cols = sorted(leftout_binary_cols_ttest.items(), key = operator.itemgetter(1))


# In[ ]:


threshold = 0.15
leftout_sig_binary_cols = [a for a,b in sorted_leftout_binary_cols if b<=threshold]


# In[ ]:


len(leftout_sig_binary_cols)


# In[ ]:


print(len(leftout_sig_binary_cols), len(final_binary_cols))


# In[ ]:


final_binary_cols = final_binary_cols + leftout_sig_binary_cols


# In[ ]:


categorical_cols = train_df.columns[2:10]


# In[ ]:


for col in categorical_cols:
    train_df[col] = train_df[col].astype('category')
    print(col, len(set(train_df[col])))


# In[ ]:


final_idvs = list(categorical_cols) + list(final_binary_cols)


# In[ ]:


#Merging train and tests to convert categorical str variables into their equivalent numeric
train_and_test = train_df[final_idvs]
print(train_and_test.shape)
train_and_test = train_and_test.append(test_df[final_idvs])
print(train_and_test.shape)


# In[ ]:


temp = pd.get_dummies(train_and_test[categorical_cols])


# In[ ]:


temp.reset_index(inplace=True)


# In[ ]:


not_present_in_train = []
not_present_in_test = []
present_in_both = []
for col in temp.columns:
    train_sum = temp.ix[:4208, col].sum()
    test_sum = temp.ix[4209:, col].sum()
    #print(train_sum, test_sum)
    if (train_sum ==0 and test_sum >0):
        print(col,"not present in train")
        not_present_in_train.append(col)
    elif (test_sum ==0 and train_sum >0):
        print(col,"not present in test")
        not_present_in_test.append(col)
    else:
        present_in_both.append(col)


# In[ ]:


print(not_present_in_train, not_present_in_test)


# In[ ]:


category_cols_mean = {}
for col in present_in_both:
    try:
        category_cols_mean[col] = temp.ix[:4208,col].mean()
    except TypeError:
        print(col,"is not an integer column")
sorted_category_cols_mean = sorted(category_cols_mean.items(), key = operator.itemgetter(1))      
threshold = 0.01
shortlisted_category_cols = [a for a,b in sorted_category_cols_mean if (b>=threshold and b<=(1-threshold))]


# In[ ]:


train_dummies = temp.ix[:4208,shortlisted_category_cols]
test_dummies = temp.ix[4209:,shortlisted_category_cols]
print(train_dummies.shape, test_dummies.shape)


# In[ ]:


train = pd.concat([train_df, train_dummies],axis=1)
print(train.shape)
test = pd.concat([test_df, test_dummies.reset_index(drop=True)],axis=1)
print(test.shape)


# In[ ]:


shortlisted_category_cols_ttest = {}

for col in shortlisted_category_cols:
    temp = train[["y",col]]
    #print(temp.head())
    y_1 = list(temp.ix[temp[col]==1,"y"])
    y_0 = list(temp.ix[temp[col]==0,"y"])
    p_value = stats.ttest_ind(a= y_1, b= y_0, equal_var=False).pvalue
    shortlisted_category_cols_ttest[col] = p_value


# In[ ]:


sorted_shortlisted_category_cols = sorted(shortlisted_category_cols_ttest.items(), key = operator.itemgetter(1))
#Selecting variables which are significant for 95% Confidence Interval
threshold = 0.3
final_category_cols = [a for a,b in sorted_shortlisted_category_cols if b<=threshold]


# In[ ]:


final_idvs = list(set(list(final_category_cols) + list(final_binary_cols)))


# In[ ]:


len(final_idvs)


# In[ ]:


from sklearn import linear_model


# In[ ]:


train[["ID","y"]+final_idvs].to_csv("train_selected_dummies.csv",index=None)


# In[ ]:


test[["ID"]+final_idvs].to_csv("test_selected_dummies.csv",index=None)


# In[ ]:


len(final_idvs)


# In[ ]:


import statsmodels.api as sm
import math


# In[ ]:


y = [math.log(each) for each in train["y"]]


# In[ ]:


math.log(130)


# In[ ]:


y_old = trai
n["y"]
train["y"] = y
model = sm.OLS(train["y"], train[final_idvs])


# In[ ]:


model = sm.OLS(train["y"], train[final_idvs])


# In[ ]:


results = model.fit()
print(results.summary())


# In[ ]:


var_imp = pd.DataFrame(results.pvalues).reset_index(inplace=False)
var_imp.columns = ["Variable","pvalue"]
refined_final_vars = list(var_imp.ix[var_imp["pvalue"] <= 0.3,"Variable"])


# In[ ]:


print(len(refined_final_vars))


# In[ ]:


model = sm.OLS(train["y"], train[refined_final_vars])
results = model.fit()
print(results.summary())


# In[ ]:


regr = linear_model.LinearRegression()
regr.fit(train[refined_final_vars], train["y"])
pred = regr.predict(train[refined_final_vars])


# In[ ]:


max(pred), min(pred)


# In[ ]:


linear_pred = [math.exp(each) for each in pred]


# In[ ]:


#print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(train[refined_final_vars]) - train["y"]) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(train[refined_final_vars], train["y"]))


# In[ ]:


from sklearn.ensemble.forest import RandomForestRegressor
regr_rf = RandomForestRegressor(n_estimators = 500)
regr_rf.fit(train[final_idvs], train["y"])


# In[ ]:


pred = list(regr_rf.predict(test[final_idvs]))


# In[ ]:


max(pred)


# In[ ]:


rf_pred = [math.exp(each) for each in pred]


# In[ ]:


import xgboost as xgb
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1
}
dtrain = xgb.DMatrix(train[refined_final_vars], train["y"], feature_names=refined_final_vars)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()


# In[ ]:


pred = model.predict(xgb.DMatrix(test[refined_final_vars], feature_names=refined_final_vars))


# In[ ]:


pred_train = model.predict(xgb.DMatrix(train[refined_final_vars], feature_names=refined_final_vars))


# In[ ]:


xgb_pred_train = [math.exp(each) for each in pred_train]


# In[ ]:


xgb_pred = [math.exp(each) for each in pred]


# In[ ]:


print(xgb_pred_train[:10],y_old[:10])


# In[ ]:


op_df = pd.DataFrame({"ID":test_df["ID"],"y":rf_pred})
op_df.to_csv("Submission16_rf.csv",index=None)


# In[ ]:


train_eval = pd.DataFrame({"ID":train["ID"],"y":train["y"],"pred":pred_train})


# In[ ]:


linear_train_pred = regr.predict(train[refined_final_vars])


# In[ ]:


rf_train_pred = list(regr_rf.predict(train[refined_final_vars]))


# In[ ]:


train_eval = pd.DataFrame({"ID":train["ID"],"y":train["y"],"xgb_pred":pred_train,"linear_pred":linear_train_pred,"rf_pred":rf_train_pred})


# In[ ]:


print(rf_train_pred[:10])


# In[ ]:


train_eval.head()


# In[ ]:


from sklearn.metrics import r2_score

print(r2_score(train_eval["linear_pred"], train_eval["y"]))
print(r2_score(train_eval["rf_pred"], train_eval["y"]))
print(r2_score(train_eval["xgb_pred"], train_eval["y"]))
print(pow(2,5))


# In[ ]:


s = 0
s_rf = 0
s_linear = 0
for i in range(len(train_eval)):
    s += pow(pow(((train_eval.ix[i,"xgb_pred"]-train_eval.ix[i,"y"])/train_eval.ix[i,"y"]),2),0.5)
    s_rf += pow(pow(((rf_train_pred[i]-train_eval.ix[i,"y"])/train_eval.ix[i,"y"]),2),0.5)
    s_linear += pow(pow(((linear_train_pred[i]-train_eval.ix[i,"y"])/train_eval.ix[i,"y"]),2),0.5)
print(s, s/len(train_eval), s_rf, s_rf/len(train_eval), s_linear, s_linear/len(train_eval))


# In[ ]:


# Following section contains Keras implementation from umbertogriffo


# In[ ]:


# preprocessing/decomposition
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, KernelPCA

# keras 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
# define custom R2 metrics for Keras backend
from keras import backend as K
# to tune the NN
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam

# model evaluation
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# feature selection
from sklearn.feature_selection import f_regression, mutual_info_regression, VarianceThreshold
model_path = 'keras_model.h5'


# In[ ]:


def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    


# In[ ]:


dimensions = len(refined_final_vars)
print(dimensions)


# In[ ]:


def model():
    model = Sequential()
    #input layer
    model.add(Dense(dimensions, input_dim=dimensions, kernel_constraint=maxnorm(5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4)) # Reduce Overfitting With Dropout Regularization
    # hidden layers
    model.add(Dense(dimensions,kernel_constraint=maxnorm(5)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    
    model.add(Dense(dimensions//2,kernel_constraint=maxnorm(5)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    
    model.add(Dense(dimensions//4))
    model.add(Activation('tanh'))
  
    # output layer (y_pred)
    model.add(Dense(1, activation='linear'))
    # Use a large learning rate with decay and a large momentum. Increase your learning rate by a factor of 10 to 100 and use a high momentum value of 0.9 or 0.99
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # compile this model
    model.compile(loss='mean_squared_error', # one may use 'mean_absolute_error' as alternative
                  optimizer='adam',
                  metrics=[r2_keras] # you can add several if needed
                 )
    
    # Visualize NN architecture
    print(model.summary())
    return model


# In[ ]:


#activation functions for hidden layers
act_func = 'tanh' # could be 'relu', 'sigmoid', ...tanh

# make np.seed fixed
# np.random.seed(seed)

# initialize estimator, wrap model in KerasRegressor
estimator = KerasRegressor(
    build_fn=model, 
    nb_epoch=300, 
    batch_size=100,
    verbose=1
)


# In[ ]:


# prepare callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=10,
        verbose=1),
    ModelCheckpoint(
        model_path, 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=0)
]


# In[ ]:


# fit estimator
history = estimator.fit(
    np.array(train.ix[:4008,refined_final_vars]), 
    np.array(train.ix[:4008,"y"]), 
    epochs=200, # increase it to 20-100 to get better results
    validation_data=(np.array(train.ix[4009:,refined_final_vars]), np.array(train.ix[4009:,"y"])),
    verbose=2,
    callbacks=callbacks,
    shuffle=True
)


# In[ ]:


res = estimator.predict(np.array(test[refined_final_vars])).ravel()
print(res)

# create df and convert it to csv
output = pd.DataFrame({'ID': test["ID"], 'y': res})
output.to_csv('submission15_keras_bl.csv', index=False)


# In[ ]:


res1 = estimator.predict(np.array(train[refined_final_vars]))


# In[ ]:


print(res1[:10])


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


mean_squared_error(res1,train["y"])


# In[ ]:




