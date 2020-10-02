#!/usr/bin/env python
# coding: utf-8

# # **Intro**
# I used this competition/kernel to just add to my portfolio a bit while also working on my skills. I didn't at all look at what others were doing for this kernel, since I wanted to work on this as a real world project where one doesn't have all this external help.  I am now going to look at what others did to learn from them, but I will still leave this as is.  The main idea for me here was just to practice and learn.  You will notice that I only used a subset of features.  I did this to try to get a simple model working without spending a ton of time.  I wasn't worried about the score as much as building something from start to finish independently.
# Thanks for checking it out, and please feel free to comment.

# # **Imports**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, matplotlib, math, copy
import seaborn as sns
import matplotlib.pyplot as plt

print(os.listdir("../input"))
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


# # Data Downloads

# In[ ]:


new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
merchants = pd.read_csv('../input/merchants.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
historical_transactions = pd.read_csv('../input/historical_transactions.csv')
Data_Dictionary = pd.read_excel('../input/Data_Dictionary.xlsx')


# # **EDA - Exploratory Data Analysis**

# In[ ]:


sample_submission.head()


# In[ ]:


sns.distplot(train['target']);


# In[ ]:


print(new_merchant_transactions.shape)
new_merchant_transactions.head(2)


# In[ ]:


print(train.shape)
train.head(2)


# In[ ]:


train.nunique()


# In[ ]:


print(test.shape)
test.head(2)


# In[ ]:


print(merchants.shape)
merchants.head(2)
#merchants.nunique()


# In[ ]:


print(historical_transactions.shape)
historical_transactions.head(2)


# Maybe it's already obvious... but we're looking for unique cardholders and just checking on the counts here

# In[ ]:


print(historical_transactions[['card_id','merchant_id']].nunique())
print(train.shape[0])
print(test.shape[0])
print((train.shape[0])+(test.shape[0]))


# In[ ]:


Data_Dictionary


# ## Data Types and Basic Visuals (for historicals_transactions)

# ### Unique Counts
# Taking a quick sample of 100,000 to choose which non-numeric features can be dummy variable based on unique counts

# In[ ]:


historical_trans_sample = historical_transactions[:100000].nunique()
data_types = historical_transactions.dtypes
features = pd.DataFrame({'unique':historical_trans_sample,'dtypes':data_types})
features


# In[ ]:


features_numeric = features[features['dtypes']!=object].index.tolist()
print(features_numeric)


# In[ ]:


features_dummy = features[(features['dtypes']==object)&(features['unique']<30)].index.tolist()
features_dummy


# In[ ]:


#note that this just pulls a sample from each
columns = 3
rows = math.ceil(len(features_numeric)/columns)

fig, ax = plt.subplots(rows,columns,figsize = (16,8))
for i, feature in list(enumerate(features_numeric)):
    sns.distplot(historical_transactions[feature].dropna().sample(20000),ax= ax[math.floor(i/columns),i%columns],axlabel=feature)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# # **Feature Engineering**

# ### purchase_amount

# In[ ]:


def tf(pd_series):
    return np.array(pd_series).reshape(-1,1)

def transform_viz(pd_series,transform_list = [QuantileTransformer(), MinMaxScaler(), StandardScaler(),PowerTransformer()]):
    transform_list = [QuantileTransformer(), MinMaxScaler(), StandardScaler(),PowerTransformer()]
    fig, ax = plt.subplots(1,len(transform_list)+1,figsize = (18,3))
    sns.distplot(pd_series.dropna().sample(1000),ax = ax[0]);
    for i, transformer in enumerate(transform_list):
        sns.distplot(transformer.fit_transform(tf(pd_series.dropna().sample(1000))),ax = ax[i+1]);


# In[ ]:


def plot_details(pd_series):
    plt.figure(figsize = (3,3))
    std, mean, skew = pd_series.std(), pd_series.mean(), pd_series.skew()
    print('shape: {}, mean: {:.2f}, std: {:.2f}, skew: {:.2f}'.format(pd_series.shape,mean,std,skew))
    sns.distplot(pd_series)
    plt.show();
    return std, mean, skew


# In[ ]:


def remove_outliers_and_skew(pd_series,skew_threshold = 1, remove_outliers = True, deviations = 2.15):
    """
    args:
        pd-series: required
        skew_threshold: skew calc above which series with be log-transformed
        remove_outliers: will return pd_series with ...
        deviations: ...
    """
    print('original distribution')
    std, mean, skew = plot_details(pd_series)
    if skew > 1:
        print('log transform original')
        new_series = np.log1p(pd_series)
        if remove_outliers:
            std, mean, skew = plot_details(new_series)
            print('update size')
            new_series = new_series[new_series.between(mean - (std * deviations),mean + (std * deviations))]
    else:
        if remove_outliers:
            print('update size')
            new_series = new_series[new_series.between(mean - (std * deviations),mean + (std * deviations))]
        else:
            new_series = pd_series
    plot_details(new_series)
    return new_series


# Will pull customers' total purchase amount

# In[ ]:


historical_transactions['purchase_amount'] = historical_transactions['purchase_amount'] - historical_transactions['purchase_amount'].min()
purchase_amount_sum = historical_transactions[['purchase_amount','card_id']].groupby('card_id').sum()['purchase_amount']
purchase_amount_sum_transformed = remove_outliers_and_skew(purchase_amount_sum,1,False,6) ##########
del purchase_amount_sum


# In[ ]:


sns.distplot(historical_transactions['month_lag'].abs().sample(20000))


# In[ ]:


historical_transactions['month_lag'] = historical_transactions['month_lag'].abs()
month_lag_abs_mean = historical_transactions[['month_lag','card_id']].groupby('card_id').mean()['month_lag'] ##########
month_lag_abs_mean = remove_outliers_and_skew(month_lag_abs_mean,1,False,6)


# In[ ]:


category_2_dummies = pd.get_dummies(historical_transactions['category_2'],prefix='cat_2')
category_2_dummies = pd.merge(category_2_dummies,historical_transactions[['card_id']],left_index = True, right_index = True)
category_2_dummies = category_2_dummies.groupby('card_id').sum()


# In[ ]:


category_1_dummies = pd.get_dummies(historical_transactions['category_1'],prefix='cat_1')
category_1_dummies = pd.merge(category_1_dummies,historical_transactions[['card_id']],left_index = True, right_index = True)
category_1_dummies = category_1_dummies.groupby('card_id').sum()


# # **Merge Data for Train and Test**

# In[ ]:


card_data = pd.merge(month_lag_abs_mean.to_frame(), purchase_amount_sum_transformed.to_frame(),left_index = True, right_index = True,how = 'inner')
print(card_data.shape)
card_data.head(2)


# In[ ]:


dummies = pd.merge(category_1_dummies, category_2_dummies, left_index = True, right_index = True, how = 'inner')
del category_1_dummies, category_2_dummies
print(dummies.shape)
dummies.head(2)


# In[ ]:


card_data = pd.merge(card_data,dummies, left_index = True, right_index = True, how = 'inner')
card_data.head(2)


# In[ ]:


columns = 2

train_numeric = list(train.dtypes[train.dtypes != object].index)
rows = math.ceil(len(train_numeric)/columns)

fig, ax = plt.subplots(rows,columns,figsize = (8,4))
for i, feature in list(enumerate(train_numeric)):
    #sns.distplot(train[feature].dropna().sample(20000),ax= ax[math.floor(i/columns),i%columns],axlabel=feature)
    sns.distplot(train[feature].dropna().sample(20000),ax= ax[math.floor(i/columns),i%columns])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


# In[ ]:


results = pd.concat([train,test])
results = results.drop(['first_active_month'],axis = 1)
print(results.shape)
results.head()


# In[ ]:


card_data.shape


# In[ ]:


results[results['target'].isnull()].shape[0] + results.dropna().shape[0] - results.shape[0]


# In[ ]:


all_data = pd.merge(card_data, results, left_index = True, right_on = 'card_id', how = 'inner')
all_data.set_index('card_id', inplace = True)


# In[ ]:


results_train = all_data.dropna()
results_train.shape


# In[ ]:


results_test = all_data[all_data['target'].isnull()]
results_test.shape


# In[ ]:


results_train.shape[0] + results_test.shape[0]


# In[ ]:


results_test.head(2)


# In[ ]:


results_test = results_test.drop('target',axis = 1)


# In[ ]:


X, y = results_train.drop(['target'],axis = 1), results_train['target']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# # **Model**

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def get_mse(model,alpha):
    run_model = model(alpha = alpha)
    run_model.fit(X_train,y_train)
    return mean_squared_error(y_pred=run_model.predict(X_test),y_true=y_test)

alphas = [.01,.05,0.1,0.5,1,2,3,5,10,20,40,100,1000,5000]
ridge_mses = [get_mse(Ridge,x) for x in alphas]
plt.plot(alphas,ridge_mses)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train,y_train)
mean_squared_error(y_pred=model.predict(X_test),y_true=y_test)


# In[ ]:


from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
model.fit(X_train,y_train)
mean_squared_error(y_pred=model.predict(X_test),y_true=y_test)


# In[ ]:


from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor()
model.fit(X_train,y_train)
mean_squared_error(y_pred=model.predict(X_test),y_true=y_test)


# In[ ]:


from sklearn.linear_model import Lasso
model = Lasso()
model.fit(X_train,y_train)
mean_squared_error(y_pred=model.predict(X_test),y_true=y_test)


# In[ ]:


from sklearn.linear_model import ElasticNet
model = ElasticNet()
model.fit(X_train,y_train)
mean_squared_error(y_pred=model.predict(X_test),y_true=y_test)


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
mean_squared_error(y_pred=model.predict(X_test),y_true=y_test)


# In[ ]:


from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X_train,y_train)
mean_squared_error(y_pred=model.predict(X_test),y_true=y_test)


# In[ ]:


from sklearn.linear_model import 
model = ExtraTreesRegressor()
model.fit(X_train,y_train)
mean_squared_error(y_pred=model.predict(X_test),y_true=y_test)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(X_train,y_train)
mean_squared_error(y_pred=model.predict(X_test),y_true=y_test)


# In[ ]:


sample_submission.head(2)


# In[ ]:


results_test.head()


# In[ ]:


predictions = model.predict(results_test)
sub = pd.DataFrame({'card_id':results_test.index, 'target':predictions})
sub.head()


# In[ ]:


sns.distplot(sub['target'])


# In[ ]:


sns.distplot(train['target'])`


# In[ ]:


sub.to_csv('submission.csv',index=False)

