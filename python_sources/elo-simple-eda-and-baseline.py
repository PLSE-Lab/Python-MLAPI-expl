#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb


# #### If you find this kernel usefull , pls do upvote. It really motivates

# ## Dataset exploration

# In[ ]:


get_ipython().system('ls ../input/')


# ## Lets peek into each of files

# In[ ]:


train = pd.read_csv("../input/train.csv")


# In[ ]:


train.head()


# Lets look at each of the columns to understand more 
# 
# Starting with card_id

# In[ ]:


train["card_id"].head()


# ## ^^Card Id is the unique id representing each user 
# ## Lets check the sanity of this data

# In[ ]:


card_id_groupby_count = train.groupby("card_id")["target","first_active_month"].agg(["count","size"]).reset_index()


# In[ ]:


card_id_groupby_count.columns = pd.Index(["_".join(col) for col in card_id_groupby_count.columns.tolist()])


# In[ ]:


card_id_groupby_count.head()


# In[ ]:


card_id_groupby_count[card_id_groupby_count["target_count"] > 1].count()


# In[ ]:


card_id_groupby_count[card_id_groupby_count["target_size"] > 1].count()


# In[ ]:


card_id_groupby_count[card_id_groupby_count["first_active_month_count"] > 1].count()


# In[ ]:


card_id_groupby_count[card_id_groupby_count["first_active_month_size"] > 1].count()


# ### ^^ Looks good

# ### Lets look into test and do similar sanity check

# In[ ]:


test = pd.read_csv("../input/test.csv")


# In[ ]:


test.head()


# In[ ]:


card_id_test_groupby_count = test.groupby("card_id")["first_active_month"].agg(["count","size"]).reset_index()


# In[ ]:


card_id_test_groupby_count[card_id_test_groupby_count["count"] > 1].count()


# In[ ]:


card_id_test_groupby_count[card_id_test_groupby_count["size"] > 1].count()


# ### ^^ All looks good here too
# 
# ####  Lests look at the distribution of first active month

# In[ ]:


train["first_active_month"] = pd.to_datetime(train["first_active_month"])
test["first_active_month"] = pd.to_datetime(test["first_active_month"])


# In[ ]:


cnt_srs = train['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in train set")
plt.show()

cnt_srs = test['first_active_month'].dt.date.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First active month', fontsize=12)
plt.ylabel('Number of cards', fontsize=12)
plt.title("First active month count in test set")
plt.show()


# ### ^^ The distribution of test and train is similar
# 
# ### Lets look at the target variable distribution

# In[ ]:


train["target"].describe()


# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train["target"].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train["target"],bins=50, kde=False, color="red")
plt.title("Histogram of Loyalty score")
plt.xlabel('Loyalty score', fontsize=12)
plt.show()


# ## ^^ There is small outlier values , but rest looks good
# 
# ### Lets look into three other features

# In[ ]:


def plotViolin(feature):
    plt.figure(figsize=(8,4))
    sns.violinplot(x=feature, y="target", data=train)
    plt.xticks(rotation='vertical')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Loyalty score', fontsize=12)
    plt.title("Feature 1 distribution")
    plt.show()

plotViolin("feature_1")
plotViolin("feature_2")
plotViolin("feature_3")


# In[ ]:


sns.distplot(train["feature_1"],kde=False)


# In[ ]:


sns.distplot(train["feature_2"],kde=False)


# In[ ]:


sns.distplot(train["feature_3"],kde=False)


# ### ^^ The data looks more of discreate than continuous , And the distribution of target variable seems to same for different values which means we dont see yet if these features are significant or not ,  we need see how models see it 

# ### Lets look at the historical transaction data

# In[ ]:


historical_transaction = pd.read_csv("../input/historical_transactions.csv")


# In[ ]:


historical_transaction.describe(include="all")


# In[ ]:


historical_transaction.columns


# In[ ]:


historical_transaction.count()


# ### Lets look at each of the above columns 
# 
# Lets start to look at authorized_flag

# In[ ]:


columns_check = ['authorized_flag', 'category_1', 'installments',                 'category_3','month_lag','category_2','state_id','subsector_id']


# In[ ]:


def plot_distribution(column,df):
    distribution  = df[column].value_counts()
    plt.figure(figsize=(10,5))
    sns.barplot(distribution.index, distribution.values, alpha=0.8)
    plt.title('Distribution of ' + column)
    plt.ylabel("distribution", fontsize=12)
    plt.xlabel(column, fontsize=12)
    plt.show()


# Lets look at the distribution of city_id flag

# In[ ]:


for col in columns_check:
    plot_distribution(col,historical_transaction)


# ## The summary of the data present in historical transaction file is as follows , more digging needed
# ###### authorized_flag               -  Looks like if the transaction was authorized or not , 
# ###### card_id                             - This is the unique card id  
# ###### city_id                               - city in which the transaction was done
# ###### category_1                        - some masked category , has only two values and one of them is dominating
# ###### category_2                       - Some masked category, Take 5 values.
# ###### category_3                       - Some masked category, Take 3 values
# ###### merchant_category_id    - The category the merchant belongs to 
# ###### installments                    - My guess is that it is the installments at which the purchase was completed , most of installments were completed in 0 , some at 1 . Not quite sure what 0 means ....
# ###### merchant_id                    - Unique Merchant Id 
# ###### month_lag                       - Month lag to reference date
# ###### purchase_amount          - Amount of transaction 
# ###### purchase_date               - purchase date
# ###### state_id                           - state in which the transaction was done
# ###### subsector_id                   - something similar to merchant_category_id .
# 
# 

# ## Lets now shift our focus to merchants.csv file

# In[ ]:


merchants = pd.read_csv("../input/merchants.csv")


# In[ ]:


merchants.head(5)


# In[ ]:


merchants.head()


# In[ ]:


columns_to_check_merchant = [
       'subsector_id', 'category_1',
       'most_recent_sales_range', 'most_recent_purchases_range',
       'category_4', 'city_id', 'state_id', 'category_2']


# In[ ]:


['merchant_group_id', 'merchant_category_id',
       'numerical_1', 'numerical_2', 'category_1',
       'most_recent_sales_range', 'most_recent_purchases_range',       
       'category_4', 'city_id', 'state_id', 'category_2']


# In[ ]:


for col in columns_to_check_merchant:
    plot_distribution(col,merchants)


# In[ ]:


merchants["numerical_1"].value_counts()


# ## ^^ most of values in column numerical_1 is less

# In[ ]:


merchants["numerical_2"].value_counts()


# #### ^^ The distribution and range of both columns numerical_1 and numerical_2 very similar , are these two related ?

# In[ ]:


merchants[["numerical_1","numerical_2"]].corr()


# #### ^^ Yes these two variables are highly co related

# In[ ]:





# ## The summary of the data present in historical transaction file is as follows , more digging needed
# #### merchant_id                                        : Unique Merchant Id 
# #### merchant_group_id                          : Yet another group Id , needs to be investigated   
# #### merchant_category_id                     : This field looks similar to the one we say in historical transaction file   , we have to find how these two are related
# #### subsector_id                                       :  This field looks similar to the one we say in historical transaction file   , we have to find how these two are related
# #### numerical_1                                         : Highly correlated with numerical_2, 
# #### numerical_2                                        :  
# #### category_1                                           :  This field looks similar to the one we say in historical transaction file   , we have to find how these two are related
# #### most_recent_sales_range              :  Yet another categorical data.       
# #### most_recent_purchases_range   :  Yet another categorical data.                       
# #### avg_sales_lag3                                  :  To look deeper
# #### avg_purchases_lag3                             :  To look deeper
# #### active_months_lag3                             :  To look deeper
# #### avg_sales_lag6                                 :  To look deeper
# #### avg_purchases_lag6                             :  To look deeper
# #### active_months_lag6                             :  To look deeper
# #### avg_sales_lag12                                :  To look deeper
# #### avg_purchases_lag12                            :  To look deeper
# #### active_months_lag12                            :  To look deeper
# #### category_4                                        : Yet another categorical data
# #### city_id                                                : city id of the merchant
# #### state_id                                             : state of the merchant 
# #### category_2                                      : This field looks similar to the one we say in historical transaction file   , we have to find how these two are related

# ## Lets look at the last file New Merchant Transactions

# In[ ]:


new_trans_df = pd.read_csv("../input/new_merchant_transactions.csv")
new_trans_df.head()


# #### ^^ these columns looks similar to ones present in historical transactions file , let verify it 

# In[ ]:


new_trans_df.columns == historical_transaction.columns


# ## Lets Quickly build a baseline model 

# ### Adding few Simple features

# In[ ]:


gdf = historical_transaction.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "num_hist_transactions"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")


# In[ ]:


gdf = historical_transaction.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")


# In[ ]:


train.columns


# In[ ]:


gdf = new_trans_df.groupby("card_id")
gdf = gdf["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
gdf.columns = ["card_id", "sum_merch_trans", "mean_merch_trans", "std_merch_trans", "min_merch_trans", "max_merch_trans"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")


# In[ ]:


gdf = new_trans_df.groupby("card_id")
gdf = gdf["purchase_amount"].size().reset_index()
gdf.columns = ["card_id", "num_merch_transactions"]
train = pd.merge(train, gdf, on="card_id", how="left")
test = pd.merge(test, gdf, on="card_id", how="left")


# In[ ]:


target_col = "target"
train["year"] = train["first_active_month"].dt.year
test["year"] = test["first_active_month"].dt.year
train["month"] = train["first_active_month"].dt.month
test["month"] = test["first_active_month"].dt.month

cols_to_use = ["feature_1", "feature_2", "feature_3", "year", "month", 
               "num_hist_transactions", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", 
               "min_hist_trans", "max_hist_trans",
               "num_merch_transactions", "sum_merch_trans", "mean_merch_trans", "std_merch_trans",
               "min_merch_trans", "max_merch_trans",
              ]

def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "min_child_weight" : 50,
        "learning_rate" : 0.05,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result

train_X = train[cols_to_use]
test_X = test[cols_to_use]
train_y = train[target_col].values

pred_test = 0
kf = model_selection.KFold(n_splits=5, random_state=2018, shuffle=True)
for dev_index, val_index in kf.split(train):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test += pred_test_tmp
pred_test /= 5.
    


# In[ ]:


fig, ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# In[ ]:


sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = pred_test
sub_df.to_csv("baseline_lgb.csv", index=False)


# In[ ]:




