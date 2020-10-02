#!/usr/bin/env python
# coding: utf-8

# # TLDR: I have not found a way to use weights or a custom evaluation function to make and lgbm model optimize for the WRMSSE. All adjustments made the baseline model perform worse. If anyone has any words to guide me in the right direction, it would be greatly appreciated. 
# 
# 
# This notebook is meant to test using weights and scales to improve the WRMSSE score. I used the same data and model hyper parameters in each model. I assembled the training data, including the weights and scaling factors in another kernal. The features came from [@kyakovlev](https://www.kaggle.com/kyakovlev). I used [lags fe](https://www.kaggle.com/kyakovlev/m5-lags-features) and [simple fe](https://www.kaggle.com/kyakovlev/m5-simple-fe). I got the weights from the m5methods. I got the scales manually using the data in simple fe and lags fe notebooks. I grouped by id, took difference between sales and a lag 1 shift, squared it, took the mean, and took the squareroot. 
# Training data: one year of data leading up to day 1886
# Validation data: 1886 to 1913
# Although the weights werent meant for this validation period, I had still thought that including the weights would help. 
# I tried two weighting schemes and two custom objective functions. 
# 5 models: 
# 1. Baseline:  normal hyper parameters taken from public notebooks. Look below for specifics. 
# 2. Weighted: took the weights/scales and input that into the training lgbm Dataset object before training
# 3. Weighted squared: took (weights/scales)^2 and input that in as before. This was done since I thought it represented the derivative of the square of the WRMSSE function, and I thought I could minimize the WRMSSE by trying to minimize its square. 
# 4. Custom WMSSE: Used custom objective function with weights as in 2
# 5. Cusom WMSSE with squared weights: Same as 4, but with the weights from 3. 
# 
# Trained all models for 1500 iterations and took the score for the validation period from every 25 booster(iteration) of each model 
# 
# Results: 1 Basline best was around .53, followed by 4, and closely by 2, with about a .57 score. 3 came in around 1.39 and number 5 was terrible with something like 5.0, 
# 
# Conclustion: I have not found a way to use weights or a custom evaluation function to make and lgbm model optimize for the WRMSSE. 
# 
# Next steps: Next, I will look into ways to improve my weighting or loss function. Some options include: 
# - Change the target to sales * price so that the products are naturally weighted to favor expensive items. 
# - Combining the last step with using scaling factors 
# - Looking into pytorch autograd to try to get the exact gradient for the total WRMSSE 
# - Trying to take the derivative of the entire WRMSSE function, maybe even as if we were just predicting a one day horizon, which might make the calculations easier. 
# - Training models to predict different levels of aggregation then "averaging" all the predictions so that they come to an optimal "agreement"

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# For this experiment will be using the 
# last 28 days of data as our validation set
# and 1 year of data to train on 
TRAIN_START = 1913 - 28 - 365 # last day - val period -length of train period
TRAIN_END = 1913 - 28 # final day subtract days of validation
NUM_ITERATIONS = 1500


# In[ ]:


# We start by reading in the data and looking at the features 
# we have to use. All features based on 28 day lag so 
# we can make one model for all 28 days of prediction
df = pd.read_pickle('/kaggle/input/df-basic-scale-weight-ready-pickle/df_basic_weight_scale.pkl')
df.columns


# In[ ]:


# Create the two columns to test 
# with our custom objective function
df['scaled_weight'] = df.weight/df.scale
# Also a squared version, which reflects 
# the derivative of the squared WRMSSE
df['scaled_weight_squared'] = (df.weight/df.scale)**2


# In[ ]:


# List category features for lgbm model
cat_feats = [ 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 
             'event_type_1', 'event_name_2', 'event_type_2']
# List the columns to drop from our training 
drop_cols = ['id', 'd', 'sales', 'scale', 'weight', 'scaled_weight', 'scaled_weight_squared']


# In[ ]:


# Make the training data, our df for one year before testing period
X_train = df[(df.d >= TRAIN_START) & (df.d <= TRAIN_END)].drop(drop_cols, axis=1)
y_train = df[(df.d >= TRAIN_START) & (df.d <= TRAIN_END)]['sales']

# Also get the weight scale columns before we delete df
weight_train = df[(df.d >= TRAIN_START) & (df.d <= TRAIN_END)]['scaled_weight']
weight_squared_train = df[(df.d >= TRAIN_START) & (df.d <= TRAIN_END)]['scaled_weight_squared']


X_test = df[(df.d > TRAIN_END) & (df.d <= TRAIN_END + 28)].drop(drop_cols, axis=1)
# y_test = OUR PREDICTIONS, TO BE SCORED AGAINST THE GROUND TRUTH

# We will also need the id column of the test set to join with predictions
X_test_id = df[(df.d > TRAIN_END) & (df.d <= TRAIN_END + 28)][['id', 'd']]

del df


# In[ ]:


X_train.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "# We will take a 5% sample for 'fake' (in train sample) validation, but \n# we are most concerned with how the models score on the \n# true validation set \nnp.random.seed(777)\n\nfake_valid_inds = np.random.choice(X_train.index.values, 500_000, replace = False)\ntrain_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)")


# In[ ]:


# Make the first training set with no weights for a baseline
import lightgbm as lgb
train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds],
                         categorical_feature=cat_feats, free_raw_data=False)
fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label = y_train.loc[fake_valid_inds],
                              categorical_feature=cat_feats,
                 free_raw_data=False)
# This is a random sample, we're not gonna apply any time series train-test-split tricks here


# In[ ]:


# Start with some reasonable params, not 
# optimal, but good enought to measure effect 
# of weigts and loss functions
params = {
        "objective" : "regression",
        "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.075,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
#         "nthread" : 4
        "metric": ["rmse"],
    'verbosity': 1,
    'num_iterations' : NUM_ITERATIONS,
    'num_leaves': 128,
    "min_data_in_leaf": 50,
}


# In[ ]:


get_ipython().run_cell_magic('time', '', 'baseline_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=100) \n# Delete data to save memory\ndel train_data, fake_valid_data')


# In[ ]:


# Lets train a model with weights
train_data = lgb.Dataset(X_train.loc[train_inds] , 
                         label = y_train.loc[train_inds],
                         weight = weight_train[train_inds], # putting in weights
                         categorical_feature=cat_feats, 
                         free_raw_data=False)
fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], 
                              label = y_train.loc[fake_valid_inds],
                              weight = weight_train[fake_valid_inds], # putting in weights
                              categorical_feature=cat_feats,
                 free_raw_data=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'weight_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=100) \n# Delete data to save memory\ndel train_data, fake_valid_data')


# In[ ]:


# Lets make another model with the squared scaled weights
train_data = lgb.Dataset(X_train.loc[train_inds] , 
                         label = y_train.loc[train_inds],
                         weight = weight_squared_train[train_inds], # putting in weights
                         categorical_feature=cat_feats, 
                         free_raw_data=False)
fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], 
                              label = y_train.loc[fake_valid_inds],
                              weight = weight_squared_train[fake_valid_inds], # putting in weights
                              categorical_feature=cat_feats,
                 free_raw_data=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'weight_squared_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=100) \n# Delete data to save memory\ndel train_data, fake_valid_data')


# In[ ]:


# Now lets create a custom loss function 
# Custom loss function 
def WMSSE(preds, train_data):
    labels = train_data.get_label()
    weight = train_data.get_weight()
    loss = weight*((preds - labels)**2)
    grad = 2 * weight * (preds - labels)
    hess = 2 * weight
    return grad, hess


# In[ ]:


# Lets train a model with weights and the 
# WMSSE obj function 
train_data = lgb.Dataset(X_train.loc[train_inds] , 
                         label = y_train.loc[train_inds],
                         weight = weight_train[train_inds], # putting in weights
                         categorical_feature=cat_feats, 
                         free_raw_data=False)
fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], 
                              label = y_train.loc[fake_valid_inds],
                              weight = weight_train[fake_valid_inds], # putting in weights
                              categorical_feature=cat_feats,
                 free_raw_data=False)

WMSSE_weight_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=100, fobj=WMSSE) 
# Delete data to save memory
del train_data, fake_valid_data


# In[ ]:


# Lets train a model with squared weights and the 
# WMSSE obj function 
train_data = lgb.Dataset(X_train.loc[train_inds] , 
                         label = y_train.loc[train_inds],
                         weight = weight_squared_train[train_inds], # putting in weights
                         categorical_feature=cat_feats, 
                         free_raw_data=False)
fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], 
                              label = y_train.loc[fake_valid_inds],
                              weight = weight_squared_train[fake_valid_inds], # putting in weights
                              categorical_feature=cat_feats,
                 free_raw_data=False)

WMSSE_weight_squared_lgb = lgb.train(params, train_data, valid_sets = [fake_valid_data], verbose_eval=100, fobj=WMSSE) 
# Delete data to save memory
del train_data, fake_valid_data


# In[ ]:


# Set up an evaluator to test our predictions 
from typing import Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm as tqdm

                                    
class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 0  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        weight_df = self.get_weight_df()

        self.group_ids = (
            'all_id',
            'state_id',
            'store_id',
            'cat_id',
            'dept_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
            'item_id',
            ['item_id', 'state_id'],
            ['item_id', 'store_id']
        )

        for i, group_id in enumerate(tqdm(self.group_ids)):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']
        weight_df = weight_df.loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            all_scores.append(lv_scores.sum())

        return np.mean(all_scores)
    
#######################################################################################################################
# Reading in data 
PATH = '../input/m5-forecasting-accuracy/'
stv = pd.read_csv(f'{PATH}sales_train_validation.csv')
cal = pd.read_csv(f'{PATH}calendar.csv')
ss = pd.read_csv(f'{PATH}sample_submission.csv')
sp = pd.read_csv(f'{PATH}sell_prices.csv')


DAYS_BACK=0 # number of days before 1913 that is the end
# of the validation period

# Creating our train_fold and valid_fold, according to how many days back 
train_fold_df = stv.iloc[:, : -(28 + DAYS_BACK)]
valid_fold_df = stv.iloc[:, -(28 + DAYS_BACK): 1919-DAYS_BACK].copy()

# Instantiating evaluators 
# ee = WRMSSEEvaluator_extra(train_fold_df, valid_fold_df, cal, sp)
e = WRMSSEEvaluator(train_fold_df, valid_fold_df, cal, sp)
del stv, cal 


# In[ ]:


# We will make a function to get the 
# predictions for every 25th booster of a 
# model
def score_model(model, evaluator, test_x, test_id, boosters):
    scores = []
    X_test_id = test_id.copy()
    print('scoring...')
    for booster in boosters: 
        
        preds = model.predict(test_x, num_iteration=booster)
        X_test_id['preds'] = preds
        preds = X_test_id.pivot('id', 'd', 'preds').reset_index()
        preds = pd.merge(ss.iloc[:30490]['id'], preds, on='id', how='left').iloc[:, 1:].values
        scores.append(evaluator.score(preds))
        
    return scores


# In[ ]:


preds = baseline_lgb.predict(X_test)
X_test_id_ = X_test_id.copy()
X_test_id_['preds'] = preds
preds = X_test_id_.pivot('id', 'd', 'preds').reset_index()
preds = pd.merge(ss.iloc[:30490]['id'], preds, on='id', how='left').iloc[:, 1:].values
# scores.append(evaluator.score(preds))


# In[ ]:


e.score(preds)


# In[ ]:


BOOSTERS = [25*i for i in range(1, NUM_ITERATIONS//25 + 1)]


# In[ ]:


scores_dict = {'baseline_lgb': score_model(baseline_lgb, e, X_test, X_test_id, BOOSTERS),
              'weight_lgb': score_model(weight_lgb, e, X_test, X_test_id, BOOSTERS),
              'weight_squared_lgb': score_model(weight_squared_lgb, e, X_test, X_test_id, BOOSTERS),
              'WMSSE_weight_lgb': score_model(WMSSE_weight_lgb, e, X_test, X_test_id, BOOSTERS),
              'WMSSE_weight_squared_lgb': score_model(WMSSE_weight_squared_lgb, e, X_test, X_test_id, BOOSTERS),}


# In[ ]:


scores_df = pd.DataFrame(scores_dict, index=BOOSTERS)


# In[ ]:


scores_df.plot(figsize=(16,6))


# In[ ]:


scores_df


# In[ ]:


scores_df.to_csv('scores_df.csv')


# In[ ]:




