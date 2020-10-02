#!/usr/bin/env python
# coding: utf-8

# Version 8: Thanks again to [JHkerwin](https://www.kaggle.com/jhkerwin), who spotted an error in my previous notebook, weights and scales for the past 2 years. I fixed it, and reran this kernal, so now the weights match perfectly! 
# 
# Version 7: Fixing error in my instansiation of WRMSSEEvaluatordashboard. I had a valid_df input that was for a period ending 28 days before it should have been. Thank you to [JHkerwin](https://www.kaggle.com/jhkerwin) for spotting this error. The result is that our custom metric now scores within .003 of the evaluator dashboard, although the weights are still slightly different and I don't know why exactly. 

# # Need it right now? 
# 1. Put [this weights and scales notebook](https://www.kaggle.com/chrisrichardmiles/weights-and-scales-for-the-past-2-years) in your data input. 
# 2. Copy and paste the next hidden cell into your notebook. It has the helper functions. 
# 3. Copy the next next hidden cell and paste it right below your dataset, which should be in the proper "format" (see below).

# In[ ]:


# import numpy as np 
# import pandas as pd 
# import random

# ########## Getting the scales and weights #############
# path = '/kaggle/input/weights-and-scales-for-the-past-2-years/'

# def get_weights_scales_level_12(df, end_test, path):
#     """Gets the scale, weight, and scaled weight in a dataframe, 
#     aligned with the 'id' column of df. 
#     ::path: input path where weight_scale_x files are
#     """
    
#     # Get the weights and scales for all the levels 
#     wdf = pd.read_csv(f'{path}weight_scale_{end_test-27}.csv')
#     # Get the sqrt of the scale, because I didn't do 
#     # that part in the previous notebook. Then divide the weight by the 
#     # sqrt(scale) to get the proper scaled_weight
#     wdf['scaled_weight'] = wdf.weight/np.sqrt(wdf.scale)

#     # For this function, we just want level 12 weights and scales
#     wdf = wdf[wdf.Level_id == 'Level12']

#     # We make an 'id' column for easy merging, df must have 'id' column
#     wdf['id'] = wdf['Agg_Level_1'] + '_' +  wdf['Agg_Level_2'] + '_validation'

#     # Taking just he columns we want to use in the merge 
#     wdf = wdf[['id', 'scale', 'weight', 'scaled_weight']]

#     # Merge with 'id' column of the df
#     wdf = pd.merge(df[['id']], wdf, on='id', how='left')
    
#     return wdf


# ############ Calculations function #############
# # We will define a function outside of the custom 
# # metric. This is because the custom function only 
# # has two inputs: preds and train_data, and we need 
# # to incorporate more than that to make it flexible. 

# def L12_WRMSSE(preds, actuals, p_horizon, num_products, scale, weight): 
    
#     actuals = actuals[-(p_horizon * num_products):]
#     preds = preds[-(p_horizon * num_products):]
#     diff = actuals - preds

#     # The first step in calculating the wrmsse is 
#     # squareing the daily error.
#     res = diff ** 2

#     # Now divide the result by the appropriate scale
#     # take values of scale to avoid converting res 
#     # to a pandas series
#     res = res/scale.values

#     # The next step is summing accross the horizon
#     # We must reshape our data to get the products
#     # in line. 
#     res = res
#     res = res.reshape(p_horizon, num_products)

#     # Now we take the mean accross the prediction horizon
#     # and take the square root of the result.
#     res = res.mean(axis=0)
#     res = np.sqrt(res)

#     # Now we multiply each result with the appropriate
#     # scaled_weight. We just need the first 30490 entries 
#     # of the scaled_weight column
#     res = res * weight
#     res = res.sum()
#     return res


# In[ ]:


# ## THIS WILL BE THE TRAINING DATA IN YOUR NOTEBOOK
# ## DONT COPY THIS PART. YOU NEED A DATA SET, grid_df, LIKE 
# ## THIS ONE, WITH AN 'id' COLUMN.


# # grid_df = grid_df[grid_df['d']<=END_TEST].reset_index(drop=True)
# # END_TEST = 1913 # last day of the validation set.

# ############### "Fit" custom metric #################
# #####################################################

# ################## Variables ######################
# # Variables needed for the metric. For other training
# # sets, replace grid_df with the training set, but 
# # make sure that the data is ordered by day and everything 
# # is in correct alignment with values for every product and 
# # every day. 

# wdf = get_weights_scales_level_12(grid_df, END_TEST, path)

# P_HORIZON = 28                       # Prediction horizon 
# NUM_PRODUCTS = grid_df.id.nunique()  # Number of products 

# scale = wdf[-P_HORIZON * NUM_PRODUCTS:].scale
# weight = wdf[-P_HORIZON * NUM_PRODUCTS:].weight[:NUM_PRODUCTS]
# weight = weight/weight.sum()
# ################### Custom metric #####################
# def custom_metric(preds, train_data):
#     actuals = train_data.get_label()
#     res = L12_WRMSSE(preds, actuals, P_HORIZON, NUM_PRODUCTS, scale, weight)
#     return 'L12_WRMSSE', res, False


# # Level 12 WRMSSE
# $$
# WRMSSE = \sum_{i=1}^{30490} \left(W_i \times \sqrt{\frac{\sum_{j=1}^{28}{(D_j)^2}}{S_i}}\right)
# $$
# * W_i: the weight of the ith series 
# * S_i: the scaling factor of the ith series 
# * D_j: The difference between sales and predicted sales for the ith series on day j
# 

# # Why custom metric?
# During training, we would like to measure our model's improvement with respect to the WRMSSE, especially to notice when our model is getting worse (early stopping). I don't know how well other metrics are tracking WRMSSE improvement, but I know they are not explicitly designed with the WRMSSE in mind. 
# 
# # Why only level 12?
# Its probably faster than adding more levels with rollups. There may be other advantages as well, if you are predicting levels separately and "reconciling" them. I think there are other reasons. What do you think? Please tell me pros and cons for single level metrics. 

# # Steps to create a level 12 custom metric that will work with any training data, even if you are using a subset of product ids: 
# 1. Prepare training data: Data must be in correct order and have values for all products and all days in the validation period, including last 28 days of training 
# 2. Weights and scales: Make a dataframe with the level 12 weights and scales for all items in your data, aligned with your data for easy use. 
# 3. Create a wrmsse_lv_12 calculation function and a custom metric for Lightgbm, utilizing the function. 

# In[ ]:


################# Obective ###################
# Create a flexible metric to measure the WRMSSE
# for level 12 only. 

# Flexible means we can use 
# data with less than all the products and still 
# get a meaningful metric. For instance, trianing 
# by store_id will give you only 3049 of 10490 products,
# so the full WRMSSE level is not possible, but 
# with weights scaled proportionatly, we can have 
# a meaningful metric. 

# The metric will depend on 
# the fact that there is a value for every product 
# for every day of the prediction horizon, because 
# we rely on reshaping of the (sales - preds) column
# for horizon aggregation. If there is missing values, 
# our data will not be aligned correctly. 

# This metric is unique in that it requires specific 
# structure of the training and validation data, because 
# it performs aggregations based on specific products. 
# The metric relies on data ordered by day, then product
# id. 

# Everything will be made clear below. 


# In[ ]:


import numpy as np 
import pandas as pd 
import random


TARGET = 'sales'      # Our Target
END_TEST = 1913      # Last day of the 28 day test period

NUM_ITERATIONS = 200
DIVISOR = 10 # verbosity will be NUM_ITERATIONS//DIVISOR

# ID_COLS = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']


# # 1. Preparing training data and baseline model
# We will need to train a model to explicitly show how we are calculating the level 12 WRMSSE. We will use the basic features created in [simple fe](https://www.kaggle.com/kyakovlev/m5-simple-fe), along with some code to make lightgbm models built on the code in [custom features](https://www.kaggle.com/kyakovlev/m5-custom-features), both authored by 
# [@kyakovlev](https://www.kaggle.com/kyakovlev). 
# 

# In[ ]:


########################### Load data
########################### Basic features were created here:
########################### https://www.kaggle.com/kyakovlev/m5-simple-fe
#################################################################################

# Read data
grid_df = pd.concat([pd.read_pickle('../input/m5-simple-fe/grid_part_1.pkl'),
                     pd.read_pickle('../input/m5-simple-fe/grid_part_2.pkl').iloc[:,2:],
                     pd.read_pickle('../input/m5-simple-fe/grid_part_3.pkl').iloc[:,2:]],
                     axis=1)

############ Truncating data set for fast testing #############

############ 90 days training
############ 28 day test period, 
############ 118 total days in dataset
############ Validation same as test set for testing purposes
##############################################################
# We reset the index so that the weights and scales df, created
# later in this kernal, will be in alignment.
grid_df = grid_df[(grid_df.d > (END_TEST - 118)) & (grid_df.d <= END_TEST)].reset_index(drop=True)



####################### Masks for data #######################
##############################################################

train_mask = grid_df['d']<=(END_TEST-28)

# Test mask, also used here as validation set in lgbm.
test_mask = grid_df['d']>(END_TEST-28)

# Also need a mask for the last 28 days in the training 
# set in order for our custom metric to work. 
train_valid_mask = train_mask & (grid_df['d']>(END_TEST-56))


#################### Feature columns ########################

remove_features = ['id','d',TARGET, 'weight', 'scale', 'sw'] 
features_columns = [col for col in list(grid_df) if col not in remove_features]
# Get the test set for models to predict. This is the same 
# as the validation set in training. We use it to see 
# the full WRMSSE.
test_x = grid_df[test_mask][features_columns]

# We also need the test_id for the get_preds function 
test_id = grid_df[test_mask][['id','d']]

########################### Baseline model
#################################################################################

# We will need some global VARS for future

SEED = 42             # Our random seed for everything
random.seed(42)     # to make all tests "deterministic"
np.random.seed(SEED)


# Features that we want to exclude from training
remove_features = ['id','d',TARGET, 'weight', 'scale', 'sw'] 

# Our baseline model serves
# to do fast checks of
# changes to our model weights and objective

# We will use LightGBM for our tests
import lightgbm as lgb


lgb_params = {
                    'boosting_type': 'gbdt',         # Standart boosting type
                    'objective': 'regression',       # Standart loss for RMSE
                    'metric': ['rmse'],              # as we will use rmse as metric "proxy"
                    'subsample': 0.8,                # Lets really speed things up
                    'subsample_freq': 1,
                    'learning_rate': 0.05,           # 0.5 is "fast enough" for us
                    'num_leaves': 2**7-1,            # We will need model only for fast check
                    'min_data_in_leaf': 2**8-1,      # So we want it to train faster even with drop in generalization 
                    'feature_fraction': 0.8,
                    'n_estimators': NUM_ITERATIONS,            # We don't want to limit training (you can change 5000 to any big enough number)
#                     'early_stopping_rounds': 30,     # We will stop training almost immediately (if it stops improving) 
                    'seed': SEED,
                    'verbose': -1
                } 

                
# Small function to test different weights, 
# objectives, and metrics
# estimator = make_fast_test(grid_df)
# it will return lgb booster for future analisys
def make_fast_test(df, lgb_params, weight_column=None, fobj=None, objective=None, lr=None, metric=None):

    features_columns = [col for col in list(df) if col not in remove_features]

    tr_x, tr_y = df[train_mask][features_columns], df[train_mask][TARGET]
        
    # Valid set is the same as the "test" set for comparison of 
    # custom metric compared to the level 12 of the WRMSSE evaluator
    vl_x, vl_y = df[test_mask][features_columns], df[test_mask][TARGET]
    
    if weight_column: 
            tr_weight = df[train_mask][weight_column]
            vl_weight = df[test_mask][weight_column]
            train_data = lgb.Dataset(tr_x, label=tr_y, weight=tr_weight)
            valid_data = lgb.Dataset(vl_x, label=vl_y, weight=vl_weight)
    else: 
        train_data = lgb.Dataset(tr_x, label=tr_y)
        valid_data = lgb.Dataset(vl_x, label=vl_y)
        
    lgb_params = lgb_params.copy()
    
    if objective: 
        if metric: 
            lgb_params.update({'objective': objective,
                               'metric': metric})
        else: 
            lgb_params.update({'objective': objective,
                               'metric': objective})
    if lr: 
        lgb_params.update({'learning_rate': lr})
    
    if fobj: 
        if metric: 
            estimator = lgb.train(
                                lgb_params,
                                train_data,
                                valid_sets = [train_data, valid_data],
                                verbose_eval = NUM_ITERATIONS//DIVISOR,
                                fobj=fobj, 
                                feval=metric
                            )
        else:
            estimator = lgb.train(
                                lgb_params,
                                train_data,
                                valid_sets = [train_data,valid_data],
                                verbose_eval = NUM_ITERATIONS//DIVISOR,
                                fobj=fobj
                            )
    else: 
        if metric:
            estimator = lgb.train(
                                    lgb_params,
                                    train_data,
                                    valid_sets = [train_data, valid_data],
                                    verbose_eval = NUM_ITERATIONS//DIVISOR,
                                    feval=metric
                                )
        else: 
            estimator = lgb.train(
                                    lgb_params,
                                    train_data,
                                    valid_sets = [train_data,valid_data],
                                    verbose_eval = NUM_ITERATIONS//DIVISOR,
                                )
    
    
    return estimator

# Make baseline model
baseline_model = make_fast_test(grid_df, lgb_params)


# # 2. Function to get proper weights and scales: 
# This utilizes [my notebook](https://www.kaggle.com/chrisrichardmiles/weights-and-scales-for-the-past-2-years), which has already generated weights and scales for the last 2 years, using the [evaluator](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834) from [sakami](https://www.kaggle.com/sakami) but you could make your own function.

# In[ ]:


########## Getting the scales and weights #############
path = '/kaggle/input/weights-and-scales-for-the-past-2-years/'

def get_weights_scales_level_12(df, end_test, path):
    """Gets the scale, weight, and scaled weight in a dataframe, 
    aligned with the 'id' column of df. 
    ::path: input path where weight_scale_x files are
    """
    
    # Get the weights and scales for all the levels 
    wdf = pd.read_csv(f'{path}weight_scale_{end_test-27}.csv')
    # Get the sqrt of the scale, because I didn't do 
    # that part in the previous notebook. Then divide the weight by the 
    # sqrt(scale) to get the proper scaled_weight
    wdf['scaled_weight'] = wdf.weight/np.sqrt(wdf.scale)

    # For this function, we just want level 12 weights and scales
    wdf = wdf[wdf.Level_id == 'Level12']

    # We make an 'id' column for easy merging, df must have 'id' column
    wdf['id'] = wdf['Agg_Level_1'] + '_' +  wdf['Agg_Level_2'] + '_validation'

    # Taking just he columns we want to use in the merge 
    wdf = wdf[['id', 'scale', 'weight', 'scaled_weight']]

    # Merge with 'id' column of the df
    wdf = pd.merge(df[['id']], wdf, on='id', how='left')
    
    return wdf


# # Evaluation metric for comparison
# In order to verify our result to the level 12 WRMSSE, we will use [WRMSSE evaluator dashboard](https://www.kaggle.com/tnmasui/m5-wrmsse-evaluation-dashboard), authored by [@tnmasui](https://www.kaggle.com/tnmasui). I made a slight code adjustment to limit the graphing to the 12 levels only. 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()
import gc

from sklearn import preprocessing
import lightgbm as lgb

from typing import Union
from tqdm.notebook import tqdm_notebook as tqdm

DATA_DIR = '/kaggle/input/m5-forecasting-accuracy/'

class WRMSSEEvaluator_dashboard(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, 
                 calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 'all'  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')]                     .columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')]                               .columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df], 
                                 axis=1, sort=False)

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
            setattr(self, f'lv{i + 1}_valid_df', valid_df.groupby(group_id)                    [valid_target_columns].sum())

            lv_weight = weight_df.groupby(group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] + self.weight_columns]                    .set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index()                   .rename(columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left',
                                    on=['item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd'])                    .unstack(level=2)['value']                    .loc[zip(self.train_df.item_id, self.train_df.store_id), :]                    .reset_index(drop=True)
        weight_df = pd.concat([self.train_df[self.id_columns],
                               weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt) 

    def score(self, valid_preds: Union[pd.DataFrame, 
                                       np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape                == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, 
                                       columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], 
                                 valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):

            valid_preds_grp = valid_preds.groupby(group_id)[self.valid_target_columns].sum()
            setattr(self, f'lv{i + 1}_valid_preds', valid_preds_grp)
            
            lv_scores = self.rmsse(valid_preds_grp, i + 1)
            setattr(self, f'lv{i + 1}_scores', lv_scores)
            
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, 
                                  sort=False).prod(axis=1)
            
            all_scores.append(lv_scores.sum())
            
        self.all_scores = all_scores

        return np.mean(all_scores)
    

    
def create_viz_df(df,lv):
    
    df = df.T.reset_index()
    if lv in [6,7,8,9,11,12]:
        df.columns = [i[0] + '_' + i[1] if i != ('index','')                       else i[0] for i in df.columns]
    df = df.merge(calendar.loc[:, ['d','date']], how='left', 
                  left_on='index', right_on='d')
    df['date'] = pd.to_datetime(df.date)
    df = df.set_index('date')
    df = df.drop(['index', 'd'], axis=1)
    
    return df

def create_dashboard(evaluator, by_level_only=False, model_name=None):
    
    wrmsses = [np.mean(evaluator.all_scores)] + evaluator.all_scores
    labels = ['Overall'] + [f'Level {i}' for i in range(1, 13)]

    ## WRMSSE by Level
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x=labels, y=wrmsses)
    ax.set(xlabel='', ylabel='WRMSSE')
    
    #######################ALTERATION##########################
    title = 'WRMSSE by Level'
    if model_name: 
        title = f'WRMSSE by Level for {model_name}'
    plt.title(title, fontsize=20, fontweight='bold')
    #######################ALTERATION-COMPLETE##########################

  
    for index, val in enumerate(wrmsses):
        ax.text(index*1, val+.01, round(val,4), color='black', 
                ha="center")
        
    #######################ALTERATION##########################
    if by_level_only:       # stops function early for quick plotting of 
        plt.show()          # for quick plotting of levels
        return
    #######################ALTERATION-COMPLETE##########################

    # configuration array for the charts
    n_rows = [1, 1, 4, 1, 3, 3, 3, 3, 3, 3, 3, 3]
    n_cols = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    width = [7, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
    height = [4, 3, 12, 3, 9, 9, 9, 9, 9, 9, 9, 9]
    
    for i in range(1,13):
        
        scores = getattr(evaluator, f'lv{i}_scores')
        weights = getattr(evaluator, f'lv{i}_weight')
        
        if i > 1 and i < 9:
            if i < 7:
                fig, axs = plt.subplots(1, 2, figsize=(12, 3))
            else:
                fig, axs = plt.subplots(2, 1, figsize=(12, 8))
                
            ## RMSSE plot
            scores.plot.bar(width=.8, ax=axs[0], color='g')
            axs[0].set_title(f"RMSSE", size=14)
            axs[0].set(xlabel='', ylabel='RMSSE')
            if i >= 4:
                axs[0].tick_params(labelsize=8)
            for index, val in enumerate(scores):
                axs[0].text(index*1, val+.01, round(val,4), color='black', 
                            ha="center", fontsize=10 if i == 2 else 8)
            
            ## Weight plot
            weights.plot.bar(width=.8, ax=axs[1])
            axs[1].set_title(f"Weight", size=14)
            axs[1].set(xlabel='', ylabel='Weight')
            if i >= 4:
                axs[1].tick_params(labelsize=8)
            for index, val in enumerate(weights):
                axs[1].text(index*1, val+.01, round(val,2), color='black', 
                            ha="center", fontsize=10 if i == 2 else 8)
                    
            fig.suptitle(f'Level {i}: {evaluator.group_ids[i-1]}', size=24 ,
                         y=1.1, fontweight='bold')
            plt.tight_layout()
            plt.show()

        trn = create_viz_df(getattr(evaluator, f'lv{i}_train_df')                            .iloc[:, -28*3:], i)
        val = create_viz_df(getattr(evaluator, f'lv{i}_valid_df'), i)
        pred = create_viz_df(getattr(evaluator, f'lv{i}_valid_preds'), i)

        n_cate = trn.shape[1] if i < 7 else 9

        fig, axs = plt.subplots(n_rows[i-1], n_cols[i-1], 
                                figsize=(width[i-1],height[i-1]))
        if i > 1:
            axs = axs.flatten()

        ## Time series plot
        for k in range(0, n_cate):

            ax = axs[k] if i > 1 else axs

            trn.iloc[:, k].plot(ax=ax, label='train')
            val.iloc[:, k].plot(ax=ax, label='valid')
            pred.iloc[:, k].plot(ax=ax, label='pred')
            ax.set_title(f"{trn.columns[k]}  RMSSE:{scores[k]:.4f}", size=14)
            ax.set(xlabel='', ylabel='sales')
            ax.tick_params(labelsize=8)
            ax.legend(loc='upper left', prop={'size': 10})

        if i == 1 or i >= 9:
            fig.suptitle(f'Level {i}: {evaluator.group_ids[i-1]}', size=24 , 
                         y=1.1, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
train_df = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
sell_prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
data_df = train_df.loc[:, :'d_' + str(END_TEST)]

train_fold_df = data_df.iloc[:, :-28]
valid_fold_df = data_df.iloc[:, -28:].copy()
# Instantiate an evaluator for scoring validation periodstarting day 1886
e = WRMSSEEvaluator_dashboard(train_fold_df, valid_fold_df, calendar, sell_prices)


# In[ ]:


# Helper function
# We only need to for comparing our predictions with an evaluator


################## Prediction handlers ###################

# Lets get the sample submission file also for our functions 
ss = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')


def get_preds(model,test_x, test_id, ss, booster=None):
    X_test_id = test_id.copy()
    if booster: 
        preds = model.predict(test_x, num_iteration=booster)
    else:
        preds = model.predict(test_x)
    X_test_id['preds'] = preds
    preds = X_test_id.pivot('id', 'd', 'preds').reset_index()
    preds = pd.merge(ss.iloc[:30490]['id'], preds, on='id', how='left').iloc[:, 1:].values
    return preds


# # Example calculations

# In[ ]:


############ Level 12 WRMSSE #############

# We want calculate the WRMSSE for level 12. 
# We will use the same 28 day horizon as the 
# competition metric, but this could be changed 
# to any horizon as long there is a value for every 
# product on every day of the prediction horizon. 
# This is because we must reshape the data to sum
# accross the prediction horizon. 

########## Get scales and weights #############
wdf = get_weights_scales_level_12(grid_df, END_TEST, path)
wdf.head()

########## Veryfying the data will work ############
# According to our parameters in our make_fast_test, 
# our validation set has 28 days of data. In order
# for our custom metric to work, 
# we can only consider the last 28 days of training
# data, and we must verify there are values for all 
# days.


# Checking that data has value for every day. 
# Both train and test shape[0]/28 should 
# be 30490. train_valid is to get only the
# last 28 days of data of the training set.
train_valid = grid_df[train_valid_mask]
print(train_valid.shape[0]/28) 
print(grid_df[test_mask].shape[0]/28)
# GOOD they are correct

########## Calculating the WRMSSE ############

# We start with a column which shows the 
# difference between actuals and predictions.
# We will simulate this with predictions from 
# the baseline model for example purposes. 
preds = baseline_model.predict(test_x)
actuals = grid_df[test_mask].sales
diff = actuals - preds

# The first step in calculating the wrmsse is 
# squaring the daily error.
res = diff ** 2

# Now divide the result by the appropriate scale.
# wdf is aligned with grid_df so this is easy.
# This will result in a pandas series.
scale = wdf[test_mask].scale
res = res/scale

# The next step is summing accross the horizon
# We must reshape our data to get the products
# in line. We must turn the pandas series into 
# an array to reshape.  
res = res.values
res = res.reshape(28, 30490)

# Now we take the mean accross the prediction horizon
# and take the square root of the result.
res = res.mean(axis=0)
res = np.sqrt(res)

# Now we multiply each result with the appropriate
# scaled_weight. We just need the first 30490 entries 
# of the weight column.
weight = wdf[test_mask].weight[:30490]
res = res * weight
res = res.sum()

print(f"We calculate the WRMSSE of level 12 to be {res}")

############## Compare to WRMSSE ###############
_ = e.score(get_preds(baseline_model, test_x, test_id, ss))
create_dashboard(e, by_level_only=True)

#################### No discrepency in results ######################
# Edit: After fixing my errors the wieghts match perefectly 
df = e.lv12_weight.reset_index()
df['id'] = df.item_id + '_' + df.store_id
print('############# Examining weights ##################\n\n')
print('Weights for the evaluator')
print(df.sort_values('id')[['id', 0]].head())
print('\n\nWeights that I am using')
print(wdf[:30490][['id', 'weight']].sort_values('id').head())
print('\nAll is good')


# # 3. Creating the custom metric

# In[ ]:


############ Calculations function #############
# We will define a function outside of the custom 
# metric. This is because the custom function only 
# has two inputs: preds and train_data, and we need 
# to incorporate more than that to make it flexible. 

def L12_WRMSSE(preds, actuals, p_horizon, num_products, scale, weight): 
    
    actuals = actuals[-(p_horizon * num_products):]
    preds = preds[-(p_horizon * num_products):]
    diff = actuals - preds

    # The first step in calculating the wrmsse is 
    # squareing the daily error.
    res = diff ** 2

    # Now divide the result by the appropriate scale
    # take values of scale to avoid converting res 
    # to a pandas series
    res = res/scale.values

    # The next step is summing accross the horizon
    # We must reshape our data to get the products
    # in line. 
    res = res
    res = res.reshape(p_horizon, num_products)

    # Now we take the mean accross the prediction horizon
    # and take the square root of the result.
    res = res.mean(axis=0)
    res = np.sqrt(res)

    # Now we multiply each result with the appropriate
    # scaled_weight. We just need the first 30490 entries 
    # of the scaled_weight column
    res = res * weight
    res = res.sum()
    return res


################## Variables ######################
# Variables needed for the metric. For other training
# sets, replace grid_df with the training set, but 
# make sure that the data is ordered by day and everything 
# is in correct alignment with values for every product and 
# every day. 

P_HORIZON = 28         # Prediction horizon 
NUM_PRODUCTS = 30490   # Number of products 

scale = wdf[train_valid_mask].scale
weight = wdf[train_valid_mask].weight[:NUM_PRODUCTS]

################### Custom metric #####################
def custom_metric(preds, train_data):
    actuals = train_data.get_label()
    res = L12_WRMSSE(preds, actuals, P_HORIZON, NUM_PRODUCTS, scale, weight)
    return 'L12_WRMSSE', res, False

################## Custom metric in action #####################
# Im leaving the RMSE metric in for now
baseline_model = make_fast_test(grid_df, lgb_params, metric=custom_metric)


# # Making it flexible

# In[ ]:


################# Making if flexible ######################
# Lets say we want to train a model, using a subset of 
# the total products, say training by store_id. 
store_mask = grid_df.store_id == 'CA_1'
grid_df_ca = grid_df[store_mask] 

# The problem is that the level 12 weights do not add 
# up to 1 now. All  we need to do is scale up the weights
# and adjust the total products. 
P_HORIZON = 28         # Prediction horizon 
NUM_PRODUCTS = grid_df_ca.id.nunique()   # Number of products 

scale = wdf[train_valid_mask][store_mask].scale
weight = wdf[train_valid_mask][store_mask].weight[:NUM_PRODUCTS]

# Normalize weights so that the sum to 1.
weight = weight/weight.sum()

################### Custom metric #####################
def custom_metric(preds, train_data):
    actuals = train_data.get_label()
    res = L12_WRMSSE(preds, actuals, P_HORIZON, NUM_PRODUCTS, scale, weight)
    return 'L12_WRMSSE', res, False

################## Custom metric in action #####################
# Im leaving the RMSE metric in for now
baseline_model = make_fast_test(grid_df_ca, lgb_params, metric=custom_metric)


# # Conclusion: 
# We can now generate a custom metric for level 12, regardless of how we subset our training data. To use this method, these conditions must be met: 
# * Training data must be ordered by day, then product id for correct alignment
# * There is a value for every product for every day in the validation period, as well as the last (length(validation)) days of the training set
# * You have a dataframe (wdf for me here) that has the correct scales and weights, aligned with your data correctly. 

# # Next steps: 
# 
# ## Test custom objectives and scaling methods, using this custom metric to evaluate the usefulness of the methods. 
# 
# ## Trying using this method for early stopping. 
