#!/usr/bin/env python
# coding: utf-8

# # Competition scoring metric
# $$
# WRMSSE = \sum_{i=1}^{42,800} \left(W_i \times \sqrt{\frac{\sum_{j=1}^{28}{({Y_t - \hat{Y_t}})^2}}{S_i}}\right)    
# $$
# 
# 
# $$
# S_i = \frac{1}{n-1}\sum_{t=2}^{n}({Y_t - Y_{t-1}})^2
# $$
# 
# * W_i is the weight of the ith series 
# 
# Therefore, the penalty for each series simplifies to 
# $$
# \frac{W}{\sqrt{S}}\sum{\sqrt{({Y_t - \hat{Y_t}})^2}}
# $$
# 

# With about 17.5 hours to go, it was 11:30 PM in my timezone I was getting ready to run tests to decide on features I would use in my final model. I was running all my tests with at least two main objective functions: tweedie, and an RMSSE (root mean squared scaled error) function. This was the RMSSE function:

# In[ ]:


# oos_scale = 1/w_12_train.oos_level_12_scale
def oos_rmsse(preds, train_data): 
        actuals = train_data.get_label()
        diff = actuals - preds
        grad = -diff * oos_scale
        hess = np.ones_like(diff)
        return grad, hess
    
# The scaling factor, w_12_train.oos_level_12_scale, 
# is the same as described in the competition, except 
# I removed zeros that I believed were due to being 
# out of stock before calculation.


# So essentially, I am using an RMSE objective with some weight for each item. With W being an items level 12 weight, W/sqrt(scaling factor) never worked well. Using 1/(scaling factor) as a weight gave reasonable results, but scored worse than tweedie about half the time, and sometimes it lost really bad, so it was looking like tweedie was going to be my objective function. But I had spent tens if not hundreds of hours working on a custom loss function that paid respect to the WRMSSE across all levels, so I decided to give it one more think. I thought,
# ### "if the answer was really simple and elegant, what would it be?"
# So no fancy aggregations during training, and no gradients that would tell my predictions to go the wronge direction to compensate for other wrong predictions in service to higher levels. I just wanted a proper weight for each item to add to my RMSE function that would would reflect the items contribution to the full WRMSSE score. 
# 
# ### For a given item how does an error affect the total score?
# We know that part of the penalty comes from each of the twelve hierarchical levels. From level 12, we know for sure that the penalty will be $\frac{W_{12}}{\sqrt{S_{12}}}$ multiplied by our error (I guess its actually more complicated than this because the error is actually the square root of the average of squared errors across time, but for simplicities sake, I continue my thinking like this). For the other levels, it appears to be more complicated, given that an error for this item could balance out another items error, but if we assume all other items are correct, then the error is simply $\frac{W_{L}}{\sqrt{S_{L}}}$, where W and S are the weight and scaling factor of the series on level L to which the particular item belongs. Therefore, I decided to try the simple formula 
# $$
# W_{item} = \sum_{L=1}^{12}\frac{W_{L}}{\sqrt{S_{L}}}
# $$
# 
# Now I take a simple RMSE function and weight each item with these weights. It turns out this loss function performed better than any other loss functions I had, including tweedie, but it did need the right early stopping function. 
# 
# ### Early stopping metric
# I tried various early stopping metrics, including the proper WRMSSE, level 12 RMSSE, level 12 'out of stock ' adjusted RMSSE, and a level 12 WRMSSE with the weights computed as they are in my loss function. Interestingly, the metric I created using the same idea as the loss function consistently performed worse as an early stopping metric, compared to both the oos_RMSSE and the WRMSSE. In the end, using the oos_RMSSE gave the best results as an early stopper. 
# 
# Below is code that starts with a dataframe with weights and scales for all series, and yields a dataframe with the proper sum of weights and scales for each level 12 item. 

# In[ ]:


from m5_helpers import * 
DATA_RAW_PATH = '../input/m5-forecasting-accuracy/'


# In[ ]:


######################### oos comp scale ################################
DATA_INTERIM_PATH = '../input/m5-oos-grid-df/'
oos_train_df = pd.read_pickle(f'{DATA_INTERIM_PATH}oos_train_df.pkl')

def get_oos_scale(oos_train_df):
    rec = oos_train_df.iloc[:, 6:-28]
    rdiff = np.diff(rec, axis=1)
    return np.sqrt(np.nanmean(rdiff**2, axis=1))


# In[ ]:


########################### Load competition data #########################
cal_df = pd.read_csv(F'{DATA_RAW_PATH}calendar.csv')
prices_df = pd.read_csv(F'{DATA_RAW_PATH}sell_prices.csv')
train_df = pd.read_csv(F'{DATA_RAW_PATH}sales_train_evaluation.csv')

START_TEST = 1942

####################### Make evaluator object #############################
# This object has a lot that of extras that you can find out about by 
# running WRMSSE?? in a code block. Most importantly, it has a dataframe 
# with all the series with weights and scales. 
e = WRMSSE(train_df, cal_df, prices_df, START_TEST)

############################ Weights and scales alignment #################
# Get a dataframe with all the weights and scales for all series. 
w_df = e.w_df.copy()

# Initialize a weight dataframe for just the level 12 series. 
w_12 = w_df.loc[12]

# For each item, for each level we want to add the scaled weight W/sqrt(S) 
# of the series to which that item belongs. 
w_12['level_1_sw'] = w_df.loc[1].scaled_weight[0]
w_12['level_2_sw'] = w_12.index.map(lambda x: w_df.loc[(2,x.split('_')[3])].scaled_weight)
w_12['level_3_sw'] = w_12.index.map(lambda x: w_df.loc[(3,x[-4:])].scaled_weight)
w_12['level_4_sw'] = w_12.index.map(lambda x: w_df.loc[(4,x.split('_')[0])].scaled_weight)
w_12['level_5_sw'] = w_12.index.map(lambda x: w_df.loc[(5, x.split('_')[0] + '_' +  x.split('_')[1])].scaled_weight)
w_12['level_6_sw'] = w_12.index.map(lambda x: w_df.loc[(6, x.split('_')[3] + '_' +  x.split('_')[0])].scaled_weight)
w_12['level_7_sw'] = w_12.index.map(lambda x: w_df.loc[(7, x.split('_')[3] + '_' +  x.split('_')[0] + '_' +  x.split('_')[1])].scaled_weight)
w_12['level_8_sw'] = w_12.index.map(lambda x: w_df.loc[(8, x.split('_')[3] + '_' +  x.split('_')[4] + '_' +  x.split('_')[0])].scaled_weight)
w_12['level_9_sw'] = w_12.index.map(lambda x: w_df.loc[(9, x.split('_')[3] + '_' +  x.split('_')[4] + '_' +  x.split('_')[0] + '_' +  x.split('_')[1])].scaled_weight)
w_12['level_10_sw'] = w_12.index.map(lambda x: w_df.loc[(10, x.split('_')[0] + '_' +  x.split('_')[1] + '_' +  x.split('_')[2])].scaled_weight)
w_12['level_11_sw'] = w_12.index.map(lambda x: w_df.loc[(11, x.split('_')[0] + '_' +  x.split('_')[1] + '_' +  x.split('_')[2] + '_' +  x.split('_')[3])].scaled_weight)
w_12['total_sw'] = w_12[['scaled_weight', 'level_1_sw', 'level_2_sw', 'level_3_sw', 'level_4_sw',
       'level_5_sw', 'level_6_sw', 'level_7_sw', 'level_8_sw', 'level_9_sw',
       'level_10_sw', 'level_11_sw']].sum(axis=1)

################ Add oos_scale #################################
w_12.index = w_12.index + '_evaluation'
w_12 = w_12.reindex(e.train_df.id)
if START_TEST == 1942:
    w_12['oos_level_12_scale'] = get_oos_scale(oos_train_df)
del w_df

####################### Output files ###########################
w_12.to_pickle(F'w_12_{START_TEST}.pkl')


# In[ ]:


w_12.head()


# # Define and fit custom objective and metric: Needs data to run
# Here is a code snippet that shows how to define the custom objective and custom metric using our new weights. 
# It is only for example. The [raw code of my final model](https://www.kaggle.com/chrisrichardmiles/77th-place-custom-loss-custom-metric) demonstrates how I use it in my final training. I will also update this notebook with code demonstrating tests for different custom objectives and metrics. 

# In[ ]:


#################### Custom objective ####################
def get_wrmsse(w_12_train):
    """w_12_train must be aligned with grid_df like
    w_12_train = w_12.reindex(grid_df[train_mask].id)
    """
      I normalise each weight with the average weight of 
      of all items. 
    weight = w_12_train['total_sw'] / w_12_train['total_sw'].mean()
    
    def wrmsse(preds, train_data): 
        actuals = train_data.get_label()
        diff = actuals - preds
        grad = -diff * weight
        hess = np.ones_like(diff)
        return grad, hess
    return wrmsse

################### Custom metric ########################
def get_oos_rmsse_metric(w_12_valid): 
    oos_scale = 1/w_12_valid.oos_level_12_scale
    
    def oos_rmsse_metric(preds, train_data): 
        actuals = train_data.get_label()
        diff = actuals - preds
        res = np.sum(diff**2 * oos_scale**2)
        return 'oos_rmsse', res, False
    return oos_rmsse_metric

################## Fit custom functions ##################
w_12_train = w_12.reindex(grid_df[train_mask].id)
w_12_valid = w_12.reindex(grid_df[valid_mask].id)

################## Objective #############################
wrmsse = get_wrmsse(w_12_train)

######################### Metrics ########################
oos_rmsse_metric = get_oos_rmsse_metric(w_12_valid)


# In[ ]:




