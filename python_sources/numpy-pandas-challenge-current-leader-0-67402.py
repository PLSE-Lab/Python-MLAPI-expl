#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
DAYS_BACK = 28


# # Top scores and links to notebook posted here: 
# *  0.67402[this notebook](https://www.kaggle.com/chrisrichardmiles/numpy-pandas-challenge-current-leader-71444/edit)
# * 0.71444 [this notebook](https://www.kaggle.com/chrisrichardmiles/numpy-pandas-challenge-current-leader-71444/edit)

# # Create a model utilizing only numpy and pandas 
# ## Goals: 
# * Create basic and explainable models
# * Create models that can improve the application of more advanced techniques
# 
# ## Allowed: 
# * Using insights gained from other analysis(visual, statistical, ML), but the final model must be constructed "by hand" with only numpy and pandas. 
# 
# 
# 
# ## Not allowed: 
# * using any other imports when constructing final csv for submission
# 
# ## Beginners: Please fork this as a starter. The functions I created to group the data and create submissions could be useful. 
# 
# ## Intermediate/Advanced: Use knowledge extracted from your advanced algorithms and apply it to create a simple, explainable model. 
#  

# # Helper functions

# In[ ]:


def add_snap_col(df_in):
    """adds a 'snap_day' column to a dataframe that contains a state_id column and the columns 'snap_CA', 'snap_TX', 
    and snap_WI"""
    df = pd.get_dummies(df_in, columns=['state_id'])
    df['snap_day'] = (df.snap_CA * df.state_id_CA) + (df.snap_WI * df.state_id_WI) + (df.snap_TX * df.state_id_TX)
    del df['state_id_WI'], df['state_id_CA'], df['state_id_TX']
    return df



def melt_merge_snap(df):
    df = df.melt(['id', 'state_id'], var_name='d', value_name='demand')
    df = df.merge(cal)
    df = add_snap_col(df)
    return df



def get_sub():
    """returns a tidy dataframe version of the sample submission, merged with the calendar data, 
    without the 'demand' column. It can be used to join to a group by series to make predictions  """
    # make a copy of the sample submission
    sub = ss.copy()
    # change the column names to match the last 28 days
    sub.columns = ['id'] + ['d_' + str(1914+x) for x in range(28)]
    # select only the rows with an id with the validation tag
    sub = sub.loc[sub.id.str.contains('validation')]
    # melt this dataframe and merge it with the calendar so we can join it with group_by series we create
    sub = sub.melt('id', var_name='d', value_name='demand')
    sub = sub.merge(cal)
    
    
    # add state_id column so that we can add the snap_day column
    sub['state_id'] = sub.id.str.split('_', expand=True)[3]
    
    # add the snap_day column
    sub = add_snap_col(sub)
    
    return sub.drop('demand', axis='columns')



def join_sub_groupby(sub, group):
    """ 
    Joins the sub dataframe created by get_sub to a groupby series
    """
    return sub.join(group, on=group.index.names)



def make_sub(df_in, ss, filename='submission.csv'): 
    """
    Takes a dataframe in the form given by join_sub_groupby, or any dataframe with the proper index and and 'd' colums.
    returns a csv submission file in the correct format
    """
    # pivot df to get it into the proper format for submission
    df = df_in.pivot(index='id', columns='d', values='demand')
    # need to reset index to take care of columns. comment next line out to see what i mean 
    df.reset_index(inplace=True)
    
    submission = ss[['id']].copy()    
    submission = submission.merge(df)    
    # we must copy the dataframe to match the format of the submission file which is twice as long as what we have
    submission = pd.concat([submission, submission], axis=0)
    # reset the id colum to have the same values as the sample submission
    submission['id'] = ss.id.values
    # rename the columns to match the sample submission format 
    submission.columns = ['id'] + ['F' + str(i) for i in range(1,29)]
    
    submission.to_csv(filename, index=False)  


# # Simple models: Just using the last known 28 days. 
# model 1: avg of last 28, grouped by id, weekday           **LB score: .75238**
# 
# model 2: avg of last 28, grouped by id, weekday, and snap **LB score: .72969**
# 
# #### same as simple models, but different time periods
# 
# model 3: avg of last 90, grouped by id, weekday, and snap **LB score: .71444**
# 
# model 3_dark_magic: avg of last 90, grouped by id, weekday, and snap**LB score: .67402**
# 
# model 4: avg of last 365, grouped by id, weekday, and snap **LB score: 0.88767**

# # Load data

# In[ ]:


PATH = '/kaggle/input/m5-forecasting-accuracy/'
cal = pd.read_csv(f'{PATH}calendar.csv')
sell_prices = pd.read_csv(f'{PATH}sell_prices.csv')
ss = pd.read_csv(f'{PATH}sample_submission.csv')
stv = pd.read_csv(f'{PATH}sales_train_validation.csv')
stv_id = stv[['id','state_id']]
stv = stv.iloc[:, :-DAYS_BACK]


# # Lets try different time ranges to use. I will always include the most recent data

# In[ ]:


last_90 = pd.concat([stv_id, stv.iloc[:,-90:]], axis=1) # we include 0, and 5 to get the id and state id columns


# # Melt the d_ columns, merge with calendar, and add a snap_day column. Snap column indicates if the item is snap eligible that day.

# In[ ]:


# last_28 = melt_merge_snap(last_28)
# last_28.head()


# In[ ]:


last_90 = melt_merge_snap(last_90)


# In[ ]:


last_90.head()


# # Model 1: group by id, and wday and average over demand

# In[ ]:


# get the demand for each product, grouped by weekday
# by_weekday = last_28.groupby(['id','wday'])['demand'].mean()


# # Model 2: groupby id, wday, snap_day

# In[ ]:


# by_weekday_snap = last_28.groupby(['id', 'wday', 'snap_day'])['demand'].mean()


# In[ ]:


by_weekday_snap_90 = last_90.groupby(['id', 'wday', 'snap_day'])['demand'].mean()


# In[ ]:


# by_weekday_snap_365 = last_365.groupby(['id', 'wday', 'snap_day'])['demand'].mean()


# # Prepare a copy of the submission file to merge with the groupby series

# In[ ]:


sub = get_sub()
sub.head()


# # Join the sub dataframe to a group_by series and create our final dataframe

# In[ ]:


# df_final_model_1 = join_sub_groupby(sub, by_weekday)


# In[ ]:


# df_final_model_2 = join_sub_groupby(sub, by_weekday_snap)


# In[ ]:


df_final_model_3 = join_sub_groupby(sub, by_weekday_snap_90)


# In[ ]:


# df_final_model_3_dark_magic = df_final_model_3.copy()
# df_final_model_3_dark_magic['demand'] = df_final_model_3_dark_magic['demand'] * 1.04


# In[ ]:


# df_final_model_4 = join_sub_groupby(sub, by_weekday_snap_365)


# # Make a submission file of the model

# In[ ]:


# make_sub(df_final_model_1, ss, filename='model1sub.csv')


# In[ ]:


# make_sub(df_final_model_2, ss, filename='model2sub.csv')


# In[ ]:


make_sub(df_final_model_3, ss, filename='model3sub.csv')


# In[ ]:


# make_sub(df_final_model_3_dark_magic, ss, filename='model3dmsub.csv')


# In[ ]:


# make_sub(df_final_model_4, ss, filename='model4sub.csv')


# # Now apply more insight to creating a simple model

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
train_df = train_df.loc[:, :'d_' + str(1913)]

train_fold_df = train_df.iloc[:, :-28]
valid_fold_df = train_fold_df.iloc[:, -28:].copy()
# Instantiate an evaluator for scoring validation periodstarting day 1886
e = WRMSSEEvaluator_dashboard(train_fold_df, valid_fold_df, calendar, sell_prices)


# In[ ]:


m = pd.read_csv('model3sub.csv')


# In[ ]:


preds = m.iloc[:30490, 1:].values * 1.01


# In[ ]:


_ = e.score(preds)
create_dashboard(e, by_level_only=True)


# In[ ]:




