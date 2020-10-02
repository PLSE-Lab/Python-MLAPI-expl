#!/usr/bin/env python
# coding: utf-8

# ## M5 Setup
# This competition is a little different from others. The true labels of the public leaderboard are now revealed.
# 
# The following is an excerpt from the [M5 Competition Guide](https://mofc.unic.ac.cy/m5-competition/):
# 
# > After the end of the validation phase, i.e., from June 1, 2020 to 30 June of the same year, the participants will be provided with the actual values of the 28 days of data used for scoring their performance during the validation phase. They will be asked then to re-estimate or adjust (if needed) their forecasting models in order to submit their final forecasts and prediction intervals for the following 28 days, i.e., the data used for the final evaluation of the participants. During this time, there will be no leaderboard, meaning that no feedback will be given to the participants about their score after submitting their forecasts. Thus, although the participants will be free to (re)submit their forecasts any time they wish (a maximum of 5 entries per day), they will not be aware of their absolute, as well as their relative performance. The final ranks of the participants will be made available only at the end of competition, when the test data will be made available. This is done in order for the competition to simulate reality as closely as possible, given that in real life forecasters do not know the future.
# 
# So while the public LB on Kaggle will either get infested by scores that use the true labels or Kaggle will freeze the public LB, we now have access to the actual labels and hence can calculate the validation scores (and rank as of 31st May, 2020) at various levels of aggregations ourselves.
# 
# The weights used in this notebook are the weights for the public LB (validation data). Note that the private LB (evaluation data) uses a different set of weights. A summary of the weights comparison is shared here: https://www.kaggle.com/rohanrao/m5-the-weighing-scale
# 
# Note that the final private LB ranking will be based on the test data at the end of the competition.
# 

# ## Validation Data
# The actual validation data (*d_1914 to d_1941*) is now available in the [sales_train_evaluation.csv](https://www.kaggle.com/c/m5-forecasting-accuracy/data?select=sales_train_validation.csv). Since this file also consists of the entire train data available earlier, we can completely ignore [sales_train_validation.csv](https://www.kaggle.com/c/m5-forecasting-accuracy/data?select=sales_train_validation.csv) for the rest of this competition.
# 

# In[ ]:


## new train data
import pandas as pd

df_train_full = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv")
df_train_full.iloc[:, -31:].head()


# ## Submission
# The test data is for the predictions from *d_1942 to d_1969* corresponding to the sample submission format *F1 to F28*.   
# We can still submit on Kaggle up to 5 times a day till the end of the competition.
# 

# ## Public LB Score
# We can use the actual validation data labels to score our models and get the exact public LB score. For predicting on the final test data it is highly recommended to rerun your models with including the new validation data available.
# 
# The code below can be used to get your public LB score. Thanks to [sakami](https://www.kaggle.com/sakami) for providing a neat class for the evaluation metric [here](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834).   
# The [dataset of M5 public LB](https://www.kaggle.com/rohanrao/m5-accuracy-final-public-lb) can be used to get your public LB rank.
# 
# I've also shared a notebook of how you can deep dive into analyzing your submission with the public LB: https://www.kaggle.com/rohanrao/m5-anatomy-of-the-public-lb
# 
# So you can now work without needing to make submissions.
# 
# I've verified the calculations below with [Konstantin Yakovlev](https://www.kaggle.com/kyakovlev)'s two public kernel submission files and corresponding scores on public LB:   
# https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic   
# https://www.kaggle.com/kyakovlev/m5-witch-time
# 

# In[ ]:


## importing packages
import numpy as np
import pandas as pd

from typing import Union
from tqdm.notebook import tqdm


# In[ ]:


## evaluation metric
## from https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/133834 and edited to get scores at all levels
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
            'cat_id',
            'state_id',
            'dept_id',
            'store_id',
            'item_id',
            ['state_id', 'cat_id'],
            ['state_id', 'dept_id'],
            ['store_id', 'cat_id'],
            ['store_id', 'dept_id'],
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

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]):
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        group_ids = []
        all_scores = []
        for i, group_id in enumerate(self.group_ids):
            lv_scores = self.rmsse(valid_preds.groupby(group_id)[self.valid_target_columns].sum(), i + 1)
            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)
            group_ids.append(group_id)
            all_scores.append(lv_scores.sum())

        return group_ids, all_scores


# In[ ]:


## public LB rank
def get_lb_rank(score):
    """
    Get rank on public LB as of 2020-05-31 23:59:59
    """
    df_lb = pd.read_csv("../input/m5-accuracy-final-public-lb/m5-forecasting-accuracy-publicleaderboard-rank.csv")

    return (df_lb.Score <= score).sum() + 1


# In[ ]:


## reading data
df_calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
df_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
df_sample_submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")
df_sample_submission["order"] = range(df_sample_submission.shape[0])

df_train = df_train_full.iloc[:, :-28]
df_valid = df_train_full.iloc[:, -28:]

evaluator = WRMSSEEvaluator(df_train, df_valid, df_calendar, df_prices)


# In[ ]:


## structure of validation data
preds_valid = df_valid.copy() + np.random.randint(100, size = df_valid.shape)
preds_valid.head()


# In[ ]:


## evaluating random submission
groups, scores = evaluator.score(preds_valid)

score_public_lb = np.mean(scores)
score_public_rank = get_lb_rank(score_public_lb)

for i in range(len(groups)):
    print(f"Score for group {groups[i]}: {round(scores[i], 5)}")

print(f"\nPublic LB Score: {round(score_public_lb, 5)}")
print(f"Public LB Rank: {score_public_rank}")


# In[ ]:


## evaluating submission from public kernel M5 - Three shades of Dark: Darker magic
## from https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic
preds_valid = pd.read_csv("../input/m5-three-shades-of-dark-darker-magic/submission_v1.csv")
preds_valid = preds_valid[preds_valid.id.str.contains("validation")]
preds_valid = preds_valid.merge(df_sample_submission[["id", "order"]], on = "id").sort_values("order").drop(["id", "order"], axis = 1).reset_index(drop = True)
preds_valid.rename(columns = {
    "F1": "d_1914", "F2": "d_1915", "F3": "d_1916", "F4": "d_1917", "F5": "d_1918", "F6": "d_1919", "F7": "d_1920",
    "F8": "d_1921", "F9": "d_1922", "F10": "d_1923", "F11": "d_1924", "F12": "d_1925", "F13": "d_1926", "F14": "d_1927",
    "F15": "d_1928", "F16": "d_1929", "F17": "d_1930", "F18": "d_1931", "F19": "d_1932", "F20": "d_1933", "F21": "d_1934",
    "F22": "d_1935", "F23": "d_1936", "F24": "d_1937", "F25": "d_1938", "F26": "d_1939", "F27": "d_1940", "F28": "d_1941"
}, inplace = True)

groups, scores = evaluator.score(preds_valid)

score_public_lb = np.mean(scores)
score_public_rank = get_lb_rank(score_public_lb)

for i in range(len(groups)):
    print(f"Score for group {groups[i]}: {round(scores[i], 5)}")

print(f"\nPublic LB Score: {round(score_public_lb, 5)}")
print(f"Public LB Rank: {score_public_rank}")


# In[ ]:


## evaluating submission from public kernel M5 - Witch Time
## from https://www.kaggle.com/kyakovlev/m5-witch-time
preds_valid = pd.read_csv("../input/m5-witch-time/submission.csv")
preds_valid = preds_valid[preds_valid.id.str.contains("validation")]
preds_valid = preds_valid.merge(df_sample_submission[["id", "order"]], on = "id").sort_values("order").drop(["id", "order"], axis = 1).reset_index(drop = True)
preds_valid.rename(columns = {
    "F1": "d_1914", "F2": "d_1915", "F3": "d_1916", "F4": "d_1917", "F5": "d_1918", "F6": "d_1919", "F7": "d_1920",
    "F8": "d_1921", "F9": "d_1922", "F10": "d_1923", "F11": "d_1924", "F12": "d_1925", "F13": "d_1926", "F14": "d_1927",
    "F15": "d_1928", "F16": "d_1929", "F17": "d_1930", "F18": "d_1931", "F19": "d_1932", "F20": "d_1933", "F21": "d_1934",
    "F22": "d_1935", "F23": "d_1936", "F24": "d_1937", "F25": "d_1938", "F26": "d_1939", "F27": "d_1940", "F28": "d_1941"
}, inplace = True)

groups, scores = evaluator.score(preds_valid)

score_public_lb = np.mean(scores)
score_public_rank = get_lb_rank(score_public_lb)

for i in range(len(groups)):
    print(f"Score for group {groups[i]}: {round(scores[i], 5)}")

print(f"\nPublic LB Score: {round(score_public_lb, 5)}")
print(f"Public LB Rank: {score_public_rank}")


# The scores and rank should match the public LB scores as of 31st May, 2020 midnight.

# In[ ]:


## Top Score: 0.42714
print(get_lb_rank(0.42713))
print(get_lb_rank(0.42714))
print(get_lb_rank(0.42715))


# Whenever your score exactly matches one already on public LB, it will be ranked below the same.

# ## Notes
# * The merge and sort with sample submission is not required if predictions are already ordered.
# * The renaming of columns is not required if predictions already have the columns *d_1914 to d_1941*.
# 
