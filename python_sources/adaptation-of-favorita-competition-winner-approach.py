#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ## Acknowledgement
# I want to start with thanks to winning team of "Favorita Groceries Sales Competition" for sharing their interesting solution which achieved highest predictive performance only using the last 3 months of the data with a very small training time and let me learn a lot from. This notebook is a adaptation of that approach to M5 forecasting competition.
# 
# ## What is interesting about this approach?
# 
# Instead of representing sales of each day as a row in the training data as in most of the public kernels, each series are represented as a row --similar to the given format by the competition organisers--. This is particularly advantagous for code efficiency as we don't need to use pandas "groupby" which is horribly slow especially when the number of groups are high as in this data -- 30490 groups --. 
# 
# I will be creating 200+ features for the purpose of this first version of the notebook. They are mostly running statistics on sales, calendar, intermittent demand and price related features -- some are adjusted versions of features for this approach from other public kernels --.
# 
# All feature creations, and training of 28 models finishes around 20mins. 
# 
# 
# ## What about predictive performance?
# 
# I have created my pipeline with regualar approach and also developed this one to discover how this interesting modelling technique works. I can say that in terms of the final WRMSSE score, almost always the regular approach performed better even though it was based on only some basic features relative to this approach. For the best --that I managed to achieve-- of each approach were not so different. My best 28days model with regular approach gave LB 0.61XX whereas with this approach, I managed to get LB 0.63XX none of them use recursive features or magic multipliers. I used quite a lot of complex features that I prefered to keep it to myself until the end of the competition though I can say that they are mainly targeting to capture SNAP-sales interactions.  
# 
# In terms of the performance, this version of the notebook is intended to be a starter code for those of you that are willing to explore this approach. I can easily say that it taught me a lot about how wide range of ways are possible to model a problem. I will try to explain how this model works in more detail in the upcoming steps. 
# 
# 
# ## How this model works?
# 
# <img src="https://i.imgur.com/Sutycbt.jpg[/img" width="1000">
# 
# 
# For each series we look back from some certain points in time -- with one week gaps in between-- and derive features to describe the behaviour of the series up until that point in time.
# ### This is actually the main difference with the regular approach. We are deriving features to explain the characteristics of the entire series up until that point in time instead of deriving features that are describing the characteristics of one single day.
# 
# Above image is showing the points in time that the features are generated. Gaps between are one week which aims to capture weekly cycle in the data. Arrows are looking back to represent that the features at that point are created by looking back in time.
# 
# 
# Once we have our features generated, we create 28 models which learns the relationship between these features of a series and the next 1,2,3....28 days sales. Each of these models are designed to learn the relationship between the same derived features and different forecasting horizons so for each training example we have 28 different labels.
# 
# ## What are potential improvements?
# - Creating linear regression models to capture the trend of the series -- because the decision trees are not able to extrapolate so struggle with trend -- and adding predictions as features would definitely help as they helped to regular approach.
# - Custom loss function -- I am planning to add this in the next version
# - Some more complex and aggregated features help!
# - Creating models to predict aggregated levels to use as scalers -- kind of similar to magic multipliers without a real magic though -- Thanks @chrisrichardmiles for suggesting that
# - This notebook that I shared [here](https://www.kaggle.com/anlgrbz/clustering-with-intermittent-demand-related-featrs) clusters the examples according to their intermittent demand related characteristics. I used this clusters to train models seperately using the regular approach and got a good boost on my local CV though harmed my LB score. I believe this could be worthy to try one more time.
# ...
# 
# ## What is the main advantage of using this approach?
# #### It is a day-by-day model -- consists of 28 seperate models-- so we have the flexibility to modify models for different days. At the same time, training is so fast that we can quickly try different ideas and see how the model performance reacts to it. I think this flexibility and pace is a rare and valuable combination by looking at the public kernels.
# Ability to quickly try ideas is critical in here because most of people are struggling to create a trust worthy CV strategy --including me-- so we can search for correlation of CV and LB much more quickly. Also, if you have an existing CV that requires hours of time and you got sick of it, you may prefer to pivot your approach to this one.
# 
# ## Notes
# - I calculated the WRMSSE weights and constants in my local computer and loaded it to the kernel using a pickle file so the additional .pkl files are just these weights.
# 
# - Even though code works quite fast, I created some check point like save_obj(), load_obj() functions and commented them out for those of you that want to avoid each time running the pre-processing and feature engineering parts of the code.
# 
# - I mainly created 4 datasets in pre-processing to derive the features from. snap_df is not used in this notebook but It is useful to develop further features.
# 
# 
# 

# In[ ]:


from datetime import date, timedelta
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
import lightgbm as lgb
import random
import warnings
warnings.filterwarnings('ignore')
import wandb


def save_obj(obj, name):
    with open(  name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('../input/weights/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def load_obj_from_sets(name):
    with open('../input/small-first-sets/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# ## Pre-Processing

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")\ncalendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")\nprice = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")\nsubmission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")\n\ncategory_cols = [\'item_id\', \'dept_id\', \'cat_id\', \'store_id\', \'state_id\']\nid_cols = ["id"] + category_cols\nid_df = train_df[id_cols]  # id_df not encoded\nprint("running")\n# Label Encoding of categorical variables\nmapper = {}\nfor col in category_cols:\n    le = LabelEncoder()\n    mapper[col] = dict(zip(le.fit_transform(train_df[col]), train_df[col]))\n    train_df[col] = le.fit_transform(train_df[col])\n\nmulti_indexes = train_df.set_index(category_cols).index  # multi_indexes are encoded\n\n# Create ordered_ids_and_weights -- Weights are pre-calculated\nagg_level_to_denominator = load_obj("denominator")\nagg_level_to_weight = load_obj("weight")\n\nfinal_multiplier = (agg_level_to_weight[11]) / agg_level_to_denominator[11]\nfinal_multiplier = final_multiplier.reset_index()\nfinal_multiplier["id"] = final_multiplier["item_id"] + "_" + final_multiplier["store_id"] + "_validation"\nfinal_multiplier.drop(["item_id", "store_id"], axis=1, inplace=True)\ndel agg_level_to_weight, agg_level_to_denominator\ngc.collect()\n\nordered_ids_and_weights = pd.merge(id_df, final_multiplier, on=["id"], how="left")\nordered_ids_and_weights = ordered_ids_and_weights[[0]]\nordered_ids_and_weights.index = multi_indexes\nordered_ids_and_weights = ordered_ids_and_weights.reset_index()\nordered_ids_and_weights = ordered_ids_and_weights.rename({0: "weights"}, axis=1)\n\n# Trainset set column names\ntrain_df.set_index(keys=id_cols, inplace=True)\nstart_date = date(2011, 1, 29)\ntrain_df.columns = pd.date_range(start_date, freq="D", periods=1913)\ntrain_df.reset_index(inplace=True)\n\n# Calendar\ncalendar["date"] = pd.to_datetime(calendar.date)\n\n# Preprocess price\nprice_df = pd.merge(price, id_df, how="left", on=["item_id", "store_id"])\n\ntmp = calendar[["wm_yr_wk", "date"]]\ntmp = tmp.groupby("wm_yr_wk").agg(list).reset_index()\nprice_df = pd.merge(price_df, tmp, how="left", on="wm_yr_wk")\nprice_df = price_df.explode("date")\nprice_df.drop(["wm_yr_wk"], axis=1, inplace=True)\n\nprice_df = price_df.set_index(id_cols + ["date"])\nprice_df = price_df[["sell_price"]].unstack()\nprice_df.columns = price_df.columns.droplevel()\nprice_df.reset_index(inplace=True)\n\n# Preprocess Calendar and SNAP\ntmp = calendar[["date", "event_name_1"]]\n\nnba_start_idx = tmp[tmp.event_name_1 == "NBAFinalsStart"].index\nnba_end_idx = tmp[tmp.event_name_1 == "NBAFinalsEnd"].index\nnba_idxs = zip(nba_start_idx, nba_end_idx)\n\nram_start_idx = tmp[tmp.event_name_1 == "Ramadan starts"].index\nram_end_idx = tmp[tmp.event_name_1 == "Eid al-Fitr"].index\nram_end_idx.append(pd.Index([(tmp.index.max())]))  # Add ending to the last\nram_idxs = zip(ram_start_idx, ram_end_idx)\n\nfor start, end in nba_idxs: tmp.iloc[start:end + 1, 1] = "NBA Finals"\nfor start, end in ram_idxs: tmp.iloc[start:end + 1, 1] = "Ramadan"\n\ntmp2 = tmp.dropna(axis=0)\ncalendar_df = pd.DataFrame(columns=pd.date_range(start_date, freq="D", periods=1913 + 56),\n                           index=tmp2.event_name_1.unique())\ntmp3 = tmp2.groupby("event_name_1").agg(list).reset_index()\na = zip(tmp3["event_name_1"], tmp3["date"])\nfor row, col in a: calendar_df.loc[row, col] = 1\n\nsnap_ca = pd.DataFrame(index=["CA"], columns=pd.date_range(start_date, freq="D", periods=1913 + 56))\nsnap_tx = pd.DataFrame(index=["TX"], columns=pd.date_range(start_date, freq="D", periods=1913 + 56))\nsnap_wa = pd.DataFrame(index=["WI"], columns=pd.date_range(start_date, freq="D", periods=1913 + 56))\n\nsnap_ca.loc["CA", :] = calendar["snap_CA"].values\nsnap_tx.loc["TX", :] = calendar["snap_TX"].values\nsnap_wa.loc["WI", :] = calendar["snap_WI"].values\nsnap_df = pd.concat([snap_ca, snap_tx, snap_wa])\n\ncalendar_df = calendar_df.fillna(0)\n\nmapper_back_state = {v: k for k, v in mapper["state_id"].items()}\nsnap_df.index = snap_df.index.map(mapper_back_state)\n\ndel snap_ca, snap_tx, snap_wa, tmp, tmp2, tmp3, calendar, price, mapper_back_state\n\nprint("pre-processing is done, saving; calendar_df, snap_df, price_df, train_df, ordered_ids_and_weights")\n\n#save_obj(calendar_df,"calendar_df")\n#save_obj(price_df,"price_df")\n#save_obj(snap_df,"snap_df")\n#save_obj(train_df,"train_df")\n#save_obj(ordered_ids_and_weights, "ordered_ids_and_weights")\n\nprint("Saved the objects")')


# ## Feature Engineering

# In[ ]:


id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
no_id_id_columns = ['item_id', 'store_id', 'cat_id', 'dept_id', 'state_id']


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df

# Select only the part of the data that will be used in training and feature engineering
def sample_from_train(train_df, start, end):
    return train_df[ id_cols + list(pd.date_range(start, end))]

# Turn leading zeros into NAs
def first_zeros_to_NAs(df):
    # df: first 5 columns are Ids -- hence starts form 6
    for i in range(len(df)):
        series = df.iloc[i, 6:].values
        to_index = (np.argmax(series != 0))
        df.iloc[i, 6:to_index] = np.NaN
        return df

# Select the data between minus days before the date_from and date_from
def get_timespan(df, date_from, minus, periods, freq="D"):
    return df[pd.date_range(date_from - timedelta(days=minus), periods=periods, freq=freq)]

# Create sales related features
def fe(df, date_from, get_label=True, name_prefix=None):
    X = dict()

    for i in [3, 7, 30, 90, 180, 365]:
        tmp = get_timespan(df, date_from, i, i)
        X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s' % i] = tmp.mean(axis=1).values
        X['median_%s' % i] = tmp.median(axis=1).values
        X['min_%s' % i] = tmp.min(axis=1).values
        X['max_%s' % i] = tmp.max(axis=1).values
        X['std_%s' % i] = tmp.std(axis=1).values

    for i in [3, 7, 30, 90, 180, 365]:
        tmp = get_timespan(df, date_from + timedelta(days=-7), i, i)
        X['diff_%s_mean_2' % i] = tmp.diff(axis=1).mean(axis=1).values
        X['mean_%s_decay_2' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
        X['mean_%s_2' % i] = tmp.mean(axis=1).values
        X['median_%s_2' % i] = tmp.median(axis=1).values
        X['min_%s_2' % i] = tmp.min(axis=1).values
        X['max_%s_2' % i] = tmp.max(axis=1).values
        X['std_%s_2' % i] = tmp.std(axis=1).values

    for i in [3, 7, 14, 30, 90, 180, 365]:
        tmp = get_timespan(df, date_from, i, i)
        X['has_sales_days_in_last_%s' % i] = (tmp > 0).sum(axis=1).values
        X['last_has_sales_day_in_last_%s' % i] = i - ((tmp > 0) * np.arange(i)).max(axis=1).values
        X['first_has_sales_day_in_last_%s' % i] = ((tmp > 0) * np.arange(i, 0, -1)).max(axis=1).values
        X['Number_of_days_to_max_sales_in_last_%s' % i] = (pd.to_datetime(date_from) - pd.to_datetime(
            get_timespan(df, date_from, i, i).idxmax(axis=1).values)).days.values
        X['Number_of_days_to_min_sales_in_last_%s' % i] = (pd.to_datetime(date_from) - pd.to_datetime(
            get_timespan(df, date_from, i, i).idxmin(axis=1).values)).days.values


    for i in range(1, 29):
        X["lag_%s" % i] = get_timespan(df, date_from, i, 1).values.ravel()

    for i in range(7):
        X["mean_4_dow_%s" % i] = get_timespan(df, date_from, 4 * 7 - i, 4, freq="7D").mean(axis=1).values
        X["mean_8_dow_%s" %i] = get_timespan(df, date_from, 8 * 7 - i, 8, freq="7D").mean(axis=1).values
        X["mean_13_dow_%s" % i] = get_timespan(df, date_from, 13 * 7 - i, 13, freq="7D").mean(axis=1).values
        X["mean_26_dow_%s" % i] = get_timespan(df, date_from, 26 * 7 - i, 26, freq="7D").mean(axis=1).values
        X["mean_52_dow_%s" % i] = get_timespan(df, date_from, 52 * 7 - i, 52, freq="7D").mean(axis=1).values

    X = pd.DataFrame(X)

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
        return X
    
    # LABELS ARE SALES VALUES OF NEXT 28 DAYS
    if get_label:
        y = df[pd.date_range(date_from, periods=28)].values
        return X, y
    else:
        return X

# Create event related features
def calendar_fe(calendar_df, date_from, number_of_series=30490, add_bulk=False):
    X={}

    for i in [7,28]:
        tmp = get_timespan(calendar_df, date_from, i, i)
        X["days_after_last_event_in_last_%s_days" %i] = np.repeat(i - ((tmp.sum() > 0) * np.arange(i)).max(), number_of_series)

    tmp = get_timespan(calendar_df, date_from + timedelta(days=28), 27, 28)
    X["days_to_next_event_in_28"] = np.repeat(28 - ((tmp.sum()) * np.arange(28, 0, -1)).max(), number_of_series)

    X = pd.DataFrame(X)
    
    # if add_bulk, Adds 28 binary features representing whether there is an event in next 28 days
    if add_bulk:
        ## Create daily calendar entries for next 28 days -- bulk binary features
        tmp = get_timespan(calendar_df, date_from + timedelta(days=28), 27, 28)
        calendar_l = []
        for i in range(28):
            series = tmp.iloc[i, :]
            idx = [tmp.index[i] + "_%s_days_later_CATEGORICAL" % j for j in range(1, 29)]
            X_tmp = pd.DataFrame(data=[series.values]*number_of_series, columns=idx, index=range(number_of_series))
            calendar_l.append(X_tmp)

        calendar_l.append(X)
        calendar_l = pd.concat(calendar_l, axis=1)

        return calendar_l

    else:
        return X

# Create price related features
def price_fe(price_df, date_from):
    X={}
    for i in [28]:
        tmp = get_timespan(price_df, date_from + timedelta(days=i), i, i)
        X["max_price_NEXT_%s_days" %i] = tmp.max(axis=1).values
        X["min_price_NEXT_%s_days" % i] = tmp.min(axis=1).values
        X["percent_price_change_NEXT_%s_days" % i] = (X["max_price_NEXT_%s_days" %i] -
                                                      X["min_price_NEXT_%s_days" % i])/X["min_price_NEXT_%s_days" % i]
        #X["price_NA_NEXT_%s_days" % i] = tmp.isna().sum(axis=1).values

    for i in [28,90,180]:
        tmp = get_timespan(price_df, date_from, i, i)
        X["max_price_last_%s_days" % i] = tmp.max(axis=1).values
        X["min_price_last_%s_days" % i] = tmp.min(axis=1).values
        X["percent_price_change_last_%s_days" % i] = (X["max_price_last_%s_days" % i] - X[
            "min_price_last_%s_days" % i]) / X["min_price_last_%s_days" % i]
        X["price_NA_last_%s_days" % i] = tmp.isna().sum(axis=1).values

    X = pd.DataFrame(X)
    return X


# Create same features for different points in time and then store them as elements of a list
# First element is the data that is derived looking back from the start of the validation day so it will be used for early stopping of LGBM
def create_train_and_val_as_list_of_df(df, calendar_df, price_df, multi_indexes, val_from, number_of_weeks):
    X_l = []
    y_l = []
    weights = ordered_ids_and_weights.copy()

    for i in tqdm(range(number_of_weeks)):
        dt_from = val_from + timedelta(days=- i*7)
        X, y = fe(df, dt_from, get_label=True)
        X_calendar = calendar_fe(calendar_df, dt_from)
        X_price = price_fe(price_df, dt_from)
        weights["weights"] *= (0.998**i)
        X_l.append(reduce_mem_usage(pd.concat([X, X_calendar, X_price, weights], axis=1)))
        y_l.append(y)

    return X_l, y_l

# Create same features this time to describe the status series just before the test start date
def create_test_as_df(df, calendar_df, price_df, multi_indexes, test_from):
    X_test = fe(df, test_from, get_label=False)
    X_calendar = calendar_fe(calendar_df, test_from)
    X_price = price_fe(price_df, test_from)
    return pd.concat([X_test, X_calendar, X_price, ordered_ids_and_weights], axis=1)

# define cost function -- From Ragnar's 
def custom_asymmetric_train(y_pred, y_true):
    y_true = y_true.get_label()
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * residual, -2 * residual * 1.15)
    hess = np.where(residual < 0, 2, 2 * 1.15)
    return grad, hess

# Function for 28 LBM training 
def train_and_predict(train_X: pd.DataFrame, train_y:np.ndarray, val_X: pd.DataFrame, val_y:pd.DataFrame, test_X: pd.DataFrame,
                      features, category_features, submission_name="submission.csv"):

    submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")
    df_val_pred = submission.iloc[:30490, :]
    df_val_label = submission.iloc[:30490, :]
    #importances = pd.DataFrame(index=features, columns=range(1,29))
    category_features = [col for col in category_features if col in features]
    params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'custom',
        #"tweedie_variance_power": 1.1,
        'n_jobs': -1,
        'seed': 236,
        "num_leaves": 63,
        'learning_rate': 0.1,
        'bagging_fraction': 0.75,
        'bagging_freq': 10,
        'colsample_bytree': 0.6,
        "num_boost_round": 2500,
        "early_stopping_rounds": 50,
        "min_data_in_leaf": 30}

    for i in range(28):
        print("=" * 50)
        print("Fold%s" % (i + 1))
        print("=" * 50)

        train_set = lgb.Dataset(train_X[features], pd.Series(train_y[:, i]), weight=train_X["weights"],
                                categorical_feature=category_features)
        val_set = lgb.Dataset(val_X[features], pd.Series(val_y[:, i]), weight=val_X["weights"],
                              categorical_feature=category_features)

        model = lgb.train(params, train_set, valid_sets= [train_set,val_set], verbose_eval=50, fobj=custom_asymmetric_train)
        
        # Store model predictions on validation data to calculate validation WRMSSE at the end of the training
        df_val_pred.iloc[:, (i + 1)] = model.predict(val_X[features])
        df_val_label.iloc[:, (i + 1)] = pd.Series(val_y[:, i])
        
        submission.iloc[:30490, (i + 1)] = model.predict(test_X[features])

    submission.to_csv(submission_name, index=False)
    # Calculate WRMSSE
    wrmsse = ordered_pred_df_to_wrmsse(df_val_pred, df_val_label)
    print("*" * 50)
    print("Validation WRMSSE: ",wrmsse)


# ## Functions to calculate WRMSSE using Validation Predictions

# In[ ]:


group_ids = ('all_id', 'state_id', 'store_id', 'cat_id', 'dept_id', ['state_id', 'cat_id'], ['state_id', 'dept_id'], ['store_id', 'cat_id'], ['store_id', 'dept_id'], 'item_id', ['item_id', 'state_id'], ['item_id', 'store_id'])
no_id_id_columns = ['item_id', 'store_id', 'cat_id', 'dept_id', 'state_id']

agg_level_to_denominator = load_obj("denominator")
agg_level_to_weight = load_obj("weight")
id_df = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
id_df = id_df[no_id_id_columns]
#number_of_weeks_minus_one = 3-1 # One week goes to validation


def ordered_pred_df_to_wrmsse(pred_df, label_df, as_eval_metric=False):
    if not as_eval_metric:
        pred_df.drop(["id"], axis=1, inplace=True)
        label_df.drop(["id"], axis=1, inplace=True)

        pred_df = pd.concat([id_df, pred_df], axis=1)
        label_df = pd.concat([id_df, label_df], axis=1)


    pred_df["all_id"] = 0
    label_df["all_id"] = 0


    total = 0
    level_total_wrmsse = dict(zip(range(12), [0] * 12))

    for i, id in enumerate(group_ids):
        rmse = (((pred_df.groupby(id).sum() - label_df.groupby(id).sum()) ** 2).mean(axis=1)) ** 0.5
        RMSSE = rmse / agg_level_to_denominator[i]
        level_total_wrmsse[i] = ((1 / 12) * RMSSE * agg_level_to_weight[i]).sum()

        total += level_total_wrmsse[i]

    return total #, level_total_wrmsse


def calculate_eval_metric(preds, train_data):
    label = train_data.get_label()
    #n=int(len(label) / 30490)

    df = pd.concat([id_df], axis=0) # *n

    df["F1"] = preds.astype("float")
    df["F2"] = label.astype("float")

    df["all_id"] = 0
    wrmsse = 0
    for i, id in enumerate(group_ids):
        rmse = (((df.groupby(id).sum()["F1"].values - df.groupby(id).sum()["F2"].values) ** 2).mean()) ** 0.5
        RMSSE = rmse / agg_level_to_denominator[i]
        wrmsse += ((1 / (12)) * RMSSE * agg_level_to_weight[i]).sum() # n*12


    return "WRMSSE", wrmsse, False


# ## Creating Train & Test Data and Model Training

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nmulti_indexes = train_df.set_index(no_id_id_columns).index\n\ntrain_start = date(2013,1,1)\ntrain_end = date(2016, 3, 27)\nvalidation_start = date(2016, 3, 28)\nvalidation_end = date(2016, 4, 24)\ntest_start =  date(2016, 4, 25)\ntest_end =  date(2016, 5, 22)\n\n\n# Select fresh examples\ntrain_df = sample_from_train(train_df, train_start, validation_end)\nprice_df = sample_from_train(price_df, train_start, test_end)\n\n# Create train, labels  and test\nX_l, y_l = create_train_and_val_as_list_of_df(df=train_df, calendar_df=calendar_df,  price_df = price_df,\n                                              multi_indexes=multi_indexes, val_from=validation_start,\n                                              number_of_weeks=90)\n\n\nX_test = create_test_as_df(train_df, calendar_df, price_df, multi_indexes, test_start)\n\nsave_obj(X_l,"X_l")\nsave_obj(y_l,"y_l")\nsave_obj(X_test,"X_test")')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# X_l = load_obj_from_sets("X_l")\n# y_l = load_obj_from_sets("y_l")\n# X_test = load_obj_from_sets("X_test")\n\n\n\n# Select validation\nval_X = X_l.pop(0)\nval_y = y_l.pop(0)\n\n# Make train_X and train labels\nX_l = pd.concat(X_l, axis=0)\ny_l = np.concatenate(y_l, axis=0)\n\n\n\nfeatures = [col for col in val_X.columns if col != "weights" ]\n\ncategory_cols = [\'item_id\', \'dept_id\', \'cat_id\', \'store_id\', \'state_id\'] + [col for col in features if "_CATEGORICAL" in col]\n                \n\nprint("Number of Features: ", len(features))\n\ntrain_and_predict(X_l, y_l, val_X, val_y, X_test, features, category_cols)')


# In[ ]:




