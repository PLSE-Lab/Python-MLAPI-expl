#!/usr/bin/env python
# coding: utf-8

# # Quantile Regression - Tensorflow CPU - CV3
# 
# > This kernel is mainly a fork of Quantile-Regression-with-Keras interesting kernel from Ulrich GOUE. It adds TimeSeries split cross validation on the last 3 folds for better regularization.
# 
# - Fork#1 from: https://www.kaggle.com/ulrich07/quantile-regression-with-keras
# - Fork#2 from: https://www.kaggle.com/chrisrichardmiles/m5u-wsplevaluator-weighted-scaled-pinball-loss
# - Some code optimization to make it simple.
# - LabelEncoder added.
# - Embedding dimensions rules updated.
# - Data starting from 2014-03-28. Float16 removed.
# - SCALED option added.
# - CV3 added for regularization.
# 
# - v1.0: 2xDense64+1xDense32, dt_week_end category added, CV3: Last 3 folds (28 days each), EPOCH=20, LB=0.1720

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm
import gc, os
from scipy.sparse import csr_matrix
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 4000)


# In[ ]:


import tensorflow.keras.layers as L
import tensorflow.keras.models as M
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
tf.random.set_seed(2020)


# In[ ]:


def add_days(date_str, days=1, fmt="%Y-%m-%d"):
    date = datetime.strptime(date_str, fmt)
    date = date + timedelta(days=days)
    return datetime.strftime(date, fmt)


# In[ ]:


os.listdir("/kaggle/input/m5-forecasting-uncertainty")
#os.listdir("/kaggle/input/m5ubasicfeatures-testscaled")


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
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
                    df[col] = df[col].astype(np.float32) # float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def autocorrelation(ys, t=1):
    return np.corrcoef(ys[:-t], ys[t:])


# In[ ]:


SCALED = True


# In[ ]:


# Datasets with all levels timeseries + time features + scale. Starting on 2014-03-28 (to fit in kernel memory)
# For level12, all leading zeros have been removed (according to calendar)
# scale is computed on all data (since 2011) with train_pd["scale"] = train_pd.groupby(['id'])["sales"].transform(lambda x: np.abs(x - x.shift(1)).mean() )
sales_train = pd.read_pickle("/kaggle/input/m5ubasicsfeaturesscaled/basic_features_2014-03-28.pkl")
sales_test = pd.read_pickle("/kaggle/input/m5ubasicfeatures-testscaled/basic_features_test_2014-03-28.pkl")

sales = pd.concat([sales_train, sales_test])
    
sales = pd.concat([sales_train, sales_test]).set_index(["id", "date"]).sort_index().reset_index()

# Save memory now as we're not going to use these features.
sales.drop(columns=['sell_price',
 'event_type_1',
 'event_name_2',
 'event_type_2',
 'dt_weekofyear',
 'dt_quarter',
 'dt_month_cursor'], inplace=True)

# Add start date for each time series.
sales["start"] = sales.groupby('id')["date"].transform(lambda x: x.min())
del sales_train, sales_test


# In[ ]:


for c in sales.columns:
    if 'dt_' in c:
        sales[c] = sales[c].astype(np.int8) # Wrong for dt_year but no impact as using as categorical
sales.head()


# In[ ]:


# 13
categoricals_col = ["snap_CA", "snap_TX", "snap_WI", "dt_weekday", "dt_month", "dt_year", "event_name_1", "dt_dayofmonth", "dt_weekend",
                   "item_id", "dept_id", "cat_id", "store_id", "state_id"]
for cat in tqdm(categoricals_col):
    if (cat == "event_name_1") or ("_id" in cat):
        sales[cat] = sales[cat].astype(str)
    le_tmp = LabelEncoder()
    le_tmp.fit(sales[cat])
    sales[cat] = le_tmp.transform(sales[cat])


# In[ ]:


if SCALED == True:
    sales["sales"] = sales["sales"] / sales["scale"]


# In[ ]:


sales = reduce_mem_usage(sales)


# In[ ]:


numericals = []

LAGS = [28, 35, 42, 49, 56, 63]
for lag in tqdm(LAGS):
    sales[f"x_{lag}"] = sales[["id", "sales"]].groupby("id")["sales"].shift(lag)
    numericals.append(f"x_{lag}")

ROLLS = [7, 14, 28]
for roll in tqdm(ROLLS):
    for q in [0.95]:
        sales["xr_q%.3f_%d" % (q, roll)] = sales[["id", "sales"]].groupby("id")["sales"].transform(lambda x: x.shift(28).rolling(roll).quantile(q))
        numericals.append("xr_q%.3f_%d" % (q, roll))


# In[ ]:


# Additional drop if needed
drop_cols = [c for c in sales.columns if c not in ["sales", "scale", "id", "date", "start"] + numericals + categoricals_col]
print(drop_cols)
sales.drop(columns=drop_cols, inplace=True)


# In[ ]:


# Embedding rules. Max dim is 50, otherwise half of the uniques.
categoricals_info = {}
for c in categoricals_col:
    total_unique = sales[c].nunique()
    categoricals_info[c] = (total_unique, min(50, (total_unique + 1) // 2))


# In[ ]:


def make_data(df, idx, numericals_cols, categoricals_cols):
    x = {}
    x["num"] = df[idx][numericals_cols].values
    for cat in categoricals_cols:
        x[cat] = df[idx][cat].values
    t = df[idx]["sales"].values
    m = df[idx][["id", "date", "scale"]]
    return x, t, m


# In[ ]:


# Train/Valid/Test
DATE = "date"
START = "start"
MAX_LAG = max(LAGS)
MAX_ROLL = max(ROLLS) + 28
FINAL_MAX = np.maximum(MAX_LAG, MAX_ROLL)
print("FINAL_MAX", FINAL_MAX)
# Add max lag to drop NaN due to rolling windows
sales[START] = sales[START] + pd.DateOffset(days=FINAL_MAX)
sales.head()


# In[ ]:


def qloss(y_true, y_pred):
    # Pinball loss for multiple quantiles
    qs = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)


# In[ ]:


def make_model(n_in, categoricals_info):    
    num = L.Input((n_in,), name="num")
    inp = {"num": num}
    p = []
    for key, value in categoricals_info.items():
        tmp_inp = L.Input((1,), name=key)
        inp[key] = tmp_inp
        p.append(L.Embedding(value[0], value[1], name="%s_3d" % key)(tmp_inp))
        
    emb = L.Concatenate(name="embds")(p)
    context = L.Flatten(name="context")(emb)
    
    x = L.Concatenate(name="x1")([context, num])
    x = L.Dense(64, activation="relu", name="d1")(x)
    x = L.Dense(64, activation="relu", name="d2")(x)
    x = L.Dense(32, activation="relu", name="d3")(x)
    
    preds = L.Dense(9, activation="linear", name="preds")(x)
    
    model = M.Model(inp, preds, name="M1")
    model.compile(loss=qloss, optimizer="adam")
    return model


# In[ ]:


print(numericals)
print(categoricals_info)


# In[ ]:


# Time Series split validation


# In[ ]:


FIRST_DATE = '2014-03-28'
FOLD_LIST = [
    (1,  {"train_start": FIRST_DATE, "train_stop": "2016-02-01", "valid_start": "2016-02-01", "valid_stop": "2016-02-29"}),
    (2,  {"train_start": FIRST_DATE, "train_stop": "2016-02-29", "valid_start": "2016-02-29", "valid_stop": "2016-03-28"}), 
    (3,  {"train_start": FIRST_DATE, "train_stop": "2016-03-28", "valid_start": "2016-03-28", "valid_stop": "2016-04-25"}),    
]

TEST_START = '2016-04-25'
TEST_STOP = '2016-05-23'


# In[ ]:


EPOCH = 20
BATCH_SIZE = 50_000

for i, FOLD in enumerate(FOLD_LIST):
    
    xt, yt, mt = make_data(sales, ((sales[DATE] >= FOLD[1]["train_start"]) & (sales[DATE] < FOLD[1]["valid_start"]) & (sales[DATE] >= sales[START]) ), numericals, categoricals_col)
    xv, yv, mv = make_data(sales, ((sales[DATE] >= FOLD[1]["valid_start"]) & (sales[DATE] < FOLD[1]["valid_stop"]) & (sales[DATE] >= sales[START])), numericals, categoricals_col)
    
    net = make_model(len(numericals), categoricals_info)
    ckpt = ModelCheckpoint("w_%d.h5" % FOLD[0], monitor='val_loss', verbose=1, save_best_only=True,mode='min')
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    #es = EarlyStopping(monitor='val_loss', patience=3)
    if i == 0:
        print(net.summary())
        tf.keras.utils.plot_model(net, to_file='model.png', show_shapes=True, show_layer_names=True)
        
    print("Train:", mt["date"].min(), mt["date"].max(), xt["num"].shape, xt["num"].dtype, xt["dt_weekday"].shape, xt["dt_weekday"].dtype, yt.shape, yt.dtype, "NaN", np.count_nonzero(np.isnan(xt["num"])))
    print("Valid:", mv["date"].min(), mv["date"].max(), xv["num"].shape, xv["num"].dtype, xv["dt_weekday"].shape, xv["dt_weekday"].dtype, yv.shape, yv.dtype, "NaN", np.count_nonzero(np.isnan(xv["num"])))
    
    net.fit(xt, yt, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(xv, yv), callbacks=[ckpt]) # [ckpt, reduce_lr, es]
    del net, xt, xv, yt, yv, mt, mv
    gc.collect()


# In[ ]:


# Reload each model and Test
xe, ye, me = make_data(sales, ((sales[DATE] >= TEST_START) & (sales[DATE] < TEST_STOP) & (sales[DATE] >= sales[START])), numericals, categoricals_col)
pe = []
for i, FOLD in enumerate(FOLD_LIST):    
    nett = make_model(len(numericals), categoricals_info)
    nett.load_weights("w_%d.h5" % FOLD[0])
    fold_pe = nett.predict(xe, batch_size=BATCH_SIZE, verbose=1)
    pe.append(fold_pe)
pe = np.array(pe)    


# In[ ]:


# Average of per-fold prediction
pe = np.mean(pe, axis=0)


# In[ ]:


pe = pe.reshape((-1, 28, 9))
se = me["scale"].values.reshape((-1, 28))
if SCALED == False:
    se = np.ones_like(se)
ids = me["id"].values.reshape((-1, 28))


# In[ ]:


names = [f"F{i+1}" for i in range(28)]
piv = pd.DataFrame(ids[:, 0], columns=["id"])
QUANTILES = ["0.005", "0.025", "0.165", "0.250", "0.500", "0.750", "0.835", "0.975", "0.995"]
VALID = []
EVAL = []

for i, quantile in tqdm(enumerate(QUANTILES)):
    t1 = pd.DataFrame(pe[:,:, i]*se, columns=names)
    t1 = piv.join(t1)
    t1["id"] = t1["id"] + f"_{quantile}_validation"
    t2 = pd.DataFrame(pe[:,:, i]*se, columns=names)
    t2 = piv.join(t2)
    t2["id"] = t2["id"] + f"_{quantile}_evaluation"
    VALID.append(t1)
    EVAL.append(t2)


# In[ ]:


sub = pd.DataFrame()
sub = sub.append(VALID + EVAL)
del VALID, EVAL, t1, t2, sales


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv("submission.csv", index=False)


# In[ ]:


def get_rollup(train_df):
    """Gets a sparse roll up matrix for aggregation and 
    an index to align weights and scales."""
    
    # Take the transpose of each dummy matrix to correctly orient the matrix
    dummy_frames = [
        pd.DataFrame({'Total': np.ones((30490,)).astype('int8')}, index=train_df.index).T, 
        pd.get_dummies(train_df.state_id, dtype=np.int8).T,                                             
        pd.get_dummies(train_df.store_id, dtype=np.int8).T,
        pd.get_dummies(train_df.cat_id, dtype=np.int8).T,
        pd.get_dummies(train_df.dept_id, dtype=np.int8).T,
        pd.get_dummies(train_df.state_id + '_' + train_df.cat_id, dtype=np.int8).T,
        pd.get_dummies(train_df.state_id + '_' + train_df.dept_id, dtype=np.int8).T,
        pd.get_dummies(train_df.store_id + '_' + train_df.cat_id, dtype=np.int8).T,
        pd.get_dummies(train_df.store_id + '_' + train_df.dept_id, dtype=np.int8).T,
        pd.get_dummies(train_df.item_id, dtype=np.int8).T,
        pd.get_dummies(train_df.item_id + '_' + train_df.state_id, dtype=np.int8).T,
        pd.get_dummies(train_df.item_id + '_' + train_df.store_id, dtype=np.int8).T
    ]

    rollup_matrix = pd.concat(dummy_frames, keys=range(1,13), names=['Level', 'id'])

    # Save the index for later use 
    rollup_index = rollup_matrix.index

    # Sparse format will save space and calculation time
    rollup_matrix_csr = csr_matrix(rollup_matrix)
    
    return rollup_matrix_csr, rollup_index

def get_w_df(train_df, cal_df, prices_df, rollup_index, rollup_matrix_csr, start_test=1914): 
    """Returns the weight, scale, and scaled weight of all series, 
    in a dataframe aligned with the rollup_index, created in get_rollup()"""
    
    d_cols = [f'd_{i}' for i in range(start_test - 28, start_test)]
    df = train_df[['store_id', 'item_id'] + d_cols]
    df = df.melt(id_vars=['store_id', 'item_id'],
                           var_name='d', 
                           value_name = 'sales')
    df = df.merge(cal_df[['d', 'wm_yr_wk']], on='d', how='left')
    df = df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    df['dollar_sales'] = df.sales * df.sell_price

    # Now we will get the total dollar sales 
    dollar_sales = df.groupby(['store_id', 'item_id'], sort=False)['dollar_sales'].sum()
    del df

    # Build a weight, scales, and scaled weight columns 
    # that are aligned with rollup_index. 
    w_df = pd.DataFrame(index = rollup_index)
    w_df['dollar_sales'] = rollup_matrix_csr * dollar_sales
    w_df['weight'] = w_df.dollar_sales / w_df.dollar_sales[0]
    del w_df['dollar_sales']

    ##################### Scaling factor #######################
    
    df = train_df.loc[:, 'd_1':f'd_{start_test-1}']
    agg_series = rollup_matrix_csr * df.values
    no_sale = np.cumsum(agg_series, axis=1) == 0
    agg_series = np.where(no_sale, np.nan, agg_series)
    scale = np.nanmean(np.diff(agg_series, axis=1) ** 2, axis=1)

    w_df['scale'] = scale
    w_df['scaled_weight'] = w_df.weight / np.sqrt(w_df.scale)
    
    ################# spl_scale ####################
    # Now we can compute the scale and add 
    # it as a column to our w_df
    scale = np.nanmean(np.abs(np.diff(agg_series)), axis = 1)
    scale.shape
    w_df['spl_scale'] = scale

    # It may also come in handy to have a scaled_weight 
    # on hand.  
    w_df['spl_scaled_weight'] = w_df.weight / w_df.spl_scale
     
    ############## sub_id for uncertainty submission ##############
    w_df = add_sub_id(w_df)
    
    return w_df


# In[ ]:


class WSPLEvaluator(): 
    """ Will generate w_df and ability to score prediction for any start_test period """
    def __init__(self, train_df, cal_df, prices_df, start_test=1914):
        self.rollup_matrix_csr, self.rollup_index = get_rollup(train_df)
                        
        self.w_df = get_w_df(
                        train_df,
                        cal_df,
                        prices_df,
                        self.rollup_index,
                        self.rollup_matrix_csr,
                        start_test=start_test,
                    )
        
        self.quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
        level_12_actuals = train_df.loc[:, f'd_{start_test}': f'd_{start_test + 27}']
        self.actuals = self.rollup_matrix_csr * level_12_actuals.values
        self.actuals_tiled = np.tile(self.actuals.T, 9).T
        
        
    def score_all(self, preds): 
        scores_df, total = wspl(self.actuals_tiled, preds, self.w_df)
        self.scores_df = scores_df
        self.total_score = total
        print(f"Total score is {total}")


############## spl scaling factor function ###############
def add_spl_scale(w_df, train_df, rollup_matrix_csr): 
    # We calculate scales for days preceding 
    # the start of the testing/scoring period. 
    start_test = 1914
    df = train_df.loc[:, 'd_1':f'd_{start_test-1}']

    # We will need to aggregate all series 
    agg_series = rollup_matrix_csr * df.values

    # Make sure leading zeros are not included in calculations
    agg_series = h.nan_leading_zeros(agg_series)

    # Now we can compute the scale and add 
    # it as a column to our w_df
    scale = np.nanmean(np.abs(np.diff(agg_series)), axis = 1)
    scale.shape
    w_df['spl_scale'] = scale

    # It may also come in handy to have a scaled_weight 
    # on hand.  
    w_df['spl_scaled_weight'] = w_df.weight / w_df.spl_scale
    
    return w_df

########## Function for all level pinball loss for quantile u ############
def spl_u(actuals, preds, u, w_df):
    """Returns the scaled pinball loss for each series"""
    pl = np.where(actuals >= preds, (actuals - preds) * u, (preds - actuals) * (1 - u)).mean(axis=1)

    # Now calculate the scaled pinball loss.  
    all_series_spl = pl / w_df.spl_scale
    return all_series_spl

########## wspl for all quantiles ############
def wspl(actuals, preds, w_df): 
    """
    :acutals:, 9 vertical copies of the ground truth for all series. 
    :preds: predictions for all series and all quantiles. Same 
    shape as actuals"""
    quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
    scores = []
    
    # In this case, preds has every series for every  
    # quantile T, so it has 42840 * 9 rows. We first 
    # break it up into 9 parts to get the wspl_T for each.
    # We also do the same for actuals. 
    preds_list = np.split(preds, 9)
    actuals_list = np.split(actuals, 9)
    
    for i in range(9):
        scores.append(spl_u(actuals_list[i], preds_list[i], quantiles[i], w_df))
        
    # Store all our results in a dataframe
    scores_df = pd.DataFrame(dict(zip(quantiles, [w_df.weight * score for score in scores])))
    
    #  We divide score by 9 
    # to get the average wspl of each quantile. 
    spl = sum(scores) / 9
    wspl_by_series = (w_df.weight * spl)
    total = wspl_by_series.sum() / 12
    
    return scores_df, total

####################################################################################
############################ formatting for submission #############################

############## sub_id function ################
def add_sub_id(w_df):
    """ adds a column 'sub_id' which will match the 
    labels in the sample_submission 'id' column. Next 
    step will be adding '_{quantile}_validation/evaluation'
    onto the sub_id column. This will be done in another 
    function. 
    
    :w_df: dataframe with the multi-index that is 
    genereated by get_rollup()
    
    Returns w_df with added 'sub_id' column"""
    # Lets add a sub_id col to w_df that 
    # we will build to match the submission file. 
    w_df['sub_id'] = w_df.index.get_level_values(1)

    ###### level 1-5, 10 change ########
    w_df.loc[1:5, 'sub_id'] = w_df.sub_id + '_X'
    w_df.loc[10, 'sub_id'] = w_df.sub_id + '_X'

    ######## level 11 change ##########
    splits = w_df.loc[11, 'sub_id'].str.split('_')
    w_df.loc[11, 'sub_id'] = (splits.str[3] + '_' +                               splits.str[0] + '_' +                               splits.str[1] + '_' +                               splits.str[2]).values
    
    return w_df



################## add quantile function ################
def add_quantile_to_sub_id(w_df, u): 
    """Used to format 'sub_id' column in w_df. w_df must 
    already have a 'sub_id' column. This used to match 
    the 'id' column of the submission file."""
    # Make sure not to affect global variable if we 
    # don't want to. 
    w_df = w_df.copy()
    w_df['sub_id'] = w_df.sub_id + f"_{u:.3f}_validation"
    return w_df


# In[ ]:


train_df = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sales_train_evaluation.csv")
cal_df = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/calendar.csv")
prices_df = pd.read_csv("/kaggle/input/m5-forecasting-uncertainty/sell_prices.csv")


# In[ ]:


e = WSPLEvaluator(train_df, 
                  cal_df,
                  prices_df,
                  start_test=1914)


# In[ ]:


sub = pd.read_csv("submission.csv").iloc[:42840 * 9]
sub.head()


# In[ ]:


QUANTS = [0.005,0.025,0.165,0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
copies = [add_quantile_to_sub_id(e.w_df, QUANTS[i]) for i in range(9)]
w_df_9 = pd.concat(copies, axis = 0)
sorted_sub = sub.set_index('id').reindex(w_df_9.sub_id)


# In[ ]:


e.score_all(sorted_sub.values)


# In[ ]:


e.total_score

