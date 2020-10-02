#!/usr/bin/env python
# coding: utf-8

# Few things are more attractive than money. Predicting the future is a big candidate, indeed. Let's try this time to get both at once by predicting stock market returns, not in the short or long run, but somewhere in the middle. I have always wanted to invest not blindly, not checking online brokers five times in the morning. The return for the next quarter looks best. Here we use [New York Stock Exchange stock prices](https://www.kaggle.com/dgawlik/nyse#prices.csv) data combined with last 10 years fundamentals for most of those companies.
# 
# Here is a list of motivations:
# 1. Find the stocks with the highest expected value increase
# 2. Understand the main factors driving price dynamics
# 3. Reveal the performance of prediction models from naive to sophisticated
# 
# 
# 

# In[ ]:


import os
import numpy as np
import pandas as pd
# import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

sns.set(color_codes=True)


# In[ ]:


FOLDER = "../input/"
PATH_WRITE = "/kaggle/working/"

folder_nyse = "nyse"
filename_securities   = "securities.csv"
filename_fundamentals = "fundamentals.csv"
filename_prices       = "prices-split-adjusted.csv"

max_rows = None


# In[ ]:


[elem for elem in os.listdir("/kaggle/working")]


# #### If you like, you can skip EDA and data preprocessing to get the persisted dataframe ready to model, by going to the checkpoint.

# In[ ]:


df_sec = pd.read_csv(os.path.join(FOLDER, folder_nyse, filename_securities), nrows=max_rows)


# In[ ]:


def parse_state(address):
    return address.split(",")[-1].strip()

df_sec["state"] = df_sec["Address of Headquarters"].apply(parse_state)


# In[ ]:


df_sec.head()


# Where are NSYE stocks from?

# In[ ]:


sns.catplot(
    y="state", kind="count", data=df_sec,
    aspect=2.0,
    order = df_sec['state'].value_counts().index
           )


# In[ ]:


sns.catplot(
    y="GICS Sub Industry", kind="count", data=df_sec,
    aspect=2.0,
    order = df_sec['GICS Sub Industry'].value_counts().index[:30]
           )


# In[ ]:


df_fun = pd.read_csv(os.path.join(FOLDER, folder_nyse, filename_fundamentals), nrows=max_rows)


# In[ ]:


df_fun.drop(columns="Unnamed: 0", inplace=True)

df_fun.head()


# In[ ]:


df_fun.info()


# In[ ]:


df_fun['Ticker Symbol'].value_counts().max()


# The fundamentals history is 4 years long. Thus, it is going to be almost useless for predicting the impact on prices. Let's search somewhere else.

# In[ ]:


folder_fundamentals10 = "some-nice-data-indeed"
filename_fundamentals10 = "sim_fin_nyse_fundamentals_2007-2016.csv"


# In[ ]:


df_fun10 = pd.read_csv(os.path.join(FOLDER, folder_fundamentals10, filename_fundamentals10),
                       nrows=max_rows)

# n_samples = len(df_fun10) if max_rows is None else max_rows
# df_fun10.sample(n=n_samples)

df_fun10.shape


# In[ ]:


df_fun10.head()


# In[ ]:


df_fun10["datetime"] = pd.to_datetime(df_fun10["date"])
df_fun10["quarter"] = df_fun10["datetime"].dt.to_period("Q")

df_fun10.head()


# In[ ]:


df_fun10["year"] = df_fun10["datetime"].dt.year.map(str)
df_fun10["year"].value_counts().plot(kind="barh")


# The data is originaly formatted as sparse records of (stock, time,  metric, value). It is transfomed into a 2-d matrix of (stock, time) x metric using pandas pivot function.

# In[ ]:


# df_fun10.set_index(["Ticker", "year_quarter"])
df_fun_pivot = df_fun10.pivot_table(
    index=["Ticker", "quarter"],
    columns="Indicator Name",
    values="Indicator Value",
    aggfunc=np.mean
)
df_fun_pivot.head()


# In[ ]:


df_fun_pivot.shape


# In[ ]:


df_pri_daily = pd.read_csv(os.path.join(FOLDER, folder_nyse, filename_prices), nrows=max_rows)


# In[ ]:


df_pri_daily["datetime"] = pd.to_datetime(df_pri_daily["date"])
df_pri_daily["quarter"] = df_pri_daily["datetime"].dt.to_period("Q")

df_pri_daily.head()


# In[ ]:


df_pri = (df_pri_daily
          .sort_values(by=["symbol", "date"], ascending=True)
          .drop_duplicates(subset=["symbol", "quarter"], keep="last")
         )


# In[ ]:


sns.catplot(
    y="quarter", kind="count", data=df_pri,
    aspect=2.0,
    order = df_pri['quarter'].value_counts().index[:30]
           )


# In[ ]:


target = "close_rel_increase_prev_quarter"

df_pri.sort_values(["symbol", "quarter"], inplace=True)
df_pri[target] = (df_pri
                  .groupby(["symbol"])["close"]
                  .transform(lambda x: x.pct_change())
                 )
df_pri.sort_index(inplace=True)


# In[ ]:


df_pri.head()


# During the evolution of this work, a silent mistake came to the surface in the shape of baseline_pred_mean beating every other model.
# The same way, models had baseline_pred_mean among the top features, sometimes accounting for almost all of the weight and importances among features.
# 
# After becaming ovbious, the root cause was the inocent use of mean without expanding aggregation. Thus, each datapoint contained illicit info about future mean of the target, instead of averaging target exclusively up to the present.
# 
# Always keep yourself safe from data leakage!

# In[ ]:


df_pri["baseline_pred_same_prev"]       = df_pri.groupby(["symbol"])[target].transform(lambda x: x.shift())
df_pri["baseline_pred_mean"]            = df_pri.groupby(["symbol"])[target].transform(lambda x: x.expanding().mean().shift())
df_pri["baseline_pred_exp_weight_mean"] = df_pri.groupby(["symbol"])[target].transform(lambda x: x.ewm(halflife=1).mean().shift())
df_pri.dropna(subset=["baseline_pred_same_prev", "baseline_pred_mean", "baseline_pred_exp_weight_mean"])


# In[ ]:


df_pri.head()


# In[ ]:


df_fun_pivot.reset_index(inplace=True)
df_fun_pivot.info()


# In[ ]:


df_fun_pivot["next_quarter"] = (df_fun_pivot["quarter"].dt.end_time + pd.Timedelta(days=365/4)
                               ).dt.to_period("Q")

df_fun_pivot.head()


# In[ ]:


baseline_cols = ["baseline_pred_same_prev", "baseline_pred_mean", "baseline_pred_exp_weight_mean"]

for col in baseline_cols + [target]:
    sns.distplot(df_pri[col].dropna(), norm_hist=True, hist=False, label=col)
plt.legend()


# It can be seen how baseline_pred_mean is remarkably the most conservative model, with estimated targets very tight around the global mean. At the other side, the actual target, close_rel_increase_prev_quarter, shows a much higher volatility with scattered estimates. Somehow, more information is needed to be incorporated into the model to safely expand the baseline_pred_mean.

# In[ ]:


corr = df_pri[baseline_cols + [target]].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap=sns.light_palette("green"), linewidths=.5)


# In[ ]:


symbols = set(df_pri["symbol"].unique())
tickers = set(df_fun_pivot["Ticker"].unique())

n_stocks_common = len(symbols.intersection(tickers))
n_symbols_unmatched = len(symbols.difference(tickers))
n_tickers_unmatched = len(tickers.difference(symbols))

print(n_stocks_common, n_symbols_unmatched, n_tickers_unmatched)


# In[ ]:


df = df_pri.merge(df_fun_pivot, left_on=["symbol", "quarter"], right_on=["Ticker", "next_quarter"])


# In[ ]:


df[df["Ticker"] == "AMZN"]


# In[ ]:


df.head()


# In[ ]:


df_fun_pivot.head(20)


# In[ ]:


df.shape


# In[ ]:


# df.to_csv(os.path.join(FOLDER, "df.csv"))


# The target is defined as:

# In[ ]:


target = "close_rel_increase_prev_quarter"
baseline_cols = ["baseline_pred_same_prev", "baseline_pred_mean", "baseline_pred_exp_weight_mean"]

# df["baseline_pred"] = df.groupby(["symbol"])[target].transform(lambda x: x.shift())


# In[ ]:


df = df.sort_values("quarter_x")

n_quarters = df["quarter_x"].nunique()

cols_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
cols_numeric.remove(target)

for col in cols_numeric:  # do not forget about expanding for avoiding data leakage!
    df[col + "_todate_mean"] = df.groupby(["symbol"])[col].transform(lambda x: x.expanding().mean())
    df[col + "_todate_var"]  = df.groupby(["symbol"])[col].transform(lambda x: x.expanding().var())
                            
df.head(20)


# In[ ]:


df = df.dropna(subset=[target] + baseline_cols, axis=0)


# ## Checkpoint: Persist dataframe

# In[ ]:


filnename_df = "df.pickle"


# In[ ]:


df.to_pickle(os.path.join(PATH_WRITE, filnename_df))


# In[ ]:


df = pd.read_pickle(os.path.join(PATH_WRITE, filnename_df))


# In[ ]:


df.shape


# In[ ]:


last_quarter = df["quarter_x"].max()
train = df[df["quarter_x"] < last_quarter]
test = df[df["quarter_x"] == last_quarter]

len(train), len(test)


# In[ ]:


cat_cols = [
    "symbol",
    "quarter_x",
    "Ticker",
    "quarter_y"
]

rejected_cols = [
    "date",
    "open",
    "close",
    "low",
    "high",
    "volume",
    "datetime",
    "next_quarter",
] + cat_cols

train.drop(columns=rejected_cols, inplace=True)
test.drop(columns=rejected_cols, inplace=True)


# In[ ]:


def plot_together(values, names):
    for pred, name in zip(values, names):
        sns.distplot(pred, norm_hist=True, hist=False, label=name)
    plt.legend()


# In[ ]:


values = [train[target].dropna(), test[target].dropna()]
names = ["train", "test"]

plot_together(values, names)


# In[ ]:


train_x = train.drop(columns=target)
train_y = train[target]
test_x = test.drop(columns=target)
test_y = test[target]


# In[ ]:


# train_x_safe = (train_x
#                 .replace([np.inf, -np.inf], np.nan)
#                 .fillna(train_x.mean())
#                )

# test_x_safe = (test_x
#                 .replace([np.inf, -np.inf], np.nan)
#                 .fillna(train_x.mean())
#                )


# In[ ]:


train_x_mean = train_x.mean()
train_x_std = train_x.std()

train_x_norm = ((train_x - train_x_mean) / train_x_std).fillna(0)
test_x_norm = ((test_x - train_x_mean) / train_x_std).fillna(0)


# In[ ]:


lgb_model = lgb.LGBMRegressor()
lgb_model.fit(train_x_norm, train_y)


# Model tunning:
# 
# Bayesian optimization is pretty nice for minimizing heavy functions, like model performance vs. hyperparameters. A reasonable analogy is stochastig gradien descent, but with bayesian inference instead of differentiation as a means to efficiently explore the domain.
# As well, dealing with naturally order data, such as time series, cross-validating to random folds is less natural than validating against a fold with data more recent that any in the training folds.

# In[ ]:


do_incremental_time_cross_validation = True
# n_folds = n_quarters - 1
n_folds = 5

do_incremental_time_cross_validation_full_quarters = "not yet implemented"

if do_incremental_time_cross_validation:
    folds = TimeSeriesSplit(n_splits=n_folds)
else:
    folds = None

    
def fit_type_params(params):
    int_params = ["num_trees", "max_depth", "num_leaves", "max_bin"]
    fitted_params = {}
    for (k, v) in params.items():
        if k in int_params:
            v = int(round(v))
        fitted_params[k] = v
    return fitted_params


def bayes_parameter_opt_lgb(train_x, train_y, init_rounds=10, optimization_rounds=10):
    
    random_seed = 42
    train_set = lgb.Dataset(data=train_x, label=train_y)
    metric = "mse"
    params_static = {
        "objective": "regression",
        "metric":     metric
    }
    
    params_ranges = {
        "num_trees":        (10, 100),
        "learning_rate":    (0.001, 0.3),
        "num_leaves":       (2, 50),
        "feature_fraction": (0.5, 1.0),
        "max_bin":          (2, 2048),
        'bagging_fraction': (0.7, 1.0),
        'lambda_l2' :       (0.0, 100.0),
    }
    
    def eval_cv_mse_negative(
        num_trees,
        learning_rate,
        num_leaves,
        feature_fraction,
        max_bin,
        bagging_fraction,
        lambda_l2
    ):
        
        params = {}
        params["num_trees"]        = num_trees
        params['learning_rate']    = learning_rate
        params['num_leaves']       = num_leaves
        params['feature_fraction'] = feature_fraction
        params['max_bin']          = max_bin
        params['bagging_fraction'] = bagging_fraction
        params['lambda_l2']        = lambda_l2
        
        params = fit_type_params({**params_static, ** params})
    
        cv_result = lgb.cv(params,
                           train_set=train_set,
                           folds=folds,
                           seed=random_seed,
                           metrics=[metric]
                          )
        mse_cv_neg = -np.array(cv_result["l2-mean"]).mean()
        return mse_cv_neg

    optimizer = BayesianOptimization(f=eval_cv_mse_negative, random_state=0, pbounds=params_ranges)
    optimizer.maximize(init_points=init_rounds, n_iter=optimization_rounds)
    
    result = pd.DataFrame.from_records(optimizer.res)
    print(result)
    best_result_row = result.loc[result["target"].idxmax()]
    best_result = {
        "params" :  best_result_row["params"],
        metric :   -best_result_row["target"]
    }
    return best_result

best_result = bayes_parameter_opt_lgb(train_x_norm[:1000], train_y[:1000], init_rounds=2, optimization_rounds=5)
best_result


# In[ ]:


lgb_model_cv = lgb.LGBMRegressor(
    **fit_type_params(best_result["params"])
)
lgb_model_cv.fit(train_x_norm, train_y)


# In[ ]:


lgb.plot_importance(lgb_model_cv, height=0.8, max_num_features=20)


# In[ ]:


lin_model = LinearRegression()
lin_model.fit(train_x_norm, train_y)


# In[ ]:


df_lin_coefs = pd.DataFrame(
    list(zip(train_x_norm.columns, lin_model.coef_)),
    columns=["feature", "linear_coef"]
).sort_values("linear_coef", ascending=False)

most_positive_feats = df_lin_coefs.head()
most_negative_feats = df_lin_coefs.tail()

pd.concat([most_positive_feats, most_negative_feats]).set_index("feature").plot.barh()


# Feature insights reveal that trading world is wininig on its self-fulfilling prophecy: "search for momentum and will get returns". Said otherwise, prev price dinamycs is better predictor than fundamentals. At the end, price is a very valuable info about actual value and gathers consensus about many well informed players.
# It can get more audacious and try to state
# * high valuation (open and close mean) -> low target
# * high volatility (intraperiod max and min) -> high target
# 
# However, there is valuable info too about driving factors within the fundamental indicators to dive into or to select when researching a company.

# In[ ]:


train_pred_lgm = lgb_model.predict(train_x_norm)
test_pred_lgm  = lgb_model.predict(test_x_norm)
mse_train_lgm  = sk_mse(train_y, train_pred_lgm)
mse_test_lgm   = sk_mse(test_y, test_pred_lgm)

train_pred_lgm_cv = lgb_model_cv.predict(train_x_norm)
test_pred_lgm_cv  = lgb_model_cv.predict(test_x_norm)
mse_train_lgm_cv  = sk_mse(train_y, train_pred_lgm_cv)
mse_test_lgm_cv   = sk_mse(test_y, test_pred_lgm_cv)

train_x_safe = (train_x_norm
                .replace([np.inf, -np.inf], np.nan)
                .fillna(train_x.mean())
               )
test_x_safe = (test_x_norm
                .replace([np.inf, -np.inf], np.nan)
                .fillna(train_x.mean())
               )
train_pred_linear = lin_model.predict(train_x_safe)
test_pred_linear = lin_model.predict(test_x_safe)

mse_train_linear = sk_mse(train_y, train_pred_linear)
mse_test_linear = sk_mse(test_y, test_pred_linear)


mse_summary = [
    ("lgm", mse_train_lgm, "train"),
    ("lgm", mse_test_lgm, "test"),
    ("linear", mse_train_linear, "train"),
    ("linear", mse_test_linear, "test"),
    ("lgm_cv", mse_train_lgm_cv, "train") ,
    ("lgm_cv", mse_test_lgm_cv, "test") ,
]


for col in baseline_cols:
    mse_train_baseline = sk_mse(train_y, train[col])
    mse_summary.append((col, mse_train_baseline, "train"))
for col in baseline_cols:
    mse_test_baseline = sk_mse(test_y, test[col])
    mse_summary.append((col, mse_test_baseline, "test"))
    
mse_summary_df = pd.DataFrame(mse_summary, columns=["model", "mean_squared_error", "dataset"])

mse_summary_df = mse_summary_df.sort_values("mean_squared_error")
# .set_index("model")
# mse_summary_df.plot.barh()
sns.barplot(x="mean_squared_error", y="model", hue="dataset",
            data=mse_summary_df,
            order = mse_summary_df[mse_summary_df["dataset"] == "test"].sort_values("mean_squared_error")["model"]
)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'sns.barplot')


# Some observations:
# * Baseline models are not expressive enough even to explain known data.
# * LightGBM with default parameters gets the most accurate picture of known data, but suffers from greatest overfit.
# * However, defult parameters of linear regression beats previous due to better generalization, as expected for a simpler model.
# * Beating linear regression is possible for lightGBM only with incremental time cross validation and bayesian grid search.
# 
# It can be concluded too that clean data approach and cautions feature engineering (not allow future data to leak into the past!) but not affraid of beeing exhaustive (yes, let's make the cloud work it out with several agggregations for every feature) can pave a golden way for understimated models, just like the right weapon and a taste of boldness gave the edge to David over Goliath.

# In[ ]:


mse_summary_df


# Coming soon: DeeP Learning Enriched Time Series!

# In[ ]:


df = pd.read_pickle(os.path.join(PATH_WRITE, filnename_df))


# In[ ]:


df.head().T


# In[ ]:


# from keras.models import Sequential
# from keras.layers import Dense, Embedding, LSTM


# In[ ]:


# train_x.head()


# In[ ]:


# train_y.head()


# In[ ]:


# model = Sequential()
# model.add(LSTM(4, input_shape=(1000, n_look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')


# In[ ]:


# model.fit(train_x_norm[:1000], train_y[:1000], epochs=100, batch_size=1, verbose=2)


# In[ ]:


from fastai.tabular import *
df = add_datepart(df, "date", drop=False)


# In[ ]:


last_quarter = df["quarter_x"].max()
train = df[df["quarter_x"] < last_quarter]
test = df[df["quarter_x"] == last_quarter]

# valid_quarter = train["quarter_x"].max()
# valid = train[train["quarter_x"] == valid_quarter]
# train = train[train["quarter_x"] < valid_quarter]

len(train), len(test)


# In[ ]:


feats_cat  = [
    "symbol",
    "quarter_x",
    "Year",
    "Month"
]

rejected_cols = [
    "Ticker",
    "date",
    "open",
    "close",
    "low",
    "high",
    "volume",
    "datetime",
    "next_quarter",
    "quarter_y"
]

train.drop(columns=rejected_cols, inplace=True)
# valid.drop(columns=rejected_cols, inplace=True)
test.drop(columns=rejected_cols, inplace=True)


# In[ ]:


feats_num = train.columns
feats_num = list(set(feats_num).difference(set(feats_cat + [target])))


# In[ ]:


train = train.reset_index()
test = test.reset_index()
valid_quarter = train["quarter_x"].max()
valid_idx = train[train["quarter_x"] == valid_quarter].index


# In[ ]:


procs = [FillMissing, Categorify, Normalize]
# procs = [Categorify]
# procs=None
data = TabularDataBunch.from_df(
    os.path.join(PATH_WRITE, "databunch"),
    df=train,
    test_df=test,
    dep_var=target,
    valid_idx=valid_idx,
    procs=procs,
    cat_names=feats_cat,
    cont_names=feats_num
)


# In[ ]:


data


# In[ ]:


layers = [2, 3, 2]
learner = tabular_learner(data, layers=layers, metrics=[RMSE()])
learning_rate = 0.01
n_epochs = 3
learner.fit_one_cycle(n_epochs, learning_rate)


# In[ ]:


test_predicts, _ = learner.get_preds(ds_type=DatasetType.Test)
test_probs = to_np(test_predicts)
pd.Series(list(test_probs)).describe()
# test_predicts


# In[ ]:


train_predicts, _ = learner.get_preds(ds_type=DatasetType.Train)
valid_predicts, _ = learner.get_preds(ds_type=DatasetType.Valid)
bound = 0.9
train_probs = to_np(train_predicts).clip(-bound, bound)
valid_probs = to_np(valid_predicts).clip(-bound, bound)

values = [train_probs, valid_probs, test_probs]
names = ["train", "valid", "test"]

plot_together(values, names)


# In[ ]:



values = [test_probs, test[target]]
names = ["test_probs", "test_truth"]

plot_together(values, names)

