#!/usr/bin/env python
# coding: utf-8

# # The worst meters
# In this experiment, we evaluate each building/meter pair separately and chart the errors so that we can see which meters contribute most to our global error. The goal is to demonstrate where we should focus the most effort -- either to clean up bad meter-reading data or to produce models which can better deal with those particular readings.
# 
# I believe that the harm done by the worst of the meters is two-fold: the impossibility of predicting them directly raises the error, but I suspect that it also degrades the overall model so that it performs less well even on the "good" meters. Further experimentation is required to figure out how to best mitigate this harm.
# 
# Note that we already eliminate the first 141 days of electrical readings for site 0. We already know that those data points are bad, and that we best deal with them by simply dropping them. This experiment is intended to find out what *further* cleanup is required after this basic process.

# ## Summary
# As expected, a small number of meters account for a moderate fraction of our overall error -- with just 4 meters accounting for 1%. Of particular note are building 1072, with the highest error contribution; and building 954, with 3 different meters in our top ten wall of shame. Interestingly, the much maligned buildings 1099 and 778 come in at #2 and #12, respectively, rather than headlining the chart. A surprising number of offenders were electrical meters (which we'd expect to be more predictable), starting with building 799 at #3.
# 
# The RMSLE and global contribution of the 10 worst meters are shown here:
# 
# |Building|Meter|RMSLE|Contribution|
# |-------:|----:|----:|-----------:|
# |1072|2|6.685|0.141|
# |1099|2|5.801|0.122|
# |799|0|5.715|0.120|
# |954|1|5.650|0.119|
# |803|0|5.839|0.119|
# |1303|2|5.442|0.114|
# |1021|3|5.199|0.109|
# |954|2|4.967|0.104|
# |802|0|4.862|0.102|
# |954|0|4.783|0.101|
# 

# # Framework
# The framework code is taken from my previous kernel: [Strategy evaluation: What helps and by how much?](https://www.kaggle.com/purist1024/strategy-evaluation-what-helps-and-by-how-much). It is described in more detail there and so, in order to get to the point, we incorporate it here without the descriptions.

# In[ ]:


import pandas as pd
import numpy as np
import os
import warnings

from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_squared_log_error
from IPython.display import HTML

pd.set_option("max_columns", 500)


def input_file(file):
    path = f"../input/ashrae-energy-prediction/{file}"
    if not os.path.exists(path): return path + ".gz"
    return path

def compress_dataframe(df):
    result = df.copy()
    for col in result.columns:
        col_data = result[col]
        dn = col_data.dtype.name
        if dn == "object":
            result[col] = pd.to_numeric(col_data.astype("category").cat.codes, downcast="integer")
        elif dn == "bool":
            result[col] = col_data.astype("int8")
        elif dn.startswith("int") or (col_data.round() == col_data).all():
            result[col] = pd.to_numeric(col_data, downcast="integer")
        else:
            result[col] = pd.to_numeric(col_data, downcast='float')
    return result

def read_train():
    df = pd.read_csv(input_file("train.csv"), parse_dates=["timestamp"])
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    return compress_dataframe(df)

def read_building_metadata():
    return compress_dataframe(pd.read_csv(
        input_file("building_metadata.csv")).fillna(-1)).set_index("building_id")

site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]

def read_weather_train(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    df = pd.read_csv(input_file("weather_train.csv"), parse_dates=["timestamp"])
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    if fix_timestamps:
        GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}
        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)
    if interpolate_na:
        site_dfs = []
        for site_id in df.site_id.unique():
            # Make sure that we include all possible hours so that we can interpolate evenly
            site_df = df[df.site_id == site_id].set_index("timestamp").reindex(range(8784))
            site_df.site_id = site_id
            for col in [c for c in site_df.columns if c != "site_id"]:
                if add_na_indicators: site_df[f"had_{col}"] = ~site_df[col].isna()
                site_df[col] = site_df[col].interpolate(limit_direction='both', method='linear')
                # Some sites are completely missing some columns, so use this fallback
                site_df[col] = site_df[col].fillna(df[col].median())
            site_dfs.append(site_df)
        df = pd.concat(site_dfs).reset_index()  # make timestamp back into a regular column
    elif add_na_indicators:
        for col in df.columns:
            if df[col].isna().any(): df[f"had_{col}"] = ~df[col].isna()
    return compress_dataframe(df).set_index(["site_id", "timestamp"])

def combined_train_data(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    Xy = compress_dataframe(read_train().join(read_building_metadata(), on="building_id").join(
        read_weather_train(fix_timestamps, interpolate_na, add_na_indicators),
        on=["site_id", "timestamp"]).fillna(-1))
    return Xy.drop(columns=["meter_reading"]), Xy.meter_reading

def _drop_electrical_zeros(X, y):
    X = X[(y > 0) | (X.meter != 0)]
    y = y.reindex(X.index)
    return X, y

def _drop_missing_site_0(X, y):
    X = X[(X.timestamp >= 3378) | (X.site_id != 0) | (X.meter != 0)]
    y = y.reindex(X.index)
    return X, y

def _add_time_features(X):
    return X.assign(tm_day_of_week=((X.timestamp // 24) % 7), tm_hour_of_day=(X.timestamp % 24))

def np_sample(a, frac):
    return a if frac == 1 else np.random.choice(a, int(len(a) * frac), replace=False)

def make_8121_splits(X, sample_frac):
    np.random.seed(0)
    time_sorted_idx = np.argsort(X.timestamp.values, kind='stable')
    sections = np.array_split(time_sorted_idx, 12)
    folds = []
    for start_ix in range(0, 12, 2):
        val_idxs = np.concatenate(sections[start_ix:start_ix + 2])  # no modulo necessary
        train_idxs = np.concatenate(
            [sections[ix % 12] for ix in range(start_ix + 3, start_ix + 11)])
        folds.append((np_sample(train_idxs, sample_frac), np_sample(val_idxs, sample_frac)))
    return folds

def make_cv_predictions(model, split, X, y, drop_electrical_zeros, verbose=True):
    preds = []
    for ix, (train_fold, val_fold) in enumerate(split):
        Xt = X.iloc[train_fold]
        yt = y.reindex_like(Xt)
        if drop_electrical_zeros:
            Xt, yt = _drop_electrical_zeros(Xt, yt)
        Xv = X.iloc[val_fold]
        yv = y.reindex_like(Xv)
        if verbose: print(f"Testing split {ix}: {len(Xt)} train rows & {len(Xv)} val rows")
        model.fit(Xt, yt)
        preds.append(pd.DataFrame(dict(target=yv, prediction=model.predict(Xv)), index=yv.index))
    result = pd.concat(preds).sort_index()
    return result.target, result.prediction

categorical_columns = [
    "building_id", "meter", "site_id", "primary_use", "had_air_temperature", "had_cloud_coverage",
    "had_dew_temperature", "had_precip_depth_1_hr", "had_sea_level_pressure", "had_wind_direction",
    "had_wind_speed", "tm_day_of_week", "tm_hour_of_day"
]

class LGBMWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, categorical_feature=None, **params):
        self.model = LGBMRegressor(**params)
        self.categorical_feature = categorical_feature

    def fit(self, X, y):
        with warnings.catch_warnings():
            cats = None if self.categorical_feature is None else list(
                X.columns.intersection(self.categorical_feature))
            warnings.filterwarnings("ignore",
                                    "categorical_feature in Dataset is overridden".lower())
            self.model.fit(X, y, **({} if cats is None else {"categorical_feature": cats}))
            self.feature_importances_ = self.model.feature_importances_
            return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {**self.model.get_params(deep), "categorical_feature": self.categorical_feature}

    def set_params(self, **params):
        ctf = params.pop("categorical_feature", None)
        if ctf is not None: self.categorical_feature = ctf
        self.model.set_params(params)


# # Code for per-building/meter evaluation
# Our `run_experiment` function evaluates model performance for each building & meter via a 6-way time-based cross-validation. It produces a chart giving the RMSLE for each model, and a "contribution" number which is the RMSLE that would be achieved over the whole dataset if we perfectly predicted every other building and meter. This makes it a crude approximation of the maximum improvement we could get if we fixed all of the problems with the given meter.
# 
# `fractions_report` describes the number of meters required to account for specific fractions of the overall error.
# 
# `plot_contributions` shows all of the contribution values in 4 easy charts. Note that matplotlib is perfectly willing to simply drop any lines which it considers "too thin", which is why we had to adjust the figure size to absurd values.

# In[ ]:


def run_experiment(n_estimators, sample_frac=1):
    X, y = combined_train_data()

    # Reduce evaluation cost by subsampling the data
    X = X.sample(frac=sample_frac).sort_index()
    y = y.reindex(X.index)

    # Additional preprocessing
    X, y = _drop_missing_site_0(X, y)
    X = compress_dataframe(_add_time_features(X))
    y = np.log1p(y)

    model = LGBMWrapper(random_state=0, n_jobs=-1, n_estimators=n_estimators,
                        categorical_feature=categorical_columns)
    contribution_chart = pd.DataFrame(columns=["building_id", "meter", "RMSLE", "contribution"])

    for building in sorted(X.building_id.unique()):
        for meter in range(4):
            X_subset = X[(X.building_id == building) & (X.meter == meter)]
            y_subset = y.reindex(X_subset.index)
            if len(y_subset) == 0: continue
            splits = make_8121_splits(X_subset, 1)  # We already subsampled, so no need to resample
            X_subset = X_subset.drop(columns="timestamp")

            cv_y, cv_prediction = make_cv_predictions(model, splits, X_subset, y_subset,
                                                      drop_electrical_zeros=False, verbose=False)
            sle = np.square(cv_y - cv_prediction).sum()
            rmsle = np.sqrt(sle / len(cv_y))
            contribution = np.sqrt(sle / len(y))
            contribution_chart.loc[len(contribution_chart)] = (building, meter, rmsle, contribution)
    return contribution_chart


# In[ ]:


def fractions_report(chart):
    contribution = chart.contribution
    cum_sum = (contribution.sort_values(ascending=False) / contribution.sum()).cumsum()
    for frac in [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 1]:
        count = cum_sum[cum_sum < frac].count()+1  # Add one to account for boundary issues
        frac_frac = count / len(cum_sum)
        print(f"{count: 5} meters ({frac_frac:6.1%} of total) account for {frac:4.0%} of error.")


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

def plot_contributions(chart):
    for meter in range(4):
        subchart = chart[chart.meter == meter].sort_values("building_id")
        ax = plt.figure(0, (24, 8)).add_subplot(111)
        ax.set(xlim=(0, chart.building_id.max() + 1), ylim=(0, chart.contribution.max() * 1.05))
        ax.bar(subchart.building_id, subchart.contribution, width=1, label=f"meter {meter}")
        ax.legend()
        plt.show()


# # Evaluation
# We evaluate the error contribution of each building/meter pair, and chart the results. The resulting "contribution chart" is also saved to a file so that you can easily view it in its entirety. We run a "light" (10 estimators) version of LGBM, both for evaluation speed, and because our experiments have shown that more estimators at the per-building level just cause overfitting.
# 
# Note that the total contributions sum to a much higher total than the actual observed error. This is presumably because the global model *does* benefit from cross-building correlations. This doesn't negate the fact that some meter readings contain a much lower signal-to-noise ratio, and that we could benefit by cleaning them up or even eliminating them.

# In[ ]:


contribution_chart = run_experiment(10, 1)
contribution_chart.sort_values("contribution", ascending=False).to_csv("contribution_chart.csv", index=False)


# ## Contributions by number of meters

# In[ ]:


fractions_report(contribution_chart)


# ## Contributions by meter type

# In[ ]:


display(contribution_chart.groupby("meter").contribution.sum().to_frame().T)


# ## Top contributions by building/meter

# In[ ]:


display(contribution_chart.sort_values("contribution", ascending=False).reset_index(drop=True).head(25))


# ## All contributions by meter

# In[ ]:


plot_contributions(contribution_chart)

