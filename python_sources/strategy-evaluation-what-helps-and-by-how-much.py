#!/usr/bin/env python
# coding: utf-8

# # Strategy Evaluation
# This competition provides an interesting grab-bag of sub-problems to solve: the data is messy and needs cleaned (site 0); the data sources are inconsistent (meter timestamps vs. weather timestamps); it asks you to predict several different vaguely related sets of values (meter type); it has hugely important categorical features (building_id); and we can find lots of external data to supplement the provided data. This makes it hard to know where to start and what areas are most important to tackle.
# 
# Like everyone, I've tried a bunch of strategies and techniques designed to tackle all of the above issues as well the usual machine-learning pitfalls. However, as I built up layers of approaches, it became hard to keep track of which ones were actually useful and which ones simply didn't hurt. Since I'm an experimentalist at heart, I decided to build a framework which let me evaluate the actual effect of a bunch of different strategies. Since this is Kaggle, I decided to share the insights with the rest of the community.

# ## Summary
# All of the code and the raw results are included below, and are well worth reading for additional insights and coding tricks. However, they can be summarized here. The percentage "effect" for each strategy is how much *worse* our results are when we don't use it.
# 
# |Strategy|Effect|Details|
# |:-------|-----:|:------|
# |**Log-scaled target**|**45.7%**|Use log1p to rescale the meter readings, and expm1 to rescale the predictions back to kWh.|
# |**Site 0 data cleanup**|**17.8%**|Simply drop site 0 electrical data from the first 141 days.
# |**LGBM category-awareness**|**9.7%**|Declare which columns are categorical data.
# |**Per-meter models**|**3.9%**|Use separate models to predict for each meter type and then re-combine the predictions.
# |**Time-based features**|**3.6%**|Perform basic feature engineering, with hour-of-day and day-of-week.
# |**Timestamp alignment**|**0.4%**|Convert the GMT-based weather data to align with local-time-based readings.
# |**Interpolated weather data**|**0.4%**|Impute missing weather values by interpolating from time-adjacent readings.
# |**Missing value indicators**|**0.1%**|Add flags to inform the model of which values were imputed.
# |**0-valued meter cleanup**|**0.0%**|Drop all 0-valued electrical readings, assuming that there's always some electrical use, and that zeros indicate anomalies.

# # Framework

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


# ## Utilities
# `input_file` is a simple wrapper that hides the input directory location and automatically handles compressed files.
# 
# `compress_dataframe` uses Pandas' built-in capabilities to downcast data into its smallest practical form.

# In[ ]:


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


# ## Data loading and preprocessing
# Code to read and combine the standard input files, converting timestamps to number of hours since the beginning of 2016.
# 
# Optionally, we perform cleanups on the weather data: adjusting timestamps to the appropriate time zone; using interpolation to fill missing values; and adding NaN presence indicators.

# In[ ]:


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


# ## Drop potentially bad data
# There are two different optional strategies here that can be used together. `_drop_electrical_zeros` removes every electrical meter reading with a value of 0; while `_drop_missing_site_0` gets rid of all electrical meter readings for site zero during the 141 days that its readings were mostly pegged to 0.

# In[ ]:


def _drop_electrical_zeros(X, y):
    X = X[(y > 0) | (X.meter != 0)]
    y = y.reindex(X.index)
    return X, y

def _drop_missing_site_0(X, y):
    X = X[(X.timestamp >= 3378) | (X.site_id != 0) | (X.meter != 0)]
    y = y.reindex(X.index)
    return X, y


# ## Add basic time features
# We add columns for "hour of day" and "day of week". Note that the source "timestamp" columns will be removed before training, so these are the only time-oriented features that will remain.

# In[ ]:


def _add_time_features(X):
    return X.assign(tm_day_of_week=((X.timestamp // 24) % 7), tm_hour_of_day=(X.timestamp % 24))


# ## Cross-validation
# We use a time-oriented split which deals in 12 equal time slices (which are almost, but not quite, months). Each fold includes a two adjacent slices for validation, 8 adjacent slices for training, and a 1 slice buffer on each side of the validation set. We use modulo arithmetic to wrap-around the year-end boundary. We thus have slices 4-11 vs 1-2; 6-12&1 vs 3-4; 8-12&1-3 vs 5-6; etc.
# 
# While the CV score is always better than the LB score, relative differences between strategies tend to align well between CV and LB.
# 
# Because the data set is large, we use the `sample frac` parameter to train and evaluate on a specified fraction of the dataset. Again, this seems to track the full set fairly well with values of 0.05 and higher.

# In[ ]:


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


# ## Per-meter models
# The `CatSplitRegressor` wrapper class hides the process of splitting the training and validation data up by unique values of a specified column (i.e. "meter"), using a different sub-model for each value, and then re-integrating the resulting predictions. 

# In[ ]:


class CatSplitRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model, col):
        self.model = model
        self.col = col

    def fit(self, X, y):
        self.fitted = {}
        importances = []
        for val in X[self.col].unique():
            X1 = X[X[self.col] == val].drop(columns=[self.col])
            self.fitted[val] = clone(self.model).fit(X1, y.reindex_like(X1))
            importances.append(self.fitted[val].feature_importances_)
            del X1
        fi = np.average(importances, axis=0)
        col_index = list(X.columns).index(self.col)
        self.feature_importances_ = [*fi[:col_index], 0, *fi[col_index:]]
        return self

    def predict(self, X):
        result = np.zeros(len(X))
        for val in X[self.col].unique():
            ix = np.nonzero((X[self.col] == val).to_numpy())
            predictions = self.fitted[val].predict(X.iloc[ix].drop(columns=[self.col]))
            result[ix] = predictions
        return result


# ## LGBM category awareness
# The LGBM uses some very effective strategies to produce better splits when it knows that a feature is categorical. Since we have a reasonable number of integer-valued features that are, in fact, unordered or cyclical categoricals, we benefit from explicitly declaring them to LGBM.
# 
# Unfortunately, the scikit wrapper from LGBM doesn't handle these declarations gracefully, and prints a warning that we've overridden its default guesses. We include extra code here to silence those warnings.

# In[ ]:


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


# ## Catch-all experimental framework
# This loads the data, runs cross-validation, reports the CV error, and displays the most important features as declared by the model.
# 
# The extra complexity comes from allowing us to switch various strategies on and off:
# * fix_timestamps: adjust for time-zone errors in weather data
# * interpolate_na: use simple linear interpolation to provide fully-populated weather data
# * add_na_indicators: add boolean columns which let the model know that values were imputed
# * drop_electrical_zeros: drops all electrical meter readings which are zero, *but only from the training set*. We still evaluate our performance on such readings.
# * drop_missing_site_0: drops all electical meter readings for the first 141 days for site 0. These are known to be bogus, so we neither train *nor* evaluate on them
# * log_target: convert the target variable to a logarithmic scale, and convert the predictions back.
# * LGBM_cataware: declare to LGBM the columns which are known to be categorical
# * add_time_features: add basic day of week and hour of day features.
# * per_meter_models: train separate models for each distinct meter type
# 
# By switching *off* one of these strategies at a time, we can get an indication of how much worse we are without them, and thus how much they helped out the baseline "all strategies on" evaluation.

# In[ ]:


def run_experiment(fix_timestamps=True, interpolate_na=True, add_na_indicators=True,
                   drop_electrical_zeros=True, drop_missing_site_0=True, log_target=True,
                   LGBM_cataware=True, add_time_features=True, per_meter_models=True,
                   sample_frac=0.2, verbose=False, baseline='return'):
    if verbose: print(f"Loading data")
    X, y = combined_train_data(fix_timestamps, interpolate_na, add_na_indicators)
    if verbose: X.info(memory_usage="deep", verbose=False)

    if add_time_features:
        if verbose: print(f"Adding time features")
        X = compress_dataframe(_add_time_features(X))
        if verbose: X.info(memory_usage="deep", verbose=False)

    if drop_missing_site_0:
        if verbose: print(f"Dropping missing site 0 data")
        # We do this here rather than without our CV, because we are *certain* that there is no
        # equivalent data in the test set.
        X, y = _drop_missing_site_0(X, y)
        if verbose: X.info(memory_usage="deep", verbose=False)

    model = LGBMWrapper(random_state=0, n_jobs=-1,
                        categorical_feature=None if not LGBM_cataware else categorical_columns)
    if per_meter_models:
        model = CatSplitRegressor(model, "meter")

    if verbose: print(f"Creating splits")
    splits = make_8121_splits(X, sample_frac)

    X = X.drop(columns="timestamp")  # Raw timestamp doesn't carry over to test data

    if log_target:
        y = np.log1p(y)
    if verbose: print("Making predictions")
    sampled_y, sampled_prediction = make_cv_predictions(model, splits, X, y, drop_electrical_zeros,
                                                        verbose=verbose)
    if log_target:
        sampled_y = np.expm1(sampled_y)
        sampled_prediction = np.expm1(sampled_prediction)
    rmsle = np.sqrt(mean_squared_log_error(sampled_y, np.clip(sampled_prediction, 0, None)))
    if baseline == 'return':
        print(f"RMSLE = {rmsle:.4f} (baseline)")
    else:
        baseline_change_pct = ((rmsle/baseline) - 1) * 100
        print(f"RMSLE = {rmsle:.4f} ({baseline_change_pct:.1f}% over baseline)")

    if verbose: print("Finding important features")
    X_feature_importance = X.sample(frac=sample_frac, random_state=0)
    y_feature_importance = y.reindex(X_feature_importance.index)
    model.fit(X_feature_importance, y_feature_importance)
    importances = pd.Series(model.feature_importances_, index=X.columns).rename("Importance")
    display(importances.sort_values(ascending=False).to_frame().T)

    if baseline == 'return':
        return rmsle


# # The experiments

# ## All strategies enabled
# This is the baseline against which we evaluate each individual strategy. All subsequent experiments will *disable* a single strategy and report how much *worse* we perform compared to this evaluation.

# In[ ]:


baseline_score = run_experiment()


# ## No log-scaled target
# Given that we are evaluated on log-error, it's not surprising that we do better when operating in log-space. The naive linear-space approach is our biggest loser at 45% worse.

# In[ ]:


run_experiment(baseline=baseline_score, log_target=False)


# ## Keep bad site 0 readings
# Those bad readings for a single site really do drag everything else down -- by a full 17%. Without pre-emptively throwing them away, we'll never achieve greatness.

# In[ ]:


run_experiment(baseline=baseline_score, drop_missing_site_0=False)


# ## No LGBM category awareness
# The category-aware capabilities of LGBM are unreasonably effective, netting us a 9% improvement. This is a lot of extra power for almost no cost.

# In[ ]:


run_experiment(baseline=baseline_score, LGBM_cataware=False)


# ## No per-meter models
# The different types of meters are pretty much completely unrelated to each other, so it makes sense to split them up and better take advantage of true commonalities between similar meters. At 4% improvement, it's not mind-bogglingly impressive, but it's still a no-brainer.

# In[ ]:


run_experiment(baseline=baseline_score, per_meter_models=False)


# ## No time features
# It's hard to imagine *not* using these basic time features -- hour of day and day of week. We all know that power usage will be lower during non-peak hours. However, the actual effect is fairly small. Oh well, c'est la vie.

# In[ ]:


run_experiment(baseline=baseline_score, add_time_features=False)


# ## No adjusted weather timestamps
# This is a surprising result. We know that the timestamps for the weather data aren't aligned with the timestamps for the meter readings, and can be off by up to 8 hours. However, it seems that we lose less than half a percent in performance if we simply ignore this appalling condition.

# In[ ]:


run_experiment(baseline=baseline_score, fix_timestamps=False)


# ## No interpolated weather
# There's a lot of missing weather data, but it seems that it really doesn't much matter whether we fix it. This seems odd.

# In[ ]:


run_experiment(baseline=baseline_score, interpolate_na=False)


# ## No NA indicators
# Given the interpolation that we did, there isn't likely to be much difference between an imputed value and the true value. Thus, flagging the imputed values doesn't really give the model much useful information. We can probably just drop the indicators to save a bit of memory.

# In[ ]:


run_experiment(baseline=baseline_score, add_na_indicators=False)


# ## Keep electrical meter zero readings
# This is a misleading result. In fact, the point of dropping all readings of 0 for electrical meters is to provide an alternate method of dealing with the bad zeros in the first part of the site 0 data. However, unlike with the "drop bad site 0 readings" strategy, we dropped these values from training but not evaluation. The fact that we do better by dropping them than by keeping them indicates that zero-readings for electricity really are misleading outliers and that dropping all of them is a valid alternative to specially handling the ones from site 0.

# In[ ]:


run_experiment(baseline=baseline_score, drop_electrical_zeros=False)


# ## Disable all strategies
# This is another baseline, showing what would happen if we just used the raw data and didn't try to improve anything. Not surprisingly, it does *really* badly.

# In[ ]:


run_experiment(baseline=baseline_score, fix_timestamps=False, interpolate_na=False,
               add_na_indicators=False, drop_electrical_zeros=False, drop_missing_site_0=False,
               log_target=False, LGBM_cataware=False, add_time_features=False,
               per_meter_models=False)

