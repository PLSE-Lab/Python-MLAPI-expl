#!/usr/bin/env python
# coding: utf-8

# ## Modelling COVID-19 with Mobility Features
# This notebook aims to provide a basic workflow of how mobility data shared by [Google](https://www.google.com/covid19/mobility/) and [Apple](https://www.apple.com/covid19/mobility) can potentially be used as features into COVID-19 models.
# 
# I will be maintaining a structured version of the dataset here: https://www.kaggle.com/rohanrao/covid19-mobility-data and also likely will use it to some extent for my final submission of this competition.
# 
# The notebook demonstrates how the dataset can be merged and used with the competition data. I've also shared the validation and LB scores with and without using the features.
# 
# **P.S.** If you plan to use the data / notebook, be careful to use the features with appropriate lagged values since the actual test data duration is 28 days.
# 

# In[ ]:


## importing packages
import lightgbm as lgb
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.preprocessing import LabelEncoder


# In[ ]:


## defining constants
PATH_TRAIN = "/kaggle/input/covid19-global-forecasting-week-5/train.csv"
PATH_TEST = "/kaggle/input/covid19-global-forecasting-week-5/test.csv"
PATH_GOOGLE_MOBILITY = "/kaggle/input/covid19-mobility-data/Google_Mobility_Data.csv"

PATH_SUBMISSION = "submission.csv"


# ## Reading and Preparing Data
# The datasets will require some basic cleaning to match the countries, states/cities and counties. I have shown how it can be used for US. Adding more preprocessing for other countries can improve quality of features.

# In[ ]:


## reading data
df_train = pd.read_csv(PATH_TRAIN)
df_test = pd.read_csv(PATH_TEST)

df_google = pd.read_csv(PATH_GOOGLE_MOBILITY, low_memory = False)


# In[ ]:


## preparing data
df_train = df_train[df_train.Date < "2020-04-27"]

df_google["Date"] = pd.to_datetime(df_google.date)

df_google.loc[df_google.country_region == "United States", "country_region"] = "US"
df_google.sub_region_2 = df_google.sub_region_2.str.replace(" County", "")

df_google["geography"] = df_google.country_region + "_" + df_google.sub_region_1 + "_" + df_google.sub_region_2
df_google.loc[df_google.sub_region_2.isna(), "geography"] = df_google[df_google.sub_region_2.isna()].country_region + "_" + df_google[df_google.sub_region_2.isna()].sub_region_1
df_google.loc[df_google.sub_region_1.isna(), "geography"] = df_google[df_google.sub_region_1.isna()].country_region

df_google["Google_Recreation_Index"] = df_google.retail_and_recreation_percent_change_from_baseline
df_google["Google_Grocery_Index"] = df_google.grocery_and_pharmacy_percent_change_from_baseline
df_google["Google_Parks_Index"] = df_google.parks_percent_change_from_baseline
df_google["Google_Transit_Index"] = df_google.transit_stations_percent_change_from_baseline
df_google["Google_Workplaces_Index"] = df_google.workplaces_percent_change_from_baseline
df_google["Google_Residential_Index"] = df_google.residential_percent_change_from_baseline

df_google = df_google[[
    "geography",
    "Date",
    "Google_Recreation_Index",
    "Google_Grocery_Index",
    "Google_Parks_Index",
    "Google_Transit_Index",
    "Google_Workplaces_Index",
    "Google_Residential_Index"
]].drop_duplicates(subset = ["geography", "Date"])


# ## Feature Engineering
# Without going into too much depth, I've created some basic features on the raw data and some lag features with and without mobility. A lot more features can be added to the pipeline to improve the performance.

# In[ ]:


## basic features
df = pd.concat([df_train, df_test])

df.Date = pd.to_datetime(df.Date)

df["geography"] = df.Country_Region + "_" + df.Province_State + "_" + df.County
df.loc[df.County.isna(), "geography"] = df[df.County.isna()].Country_Region + "_" + df[df.County.isna()].Province_State
df.loc[df.Province_State.isna(), "geography"] = df[df.Province_State.isna()].Country_Region

le = LabelEncoder()
df.Country_Region = le.fit_transform(df.Country_Region.astype(str))
df.Province_State = le.fit_transform(df.Province_State.astype(str))
df.County = le.fit_transform(df.County.astype(str))
df.Target = le.fit_transform(df.Target.astype(str))

df = df.merge(df_google, on = ["geography", "Date"], how = "left")


# In[ ]:


## lag features
df.sort_values(["geography", "Date", "Target"], inplace = True)

for lag in range(1, 10):
    df[f"lag_target_{lag}"] = df.groupby(["geography", "Target"])["TargetValue"].shift(lag)
    df[f"lag_recreation_index_{lag}"] = df.groupby(["geography"])["Google_Recreation_Index"].shift(2 * lag)
    df[f"lag_grocery_index_{lag}"] = df.groupby(["geography"])["Google_Grocery_Index"].shift(2 * lag)
    df[f"lag_parks_index_{lag}"] = df.groupby(["geography"])["Google_Parks_Index"].shift(2 * lag)
    df[f"lag_transit_index_{lag}"] = df.groupby(["geography"])["Google_Transit_Index"].shift(2 * lag)
    df[f"lag_workplaces_index_{lag}"] = df.groupby(["geography"])["Google_Workplaces_Index"].shift(2 * lag)
    df[f"lag_residential_index_{lag}"] = df.groupby(["geography"])["Google_Residential_Index"].shift(2 * lag)


# In[ ]:


df["lag_target_1_3"] = df[["lag_target_1", "lag_target_2", "lag_target_3"]].mean(axis = 1)
df["lag_target_1_5"] = df[["lag_target_1", "lag_target_2", "lag_target_3", "lag_target_4", "lag_target_5"]].mean(axis = 1)
df["lag_target_1_9"] = df[["lag_target_1", "lag_target_2", "lag_target_3", "lag_target_4", "lag_target_5",
                           "lag_target_6", "lag_target_7", "lag_target_8", "lag_target_9"]].mean(axis = 1)

df["lag_recreation_index_1_3"] = df[["lag_recreation_index_1", "lag_recreation_index_2", "lag_recreation_index_3"]].mean(axis = 1)
df["lag_grocery_index_1_3"] = df[["lag_grocery_index_1", "lag_grocery_index_2", "lag_grocery_index_3"]].mean(axis = 1)
df["lag_parks_index_1_3"] = df[["lag_parks_index_1", "lag_parks_index_2", "lag_parks_index_3"]].mean(axis = 1)
df["lag_transit_index_1_3"] = df[["lag_transit_index_1", "lag_transit_index_2", "lag_transit_index_3"]].mean(axis = 1)
df["lag_workplaces_index_1_3"] = df[["lag_workplaces_index_1", "lag_workplaces_index_2", "lag_workplaces_index_3"]].mean(axis = 1)
df["lag_residential_index_1_3"] = df[["lag_residential_index_1", "lag_residential_index_2", "lag_residential_index_3"]].mean(axis = 1)

df["lag_recreation_index_1_5"] = df[["lag_recreation_index_1", "lag_recreation_index_2", "lag_recreation_index_3",
                                     "lag_recreation_index_4", "lag_recreation_index_5"]].mean(axis = 1)
df["lag_grocery_index_1_5"] = df[["lag_grocery_index_1", "lag_grocery_index_2", "lag_grocery_index_3",
                                  "lag_grocery_index_4", "lag_grocery_index_5"]].mean(axis = 1)
df["lag_parks_index_1_5"] = df[["lag_parks_index_1", "lag_parks_index_2", "lag_parks_index_3",
                                "lag_parks_index_4", "lag_parks_index_5"]].mean(axis = 1)
df["lag_transit_index_1_5"] = df[["lag_transit_index_1", "lag_transit_index_2", "lag_transit_index_3",
                                  "lag_transit_index_4", "lag_transit_index_5"]].mean(axis = 1)
df["lag_workplaces_index_1_5"] = df[["lag_workplaces_index_1", "lag_workplaces_index_2", "lag_workplaces_index_3",
                                     "lag_workplaces_index_4", "lag_workplaces_index_5"]].mean(axis = 1)
df["lag_residential_index_1_5"] = df[["lag_residential_index_1", "lag_residential_index_2", "lag_residential_index_3",
                                      "lag_residential_index_4", "lag_residential_index_5"]].mean(axis = 1)

df["lag_recreation_index_1_9"] = df[["lag_recreation_index_1", "lag_recreation_index_2", "lag_recreation_index_3",
                                     "lag_recreation_index_4", "lag_recreation_index_5", "lag_recreation_index_6",
                                     "lag_recreation_index_7", "lag_recreation_index_8", "lag_recreation_index_9"]].mean(axis = 1)
df["lag_grocery_index_1_9"] = df[["lag_grocery_index_1", "lag_grocery_index_2", "lag_grocery_index_3",
                                  "lag_grocery_index_4", "lag_grocery_index_5", "lag_grocery_index_6",
                                  "lag_grocery_index_7", "lag_grocery_index_8", "lag_grocery_index_9"]].mean(axis = 1)
df["lag_parks_index_1_9"] = df[["lag_parks_index_1", "lag_parks_index_2", "lag_parks_index_3",
                                "lag_parks_index_4", "lag_parks_index_5", "lag_parks_index_6",
                                "lag_parks_index_7", "lag_parks_index_8", "lag_parks_index_9"]].mean(axis = 1)
df["lag_transit_index_1_9"] = df[["lag_transit_index_1", "lag_transit_index_2", "lag_transit_index_3",
                                  "lag_transit_index_4", "lag_transit_index_5", "lag_transit_index_6",
                                  "lag_transit_index_7", "lag_transit_index_8", "lag_transit_index_9"]].mean(axis = 1)
df["lag_workplaces_index_1_9"] = df[["lag_workplaces_index_1", "lag_workplaces_index_2", "lag_workplaces_index_3",
                                     "lag_workplaces_index_4", "lag_workplaces_index_5", "lag_workplaces_index_6",
                                     "lag_workplaces_index_7", "lag_workplaces_index_8", "lag_workplaces_index_9"]].mean(axis = 1)
df["lag_residential_index_1_9"] = df[["lag_residential_index_1", "lag_residential_index_2", "lag_residential_index_3",
                                      "lag_residential_index_4", "lag_residential_index_5", "lag_residential_index_6",
                                      "lag_residential_index_7", "lag_residential_index_8", "lag_residential_index_9"]].mean(axis = 1)


# ## Modelling without Mobility
# Modelling without using any mobility features and data.

# In[ ]:


## modelling without mobility features
features = [
    "Country_Region",
    "Province_State",
    "County",
    "Population",
    "Target",
    "lag_target_1",
    "lag_target_2",
    "lag_target_3",
    "lag_target_4",
    "lag_target_5",
    "lag_target_6",
    "lag_target_7",
    "lag_target_8",
    "lag_target_9",
    "lag_target_1_3",
    "lag_target_1_5",
    "lag_target_1_9"
]

categorical_features = [
    "Country_Region",
    "Province_State",
    "County",
    "Target"
]

df_train = df[~df.TargetValue.isna()]
df_build = df_train[df_train.Date < datetime(2020, 4, 20)]
df_val = df_train[df_train.Date >= datetime(2020, 4, 20)]
df_test = df[df.TargetValue.isna()]

y_train = df_train.TargetValue.values
y_build = df_build.TargetValue.values
y_val = df_val.TargetValue.values

test_ids = df_test.ForecastId.astype(int).astype(str).values

df_train = df_train[features]
df_build = df_build[features]
df_val = df_val[features]
df_test = df_test[features]

print("Build shape: ", df_build.shape)
print("Val shape: ", df_val.shape)
print("Train shape: ", df_train.shape)
print("Test shape: ", df_test.shape)

dtrain = lgb.Dataset(df_train, label = y_train, categorical_feature = categorical_features)
dbuild = lgb.Dataset(df_build, label = y_build, categorical_feature = categorical_features)
dval = lgb.Dataset(df_val, label = y_val, categorical_feature = categorical_features)

params = {
    "objective": "regression_l1",
    "num_leaves": 7,
    "learning_rate": 0.013,
    "bagging_fraction": 0.91,
    "feature_fraction": 0.81,
    "reg_alpha": 0.13,
    "reg_lambda": 0.13,
    "metric": "mae",
    "seed": 838861
}

model_lgb_val = lgb.train(params, train_set = dbuild, valid_sets = [dval], num_boost_round = 2000, early_stopping_rounds = 100, verbose_eval = 100)
model_lgb_without_mobility = lgb.train(params, train_set = dtrain, num_boost_round = model_lgb_val.best_iteration)
    
y_pred_without_mobility = model_lgb_without_mobility.predict(df_test)


# ## Modelling with Mobility
# Modelling with mobility features and data.

# In[ ]:


## modelling with mobility features
features_with_mobility = features + [
    "lag_recreation_index_1",
    "lag_recreation_index_2",
    "lag_recreation_index_3",
    "lag_recreation_index_4",
    "lag_recreation_index_5",
    "lag_recreation_index_6",
    "lag_recreation_index_7",
    "lag_recreation_index_8",
    "lag_recreation_index_9",
    "lag_grocery_index_1",
    "lag_grocery_index_2",
    "lag_grocery_index_3",
    "lag_grocery_index_4",
    "lag_grocery_index_5",
    "lag_grocery_index_6",
    "lag_grocery_index_7",
    "lag_grocery_index_8",
    "lag_grocery_index_9",
    "lag_parks_index_1",
    "lag_parks_index_2",
    "lag_parks_index_3",
    "lag_parks_index_4",
    "lag_parks_index_5",
    "lag_parks_index_6",
    "lag_parks_index_7",
    "lag_parks_index_8",
    "lag_parks_index_9",
    "lag_transit_index_1",
    "lag_transit_index_2",
    "lag_transit_index_3",
    "lag_transit_index_4",
    "lag_transit_index_5",
    "lag_transit_index_6",
    "lag_transit_index_7",
    "lag_transit_index_8",
    "lag_transit_index_9",
    "lag_workplaces_index_1",
    "lag_workplaces_index_2",
    "lag_workplaces_index_3",
    "lag_workplaces_index_4",
    "lag_workplaces_index_5",
    "lag_workplaces_index_6",
    "lag_workplaces_index_7",
    "lag_workplaces_index_8",
    "lag_workplaces_index_9",
    "lag_residential_index_1",
    "lag_residential_index_2",
    "lag_residential_index_3",
    "lag_residential_index_4",
    "lag_residential_index_5",
    "lag_residential_index_6",
    "lag_residential_index_7",
    "lag_residential_index_8",
    "lag_residential_index_9",
    "lag_recreation_index_1_3",
    "lag_grocery_index_1_3",
    "lag_parks_index_1_3",
    "lag_transit_index_1_3",
    "lag_workplaces_index_1_3",
    "lag_residential_index_1_3",
    "lag_recreation_index_1_5",
    "lag_grocery_index_1_5",
    "lag_parks_index_1_5",
    "lag_transit_index_1_5",
    "lag_workplaces_index_1_5",
    "lag_residential_index_1_5",
    "lag_recreation_index_1_9",
    "lag_grocery_index_1_9",
    "lag_parks_index_1_9",
    "lag_transit_index_1_9",
    "lag_workplaces_index_1_9",
    "lag_residential_index_1_9"
]

df_train = df[~df.TargetValue.isna()]
df_build = df_train[df_train.Date < datetime(2020, 4, 20)]
df_val = df_train[df_train.Date >= datetime(2020, 4, 20)]
df_test = df[df.TargetValue.isna()]

y_train = df_train.TargetValue.values
y_build = df_build.TargetValue.values
y_val = df_val.TargetValue.values

test_ids = df_test.ForecastId.astype(int).astype(str).values

df_train = df_train[features_with_mobility]
df_build = df_build[features_with_mobility]
df_val = df_val[features_with_mobility]
df_test = df_test[features_with_mobility]

print("Build shape: ", df_build.shape)
print("Val shape: ", df_val.shape)
print("Train shape: ", df_train.shape)
print("Test shape: ", df_test.shape)

dtrain = lgb.Dataset(df_train, label = y_train, categorical_feature = categorical_features)
dbuild = lgb.Dataset(df_build, label = y_build, categorical_feature = categorical_features)
dval = lgb.Dataset(df_val, label = y_val, categorical_feature = categorical_features)

params = {
    "objective": "regression_l1",
    "num_leaves": 9,
    "learning_rate": 0.013,
    "bagging_fraction": 0.81,
    "feature_fraction": 0.81,
    "reg_alpha": 0.13,
    "reg_lambda": 0.13,
    "metric": "mae",
    "seed": 838861
}

model_lgb_val = lgb.train(params, train_set = dbuild, valid_sets = [dval], num_boost_round = 2000, early_stopping_rounds = 100, verbose_eval = 100)
model_lgb_with_mobility = lgb.train(params, train_set = dtrain, num_boost_round = model_lgb_val.best_iteration)
    
y_pred_with_mobility = model_lgb_with_mobility.predict(df_test)


# ## Feature Importance
# Comparing the feature importance of the two models.

# In[ ]:


from bokeh.io import output_notebook, show
from bokeh.layouts import column
from bokeh.plotting import figure

output_notebook()

df_model_without_mobility = pd.DataFrame({"feature": features, "importance": model_lgb_without_mobility.feature_importance()})
df_model_with_mobility = pd.DataFrame({"feature": features_with_mobility, "importance": model_lgb_with_mobility.feature_importance()})

df_model_without_mobility.sort_values("importance", ascending = False, inplace = True)
df_model_with_mobility.sort_values("importance", ascending = False, inplace = True)

v1 = figure(plot_width = 800, plot_height = 400, x_range = df_model_without_mobility.feature, title = "Feature Importance of LGB Model without Mobility features")
v1.vbar(x = df_model_without_mobility.feature, top = df_model_without_mobility.importance, width = 0.81)
v1.xaxis.major_label_orientation = 1.3

v2 = figure(plot_width = 800, plot_height = 400, x_range = df_model_with_mobility.feature, title = "Feature Importance of LGB Model with Mobility features")
v2.vbar(x = df_model_with_mobility.feature, top = df_model_with_mobility.importance, width = 0.81)
v2.xaxis.major_label_orientation = 1.3

show(column(v1, v2))


# ## Submission
# Choosing the appropriate final submission. Instead of using a static multiplier, the different quantiles can be optimized to improve score.

# In[ ]:


## submission
y_pred = y_pred_with_mobility

df_pred_q05 = pd.DataFrame({"ForecastId_Quantile": test_ids + "_0.05", "TargetValue": 0.85 * y_pred})
df_pred_q50 = pd.DataFrame({"ForecastId_Quantile": test_ids + "_0.5", "TargetValue": y_pred})
df_pred_q95 = pd.DataFrame({"ForecastId_Quantile": test_ids + "_0.95", "TargetValue": 1.15 * y_pred})

df_submit = pd.concat([df_pred_q05, df_pred_q50, df_pred_q95])


# In[ ]:


print(df_submit.shape)
df_submit.to_csv(PATH_SUBMISSION, index = False)

