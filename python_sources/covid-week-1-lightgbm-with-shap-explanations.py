#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Data Gathering

# In[ ]:


pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)


# In[ ]:


country_info = pd.read_csv("/kaggle/input/countryinfo/covid19countryinfo.csv")
country_info = country_info.drop([col for col in country_info.columns if "Unnamed" in col], axis=1)
country_info["pop"] = country_info["pop"].str.replace(',', '').astype(float)

pollution = pd.read_csv("/kaggle/input/pollution-by-country-for-covid19-analysis/region_pollution.csv")
pollution = pollution.rename({"Region": "country",
                             "Outdoor Pollution (deaths per 100000)": "outdoor_pol",
                             "Indoor Pollution (deaths per 100000)": "indoor_pol"}, axis=1)

economy = pd.read_csv("/kaggle/input/the-economic-freedom-index/economic_freedom_index2019_data.csv", engine='python')
economy_cols = [col for col in economy.columns if economy[col].dtype == "float64"] + ["Country"]
economy = economy[economy_cols]
economy = economy.rename({"Country": "country"}, axis=1)

def append_external_data(df):
    df = pd.merge(df, country_info, on="country", how="left")
    df = pd.merge(df, pollution, on="country", how="left")
    df = pd.merge(df, economy, on="country", how="left")
    return df


# In[ ]:


country_info.head()


# In[ ]:


def aggregate_label(df):
    country_df = df[["country", "Date", "ConfirmedCases", "Fatalities"]].groupby(["country", "Date"], as_index=False).sum()
    country_df = country_df.rename({"ConfirmedCases": "country_cases", "Fatalities": "country_fatalities"}, axis=1)
    df = pd.merge(df, country_df, on=["country", "Date"], how="left")
    return df


# # Feature Engineering

# In[ ]:


def calculate_days_since_event(df, feature_name, casualties, casualties_amount, groupby=["country"]):
    cases_df = df.loc[df[casualties] > casualties_amount][groupby + ["Date"]].groupby(groupby, as_index=False).min()
    cases_df = cases_df.rename({"Date": "relevant_date"}, axis=1)
    df = pd.merge(df, cases_df, on=groupby, how="left")
    df[feature_name] = (pd.to_datetime(df["Date"]) - pd.to_datetime(df["relevant_date"])).dt.days
    df.loc[df[feature_name] < 0, feature_name] = 0
    df = df.drop("relevant_date", axis = 1)
    return df


# In[ ]:


def generate_time_features(df):
    df = calculate_days_since_event(df, "days_from_first_death", "Fatalities", 0, ["country"])
    df = calculate_days_since_event(df, "days_from_first_case", "ConfirmedCases", 0, ["country"])
    df = calculate_days_since_event(df, "days_from_first_case_province", "ConfirmedCases", 0, ["country", "state"])
    df = calculate_days_since_event(df, "days_from_first_death_province", "Fatalities", 0, ["country", "state"])
    df = calculate_days_since_event(df, "days_from_centenary_case", "ConfirmedCases", 99, ["country"])
    df = calculate_days_since_event(df, "days_from_centenary_case_province", "Fatalities", 99, ["country", "state"])
    df = calculate_days_since_event(df, "days_from_centenary_daily_cases_province", "ConfirmedCases_daily", 99, ["country", "state"])
    df = calculate_days_since_event(df, "days_from_centenary_daily_cases", "ConfirmedCases_daily", 99, ["country"])
    
    # Days from first detected case
    df["days_from_first_ever_case"] = (pd.to_datetime(df["Date"]) - pd.to_datetime("2019-12-01")).dt.days
    df.loc[df["days_from_first_ever_case"] < 0, "days_from_first_ever_case"] = 0
    
    #Days from quarantine, school closures and restrictions
    df["days_from_quarantine"] = (pd.to_datetime(df["Date"]) - pd.to_datetime(df["quarantine"])).dt.days
    df["days_from_quarantine"].fillna(0)
    df.loc[df["days_from_quarantine"] < 30, "days_from_quarantine"] = 0
    df.loc[df["days_from_quarantine"] >= 30, "days_from_quarantine"] = 1

    df["days_from_school"] = (pd.to_datetime(df["Date"]) - pd.to_datetime(df["schools"])).dt.days
    df["days_from_school"].fillna(df["days_from_quarantine"])
    df.loc[df["days_from_school"] < 30, "days_from_school"] = 0
    df.loc[df["days_from_school"] >= 30, "days_from_school"] = 1

    df["days_from_publicplace"] = (pd.to_datetime(df["Date"]) - pd.to_datetime(df["publicplace"])).dt.days
    df["days_from_publicplace"].fillna(df["days_from_quarantine"])
    df.loc[df["days_from_publicplace"] < 30, "days_from_publicplace"] = 0
    df.loc[df["days_from_publicplace"] >= 30, "days_from_publicplace"] = 1
    
    df["days_from_gathering"] = (pd.to_datetime(df["Date"]) - pd.to_datetime(df["gathering"])).dt.days
    df["days_from_gathering"].fillna(df["days_from_quarantine"])
    df.loc[df["days_from_gathering"] < 30, "days_from_gathering"] = 0
    df.loc[df["days_from_gathering"] >= 30, "days_from_gathering"] = 1
    
    df["days_from_nonessential"] = (pd.to_datetime(df["Date"]) - pd.to_datetime(df["nonessential"])).dt.days
    df["days_from_nonessential"].fillna(df["days_from_quarantine"])
    df.loc[df["days_from_nonessential"] < 30, "days_from_nonessential"] = 0
    df.loc[df["days_from_nonessential"] >= 30, "days_from_nonessential"] = 1
    
    return df


# In[ ]:


def generate_ar_features(df, group_by_cols, value_cols):
    
    # Daily cases
    diff_df = df.groupby(group_by_cols)[value_cols].diff().fillna(0)
    diff_df.columns = [col + "_daily" for col in value_cols]
    value_cols += [col + "_daily" for col in value_cols]
    df = pd.concat([df, diff_df], axis=1)
    
    # Daily percentage increase
    pct_df = df.groupby(group_by_cols)[value_cols].pct_change().fillna(0)
    pct_df.columns = [col + "_pct_change" for col in value_cols]
    value_cols += [col + "_pct_change" for col in value_cols]
    df = pd.concat([df, pct_df], axis=1)

    # Shift to yesterday's data
    yesterday_df = df.groupby(group_by_cols)[value_cols].shift()
    value_cols = [col + "_yesterday" for col in value_cols]
    yesterday_df.columns = value_cols
    df = pd.concat([df, yesterday_df], axis=1)

    # Average of the percentage change in the last 3 days
    three_days_avg = df.groupby(group_by_cols)[value_cols].rolling(3).mean()
    three_days_avg = three_days_avg.reset_index()[value_cols]
    three_days_avg.columns = [col + "_3_day_avg" for col in three_days_avg.columns]
    df = pd.concat([df, three_days_avg], axis=1)

    # Average of the percentage change in the last 7 days
    seven_days_avg = df.groupby(group_by_cols)[value_cols].rolling(7).mean()
    seven_days_avg = seven_days_avg.reset_index()[value_cols]
    seven_days_avg.columns = [col + "_7_day_avg" for col in seven_days_avg.columns]
    df = pd.concat([df, seven_days_avg], axis=1)
    
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


# In[ ]:


def generate_features(df):
    group_by_cols = ["state","country"]
    value_cols = ["ConfirmedCases", "Fatalities", "country_cases", "country_fatalities"]
    
    df = aggregate_label(df)
    df = append_external_data(df)
    df = generate_ar_features(df, group_by_cols, value_cols)
    df = generate_time_features(df)
    df["dow"] = pd.to_datetime(df["Date"]).dt.dayofweek
    df = df.fillna(-1)
    df.loc[df["ConfirmedCases_yesterday"]<0, "ConfirmedCases_yesterday"] = 0
    df.loc[df["Fatalities_yesterday"]<0, "Fatalities_yesterday"] = 0
    return df


# In[ ]:


train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
train = train.rename({"Province/State": "state", "Country/Region": "country"}, axis=1)
train.loc[train["state"].isna(), "state"] = "Unknown"
train = generate_features(train)
print(train["Date"].min(), "-", train["Date"].max())
train.loc[train["country"] == "Italy"].tail()


# In[ ]:


test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
test = test.rename({"Province/State": "state", "Country/Region": "country"}, axis=1)
test.loc[test["state"].isna(), "state"] = "Unknown"
print(test["Date"].min(), "-", test["Date"].max())
test.head()


# # Modeling

# In[ ]:


train.loc[train["Date"]<"2020-03-19", "split"] = "train"
train.loc[train["Date"]>="2020-03-19", "split"] = "test"


# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
features = [col for col in train.columns if ("yesterday" in col) | ("days_from" in col)]
features += country_info.select_dtypes(include=numerics).columns.tolist()
features += pollution.select_dtypes(include=numerics).columns.tolist()
features += economy.select_dtypes(include=numerics).columns.tolist()
features += ["Lat", "Long", "dow"]


# In[ ]:


len(features)


# ## LightGBM

# In[ ]:


import lightgbm as lgb
from sklearn.metrics import mean_squared_error


# In[ ]:


def train_model(df, label, base_label, features=features, **kwargs):
    X_train = df.loc[df["split"] == "train"][features]
    y_train = np.log(df.loc[df["split"] == "train"][label] + 1)
    b_train = np.log(df.loc[df["split"] == "train"][base_label] + 1)
    X_test = df.loc[df["split"] == "test"][features]
    y_test = np.log(df.loc[df["split"] == "test", label] + 1)
    b_test = np.log(df.loc[df["split"] == "test", base_label] + 1)
    print(kwargs)
    model = lgb.LGBMRegressor(**kwargs)
    model.fit(X_train, y_train, init_score = b_train)
    y_pred = model.predict(X_test)
    print(np.sqrt(mean_squared_error(y_test, y_pred + b_test)))
    return model


# ## Cases

# In[ ]:


lgb_model_cases = train_model(train, "ConfirmedCases", "ConfirmedCases_yesterday",
                                   num_leaves=100,
                                   colsample_bytree=0.8,
                                   learning_rate=0.1,
                                   n_estimators=1000,
                                   subsample=0.8,
                                   min_data_in_leaf=1)


# In[ ]:


import shap
explainer = shap.TreeExplainer(lgb_model_cases)
shap_values = explainer.shap_values(train.loc[train["split"] == "test"][features])
shap.summary_plot(
    shap_values,
    train.loc[train["split"] == "test"][features],
    max_display=110,
    show=True,
)


# ## Fatalities

# In[ ]:


lgb_model_fatalities = train_model(train, "Fatalities", "Fatalities_yesterday", features,
                                   num_leaves=100,
                                   colsample_bytree=0.8,
                                   learning_rate=0.1,
                                   n_estimators=1000,
                                   subsample=0.8,
                                   min_data_in_leaf=1                             
                                  )


# In[ ]:


import shap
explainer = shap.TreeExplainer(lgb_model_fatalities)
shap_values = explainer.shap_values(train.loc[train["split"] == "test"][features])
shap.summary_plot(
    shap_values,
    train.loc[train["split"] == "test"][features],
    max_display=110,
    show=True,
)


# ## Train on full data

# In[ ]:


X_train = train[features]
y_train = np.log(train["ConfirmedCases"] + 1)
b_train = np.log(train["ConfirmedCases_yesterday"] + 1)
cases_model = lgb.LGBMRegressor(num_leaves=100,
                                   colsample_bytree=0.8,
                                   learning_rate=0.1,
                                   n_estimators=1000,
                                   subsample=0.8,
                                   min_data_in_leaf=1)
cases_model.fit(X_train, y_train, init_score = b_train)

X_train = train[features]
y_train = np.log(train["Fatalities"] + 1)
b_train = np.log(train["Fatalities_yesterday"] + 1)
fatalities_model = lgb.LGBMRegressor(num_leaves=100,
                                   colsample_bytree=0.8,
                                   learning_rate=0.1,
                                   n_estimators=1000,
                                   subsample=0.8,
                                   min_data_in_leaf=1)
fatalities_model.fit(X_train, y_train, init_score = b_train)


# ## Predict submission dates

# In[ ]:


base_df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
base_df = base_df.rename({"Province/State": "state", "Country/Region": "country"}, axis=1)
base_df.loc[base_df["state"].isna(), "state"] = "Unknown"
scoring_dates = test.loc[test["Date"]>"2020-03-24", "Date"].unique()
base_df = base_df.loc[base_df["Date"]<="2020-03-24"]


# In[ ]:


scoring_dates


# In[ ]:


pred_df = pd.DataFrame(columns=base_df.columns)
for date in scoring_dates.tolist():
    print(date)
    new_df = base_df.copy()
    curr_date_df = test.loc[test["Date"] == date].copy()
    curr_date_df["ConfirmedCases"] = 0
    curr_date_df["Fatalities"] = 0
    new_df = new_df.append(curr_date_df).reset_index(drop=True)
    new_df = generate_features(new_df)
    predictions = cases_model.predict(new_df[features]) + np.log(new_df["ConfirmedCases_yesterday"] + 1)
    new_df["predicted_cases"] = round(np.maximum(np.exp(predictions) - 1, new_df["ConfirmedCases_yesterday"]))
    predictions = fatalities_model.predict(new_df[features]) + np.log(new_df["Fatalities_yesterday"] + 1)
    new_df["predicted_fatalities"] = np.maximum(np.exp(predictions) - 1, new_df["Fatalities_yesterday"])
    new_df["predicted_fatalities"] = round(np.minimum(new_df["predicted_fatalities"], new_df["predicted_cases"]*0.2))
    new_df.loc[new_df["Date"] == date, "ConfirmedCases"] = new_df.loc[new_df["Date"] == date, "predicted_cases"]
    new_df.loc[new_df["Date"] == date, "Fatalities"] = new_df.loc[new_df["Date"] == date, "predicted_fatalities"]
    pred_df = pred_df.append(new_df.loc[new_df["Date"] == date][pred_df.columns.tolist()])
    base_df = base_df.append(new_df.loc[new_df["Date"] == date][base_df.columns.tolist()])


# In[ ]:


pred_df.loc[pred_df["state"] == "Hubei"]


# In[ ]:


pred_df.loc[pred_df["country"] == "Italy"]


# In[ ]:


pred_df.loc[pred_df["country"] == "Israel"]


# In[ ]:


pred_df.loc[pred_df["country"] == "Argentina"]


# In[ ]:


pred_df.loc[pred_df["country"] == "Uruguay"]
