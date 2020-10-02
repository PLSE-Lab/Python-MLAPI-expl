import pandas as pd
import numpy as np
import lightgbm as lgb


data = pd.read_csv("../input/train.csv")
data["is_test"] = 0
test = pd.read_csv("../input/test.csv")
test["is_test"] = 1
data = data.append(test)

data.sort_values(by=["year", "month", "day", "sched_dep_time"], inplace=True)
data["date"] = data.apply(lambda x: str(x["year"]) + "_" + str(x["month"]) + "_" + str(x["day"]), axis=1)

data["sched_arr_hr"] = data["sched_arr_time"].apply(lambda x: int(x / 100))
data["sched_dep_hr"] = data["sched_dep_time"].apply(lambda x: int(x / 100))
weather = pd.read_csv("../input/weather.csv")
weather.drop_duplicates(subset=["year", "month", "day"], keep="first", inplace=True)
weather["date"] = weather.apply(lambda x: str(x["year"]) + "_" + str(x["month"]) + "_" + str(x["day"]), axis=1)
weather.drop(["year", "month", "day", "hour"], axis=1, inplace=True)

data = data.merge(weather, on=["origin", "date"], how="left")

data["precip"].fillna("NA", inplace=True)
data["precip"] = data["precip"].apply(lambda x: -999 if str(x) == "NA" else x)
data["temp"].fillna("NA", inplace=True)
data["temp"] = data["temp"].apply(lambda x: -999 if str(x) == "NA" else x)
data["humid"].fillna("NA", inplace=True)
data["humid"] = data["humid"].apply(lambda x: -999 if str(x) == "NA" else x)
data["wind_speed"].fillna("NA", inplace=True)
data["wind_speed"] = data["wind_speed"].apply(lambda x: -999 if str(x) == "NA" else x)
data["wind_gust"].fillna("NA", inplace=True)
data["wind_gust"] = data["wind_gust"].apply(lambda x: -999 if str(x) == "NA" else x)

delay_dict = {}
origin_vals = data["origin"].values
date_vals = data["date"].values
delay_vals = data["is_delayed"].values
is_test = data["is_test"].values
for n in range(0, len(data)):
    if origin_vals[n] not in delay_dict:
        delay_dict[origin_vals[n]] = {}
    if date_vals[n] not in delay_dict[origin_vals[n]]:
        delay_dict[origin_vals[n]][date_vals[n]] = {"total":0, "train":[]}
    if is_test[n] == 0:
        delay_dict[origin_vals[n]][date_vals[n]]["train"].append(delay_vals[n])
    delay_dict[origin_vals[n]][date_vals[n]]["total"] += 1

count_day = np.zeros(len(data))
delay_day = np.zeros(len(data))
for n in range(0, len(data)):
    count_day[n] = delay_dict[origin_vals[n]][date_vals[n]]["total"]
    delay_day[n] = np.average(delay_dict[origin_vals[n]][date_vals[n]]["train"])
data["count_day"] = count_day
data["delay_day"] = delay_day

data["origin"] = pd.Categorical(data["origin"])
data["dest"] = pd.Categorical(data["dest"])
data["carrier"] = pd.Categorical(data["carrier"])

train = data[data["is_test"] == 0]
test = data[data["is_test"] == 1]

print(len(train), len(test))

use_features = ["distance", "origin", "dest", "carrier", "sched_arr_hr", "sched_dep_hr", "precip", "temp", "humid", "wind_speed", "count_day", "delay_day"]

data[use_features].applymap(np.isreal)
for col in use_features:
    print(col, np.sum(np.isreal(data[col])))

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 101,
    'max_depth': 7,
    'learning_rate': 0.05,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'verbose': -1,
}

train["is_val"] = np.random.randint(low=0, high=9, size=len(train))

lgb_train = lgb.Dataset(train[use_features], train["is_delayed"])
gbm = lgb.train(params, lgb_train, num_boost_round=1300)
pred = gbm.predict(test[use_features])

test["is_delayed"] = pred
test[["id" ,"is_delayed"]].to_csv("sub1.csv", index=False)






