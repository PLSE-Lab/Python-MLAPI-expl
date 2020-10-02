#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

df = pd.read_csv("../input/covid19-global-forecasting-week-5/train.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-5/test.csv")

print(df.shape, test_df.shape)
df.head()


# In[ ]:


NUM_ITER = 5


# In[ ]:


test_df["Date"] = pd.to_datetime(test_df['Date'])

test_df["Date"].min(), test_df["Date"].max()


# In[ ]:


df.loc[df["Target"] == "Fatalities", "TargetValue"] *= 10

df["TargetValue"] = np.clip(df["TargetValue"], 0, None)

scale = 250

df["TargetValue"] /= scale
scale


# In[ ]:


loc_group = ["County", "Province_State", "Country_Region"]
key_cols = loc_group + ["Date", "Population"]
targets =["ConfirmedCases", "Fatalities"]


def hash_loc(df):
    return df["Country_Region"].fillna("*") + "_" + df["Province_State"].fillna("*") + "_" + df["County"].fillna("*")

def convert_df(df):
    temp_df = df[key_cols].drop_duplicates()
    for target in targets:
        temp_df = temp_df.merge(df[df["Target"] == target][key_cols + ["TargetValue"]], on=key_cols, how="left")
        temp_df = temp_df.rename(columns={"TargetValue": target})
        for q in ["_05", "_5", "_95"]:
            temp_df[target + q] = temp_df[target].values
    temp_df["Date"] = pd.to_datetime(temp_df['Date'])
    temp_df["loc"] = hash_loc(temp_df)
    return temp_df.sort_values(["loc", "Date"])

df = convert_df(df)
test_df["loc"] = hash_loc(test_df)
print(df.shape)
df.head()


# In[ ]:


from datetime import timedelta


NUM_SHIFT = 14
FIRST_PRED_DAY = df["Date"].max() + timedelta(days=1)
print(FIRST_PRED_DAY)

temp_df = df[df["Date"] == df["Date"].max()][loc_group + ["loc", "Population"]].copy()
temp_df["Date"] = FIRST_PRED_DAY


df = df.append(temp_df, sort=False)


ts_features = {targets[0]:[], targets[1]:[]}
for s in range(NUM_SHIFT, 0, -1):
    for col in targets:
        df["prev_{}_{}".format(col, s)] = df.groupby("loc")[col].shift(s)
        ts_features[col].append("prev_{}_{}".format(col, s))
        
pred_df, df = df[df["Date"] == FIRST_PRED_DAY].copy(), df[df["Date"] < FIRST_PRED_DAY].copy()
print(pred_df.shape, df.shape)
pred_df.head()


# In[ ]:


X_hist = np.zeros((pred_df.shape[0], NUM_SHIFT, len(targets)))

for i in range(len(targets)):
    X_hist[:, :, i] = pred_df[ts_features[targets[i]]].values
    
X_hist.shape


# In[ ]:


from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model


def get_model():
    seq_inp = KL.Input(shape=(14, 2))
    
    x = KL.Conv1D(8, 3, activation="relu")(seq_inp)
    x = KL.Conv1D(32, 3, activation="relu", strides=3)(x)
    x = KL.Conv1D(128, 3, activation="relu")(x)
    
    x = KL.Flatten()(x)
    
    out1 = KL.Dense(3, activation="relu")(x)
    out1 = KL.Lambda(lambda x: K.cumsum(x, axis=1), name=targets[0])(out1)
    
    out2 = KL.Dense(3, activation="relu")(x)
    out2 = KL.Lambda(lambda x: K.cumsum(x, axis=1), name=targets[1])(out2)
    
    model = Model(inputs=seq_inp, outputs=[out1, out2])
    return model


get_model().summary()


# In[ ]:


last_train_date = df["Date"].max()
days_to_predict = (test_df["Date"].max() - last_train_date).days

last_train_date, days_to_predict


# In[ ]:


def get_pred_cols(target):
    return [target + q for q in ["_05", "_5", "_95"]]

model = get_model()
temp_df = pred_df.copy()
future_df = pd.DataFrame()
for days_ahead in range(1, days_to_predict + 1):
    preds = []
    for it in range(NUM_ITER):
        model.load_weights("../input/covid-19-w5-training/weights/model_{d}_{it}.h5".format(d=days_ahead, it=it))
        preds.append(model.predict(X_hist, batch_size=128))
    
    temp_df[get_pred_cols(targets[0])] = sum([pred[0] for pred in preds])/len(preds)
    temp_df[get_pred_cols(targets[1])] = sum([pred[1] for pred in preds])/len(preds)
        
    temp_df["Date"] = last_train_date + timedelta(days=days_ahead)

    future_df = future_df.append(temp_df, sort=False)
    print(days_ahead, future_df.shape)


# In[ ]:


SMOOTH = True

if SMOOTH:
    for col in get_pred_cols(targets[0]) + get_pred_cols(targets[1]):
        smoothed_pred = 0.6*future_df[col]
        smoothed_pred += 0.2*future_df.groupby("loc")[col].shift(1).fillna(future_df[col]).values
        smoothed_pred += 0.2*future_df.groupby("loc")[col].shift(-1).fillna(future_df[col]).values
        
        future_df[col] = smoothed_pred


# In[ ]:


df = df.append(future_df, sort=False)

df[get_pred_cols(targets[0])] *= scale
df[get_pred_cols(targets[1])] *= scale/10

for col in get_pred_cols(targets[0]) + get_pred_cols(targets[1]):
    df[col] = df[col].round().astype(int)


# In[ ]:


sub_dfs = dict()
for target in targets:
    sub_dfs[target] = test_df[test_df["Target"] == target].merge(df[get_pred_cols(target) + ["Date", "loc"]], on=["Date", "loc"])
    sub_dfs[target] = sub_dfs[target].melt(id_vars=["ForecastId"], value_vars=get_pred_cols(target)).sort_values("ForecastId")
    sub_dfs[target]["ForecastId"] = sub_dfs[target]["ForecastId"].astype(str) + "_0." + sub_dfs[target]["variable"].apply(lambda x: x.split("_")[1])
    sub_dfs[target].drop("variable", axis=1, inplace=True)
    sub_dfs[target].rename(columns={"ForecastId": "ForecastId_Quantile", "value": "TargetValue"}, inplace=True)

    
assert sub_dfs[targets[0]].shape[0] + sub_dfs[targets[1]].shape[0] == test_df.shape[0]*3
sub_dfs[targets[0]].head()


# In[ ]:


sub_dfs[targets[0]].append(sub_dfs[targets[1]], sort=False).to_csv("submission.csv", index=False)


# In[ ]:


df[(df["Country_Region"] == "Turkey") & (df["Date"] > "2020-04-23")][["Date"] + get_pred_cols(targets[0])].plot(x="Date")


# In[ ]:


df[(df["Country_Region"] == "Turkey") & (df["Date"] > "2020-04-23")][["Date"] + get_pred_cols(targets[1])].plot(x="Date")


# In[ ]:


df[(df["loc"] == "Netherlands_*_*") & (df["Date"] > "2020-04-23")][["Date"] + get_pred_cols(targets[0])].plot(x="Date")


# In[ ]:


df[(df["loc"] == "Netherlands_*_*") & (df["Date"] > "2020-04-23")][["Date"] + get_pred_cols(targets[1])].plot(x="Date")


# In[ ]:


df[(df["loc"] == "Russia_*_*") & (df["Date"] > "2020-04-23")][["Date"] + get_pred_cols(targets[0])].plot(x="Date")


# In[ ]:


df[(df["loc"] == "Russia_*_*") & (df["Date"] > "2020-04-23")][["Date"] + get_pred_cols(targets[1])].plot(x="Date")


# In[ ]:


df[(df["loc"] == "China_*_*") & (df["Date"] > "2020-04-23")][["Date"] + get_pred_cols(targets[0])].plot(x="Date")


# In[ ]:


df[(df["loc"] == "China_*_*") & (df["Date"] > "2020-04-23")][["Date"] + get_pred_cols(targets[1])].plot(x="Date")


# In[ ]:




