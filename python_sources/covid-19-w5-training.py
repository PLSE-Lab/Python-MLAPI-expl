#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

os.mkdir("weights")


# In[ ]:


NUM_MODELS = 35
NUM_ITER = 5


# In[ ]:


import numpy as np
import pandas as pd

df = pd.read_csv("../input/covid19-global-forecasting-week-5/train.csv")
print(df.shape)
df.head()


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

def convert_df(df):
    temp_df = df[key_cols].drop_duplicates()
    for target in targets:
        temp_df = temp_df.merge(df[df["Target"] == target][key_cols + ["Id", "TargetValue"]], on=key_cols, how="left")
        temp_df = temp_df.rename(columns={"Id": "Id_{}".format(target), "TargetValue": target})
    temp_df["Date"] = pd.to_datetime(temp_df['Date'])
    temp_df["loc"] = temp_df["Country_Region"].fillna("*") + "_" + temp_df["Province_State"].fillna("*") + "_" + temp_df["County"].fillna("*")
    temp_df["w"] = 1/np.log1p(temp_df["Population"])
    return temp_df.sort_values(["loc", "Date"])

df = convert_df(df)
print(df.shape)
df.head()


# In[ ]:


df[df["Country_Region"] == "Turkey"].tail(10)[targets]*scale


# In[ ]:


from datetime import timedelta

NUM_SHIFT = 14

ts_features = {targets[0]:[], targets[1]:[]}
for s in range(NUM_SHIFT, 0, -1):
    for col in targets:
        df["prev_{}_{}".format(col, s)] = df.groupby("loc")[col].shift(s)
        ts_features[col].append("prev_{}_{}".format(col, s))
        
df = df[df["Date"] >= df["Date"].min() + timedelta(days=NUM_SHIFT)]
print(df.shape)
df.head()


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(df[(df["Country_Region"] == "Turkey") & (df["Date"] == "2020-05-01")][ts_features[targets[0]]].values[0])
plt.plot(df[(df["Country_Region"] == "Turkey") & (df["Date"] == "2020-05-01")][ts_features[targets[1]]].values[0])


# In[ ]:


get_ipython().system('pip install tensorflow-addons')


# In[ ]:


from tensorflow.keras.optimizers import Nadam
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

from tensorflow_addons.losses import pinball_loss
from tensorflow_addons.optimizers import SWA


def get_custom_loss(q):
    def custom_loss(y_true, y_pred):
        return pinball_loss(y_true, y_pred, q)
    return custom_loss


def get_model():
    seq_inp = KL.Input(shape=(14, 2))
    
    x = KL.Conv1D(8, 3, activation="relu")(KL.GaussianDropout(0.1)(seq_inp))
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


def get_weight(df):
    weights = df.shape[0]*df["w"]/df["w"].sum()
    return [weights.values, weights.values]

def get_output(df):
    return [df["target_{}".format(targets[0])].values, df["target_{}".format(targets[1])].values]

def train_model(df, days_ahead, num_iter):
    print("Training {}...".format(days_ahead))
    
    for target in targets:
        df["target_{}".format(target)] = df.groupby("loc")[target].shift(1-days_ahead)
        
    train_df = df[df["target_{}".format(targets[0])].notnull()].copy()
    X_hist_train = np.zeros((train_df.shape[0], NUM_SHIFT, len(targets)))

    for i in range(len(targets)):
        X_hist_train[:, :, i] = train_df[ts_features[targets[i]]].values

        
    epochs = [10, 4]
    if days_ahead >= 20:
        epochs = [6, 2]
    
    for it in range(num_iter):
        print("Iteration:", it)
        K.clear_session()
        model = get_model()
        for loss, lr, epoch in [(get_custom_loss([0.15, 0.50, 0.85]), 0.0001, epochs[0]), (get_custom_loss([0.05, 0.50, 0.95]), 0.00005, epochs[1])]:
            model.compile(loss=loss, optimizer=SWA(Nadam(lr=lr), average_period=2))
            model.fit(X_hist_train, get_output(train_df), sample_weight=get_weight(train_df),
                      shuffle=True, batch_size=128, epochs=epoch, verbose=0)
    
        model.save_weights("weights/model_{d}_{it}.h5".format(d=days_ahead, it=it))


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfor days_ahead in range(1, NUM_MODELS+1):\n    train_model(df, days_ahead, num_iter=NUM_ITER)\n    print()')


# In[ ]:




