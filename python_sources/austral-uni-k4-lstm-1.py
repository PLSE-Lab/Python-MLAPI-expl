#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score
from IPython.display import display
from sklearn.metrics import roc_auc_score

get_ipython().run_line_magic('matplotlib', 'inline')
data_path="../input"


# In[ ]:


data = pd.concat([pd.read_csv(os.path.join(data_path, "application_train.csv"), index_col="SK_ID_CURR"), pd.read_csv(os.path.join(data_path, "application_test.csv"), index_col="SK_ID_CURR")], axis=0)


# In[ ]:


data_embeddings = data.select_dtypes("object").columns
data_embeddings
for c in data.drop(data_embeddings.tolist() + ["TARGET"], axis=1):
    mean = data[c].dropna().mean()
    std = data[c].dropna().std()
    data[c] = (data[c].fillna(mean) - mean) / std


# In[ ]:


for c in data_embeddings:
    data[c].fillna("N/A", inplace=True)
    a = pd.get_dummies(data[c])
    a.columns = [c + "_" + d for d in a.columns]
    data = data.drop(c, axis=1).join(a)


# In[ ]:


data.dtypes.value_counts()


# In[ ]:


from keras.layers import Input, Dense, TimeDistributed, concatenate, Flatten, Lambda, LSTM
from keras.models import Sequential, Model


# In[ ]:


model = Sequential()
model.add(Dense(256, activation="relu", input_shape=(data.shape[1] - 1,)))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))


# In[ ]:


from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, rankdir='LR')
from IPython.display import Image
Image("model.png")


# In[ ]:


train = data[data.TARGET.notnull()]
valid = train.sample(frac=0.1)
train = train.drop(valid.index)
X_train = train.drop("TARGET", axis=1)
y_train = train.TARGET.values
X_valid = valid.drop("TARGET", axis=1)
y_valid = valid.TARGET.values


# In[ ]:


model.compile(optimizer='RMSprop', loss='binary_crossentropy')
for i in range(10):
    model.fit(X_train, y_train, batch_size=1024, validation_data=(X_valid, y_valid))
    print("ROC_AUC para iteracion", i, ":", roc_auc_score(y_train, model.predict(X_train)[:, -1]), " - ", roc_auc_score(y_valid, model.predict(X_valid)[:, -1]))


# In[ ]:


bureau = pd.read_csv(os.path.join(data_path, "bureau.csv"))


# In[ ]:


bureau_embeddings = bureau.select_dtypes("object").columns
for c in bureau.drop(bureau_embeddings, axis=1):
    if c == "SK_ID_CURR": continue
    mean = bureau[c].dropna().mean()
    std = bureau[c].dropna().std()
    bureau[c] = (bureau[c].fillna(mean) - mean) / std
for c in bureau_embeddings:
    bureau[c].fillna("N/A", inplace=True)
    a = pd.get_dummies(bureau[c])
    a.columns = [c + "_" + d for d in a.columns]
    bureau = bureau.drop(c, axis=1).join(a)


# In[ ]:


app_model_input = Input(shape=(data.shape[1] - 1,), name="app")
app_model = Dense(256, activation="relu")(app_model_input)
app_model = Dense(128, activation="relu")(app_model)

bur_model_input = Input(shape=(None, bureau.shape[1]), name="bur")
bur_model = TimeDistributed(Dense(128, activation="relu"))(bur_model_input)
bur_model = TimeDistributed(Dense(64, activation="relu"))(bur_model)
bur_model = LSTM(32)(bur_model)

final_fully_connected = concatenate([app_model, bur_model])
final_fully_connected = Dense(128, activation="relu")(final_fully_connected)
final_fully_connected = Dense(64, activation="relu")(final_fully_connected)
final_fully_connected = Dense(32, activation="relu")(final_fully_connected)
final_fully_connected = Dense(1, activation="sigmoid")(final_fully_connected)
model = Model(inputs=[app_model_input, bur_model_input], outputs=final_fully_connected)
plot_model(model, to_file='model.png', show_shapes=True, rankdir='TB')
from IPython.display import Image
Image("model.png")


# In[ ]:


data = data.join(bureau.groupby("SK_ID_CURR").apply(lambda x: x.values[None, :, :]).rename("bur"))
data


# In[ ]:


data["bur"] = data.bur.apply(lambda x: x if x is not np.nan else np.zeros((1, 1, bureau.shape[1])))


# In[ ]:


def data_gen(data): 
    while True:
        temp = data.sample(1)
        yield {
            "app": temp.drop(["TARGET", "bur"], axis=1).values,
            "bur": temp.bur.values[0]
        }, temp.TARGET.values


# In[ ]:


next(data_gen(data))


# In[ ]:


train = data[data.TARGET.notnull()]
valid = train.sample(frac=0.1)
train = train.drop(valid.index)
valid_gen = data_gen(valid)
train_gen = data_gen(train)


# In[ ]:


model.compile(optimizer='RMSprop', loss='binary_crossentropy')
get_ipython().run_line_magic('pinfo', 'model.fit_generator')


# In[ ]:


model.fit_generator(train_gen, steps_per_epoch=100, validation_data=valid_gen, validation_steps=100)


# In[ ]:


data.bur.apply(lambda x: x.shape[1]).value_counts().sort_index().cumsum()


# In[ ]:


data["bur"] = data.bur.apply(lambda x: np.pad(x, ((0, 0), (10 - x.shape[1], 0), (0, 0)), "constant") if x.shape[1] < 10 else x[:, :10, :])


# In[ ]:


train = data[data.TARGET.notnull()]
valid = train.sample(frac=0.1)
train = train.drop(valid.index)

X_train = {
    "app": train.drop(["TARGET", "bur"], axis=1).values,
    "bur": np.concatenate(train.bur.values)
}
y_train = train.TARGET.values
X_valid = {
    "app": valid.drop(["TARGET", "bur"], axis=1).values,
    "bur": np.concatenate(valid.bur.values)
}
y_valid = valid.TARGET.values


# In[ ]:


X_train["app"].shape, X_train["bur"].shape


# In[ ]:


app_model_input = Input(shape=(data.shape[1] - 2,), name="app")
app_model = Dense(256, activation="relu")(app_model_input)
app_model = Dense(128, activation="relu")(app_model)

bur_model_input = Input(shape=X_train["bur"].shape[1:], name="bur")
bur_model = TimeDistributed(Dense(128, activation="relu"))(bur_model_input)
bur_model = TimeDistributed(Dense(64, activation="relu"))(bur_model)
bur_model = LSTM(32)(bur_model)

final_fully_connected = concatenate([app_model, bur_model])
final_fully_connected = Dense(128, activation="relu")(final_fully_connected)
final_fully_connected = Dense(64, activation="relu")(final_fully_connected)
final_fully_connected = Dense(32, activation="relu")(final_fully_connected)
final_fully_connected = Dense(1, activation="sigmoid")(final_fully_connected)
model = Model(inputs=[app_model_input, bur_model_input], outputs=final_fully_connected)


# In[ ]:


model.compile(optimizer='RMSprop', loss='binary_crossentropy')
for i in range(10):
    model.fit(X_train, y_train, batch_size=1024, validation_data=(X_valid, y_valid))
    print("ROC_AUC para iteracion", i, ":", roc_auc_score(y_train, model.predict(X_train)[:, -1]), " - ", roc_auc_score(y_valid, model.predict(X_valid)[:, -1]))


# In[ ]:


data.drop("bur", axis=1, inplace=True)

