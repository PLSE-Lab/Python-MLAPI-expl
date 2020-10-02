#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os


# ## Load data

# In[ ]:


path = "../input/m5-forecasting-accuracy"
calendar = pd.read_csv(os.path.join(path, "calendar.csv"))
selling_prices = pd.read_csv(os.path.join(path, "sell_prices.csv"))
sample_submission = pd.read_csv(os.path.join(path, "sample_submission.csv"))
sales = pd.read_csv(os.path.join(path, "sales_train_validation.csv"))


# In[ ]:


calendar.head(3)


# In[ ]:


for i, var in enumerate(["year", "weekday", "month", 
                          "snap_CA", "snap_TX", "snap_WI"]):
    plt.figure()
    g = sns.countplot(calendar[var])
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set_title(var)


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder

def prep_calendar(df):
    df = df.copy()
    temp = ["wday", "month", "year", "event_name_1", "event_type_1"]
    df = df[["wm_yr_wk", "d"] + temp]
    df.fillna("missing", inplace=True)
    df[temp] = OrdinalEncoder().fit_transform(df[temp])
    for v in temp:
        df[temp] = df[temp].astype("uint8")
    df.wm_yr_wk = df.wm_yr_wk.astype("uint16")
    return df

calendar = prep_calendar(calendar)


# In[ ]:


calendar.head(3)


# #### Notes for modeling
# 
# 
# - wday -> integer coding & embedding
# 
# - year(?) -> integer coding & embedding
# 
# - month(?) -> integer coding & embedding
# 
# - "event_name_1", "event_type_1"": simple imputer & integer coding & embedding
# 

# In[ ]:


sales.head(3)


# In[ ]:


for i, var in enumerate(["state_id", "store_id", "cat_id", "dept_id"]):
    plt.figure()
    g = sns.countplot(sales[var])
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set_title(var)


# In[ ]:


sales.item_id.value_counts()


# #### Drop dates to save space. Offline, this is not required.

# In[ ]:


sales.drop(["d_" + str(i+1) for i in range(800)], axis=1, inplace=True)


# #### Reshaping
# 
# We now reshape the data from wide to long, using "id" as fixed and swapping "d_1", to "d_1913". 

# In[ ]:


sales[sales.item_id=="HOBBIES_1_001"]


# In[ ]:


def melt_sales(df):
    df = df.drop(["item_id", "dept_id", "cat_id", "store_id", "state_id"], axis=1).melt(
        id_vars=['id'], var_name='d', value_name='demand')
    return df

sales = melt_sales(sales)


# In[ ]:


sales.head()


# #### Add reshaped submission file
# 
# So that it sneaks through data preprocessing easily.

# In[ ]:


sample_submission.head()


# In[ ]:


# Turn strings like "F1" to "d_1914"
def map_f2d(d_col, id_col):
    eval_flag = id_col.str.endswith("evaluation")
    return "d_" + (d_col.str[1:].astype("int") + 1913 + 28 * eval_flag).astype("str")

# Reverse
def map_d2f(d_col, id_col):
    eval_flag = id_col.str.endswith("evaluation")
    return "F" + (d_col.str[2:].astype("int") - 1913 - 28 * eval_flag).astype("str")

# Example
map_f2d(pd.Series(["F1", "F2", "F28", "F1", "F2", "F28"]), 
        pd.Series(["validation", "validation", "validation", "evaluation", "evaluation", "evaluation"]))


# In[ ]:


submission = sample_submission.melt(id_vars="id", var_name="d", value_name="demand").assign(
    demand=np.nan,
    d = lambda df: map_f2d(df.d, df.id))
submission.head()


# In[ ]:


sales = pd.concat([sales, submission])
sales.tail()


# #### Change "evaluation" to "validation"...

# In[ ]:


sales.id = sales.id.str.replace("evaluation", "validation")


# In[ ]:


from sklearn.preprocessing import StandardScaler

def add_lagged_features(df):
    df['lag_t56'] = df.groupby('id')['demand'].transform(lambda x: x.shift(56))
    df['rolling_mean_t30'] = df.groupby('id')['demand'].transform(lambda x: x.shift(56).rolling(30, min_periods=1).mean())
  
    temp = ["lag_t56", "rolling_mean_t30"]
    df.dropna(subset=temp, inplace=True)    
    df[temp] = StandardScaler().fit_transform(df[temp])
    for v in temp:
        df[v] = df[v].astype("float32")
    return df

sales = add_lagged_features(sales)


# In[ ]:


sales.head()


# #### Add relevant id information
# 
# After combination of training and test data, we can join further info.

# In[ ]:


def expand_id(id):
    return id.str.split("_", expand=True).assign(
        dept_id=lambda df: df.iloc[:,0] + "_" + df.iloc[:,1], 
        item_id=lambda df: df.iloc[:,0] + "_" + df.iloc[:,1] + "_" + df.iloc[:, 2],
        store_id=lambda df: df.iloc[:,3] + "_" + df.iloc[:,4]).drop(np.arange(6), axis=1)

# Example
expand_id(sales["id"].head())


# In[ ]:


uid = pd.Series(sales["id"].unique())
id_lookup = expand_id(uid)
id_lookup["id"] = uid

encode_item_id = OrdinalEncoder()
encode_dept_id = OrdinalEncoder()
encode_store_id = OrdinalEncoder()
id_lookup["item_id"] = encode_item_id.fit_transform(id_lookup[["item_id"]]).astype("uint16")
id_lookup["dept_id"] = encode_dept_id.fit_transform(id_lookup[["dept_id"]]).astype("uint8")
id_lookup["store_id"] = encode_store_id.fit_transform(id_lookup[["store_id"]]).astype("uint8")

id_lookup.head()


# In[ ]:


sales = sales.merge(id_lookup, on="id", how="left")
del sales["id"]


# In[ ]:


sales.head()


# ### Selling prices
# 
# Contains selling prices for each store_id, item_id_wm_yr_wk combination.

# In[ ]:


selling_prices.head()


# Derive some time related features:

# In[ ]:


# Add relative change
def prep_selling_prices(df):
    df = df.copy()
    df["store_id"] = encode_store_id.transform(df[["store_id"]]).astype("uint8")
    df["item_id"] = encode_item_id.transform(df[["item_id"]]).astype("uint16")
    df["wm_yr_wk"] = df["wm_yr_wk"].astype("uint16")
    
    df["sell_price_rel_diff"] = df.groupby(["store_id", "item_id"])["sell_price"].pct_change()
    sell_price_cummin = df.groupby(["store_id", "item_id"])["sell_price"].cummin()
    sell_price_cummax = df.groupby(["store_id", "item_id"])["sell_price"].cummax()
    df["sell_price_cumrel"] = (df["sell_price"] - sell_price_cummin) / (sell_price_cummax - sell_price_cummin)
    df.fillna({"sell_price_rel_diff": 0, "sell_price_cumrel": 1}, inplace=True)
    floats = ["sell_price_cumrel", "sell_price_rel_diff", "sell_price"]
    sc = StandardScaler()
    df[floats] = sc.fit_transform(df[floats])
    for v in floats:
        df[v] = df[v].astype("float32")
    return df

selling_prices = prep_selling_prices(selling_prices)


# In[ ]:


selling_prices.head()


# #### Notes for modeling
# 
# **Features**:
# 
# - sell_price: numeric
# 
# - relative change to last date (per store and item): numeric
# 
# - price position between cummin and cummax (per store and item): numeric
# 
# **Reshape**: No
# 
# **Merge key(s)**: to sales data by store_id, item_id, wm_yr_wk (through calendar data)

# ### Combine all

# In[ ]:


gc.collect()
sales = sales.merge(calendar, how="left", on="d")
del sales["d"]


# In[ ]:


gc.collect()
sales = sales.merge(selling_prices, how="left", on=["wm_yr_wk", "store_id", "item_id"])
del sales["wm_yr_wk"]


# In[ ]:


sales.fillna({"sell_price": 0, "sell_price_rel_diff": 0, "sell_price_cumrel": 0}, inplace=True)


# In[ ]:


sales.head()


# In[ ]:


sales.tail()


# In[ ]:


gc.collect()


# ## Modelling
# 
# We will now use Tensorflow & Keras to model sales demand as a function of the prepared input. Key pieces are the categorical predictors prepared above. They will be fed through embedding layers and combined to dense numeric features.
# 
# For simplicity, we use MSE as evaluation criterion. This will most certainly change in future commits.

# ### Create input dicts for multi-input

# In[ ]:


training_flag = pd.notna(sales.demand)


# In[ ]:


def make_Xy(df, ind=None, return_y = True):
    if ind is not None:
        df = df[ind]
    X = {"dense1": df[["lag_t56", "rolling_mean_t30", "sell_price", "sell_price_rel_diff", 
                       "sell_price_cumrel"]].to_numpy(dtype="float32"),
         "item_id": df[["item_id"]].to_numpy(dtype="uint16")}
    for i, v in enumerate(["wday", "month", "year", "event_name_1", "event_type_1", "dept_id", "store_id"]):
        X[v] = df[[v]].to_numpy(dtype="uint8")
    if return_y:
        return X, df.demand.to_numpy(dtype="float32")
    else:
        return X


# In[ ]:


# sales.to_csv("sales.csv", index=False)


# In[ ]:


X_train, y_train = make_Xy(sales, training_flag) # make_Xy(sales[0:1000000])
y_train.shape


# In[ ]:


X_test = make_Xy(sales, ~training_flag, return_y=False)


# In[ ]:


del sales
gc.collect()


# ### The model

# In[ ]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow.keras.layers import Dense, Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Flatten
from tensorflow.keras import regularizers


# In[ ]:


# Dense part
dense_input = Input(shape=(5, ), name='dense1')
dense_branch = Dense(30, activation="relu")(dense_input)
dense_branch = Dense(30, activation="relu")(dense_branch)

# Embedded input
wday_input = Input(shape=(1,), name='wday')
month_input = Input(shape=(1,), name='month')
year_input = Input(shape=(1,), name='year')
event_name_1_input = Input(shape=(1,), name='event_name_1')
event_type_1_input = Input(shape=(1,), name='event_type_1')
item_id_input = Input(shape=(1,), name='item_id')
dept_id_input = Input(shape=(1,), name='dept_id')
store_id_input = Input(shape=(1,), name='store_id')

# Embedding layers
wday_emb = Flatten()(Embedding(7, 3, )(wday_input))
month_emb = Flatten()(Embedding(12, 3)(month_input))
year_emb = Flatten()(Embedding(6, 3)(year_input))
event_name_1_emb = Flatten()(Embedding(31, 5)(event_name_1_input))
event_type_1_emb = Flatten()(Embedding(5, 2)(event_type_1_input))
item_id_emb = Flatten()(Embedding(len(encode_item_id.categories_[0]), 50)(item_id_input))
item_id_emb = Dense(20, activation="relu", kernel_regularizer=regularizers.l2(0.01))(item_id_emb)
dept_id_emb = Flatten()(Embedding(7, 3)(dept_id_input))
store_id_emb = Flatten()(Embedding(10, 4)(store_id_input))

x = concatenate([dense_branch, wday_emb, month_emb, year_emb, event_name_1_emb,
                event_type_1_emb, item_id_emb, dept_id_emb, store_id_emb])
x = Dense(100, activation="relu")(x)
x = Dense(20, activation="relu")(x)
prediction = Dense(1, activation="linear", name='output')(x)

model = Model(inputs={"dense1": dense_input, "wday": wday_input, "month": month_input,
                      "year": year_input, "event_name_1": event_name_1_input, "event_type_1": event_type_1_input,
                      "item_id": item_id_input, "dept_id": dept_id_input, "store_id": store_id_input},
              outputs=prediction)


# In[ ]:


model.summary()

keras.utils.plot_model(model, 'model.png', show_shapes=True)


# In[ ]:


model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['mse'])

history = model.fit(X_train, 
                    y_train,
                    batch_size=4096,
                    epochs=10,
                    validation_split=0.1)


# In[ ]:


# Plot the evaluation metrics over epochs


# In[ ]:


model.save('modelM5.h5')


# ## Submission

# In[ ]:


pred = model.predict(X_test, batch_size=4096)


# In[ ]:


pred.shape


# In[ ]:


submission.shape


# In[ ]:


submission.tail()


# In[ ]:


submission = submission.assign(
    demand = np.clip(pred, 0, None),
    d = lambda df: map_d2f(df.d, df.id))
submission.head()


# In[ ]:


# Right column order
col_order = ["id"] + ["F" + str(i + 1) for i in range(28)]
submission = submission.pivot(index="id", columns="d", values="demand").reset_index()[col_order]

# Right row order
submission = sample_submission[["id"]].merge(submission, how="left", on="id")


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv("submission.csv", index=False)

