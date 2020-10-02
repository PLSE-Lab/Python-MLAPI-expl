#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Loading Data

# In this competition, we need to predict the sales for [d_1942 - d_1969]. These rows form the evaluation set. The rows [d_1914 - d_1941] form the validation set, and the remaining rows form the training set. Now since we understand the dataset and know what to predict, let us visualize the dataset.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPool1D, Dropout
from tensorflow.keras.utils import to_categorical 

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from numpy import hstack, array

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# In[ ]:


sample_submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")


# ##### Calendar - Dates on which products sold.

# In[ ]:


cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
cal.shape


# In[ ]:


cal.head()


# ##### Sales Data

# 1. id: Validation ID
# 
# 2. item_id: Item ID
# 
# 3. cat_id: Category (Hobbies, Household, Foods)
# 
# 4. store_id: Store ID
# 
# 5. state_id: State (CA, TX, WI)

# In[ ]:


train = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")
train.shape


# In[ ]:


train.head(10)


# ##### Sales Price

# In[ ]:


sale = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
sale.shape


# In[ ]:


sale.head()


# # EDA (Exploratory Data Analysis)

# In[ ]:


ids = [i for i in range(train["id"].shape[0])]
len(ids)


# In[ ]:


# Only numerical data taken from train data

num = train.select_dtypes(exclude = ["object"]).columns
print(num)


# In[ ]:


train['item_sum'] = train[num].sum(axis=1)
train.head()


# In[ ]:


mix = pd.concat([train, cal], axis =1)
mix.head()


# #### Number of sales of each item vs Store

# In[ ]:


plt.figure(figsize=(10,6))
plt.title("Total Sale of Items in Store")

sns.barplot(x = mix["store_id"], y = mix["item_sum"])


# In[ ]:


plt.figure(figsize=(10,6))
plt.title("Total Sale of Categorical Items in Stores")

sns.barplot(x = mix["store_id"], y = mix["item_sum"], hue = mix["cat_id"])


# #### Using Calendar to analyse time series data

# In[ ]:


plt.figure(figsize=(10,6))
plt.title("Sale of Item by Year and Category")

sns.lineplot(x = mix["year"], y = mix["item_sum"], hue = mix["cat_id"])


# In[ ]:


plt.figure(figsize=(10,6))
plt.title("Sale of items by Month")

sns.lineplot(x = mix["month"], y = mix["item_sum"], hue = mix["cat_id"])


# In[ ]:


plt.figure(figsize=(10,6))
plt.title("Sale of items by Month")

sns.barplot(x = mix["month"], y = mix["item_sum"], hue = mix["cat_id"])


# In[ ]:


plt.figure(figsize=(10,6))
plt.title("Sale of items by Day of Week")

sns.lineplot(x = mix["weekday"], y = mix["item_sum"], hue = mix["cat_id"])


# ##### Results
# 
# 1. Stores in California perform much better than those in Texas and Wisconsin. CA_3 outperforms all.
# 2. Food Items are the highest selling products.
# 3. First two months and the last 4 months have much higher sale record.
# 4. Sunday and Thursday are the days when sales peak the most.

# # Preparation of Data for prediction

# In[ ]:


train.head()


# In[ ]:


# item_id = train['item_id']
# del train['item_id']
# del train['dept_id']
# del train['id']


# In[ ]:


# train = train.drop(['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'], axis=1)
# train = train.drop(['snap_CA', 'snap_TX', 'snap_WI', 'date', 'wm_yr_wk'], axis=1)


# In[ ]:


# train = train.drop(['item_sum'], axis=1)


# In[ ]:


# train["cat_id"] = train["cat_id"].apply(cat)


# In[ ]:


def cat(str):
    if str == 'HOBBIES':
        return 1
    elif str == 'HOUSEHOLD':
        return 2
    else:
        return 3


# In[ ]:


def melt_sales(df):
    df = df.drop(["item_id", "dept_id", "cat_id", "store_id", "state_id", "item_sum"], axis=1).melt(
        id_vars=['id'], var_name='d', value_name='demand')
    return df

sales = melt_sales(train)


# In[ ]:


sales.head()


# In[ ]:


sales_trend = train.drop(columns = ['id','item_id','dept_id','cat_id','store_id','state_id', 'item_sum']).mean().reset_index()
sales_trend.plot()


# In[ ]:


sales_trend.rename(columns={'index':'d', 0: 'sales'}, inplace=True)
sales_trend = sales_trend.merge(cal[["wday","month","year","d"]], on="d",how='left')
sales_trend = sales_trend.drop(columns = ["d"])


# In[ ]:


sales_trend.head()


# In[ ]:


sales.head()


# In[ ]:


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# In[ ]:


in_seq1 = np.array(sales_trend['wday'])
in_seq2 = np.array(sales_trend['month'])
in_seq3 = np.array(sales_trend['year'])
out_seq = np.array(sales_trend['sales'])
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2, in_seq3, out_seq))
n_steps = 7
X, y = split_sequences(dataset, n_steps)


# In[ ]:


X = X
Y = y


# In[ ]:


X_train = X[:-30]
Y_train = Y[:-30]
X_test = X[-30:]
Y_test = Y[-30:]


# In[ ]:


X_train.shape


# In[ ]:


for i in range(Y_train.shape[0]):
    if Y_train[i] >=1 :
        Y_train[i] = 1
    else:
        Y_train[i] = 0


# In[ ]:


for i in range(Y_test.shape[0]):
    if Y_test[i] >=1 :
        Y_test[i] = 1
    else:
        Y_test[i] = 0


# In[ ]:


n_features = X_train.shape[2]


# # Training Model

# In[ ]:


model = Sequential()


# In[ ]:


model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPool1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='tanh'))
model.add(Dense(1, activation = "softmax"))


# In[ ]:


model.compile(optimizer = 'adam', loss = 'mse', metrics = ["accuracy"])


# In[ ]:


model.fit(X_train, Y_train, epochs = 100, batch_size = 50)


# In[ ]:


predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print(score)

