#!/usr/bin/env python
# coding: utf-8

# # Electricity - Demo training of a feed-forward neural network in TensorFlow
# 
# In this tutorial:
# 
#   1. We read the available labelled data from the given csv.
#   2. We will add some extra features related to date and time.
#   3. We will transofrm the data into PyTorch tensors and normalize along some input dimensions.
#   4. Prepare a small neural network.
#   5. Optimize the neural network's parameters to minimize the RMSE.
# 
# **Be careful!**
# 
# We will use some global variables (ugly, but convenient when using notebooks) such as: `df`, `train_data`, `train_labels`, `model`, `optimizer`, etc.

# In[ ]:


import tensorflow as tf


# In[ ]:


## 1. Reading data into a pandas DataFrame, and inspecting the columns a bit

import pandas as pd

df = pd.read_csv("../input/train_electricity.csv")  # <-- only this is important

print("Dataset has", len(df), "entries.")

print(f"\n\t{'Column':20s} | {'Type':8s} | {'Min':12s} | {'Max':12s}\n")
for col_name in df.columns:
    col = df[col_name]
    print(f"\t{col_name:20s} | {str(col.dtype):8s} | {col.min():12.1f} | {col.max():12.1f}")


# In[ ]:


import numpy as np

print("You shold score better than:")
print("RMS distance between Production_MW and Consumption_MW")
np.sqrt(np.mean((df["Production_MW"] - df["Consumption_MW"]) ** 2))


# In[ ]:


## 2. Adding some datetime related features

def add_datetime_features(df):
    features = ["Year", "Week", "Day", "Dayofyear", "Month", "Dayofweek",
                "Is_year_end", "Is_year_start", "Is_month_end", "Is_month_start",
                "Hour", "Minute", "Quarter"]
    one_hot_features = ["Month", "Dayofweek", "Quarter"]

    datetime = pd.to_datetime(df.Date * (10 ** 9))

    df['Datetime'] = datetime  # <-- We won't use this for training, but we'll remove it later

    for feature in features:
        new_column = getattr(datetime.dt, feature.lower())
        if feature in one_hot_features:
            df = pd.concat([df, pd.get_dummies(new_column, prefix=feature)], axis=1)
        else:
            df[feature] = new_column
            
    return df

df = add_datetime_features(df)


df.columns


# In[ ]:


# Drop some columns

to_drop = ['Coal_MW', 'Gas_MW', 'Hidroelectric_MW', 'Nuclear_MW', 'Wind_MW', 'Solar_MW', 'Biomass_MW']
df.drop(columns=to_drop, inplace=True)


# In[ ]:


## 3. Split data into train / validation (leaving the last six months for validation)

from dateutil.relativedelta import relativedelta
import numpy as np

eval_from = df['Datetime'].max() + relativedelta(months=-6)  # Here we set the 6 months threshold
train_df = df[df['Datetime'] < eval_from].copy()
valid_df = df[df['Datetime'] >= eval_from].copy()

print(f"Train data: {train_df['Datetime'].min()} -> {train_df['Datetime'].max()} | {len(train_df)} samples.")
print(f"Valid data: {valid_df['Datetime'].min()} -> {valid_df['Datetime'].max()} | {len(valid_df)} samples.")
      
target_col = "Consumption_MW"
to_drop = ["Date", "Datetime", target_col]

# Create torch tensors with inputs / labels for both train / validation 
      
train_data = train_df.drop(columns=to_drop).values.astype(np.float)
valid_data = valid_df.drop(columns=to_drop).values.astype(np.float)
train_labels = train_df[target_col].values[:, None].astype(np.float)
valid_labels = valid_df[target_col].values[:, None].astype(np.float)


# In[ ]:


## 4. Normalize features (except one-hot ones)  - uncomment if you want to normalize this


# for idx in range(train_data.shape[1]):
#     if (train_data[:, idx].min() < 0 or train_data[:, idx].max() > 1):
#         mean, std = train_data[:, idx].mean(), train_data[:, idx].std()
#         train_data[:, idx] -= mean
#         train_data[:, idx] /= std
#         valid_data[:, idx] -= mean
#         valid_data[:, idx] /= std


# In[ ]:


## 5. Prepare a simple model
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, ReLU, LSTMCell,     BatchNormalization, Dropout

HISTORY = 10  # How many time steps to look back into the recent history y_t = f(x_t-H+1, x_t-H+2, ..., x_t)

nfeatures = train_data.shape[1]

class FeedForward(Model):
    
    def __init__(self):
        super().__init__()
        self.training = tf.placeholder(tf.bool)
        self.model = Sequential([
            Dense(300, input_shape = (nfeatures * HISTORY,)),
            ReLU(),
            Dropout(0.5),
            Dense(300),
            ReLU(),
            Dropout(0.5),
            Dense(100),
            ReLU(),
            Dense(1)
        ])
        
    def call(self, x):
        x = tf.reshape(x, [-1, nfeatures * HISTORY])
        y = self.model(x, training = self.training)
        return y + x[:, nfeatures * (HISTORY - 1):nfeatures * (HISTORY - 1) + 1]
    
    
class RNN(Model):
    
    def __init__(self):
        super().__init__()
        self.training = tf.placeholder(tf.bool)
        self.hsize = hsz = 50
        
        self.lstm = LSTMCell(hsz, input_shape=(nfeatures,))
        self.head = Sequential([
            Dense(100, input_shape = (hsz,)),
            ReLU(),
            Dense(1)
        ])
        
    def call(self, x):
        x = tf.transpose(x, [1,0,2])
        hidden = self.lstm.get_initial_state(batch_size = tf.shape(x)[1],
                                             dtype=tf.float32)
        for i in range(x.shape[0]):
            o, hidden = self.lstm(x[i], hidden)
        return self.head(o) + x[-1, :, 0:1]


s = tf.InteractiveSession()   
model = FeedForward()

input_ = tf.placeholder(tf.float32, [None, HISTORY, nfeatures])
label_ = tf.placeholder(tf.float32, [None,1])

output = model(input_)
loss = tf.reduce_mean(tf.square(output-label_))

optim = tf.train.AdamOptimizer(learning_rate = 0.001)
optim_step = optim.minimize(loss)

s.run(tf.global_variables_initializer())


# In[ ]:


# Here we write a validation routine. Observation: `model`, `valid_data`, `valid_labels` are global variables.

VALID_BATCH_SIZE = 5000

def validate():
    nvalid = len(valid_data) - HISTORY + 1  # This is the number of validation examples
    losses = []

  
    for start_idx in range(0, nvalid, VALID_BATCH_SIZE):
        end_idx = min(nvalid, start_idx + VALID_BATCH_SIZE)
        idxs = np.arange(start_idx, end_idx)
        all_idxs = (idxs[:,None] + np.arange(HISTORY)[None,:])
        data = valid_data[all_idxs]
        label = valid_labels[idxs + HISTORY -1]
        l = s.run(loss, feed_dict={input_: data,
                                  label_: label,
                                  model.training: False})
        losses.append(l)
    return np.sqrt(np.mean(np.array(losses)))


# In[ ]:


## 5. Train the model

STEPS_NO = 10000
REPORT_FREQ = 250
BATCH_SIZE = 64

nexamples = len(train_data) - HISTORY + 1

train_losses = []

for step in range(1, STEPS_NO + 1):
    # prepare batch: sample i=t-H+1 to concat x_{t-H+1} .. x_t for input, and y_t for label
    
    idxs = np.random.randint(nexamples, size=(BATCH_SIZE,))
    all_idxs = idxs[:, None] + np.arange(HISTORY)[None, :]
    
    data = train_data[all_idxs]
    label = train_labels[idxs + HISTORY - 1]
    
    # optimize using current batch
    _, l = s.run([optim_step, loss], feed_dict={input_: data,
                                                label_: label,
                                                model.training: True})

    train_losses.append(l)
    # report and monitor training
    if step % REPORT_FREQ == 0:
        valid_loss = validate()
        print(f"Step {step:4d}: Train RMSE={np.sqrt(np.mean(train_losses)):7.2f} | Valid RMSE={valid_loss:7.2f}")
        train_losses.clear()
    


# In[ ]:





# In[ ]:




