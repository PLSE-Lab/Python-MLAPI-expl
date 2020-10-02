#!/usr/bin/env python
# coding: utf-8

# This is part of an experiment:  since the full run times out on Kaggle, how about running two halves separately and then combining the results?  We'll see.  This is the second half.  (The is the second half of stage 2.  Stage 1 was data preparation, as described in the next paragraph.  Stage 3 will be combining the results from stage 2.)

# Running a neural network using the data prepared by [this script](https://www.kaggle.com/aharless/preparing-data-for-lgbm-or-something-else/output).  The data represent the fitting and validation scheme derived from Ceshine's [LGBM starter](https://www.kaggle.com/ceshine/lgbm-starter) script with the additional features created in [Lingzhi's upgraded version](https://www.kaggle.com/vrtjso/lgbm-one-step-ahead).  So far I've made no attempt to preprocess them further for optimal use by the neural network.  (I used [kaggleslayer's code](https://www.kaggle.com/kaggleslayer/grocery-prediction-with-neural-network) as a starting point for the structure of the network itself, but now I've made some revisions.)
# 

# In[ ]:


N_EPOCHS = 10
N_DAYS = 8  # Limit to first half


# In[ ]:


from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


indir = '../input/preparing-data-for-lgbm-or-something-else/'
indir2 = '../input/favorita-grocery-sales-forecasting/'


# In[ ]:


X_test = pd.read_csv(indir + 'X_test.csv')
X_val = pd.read_csv(indir + 'X_val.csv')
X_train = pd.read_csv(indir + 'X_train.csv')
y_train = np.array(pd.read_csv(indir + 'y_train.csv'))
y_val = np.array(pd.read_csv(indir + 'y_val.csv'))
stores_items = pd.read_csv(indir + 'stores_items.csv', index_col=['store_nbr','item_nbr'])
test_ids = pd.read_csv( indir + 'test_ids.csv',  parse_dates=['date']).set_index(
                        ['store_nbr', 'item_nbr', 'date'] )
items = pd.read_csv( indir2 + 'items.csv' ).set_index("item_nbr")
items = items.reindex( stores_items.index.get_level_values(1) )


# In[ ]:


model = Sequential()
model.add(Dense(32, kernel_initializer='normal', input_shape=(X_train.shape[1],)))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(.25))
model.add(Dense(16, kernel_initializer='normal'))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(.18))
model.add(Dense(8, kernel_initializer='normal'))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(.12))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])


# Note: See this [stack overflow post](https://stackoverflow.com/questions/47802601/nonetype-error-on-saving-model-in-keras) on the problem that led me to comment out the early stopping code in the block below.

# In[ ]:


val_pred = []
test_pred = []
# wtpath = 'weights.hdf5'  # To save best epoch. But need Keras bug to be fixed first.
sample_weights=np.array( pd.concat([items["perishable"]] * 6) * 0.25 + 1 )
for i in range(N_DAYS, 16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    y = y_train[:, i]
    xv = np.array(X_val)
    yv = y_val[:, i]
#    bestepoch = ModelCheckpoint( filepath=wtpath, verbose=1, save_best_only=True )
    model.fit( np.array(X_train), y, batch_size = 128, epochs = N_EPOCHS, verbose=2,
               sample_weight=sample_weights, validation_data=(xv,yv) ) 
             #, callbacks=[bestepoch] # bestepoch doesn't work: keras bug
#    model.load_weights( wtpath )
    val_pred.append(model.predict(X_val))
    test_pred.append(model.predict(X_test))


# In[ ]:


weights=pd.concat([items["perishable"]]) * 0.25 + 1
print("Unweighted validation mse: ", mean_squared_error(
    y_val[:,N_DAYS:], np.array(val_pred).squeeze(axis=2).transpose()) )
print("Partial weighted mse:      ", mean_squared_error(
    y_val[:,N_DAYS:], np.array(val_pred).squeeze(axis=2).transpose(), sample_weight=weights) )


# In[ ]:


pd.DataFrame(np.array(val_pred).squeeze(axis=2).transpose()).to_csv('nn_val_second_half.csv', 
                                                                    float_format='%.5f', 
                                                                    index=None)


# In[ ]:


START = "2017-08-24"
days = 16 - N_DAYS
y_test = np.array(test_pred).squeeze(axis=2).transpose()
df_preds = pd.DataFrame(
    y_test, index=stores_items.index,
    columns=pd.date_range(START, periods=days)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)


# In[ ]:


submission = test_ids.join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('nn_sub_second_half.csv', float_format='%.5f', index=None)

