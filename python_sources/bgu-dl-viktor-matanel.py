#!/usr/bin/env python
# coding: utf-8

# # THIS ISNT THE FINAL VERSION!!!!!
# # We were unable to revert to the final version , to see the running outcome please look at version 3

# In[ ]:


import pandas as pd
import numpy as np
import os
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Embedding, Dense, Dropout, concatenate, Flatten, LSTM, BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from sklearn.linear_model import LinearRegression


# In[ ]:


def plt_model(model_hist):
        acc = model_hist.history['mean_squared_error']
        val_acc = model_hist.history['val_mean_squared_error']
        loss = model_hist.history['loss']
        val_loss = model_hist.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(15, 6));
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, color='green', label='Training MSE')
        plt.plot(epochs, val_acc, color='blue', label='Validation MSE')
        plt.title('Training and Validation MSE')
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, color='green', label='Training loss')
        plt.plot(epochs, val_loss, color='blue', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.show()


# Loading the data from CSV files
# 

# In[ ]:


df_sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
df_item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
df_items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
df_sample_submission=pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")
df_shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
df_test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')


# Lets see the shape of our data frames
# 

# In[ ]:


print(f'df_sales shape: {df_sales.shape}')
print(f'df_items shape: {df_items.shape}')
print(f'df_item_categories shape: {df_item_categories.shape}')
print(f'df_shops shape: {df_shops.shape}')
print(f'df_test shape: {df_test.shape}')


# We would also like to see the columns in each data frame

# In[ ]:


print(f'df_sales structure\n\n{df_sales.head(3)}\n\n\n')
print(f'df_items structure\n\n{df_items.head(3)}\n\n\n')
print(f'df_item_categories structure\n\n{df_item_categories.head(3)}\n\n\n')
print(f'df_shops structure\n\n{df_shops.head(3)}\n\n\n')
print(f'df_test structure\n\n{df_test.head(3)}\n\n\n')
print(f'df_test structure\n\n{df_sample_submission.head(3)}\n\n\n')


# Lets perform some data preproccessing prior to training our model

# Since the data we have shows the sales per day, we would like to aggregate it into sales pre month

# In[ ]:


df_month_sales = df_sales.groupby(['date_block_num', "shop_id", "item_id"]).agg({"item_cnt_day" : 'sum'}).reset_index()
df_month_sales = df_month_sales.rename(columns={'item_cnt_day': 'sales_per_month'})


# Lets see how many total sales per month we have

# In[ ]:


df_total_month_sales = df_month_sales.groupby(['date_block_num']).agg({'sales_per_month' : 'sum'}).reset_index()
df_total_month_sales = df_total_month_sales.rename(columns={'sales_per_month': 'total_sales_per_month'})
plt.figure(num=None, figsize=(10, 8), dpi=80)
plt.bar(df_total_month_sales['date_block_num'], df_total_month_sales['total_sales_per_month'])


# Since we will predict a number between 0 to 20, we will clip all values to match that range

# In[ ]:


df_month_sales['sales_per_month'] = df_month_sales['sales_per_month'].clip(0,20)


# We can see that our orignal data does not include days in which 0 items were sold.
# This is a problem because we might be missing some months where there were no items sold, we would like to add those months.

# In[ ]:


cols = ["date_block_num", "shop_id", "item_id"]

all_months = pd.DataFrame(np.array(list(product(range(34), df_sales['shop_id'].unique(), df_sales['item_id'].unique()))), columns = cols)

df_month_sales = pd.merge(all_months, df_month_sales, on=cols, how='left').fillna(0)




# In[ ]:


df_month_sales = df_month_sales.sort_values(by=cols)


# We add item category and previous month sales aggregation for better results

# In[ ]:


sales_for_month_0 = df_month_sales[df_month_sales['date_block_num'] == 0]
sales_for_month_0['sales_for_previous_month'] = 0
sales_for_month_0 = pd.merge(sales_for_month_0, df_items[['item_id', 'item_category_id']], on=['item_id'], how='left')
dfs = [sales_for_month_0]
for i in range(1,34):
    sales_for_month_i = df_month_sales[df_month_sales['date_block_num'] == i]
    sales_for_month_i['sales_for_previous_month'] = dfs[i-1]['sales_per_month'].to_numpy()
    sales_for_month_i = pd.merge(sales_for_month_i, df_items[['item_id', 'item_category_id']], on=['item_id'], how='left').reset_index()
    dfs.append(sales_for_month_i)
    
df_month_sales = pd.concat(dfs)
df_month_sales


# Changing the test data to match the training

# In[ ]:


fixed_test = df_test[['shop_id', 'item_id']]
fixed_test = pd.merge(fixed_test, df_items[['item_id', 'item_category_id']], on=['item_id'], how='left')
sales_for_month_33 = df_month_sales[df_month_sales['date_block_num'] == 33][['item_id', 'shop_id', 'sales_per_month']]
sales_for_month_33.rename(columns={'sales_per_month':'sales_for_previous_month'}, inplace=True)
fixed_test = pd.merge(fixed_test, sales_for_month_33, on=['item_id', 'shop_id'], how='left').fillna(0)
fixed_test.insert(0, 'date_block_num', 34) 




# 1. We split our training data to train and validation, 20% for validation

# In[ ]:


cols = ["date_block_num", "shop_id", "item_id", "item_category_id", "sales_for_previous_month"]
data_train = df_month_sales[df_month_sales['date_block_num']%5 != 0]
data_val = df_month_sales[df_month_sales['date_block_num']%5 == 0]

X_train = data_train[cols]
y_train = data_train['sales_per_month']

X_val = data_val[cols]
y_val = data_val['sales_per_month']


# We will use a regressor to set the ML benchmark

# In[ ]:


dt = DecisionTreeRegressor()
dt.fit(X_train,y_train)


# In[ ]:


preds = dt.predict(X_val)
rmse = sqrt(mean_squared_error(y_val, preds))
print(rmse)


# In[ ]:


preds_test = dt.predict(fixed_test)


# In[ ]:


df_sample_submission['item_cnt_month'] = preds_test
df_sample_submission.to_csv('./data/first_submission.csv', index=False)


# ![first_submission.PNG](attachment:first_submission.PNG)

# Lets try improving our results by using embedding

# Inputs

# In[ ]:


item_input = Input(shape=(1,), dtype='int64', name='item_input')
months_input = Input(shape=(1,), dtype='int64', name='months_input')
shops_input = Input(shape=(1,), dtype='int64', name='shops_input')
item_category_input = Input(shape=(1,), dtype='int64', name='item_category_input')
previous_month_input = Input(shape=(1,), dtype='int64', name='previous_month_input')


# In[ ]:


We will use a simple model, without LSTM 


# In[ ]:


num_of_items = max(data_train['item_id'])+1
num_of_months = max(data_train['date_block_num'])+1
num_of_shops = max(data_train['shop_id'])+1
num_of_categories = max(data_train['item_category_id'])+1
i = Embedding(num_of_items,int(sqrt(num_of_items)), input_length=1, embeddings_regularizer=l2(1e-4))(item_input)
m = Embedding(num_of_months, int(sqrt(num_of_months)), input_length=1, embeddings_regularizer=l2(1e-4))(months_input)
s = Embedding(num_of_shops, int(sqrt(num_of_shops)), input_length=1, embeddings_regularizer=l2(1e-4))(shops_input)
ic = Embedding(num_of_categories, int(sqrt(num_of_categories)), input_length=1, embeddings_regularizer=l2(1e-4))(item_category_input)
pm = Embedding(20, int(sqrt(20)), input_length=1, embeddings_regularizer=l2(1e-4))(previous_month_input)
x = concatenate([m,s,i,ic, pm])
x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(1)(x)
nn = Model([months_input,shops_input,item_input,item_category_input, previous_month_input],x)
nn.compile(optimizer=Adam(0.001), loss='mse',metrics=["mean_squared_error"])
es = EarlyStopping(monitor='val_mean_squared_error', mode='min', verbose=1, patience=2)
checkpoint = ModelCheckpoint('model3.hdf5' , monitor='val_mean_squared_error', mode='min', save_best_only=True)
history = nn.fit([X_train['date_block_num'],X_train['shop_id'],X_train['item_id'],X_train['item_category_id'], X_train['sales_for_previous_month']], y_train, batch_size=4096, epochs=8, 
          validation_data=([X_val['date_block_num'],X_val['shop_id'],X_val['item_id'],X_val['item_category_id'], X_val['sales_for_previous_month']], y_val),callbacks=[es, checkpoint])


# In[ ]:


nn.summary()


# In[ ]:


plt_model(history)


# A method to plot the model information
# 

# In[ ]:


preds = nn.predict([fixed_test['date_block_num'],fixed_test['shop_id'],fixed_test['item_id'],fixed_test['item_category_id'], fixed_test['sales_for_previous_month']])


# In[ ]:


df_sample_submission['item_cnt_month'] = preds
df_sample_submission.to_csv('second_submission.csv', index=False)


# ![second_submission.PNG](attachment:second_submission.PNG)
# 

# [](http://)Now lets add LSTM layer and wee if it improves our results
# 

# In[ ]:


item_input = Input(shape=(1,), dtype='int64', name='item_input')
months_input = Input(shape=(1,), dtype='int64', name='months_input')
shops_input = Input(shape=(1,), dtype='int64', name='shops_input')
item_category_input = Input(shape=(1,), dtype='int64', name='item_category_input')
previous_month_input = Input(shape=(1,), dtype='int64', name='previous_month_input')


# In[ ]:


num_of_items = max(data_train['item_id'])+1
num_of_months = max(data_train['date_block_num'])+1
num_of_shops = max(data_train['shop_id'])+1
num_of_categories = max(data_train['item_category_id'])+1
i = Embedding(num_of_items,int(sqrt(num_of_items)), input_length=1, embeddings_regularizer=l2(1e-4))(item_input)
m = Embedding(num_of_months, int(sqrt(num_of_months)), input_length=1, embeddings_regularizer=l2(1e-4))(months_input)
s = Embedding(num_of_shops, int(sqrt(num_of_shops)), input_length=1, embeddings_regularizer=l2(1e-4))(shops_input)
ic = Embedding(num_of_categories, int(sqrt(num_of_categories)), input_length=1, embeddings_regularizer=l2(1e-4))(item_category_input)
pm = Embedding(20, int(sqrt(20)), input_length=1, embeddings_regularizer=l2(1e-4))(previous_month_input)
x = concatenate([m,s,i,ic,pm])
x = BatchNormalization()(x)
x = LSTM(32)(x)
x = Dropout(0.2)(x)
x = Dense(1)(x)
nn = Model([months_input,shops_input,item_input,item_category_input, previous_month_input],x)
nn.compile(optimizer=Adam(0.0001), loss='mse',metrics=["mean_squared_error"])


# In[ ]:


es = EarlyStopping(monitor='val_mean_squared_error', mode='min', verbose=1, patience=2)
checkpoint = ModelCheckpoint('model4.hdf5' , monitor='val_mean_squared_error', mode='min', save_best_only=True)
history1 = nn.fit([X_train['date_block_num'],X_train['shop_id'],X_train['item_id'],X_train['item_category_id'], X_train['sales_for_previous_month']], y_train, batch_size=4096, epochs=8, 
          validation_data=([X_val['date_block_num'],X_val['shop_id'],X_val['item_id'],X_val['item_category_id'], X_val['sales_for_previous_month']], y_val),callbacks=[es, checkpoint])


# In[ ]:


nn.load_weights('model4.hdf5')


# In[ ]:


nn.summary()


# In[ ]:


plt_model(history1)


# In[ ]:


preds = nn.predict([fixed_test['date_block_num'],fixed_test['shop_id'],fixed_test['item_id'],fixed_test['item_category_id'], fixed_test['sales_for_previous_month']])
df_sample_submission['item_cnt_month'] = preds
df_sample_submission.to_csv('final_submission.csv', index=False)


# ![third_submission.PNG](attachment:third_submission.PNG)
# 

# Now we will use feature extraction

# In[ ]:


item_input = Input(shape=(1,), dtype='int64', name='item_input')
months_input = Input(shape=(1,), dtype='int64', name='months_input')
shops_input = Input(shape=(1,), dtype='int64', name='shops_input')
item_category_input = Input(shape=(1,), dtype='int64', name='item_category_input')
previous_month_input = Input(shape=(1,), dtype='int64', name='previous_month_input')


# In[ ]:


extractor = Model(input=nn.input, output=nn.get_layer("lstm_1").output)

lr = LinearRegression(n_jobs=-1)
print('predicting training features')
train_preds = extractor.predict([X_train['date_block_num'],X_train['shop_id'],X_train['item_id'],X_train['item_category_id'], X_train['sales_for_previous_month']])

print("Training ml model with features")
lr.fit(train_preds, y_train)


# In[ ]:


print("Predicting features for validation")
val_preds = extractor.predict([X_val['date_block_num'],X_val['shop_id'],X_val['item_id'],X_val['item_category_id'], X_val['sales_for_previous_month']])
print("Predicting validation data")
preds = lr.predict(val_preds)
rmse = sqrt(mean_squared_error(y_val, preds))
print(rmse)


# # Final thoughts
# 1. We were able to improve out rmse greatly by adding extra data by filling all of the months with sales (in the original dataset, months with no sales had no corresponding rows)
#    This change, on the other hand, led to big differences between the train+validation results to the test results. (thats because calculating mse when having lows of zeros in our
#    data makes it a lo).
#    We decided to keep this change, even though it messes with the training rmse, because eventually the test rmse was much better.
# 2. At first we only used the basic categorical features but then we realized that adding features like sales for previous month and item category improve our results.
#    We also believe that adding more features like sales per month (not per store) and year could improve our results even further.
#    The most important part is finding the correct categorical features to rely upon.
# 3. We saw an improvement using LSTM layer, which yielded our best result:
# ![best.PNG](attachment:best.PNG)
# 

# 
