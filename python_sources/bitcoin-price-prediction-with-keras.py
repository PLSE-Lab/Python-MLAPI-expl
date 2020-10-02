#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Import libraries\nimport numpy as np # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\nfrom plotly.subplots import make_subplots\nimport plotly.graph_objects as go\n\nfrom keras.models import Sequential\nfrom keras.layers import LSTM,Dense,Dropout,AveragePooling1D,Reshape\n\nfrom sklearn.metrics import mean_absolute_error')


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Import data\ndata = pd.read_csv('../input/historical-data-on-the-trading-of-cryptocurrencies/crypto_tradinds.csv')\ndata.tail()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Select Bitcoin Data\nbtc_data = data[data['ticker']=='BTC']\nbtc_data.tail()")


# In[ ]:


get_ipython().run_cell_magic('time', '', '#Check data\nbtc_data.nunique()')


# In[ ]:


get_ipython().run_cell_magic('time', '', "btc_data['price_btc'].unique()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Check value 0 in 'price_btc'\nbtc_data_0 = btc_data[data['price_btc']==0]\nbtc_data_0.tail()")


# Previous code return only 1 row. Value '0' in column 'price_btc' for BTC must be mistake in dataset.

# In[ ]:


get_ipython().run_cell_magic('time', '', "#Drop columns with 1 value (and 'price_btc' with one mistake)\ndrop_columns_list = btc_data.nunique()[btc_data.nunique()<=2].index\nbtc_data.drop(drop_columns_list, axis=1, inplace=True)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Print full graph of bitcoin price\nfig = go.Figure(data=go.Scatter(x=btc_data['trade_date'], y=btc_data['price_usd']))\nfig.show()")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# fix random seed for reproducibility\nnp.random.seed(42)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "def data_preproc_and_split(data,n):\n    #define variables\n    col = []\n    for i in range(n):\n        col.append('price' + str(i))\n        col.append('volume' + str(i))\n    train = pd.DataFrame(columns = col)\n    target = pd.DataFrame(columns = ['date','price'])\n    pred_convert = pd.DataFrame(columns = ['date','price'])\n    \n    #Preprocessing of data\n    for i in range(1,len(data)-n-1):\n        def_nom = data.loc[i-1, 'price_usd']\n        for j in range(n):\n            train.loc[i, 'price' + str(j)] = data.loc[i+j, 'price_usd']/def_nom-1\n            train.loc[i, 'volume' + str(j)] = data.loc[i+j, 'volume']/data.loc[i+j, 'market_cap']\n        target.loc[i, 'price'] = data.loc[i+n+1, 'price_usd']/def_nom-1\n        target.loc[i, 'date'] = data.loc[i+n+1, 'trade_date']  \n        #Save start prices for convertation prediction resalt to valid prices\n        pred_convert.loc[i, 'price'] = def_nom\n        pred_convert.loc[i, 'date'] = data.loc[i+n+1, 'trade_date'] \n\n    #Data split\n    x_train = train.iloc[:train.shape[0]-100]\n    x_valid = train.iloc[train.shape[0]-100:]\n    y_train = target.iloc[:target.shape[0]-100]\n    y_valid = target.iloc[target.shape[0]-100:]\n    y_train.drop(['date'], axis=1, inplace=True)\n    y_valid.drop(['date'], axis=1, inplace=True)\n    \n    #Convert shape of data for LSTM model\n    x_train = x_train.to_numpy().reshape((x_train.shape[0],n,2))\n    x_valid = x_valid.to_numpy().reshape((x_valid.shape[0],n,2))\n    return x_train,x_valid,y_train,y_valid,target,pred_convert")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#model\ndef model_gen(x,y,n,e=75,v=0):\n    mod = Sequential()\n    mod.add(LSTM(32,return_sequences=True,input_shape=(n,2)))\n    mod.add(LSTM(64))\n    mod.add(Dropout(0.35))\n    mod.add(Dense(128, activation='relu'))\n    mod.add(Dense(1))\n    mod.compile(optimizer='adam',loss='mse')\n    mod.fit(x,y,epochs=e,shuffle=False,verbose=v)\n    return mod")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Tuning, step 1: found optimal epoch for model\nx_train,x_valid,y_train,y_valid,target,pred_convert = data_preproc_and_split(btc_data,15) #Create necessary datasets from start data (preprocessing and split)\ny_pred = {}\nmae = {'before_convert': {}, 'after_convert': {}}\nR = range(25,130,25)\nfor i in R:\n    #print (i)\n    model = model_gen(x_train,y_train,15,i,0) #Model generation without output\n    preds = model.predict(x_valid) #Prediction\n    y_pred[i] = pd.DataFrame(preds, index=y_valid.index, columns = ['price']) #Create DataFrame from prediction results\n    y_pred[i]['date'] = target['date'] #Add date column to results\n    mae['before_convert'][i] = mean_absolute_error(y_valid['price'],y_pred[i]['price']) #Save Mean absolute error before price convertation\n    y_pred[i]['price'] = pred_convert['price']*(y_pred[i]['price']+1) #Convert prediction results to valid price\n    mae['after_convert'][i] = mean_absolute_error(btc_data.iloc[btc_data.shape[0]-100:]['price_usd'],y_pred[i]['price']) #Save Mean absolute error after price convertation\n\n#Print results of prediction\nfig = go.Figure()\nfig.add_trace(go.Scatter(x=btc_data.iloc[btc_data.shape[0]-100:]['trade_date'], y=btc_data.iloc[btc_data.shape[0]-100:]['price_usd'], name='Real price'))\nfor i in R:\n    fig.add_trace(go.Scatter(x=y_pred[i]['date'], y=y_pred[i]['price'], name='Epoch = ' + str(i)))\nfig.show()\nfor i in R:\n    print('Epoch = ' + str(i) + '. Mean absolute error before price convertation: ' + str(mae['before_convert'][i]) + '. Mean absolute error after price convertation: ' + str(mae['after_convert'][i]) + '.') ")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Tuning, step 2: found optimal size of timeframe\ny_pred = {}\nmae = {'before_convert': {}, 'after_convert': {}}\nR = range(10,31,5)\nfor i in R:\n    #print (i)\n    x_train,x_valid,y_train,y_valid,target,pred_convert = data_preproc_and_split(btc_data,i) #Create necessary datasets from start data (preprocessing and split)\n    model = model_gen(x_train,y_train,i,75,0) #Model generation without output\n    preds = model.predict(x_valid) #Prediction\n    y_pred[i] = pd.DataFrame(preds, index=y_valid.index, columns = ['price']) #Create DataFrame from prediction results\n    y_pred[i]['date'] = target['date'] #Add date column to results\n    mae['before_convert'][i] = mean_absolute_error(y_valid['price'],y_pred[i]['price']) #Save Mean absolute error before price convertation\n    y_pred[i]['price'] = pred_convert['price']*(y_pred[i]['price']+1) #Convert prediction results to valid price\n    mae['after_convert'][i] = mean_absolute_error(btc_data.iloc[btc_data.shape[0]-100:]['price_usd'],y_pred[i]['price']) #Save Mean absolute error after price convertation\n\n#Print results of prediction\nfig = go.Figure()\nfig.add_trace(go.Scatter(x=btc_data.iloc[btc_data.shape[0]-100:]['trade_date'], y=btc_data.iloc[btc_data.shape[0]-100:]['price_usd'], name='Real price'))\nfor i in R:\n    fig.add_trace(go.Scatter(x=y_pred[i]['date'], y=y_pred[i]['price'], name='TimeFrame Size = ' + str(i)))\nfig.show()\nfor i in R:\n    print('TimeFrame Size = ' + str(i) + '. Mean absolute error before price convertation: ' + str(mae['before_convert'][i]) + '. Mean absolute error after price convertation: ' + str(mae['after_convert'][i]) + '.')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Print final result\nfig = go.Figure()\nfig.add_trace(go.Scatter(x=btc_data.iloc[btc_data.shape[0]-100:]['trade_date'], y=btc_data.iloc[btc_data.shape[0]-100:]['price_usd'], name='Real price'))\nfig.add_trace(go.Scatter(x=y_pred[15]['date'], y=y_pred[15]['price'], name='Predict price'))\nfig.show()")
