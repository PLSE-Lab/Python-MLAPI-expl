#!/usr/bin/env python
# coding: utf-8

# # Analyzing the Ethereum Network #

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt # converting timestamp to date
import seaborn as sns # Visualizer for data
import matplotlib.pyplot as plt # matplot
import matplotlib.dates as mdates # for plotting dates
get_ipython().run_line_magic('matplotlib', 'inline')
from subprocess import check_output
# What are we working with?
print('File: \n', check_output(["ls", "../input"]).decode("utf8"))


# #We are going to take the CSVs one at a time and create a DataFrame for each of them#

# In[ ]:


# Input files i'm using
address_gr = pd.read_csv('../input/EthereumUniqueAddressGrowthRate.csv')
blocksize_hist = pd.read_csv('../input/EthereumBlockSizeHistory.csv')
etherprice_usd = pd.read_csv('../input/EtherPriceHistory(USD).csv')
hashrate_gr = pd.read_csv('../input/EthereumNetworkHashRateGrowthRate.csv')
marketcap = pd.read_csv('../input/EtherMarketCapChart.csv')
tx_hist = pd.read_csv('../input/EthereumTransactionHistory.csv')
# Going to iterate and plot everything, except those with abnormalities
things_to_plot = [(blocksize_hist,"Blocksize History"),
                  (etherprice_usd, "Etherprice - USD"),
                  (hashrate_gr,"Hashrate Growth Rate"),
                # (address_gr, "Address Growth Rate"),
                # (marketcap, "Market Capital"),
                  (tx_hist, "Transaction History")]


# ### This function was to convert the timestamps, but they no longer use them. ###
# 

# In[ ]:


# the timestamp in the method is a dataframe column
# it returns a list of the format which can then be ploted if needed
def timeConvert(timestamps):
    timeValue = list(range(len(timestamps)))
    for i in range(len(timestamps)):
        timeValue[i] = (dt.datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d'))
    return timeValue;


# # Price plot #
# Note: they are USD

# In[ ]:


# Lets see:
print(marketcap.columns)


# 1. # Function to plot data #

# In[ ]:


def plotit(data, title):
    # makes numpy array
    r = data.values#.view(np.recarray)
    #grab dates - convert to format
    date_df = r[:,0]
    date_df = pd.to_datetime(date_df)
    #grab values
    value_df = r[:,2]
    # make new plots
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_title(title)
    ax.plot(date_df, value_df)
    ax.grid(False)
    # matplotlib date format object
    hfmt = mdates.DateFormatter('%Y - - %m')
    # format the ticks
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(hfmt)
    # format the coords message box
    def yvals(x):
        return '$%1.2f' % x
    ax.format_xdata = hfmt
    ax.format_ydata = yvals
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()
    plt.show()


# In[ ]:


plotit(etherprice_usd, "Etherprice - USD")


# In[ ]:


for plot,title in things_to_plot:
    plotit(plot, title)


# ### Plot the things we missed ###

# In[ ]:


mkp = marketcap.values#.view(np.recarray)

date_df = mkp[:,0]
date_df = pd.to_datetime(date_df)
value_df = mkp[:,3]
prices_df = mkp[:,4]

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(15, 7))
ax.set_title("Market Capital")
ax.set_ylabel("(USD) Millions")
ax.plot(date_df, value_df)
ax.grid(False)
# Format dates
hfmt = mdates.DateFormatter('%Y - - %m')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(hfmt)
def yvals(x):
    return '$%1.2f' % x
ax.format_xdata = hfmt
ax.format_ydata = yvals
fig.autofmt_xdate()
plt.show()


# In[ ]:


txs = tx_hist.copy()
txs['Date(UTC)'] = pd.to_datetime(txs['Date(UTC)']).dt.year
#txs['Date(UTC)'] = txs['Date(UTC)'].dt.year
txs = txs.groupby('Date(UTC)')['Value'].apply(lambda x: (x.unique().sum()))
txs


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))
shap = txs
labels = '2015','2016','2017','2018'
explode = (0, 0, 0, 0.1)
ax.pie(shap, explode=explode, labels=labels, shadow=True)
plt.title('Transactions per year')
plt.show()


# #### Good stuff, obviously there is so much more we could do with this ####

# ## This is one of my first ML applications and it was years ago ##
# ## Don't mind it, I was only 15 ##
# # Machine learning (can't use kaggle):#
#     from keras.layers.core import Dense, Activation, Dropout
#     from keras.layers.recurrent import LSTM
#     from keras.models import Sequential
#     import lstm, time #helper libraries
# **Using TensorFlow backend.**
# 
#     # Load data
#     X_train, y_train, X_test, y_test = lstm.load_data('../input/ether.csv', 30, True)
# 
# 
#     # Build model
#     model = Sequential()
#     
#     model.add(LSTM(
#         input_dim=1,
#         output_dim=30,
#         return_sequences=True,))
#     model.add(Dropout(0.14))
#     
#     model.add(LSTM(
#         100,
#         return_sequences=False))
#     model.add(Dropout(0.14))
#     model.add(Dense(
#         output_dim=1))
#     model.add(Activation('linear'))
#     
#     start = time.time()
#     model.compile(loss='mse', optimizer='rmsprop')
#     print('Compilation time: ', time.time()-start)
# **Compilation time:  0.010754108428955078**
# 
#     # Train model
#     model.fit(
#         X_train,
#         y_train,
#         batch_size=512,
#         nb_epoch=10,
#         validation_split=0.05)
# 
# **Train on 498 samples, validate on 27 samples
# Epoch 1/10
# 498/498 [==============================] - 0s - loss: 0.6696 - val_loss: 1.0974
# Epoch 2/10
# 498/498 [==============================] - 0s - loss: 0.4631 - val_loss: 0.8331
# Epoch 3/10
# 498/498 [==============================] - 0s - loss: 0.3102 - val_loss: 0.5372
# Epoch 4/10
# 498/498 [==============================] - 0s - loss: 0.1673 - val_loss: 0.3015
# Epoch 5/10
# 498/498 [==============================] - 0s - loss: 0.1224 - val_loss: 0.4458
# Epoch 6/10
# 498/498 [==============================] - 0s - loss: 0.1267 - val_loss: 0.2709
# Epoch 7/10
# 498/498 [==============================] - 0s - loss: 0.1037 - val_loss: 0.3876
# Epoch 8/10
# 498/498 [==============================] - 0s - loss: 0.1067 - val_loss: 0.2650
# Epoch 9/10
# 498/498 [==============================] - 0s - loss: 0.0929 - val_loss: 0.3726
# Epoch 10/10
# 498/498 [==============================] - 0s - loss: 0.0967 - val_loss: 0.2667
# <keras.callbacks.History at 0x7fe404259e48>**
# 
#     # Try to predict
#     predictions = lstm.predict_sequences_multiple(model, X_test, 11, 13)
#     lstm.plot_results_multiple(predictions, y_test, 12)
# **Amount preedicted:  58  with true data: 58 , and prediction length:  12**
# 
# ![Prediction output][1]
# 
# 
#   [1]: https://raw.githubusercontent.com/kingburrito666/Ethereum-Blockchain-Analysis/master/prediction.png

# # Obviously not the best...
