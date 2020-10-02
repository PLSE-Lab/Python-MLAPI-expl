#!/usr/bin/env python
# coding: utf-8

# # Predict BTC Price Movement
# 
# We will use a simple approach to predict whether the **next minute price** goes up or down based on the previous movements

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from subprocess import check_output

print("Available data:\n")
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


available_data = {
    'bitstamp': pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2017-05-31.csv"),
    'coinbase': pd.read_csv("../input/coinbaseUSD_1-min_data_2014-12-01_to_2017-05-31.csv"),
    'btce': pd.read_csv("../input/btceUSD_1-min_data_2012-01-01_to_2017-05-31.csv"),
    'kraken': pd.read_csv("../input/krakenUSD_1-min_data_2014-01-07_to_2017-05-31.csv")
}


# In[ ]:


print("Bitstamp data shape: {0}\nCoinbase data shape: {1}\nBTCe data shape: {2}\nKraken data shape: {3}".format(
    available_data['bitstamp'].shape,
    available_data['coinbase'].shape,
    available_data['btce'].shape,
    available_data['kraken'].shape))


# #### We will use Bitstamp data as our default BTC data

# In[ ]:


btc = available_data['bitstamp']


# In[ ]:


# Show how the data is structured

btc.head()


# We need to clean the data by filling the *NaNs* with the last available values. This is accomplished with [.ffill()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.ffill.html)

# In[ ]:


# Fill the value gaps forward

btc[btc.columns.values] = btc[btc.columns.values].ffill()


# In[ ]:


# Plot how the Open prices look

btc['Open'].plot()


# Now, we compute the price movement during the 1 minute interval

# In[ ]:


btc['Delta'] = btc['Close'] - btc['Open']


# In[ ]:


# And we plot the per-minute movements

btc['Delta'].plot(kind='line')


# Let's take a look at some strong movements shown by the previous graphs (probably flash crashes). The *100* limit is arbitrary

# In[ ]:


btc[abs(btc['Delta']) >= 100]


# ## Preparing the data for classification
# We attach a label to each movement: 
# * **1** if the price goes up
# * **0** if the price goes down (or stays the same)

# In[ ]:


def digitize(n):
    if n > 0:
        return 1
    return 0
    
btc['to_predict'] = btc['Delta'].apply(lambda d: digitize(d))


# In[ ]:


# Show the last 5 elements of the btc dataframe

btc.tail()


# Now, we need the data as a `numpy` matrix

# In[ ]:


btc_mat = btc.as_matrix()


# ## Here's the idea:
# We will train a classifier providing it with a window of previous `'Delta'` movements of length `WINDOW_SIZE`.
# The outcome to predict is the next `'to_predict'` value

# In[ ]:


def rolling_window(a, window):
    """
        Takes np.array 'a' and size 'window' as parameters
        Outputs an np.array with all the ordered sequences of values of 'a' of size 'window'
        e.g. Input: ( np.array([1, 2, 3, 4, 5, 6]), 4 )
             Output: 
                     array([[1, 2, 3, 4],
                           [2, 3, 4, 5],
                           [3, 4, 5, 6]])
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


WINDOW_SIZE = 22


# In[ ]:


# Generate the X dataset (the 'Delta' column is the 8th)
# Remove the last row since it can't have its Y value

X = rolling_window(btc_mat[:,8], WINDOW_SIZE)[:-1,:]


# In[ ]:


# Let's see how it looks

btc['Delta'].tail(10)


# In[ ]:


# And now let's compare the above with the X matrix

print("{0}\n\nShape: {1}".format(X, X.shape))


# In[ ]:


# We generate the corresponding Y array and check if X and Y shapes are compatible

Y = btc['to_predict'].as_matrix()[WINDOW_SIZE:]
print("{0}\n\nShape: {1}".format(Y, Y.shape))


# ## It's time for some Random Forest

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# First, we create the *train* and *test* datasets, making the second **25%** of the whole data. Also, we need to make sure to balance the two datasets (`stratify=Y`)

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=4284, stratify=Y)


# In[ ]:


# Let's see how it looks

y_test[:100]


# Now, we fit a random forest classifier to the `X_train` and `y_train` data

# In[ ]:


clf = RandomForestClassifier(random_state=4284, n_estimators=50)
clf.fit(X_train, y_train)


# We predict the values from the `X_test` data and we are ready the assess the model

# In[ ]:


predicted = clf.predict(X_test)


# ## Model Evaluation

# In[ ]:


print(classification_report(y_test, predicted))


# In[ ]:


conf_mat = confusion_matrix(y_test, predicted)

# Confusion matrix in percentages
pct_conf_mat = conf_mat/np.sum(conf_mat) * 100

print("Pred:  0\t\t1\n{}".format(pct_conf_mat))


# ## Conclusions
# 
# The classification report for this simple model is encouraging, even though it's far from perfect.
# 
# ### Some suggestions
# * Run multiple models for different batches of time - this is because the dynamics of any financial market can change dramatically over time and the range considered in this analysys is more than 5 years.
# * Use other classification models (NN, SVMs, maybe even k-NN).
# * Cross-validate! Tweak the hyperparameters of the Random Forest classifier and the `WINDOW_SIZE`.
# 
# *Feel free to fork and implement your ideas. Enjoy!*
