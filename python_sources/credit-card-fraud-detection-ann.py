#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# 
# A fradulent transaction is the result of identity theft and involves the use of your debit or credit card for charges that a customer did not authorise. It is important that credit card companies are able to recognise fradulent credit card transactions so that customers are not charged for items that they did not purchase. The fradulent transaction is not only financial crime which can risk a company's reputation and trust with customers, but also results in inconvenicence for customers to dispute those transactions or potentially financial loss.
# 
# In this sense, it is fortunate that data scientists can help to detect abnormal transactions using machine learning, especially this project uses feedforward neural network. This network is the simplest type of neural network devised and the input information moves in only one direction, forward, from the input nodes through hidden nodes and to the output nodes. There are no cycles to give feedback backward unlike recurrent neural netoworks.
# 
# ![Feedforward Neural Network](https://ds055uzetaobb.cloudfront.net/brioche/uploads/uzLXsnBLTI-fully_connected_mlp.png?width=1200)
# 
# 
# # EDA
# 
# The datasets contain transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occured in two days, where we have 492 frauds out of 284,807 transactiosn. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. Due to confidentiality issues, the original features are masked as V1,V2,...V28. The column 'Class' indicate if a transaction is fradulent as boolean. 
# 
# Looking at feature distributions below, I can find scales are within similar ranges for most of features. But, there are some to be addressed.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', -1) 

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

# feature columns
print(df.columns)

# feature distribution
df.describe()


# After pearson correlation matrix calculated, at a glance I can see there is no dominant feature over others in relation to if a transaction is fradulent. Note that last feature in x-axis is a column that indicates a fradulent one.

# In[ ]:


df.corrwith(df.Class).plot.bar(title='corr with class', x= 'feature', y='corr (pearson)', grid=True, fontsize=12, rot=30, figsize=(15, 4))


# In[ ]:


import seaborn as sn
sn.set_style('white')
f, ax = plt.subplots(figsize=(7, 7))
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sn.diverging_palette(220, 10, as_cmap=True)
sn.heatmap(df.corr(), mask=mask, cmap="YlGnBu", vmax=.5, square=True)


# # Feature Scailing
# 
# The description table above shows that there are features in large scale - we are going to address them so as to be in similar scale. The normalisation is not only good for fast convergence performance during training but also contributes to precise convergence.

# In[ ]:


feature_cols = np.array(df.columns)
feature_cols = feature_cols[~np.isin(feature_cols, ['Class', 'Time', 'Amount'])]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

df[feature_cols].describe()


# Quickly checking if there is any empty values.

# In[ ]:


# blank values
df.isna().sum()


# # Oversampling
# 
# The fraudaulent transactions only takes 0.172% of total transactions in dataset, which results in unbalanced datasets. For that reason, I decide to ovesample a less represented group by replicating them in dataframe. Replicating by 10 times, the amount will be nearly 2% in total.
# It is still not that much considering total but will see how it plays in a result.

# In[ ]:


# oversampling a less represented group
df[df.Class == 1].shape
df = df.append([df[df.Class == 1]]*10, ignore_index=True)
df[df.Class == 1].shape


# # Train Model
# 
# Prior to modelling, I am going to split data into train by 80% and test by 20%.

# In[ ]:


# split train test
X = df.iloc[:, df.columns.isin(feature_cols)]
y = df.iloc[:, df.columns == 'Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# The model that I uses is a feedforward neural network which doesn't have a cycle to update a model backward, and I am going to intentionally create a bottleneck in a architecture to reduce representation with less nodes then reconstruct back, so called an autoencoder. This helps to summarise representation in a model and better understand dominant events.

# In[ ]:


import tensorflow as tf
from tensorflow.keras import Model, Sequential, backend as K
from tensorflow.keras.layers import Input, Dense, Dropout

inp = Input(shape=(len(X.columns),))
dense_1 = Dense(100, activation='relu')(inp)
dense_2 = Dense(50, activation='relu')(dense_1)
dense_3 = Dense(100, activation='relu')(dense_2)
out = Dense(1, activation='sigmoid')(dense_3)
model = Model(inp, out)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=200, epochs=500, verbose=0)
score = model.evaluate(X_test, y_test)


# # Backtest
# 
# The predictive model should be tested on historical data, and the predictions can be compared with the actual results to understand error rate. As part of backtesting, we predict with test data and compare its results with the actual. The results are visualised with confusion matrix which allows us to check 100% accurate prediction results as well as false-positive / true-negative ones at once.

# In[ ]:


y_pred = model.predict(X_test)
y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=('Non-Fraud (Actual)', 'Fraud (Actual)'), columns=('Non-Fraud (Predicted)', 'Fraud (Predicted)'))
sn.set_style('white')
plt.figure(figsize = (7,7))
sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
print(df_cm)


# In the aforementioned matrix, you can understand the followings:
# * The model correctly classified 1096 transactions as fraud, while incorrectly 15 as non-fraud. (Percentage error = 1.36%)
# * The model correctly classified 56850 transactions as non-fraud, while incorrectly 25 as fraud. (Percentage error = 0.043%)

# In[ ]:


print("Backtest Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

