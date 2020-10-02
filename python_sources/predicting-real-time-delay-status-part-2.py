#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Load feature and target matrices from part 1

# In[ ]:


X = pd.read_csv('/kaggle/input/predicting-real-time-delay-status-part-1/X.csv', index_col=0)
y = pd.read_csv('/kaggle/input/predicting-real-time-delay-status-part-1/y.csv', index_col=0)


# # Section 5: Pre-pocessing feature and target matrix

# ## 5.1 Partial pairplot of feature matrix

# In[ ]:


col_to_keep = ['last_status','avg_station_same', 'avg_station_opp', 'avg_sys']
pair_plot_df = X[col_to_keep]
sns.pairplot(pair_plot_df)


# ## 5.2 Split train and test sets

# Split train and test sets with test size = 20%.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)


# Release the RAM used by X and y.

# In[ ]:


X = []
y = []


# ## 5.3 Standardize feature matrix

# Standardize training feature matrix.

# In[ ]:


standardize = StandardScaler()
X_train = standardize.fit_transform(X_train)


# Use the same settings to standardize test set.

# In[ ]:


X_test = standardize.transform(X_test)


# ## 5.4 Apply PCA

# Apply PCA and first set number of compenents to original dimension. Plot cummulative variance over number of components.

# In[ ]:


pca = PCA(n_components=51)
pca.fit(X_train)
pc_vs_variance = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=[8, 8])
plt.plot(pc_vs_variance)
ax.set_xlabel('Num of components')
ax.set_ylabel('Cummulative variance')
plt.show()


# In[ ]:


print(pc_vs_variance[46])
print('According to the plot, n = 47 should be enough to capture 100% of variance')


# Set n_compnents to 47 and re-apply PCA to training set. Then, use the same projection to transform test set.

# In[ ]:


pca = PCA(n_components=47)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# # Section 6: ML models

# ## 6.1 Linear regression

# Apply linear regression and print out mean squared error and r2 on both training and test sets.

# In[ ]:


lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('mean squared error on train sets:', mean_squared_error(y_train, lr.predict(X_train)))
print('mean squared error on test sets:', mean_squared_error(y_test, y_pred))
print('r2 on train sets:', r2_score(y_train, lr.predict(X_train)))
print('r2 on test sets:', r2_score(y_test, y_pred))


# ## 6.2 Neural Network Model

# ### 6.2.1 K-fold validation

# Use K-fold validation to validate different models. First initialize K-fold validation.

# In[ ]:


kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train_np = np.array(y_train)


# #### Model 1: Intial model

# Come up with an initial model:
# * Number of hidden layers: 3
# * Number of nodes each layer: 40
# * Activation function type: relu
# * Loss function: Mean squared error
# * Epochs: 10
# * Mini-batch size: 128

# In[ ]:


cvscores = []

for train, test in kfold.split(X_train, y_train_np):
  model = Sequential()
  model.add(Dense(40, input_dim=47, kernel_initializer='normal', activation='relu'))
  model.add(Dense(40, kernel_initializer='normal', activation='relu'))
  model.add(Dense(40, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))    
  # Compile model
  model.compile(loss='mse', optimizer='adam')
  print('-------------')
  # Fit the model
  model.fit(X_train[train], y_train_np[train], epochs=10, batch_size=128)
  print('-------------')
  # evaluate the model
  scores = model.evaluate(X_train[test], y_train_np[test])
  cvscores.append(scores)


# Print out validation score.

# In[ ]:


print('loss: %.4f' % np.mean(cvscores), '(+/-%.3f)' % np.std(cvscores))


# #### Model 2: w/ less hidden layer

# Test second model:
# * **Number of hidden layers: 2 (only difference between 1 & 2)**
# * Number of nodes each layer: 40
# * Activation function type: relu
# * Loss function: Mean squared error
# * Epochs: 10
# * Mini-batch size: 128

# In[ ]:


cvscores2 = []

for train, test in kfold.split(X_train, y_train_np):
  model = Sequential()
  model.add(Dense(40, input_dim=47, kernel_initializer='normal', activation='relu'))
  model.add(Dense(40, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))    
  # Compile model
  model.compile(loss='mse', optimizer='adam')
  # Fit the model
  print('-------------')
  model.fit(X_train[train], y_train_np[train], epochs=10, batch_size=128)
	# evaluate the model
  print('-------------')
  scores = model.evaluate(X_train[test], y_train_np[test])
  cvscores2.append(scores)


# Print out validation score.

# In[ ]:


print('loss: %.4f' % np.mean(cvscores2), '(+/-%.3f)' % np.std(cvscores2))


# The results show that Model 2 is better than Model 1.

# #### Model 3: Further simplify the model. Decrease the neurons of each hidden layer to 30.

# Test second model:
# * Number of hidden layers: 2 
# * **Number of nodes each layer: 30 (only difference between 1 & 2)**
# * Activation function type: relu
# * Loss function: Mean squared error
# * Epochs: 10
# * Mini-batch size: 128

# In[ ]:


cvscores3 = []

for train, test in kfold.split(X_train, y_train_np):
  model = Sequential()
  model.add(Dense(30, input_dim=47, kernel_initializer='normal', activation='relu'))
  model.add(Dense(30, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))    
  # Compile model
  model.compile(loss='mse', optimizer='adam')
  # Fit the model
  print('-------------')
  model.fit(X_train[train], y_train_np[train], epochs=10, batch_size=128)
	# evaluate the model
  print('-------------')
  scores = model.evaluate(X_train[test], y_train_np[test])
  cvscores3.append(scores)


# Print out validation score.

# In[ ]:


print('loss: %.4f' % np.mean(cvscores3), '(+/-%.3f)' % np.std(cvscores3))


# The results show that Model 2 is better than Model 3.

# ### 6.2.2 Neural Network Model

# Based on the the cross-validation results, I chose the model 2 for the final model of neural network.

# Re-build model 2 and train with the whole training set.

# In[ ]:


model = Sequential()
model.add(Dense(40, input_dim=47, kernel_initializer='normal', activation='relu'))
model.add(Dense(40, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))    
# Compile model
model.compile(loss='mse', optimizer='adam')
# Fit the model
model.fit(X_train, y_train_np, epochs=10, batch_size=128)


# Predict based on test set.

# In[ ]:


y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

print('mean squared error on train sets:', mean_squared_error(y_train, y_pred_train))
print('mean squared error on test sets:', mean_squared_error(y_test, y_pred))
print('r2 on train sets:', r2_score(y_train, y_pred_train))
print('r2 on test sets:', r2_score(y_test, y_pred))


# # Section 7: Conclusion and Discussion

# ## 7.1 Model comparison

# Based on the mean squared error and r2 scores, the Neural Network model is better than Linear regression results. I also tried random forest regressor in Google colab, which has much better hardware than my laptop, and it shows slightly worse result compared with the neural network but still better than the linear regression.

# ## 7.2 Description of challenges/ Obstacles faced

# The first challenge I came across is from the limit of kaggle notebook. The RAM limit is 14 GB, which is clearly not enough for such a huge dataset.
# 
# To test in a safer way, I am personally more likely to use the function .copy() to make sure that I will not ruin my original dataframe. However, it would directly double the usage of RAM. I have used the RAM up several times before going into model section. And eventually, it ended up with spliting into two separat notebooks.
# 
# After making sure all the processing is correct, I removed the .copy() function and re-run program to release RAM. However, in the model section, the fit process also takes a lot of memory. My notebook crashed once with Random Forest Regressor (even in Google colab with 25-G RAM) and crashed more than 5 times with neural network.

# The second challenge is the time. At the very beginning, I used a lot of apply function to process the data, but noticed that it occupied a lot of time. I re-wrote the program by replacing apply functions with built-in pandas methods, and it worked pretty well.
# 
# However, there are still some time-consuming process which are inevitable like GridSearchCV (10 hours + in Google Colab), training of Random Forest Regressor (30 minutes + in Google Colab), training of Nerual Network (20 minutes + in Google Colab). So, if I have more time, I could further improve my work.

# ### 7.3 Potential Next Steps/ Future Direction

# As for future work, the first thing I want to mention is that all the models here can be further improved if more time is given to tune hyperparameters and test more models with cross-validation.

# And I am pretty satisfied with the current performance of models described in this notebook, since the mean squared error is around 1.4 in minutes. So I think it could be potentially applied to the SEPTA system to predict delay time.
# 
# In order to do so, the models need to ba adjusted to receive real-time data, re-run machine learning program, and give out real-time prediction. This would require stream processing.
