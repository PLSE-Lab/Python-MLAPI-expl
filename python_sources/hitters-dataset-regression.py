#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("/kaggle/input/Hitters.csv")
data.head()


# In[ ]:


df = data.copy()


# In[ ]:


df.describe().T


# In[ ]:


corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
plt.show()


# In[ ]:


df.isna().sum()


# In[ ]:


df.League.value_counts()


# In[ ]:


df.Division.value_counts()


# In[ ]:


df.NewLeague.value_counts()


# # Filling NaN Values

# In[ ]:


df.groupby(['League', 'Division']).agg({'Salary':'mean'})


# In[ ]:


# Fill nan salaries by mean of group(League and Division)
df['Salary'] = df.groupby(['League', 'Division'])['Salary'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


df.isna().sum()


# In[ ]:


df.info()


# # Label Encoder

# In[ ]:


le = LabelEncoder()
df['League'] = le.fit_transform(df['League'])
df['Division'] = le.fit_transform(df['Division'])
df['NewLeague'] = le.fit_transform(df['NewLeague'])
df.head()


# In[ ]:


y = df[['Salary']]
X = df.drop('Salary', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 46)
print("X_train: ", X_train.shape, "X_test: ", X_test.shape, "y_train: ", y_train.shape, "y_test: ", y_test.shape)


# # Linear Regression

# In[ ]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)
linreg_error = np.sqrt(mean_squared_error(y_test, y_pred_linreg))
print(linreg_error)


# # Ridge Regression

# In[ ]:


ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
ridge_error = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(ridge_error)


# In[ ]:


alphas = [0.01, 0.001, 0.1, 1, 2, 1.1]
tuned_ridge = RidgeCV(alphas=alphas).fit(X_train, y_train)
y_pred_tuned_ridge = tuned_ridge.predict(X_test)
tuned_ridge_error = np.sqrt(mean_squared_error(y_test, y_pred_tuned_ridge))
print(tuned_ridge_error)


# In[ ]:


tuned_ridge.alpha_


# # Lasso Regression

# In[ ]:


lasso = Lasso() 
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
lasso_error = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print(lasso_error)


# In[ ]:


alphas = [0.01, 0.001, 0.1, 1, 2, 1.1]
tuned_lasso = LassoCV(alphas=alphas).fit(X_train, y_train)
y_pred_tuned_lasso = tuned_lasso.predict(X_test)
tuned_lasso_error = np.sqrt(mean_squared_error(y_test, y_pred_tuned_lasso))
print(tuned_lasso_error)


# # ElasticNet

# In[ ]:


enet = ElasticNet()
enet.fit(X_train, y_train)
y_pred_enet = enet.predict(X_test)
enet_error = np.sqrt(mean_squared_error(y_test, y_pred_enet))
print(enet_error)


# In[ ]:


alphas = [0.01, 0.001, 0.1, 1, 2, 1.1]
ratios = [.1, .5, .7, .9, .95, .99, 1]
tuned_enet = ElasticNetCV(alphas=alphas, l1_ratio= ratios).fit(X_train, y_train)
y_pred_tuned_enet = tuned_enet.predict(X_test)
tuned_enet_error = np.sqrt(mean_squared_error(y_test, y_pred_tuned_enet))
print(tuned_enet_error)


# # Plot

# In[ ]:


# plot
scores = [linreg_error, tuned_ridge_error, tuned_lasso_error, tuned_enet_error]
plt.plot(scores, marker="o" ) 
plt.legend()
plt.show()


# # LOF

# In[ ]:


clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[:30]


# In[ ]:


esik_deger = np.sort(df_scores)[4]


# In[ ]:


aykiri_tf = df_scores > esik_deger
aykiri_tf[:10]


# In[ ]:


yeni_df = df[aykiri_tf]
yeni_df


# We dropped 5 outlier rows. 

# In[ ]:


y = yeni_df[['Salary']]
X = yeni_df.drop('Salary', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 46)
print("X_train: ", X_train.shape, "X_test: ", X_test.shape, "y_train: ", y_train.shape, "y_test: ", y_test.shape)


# ## Lof_Linear_Regression

# In[ ]:


linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_linreg = linreg.predict(X_test)
linreg_error = np.sqrt(mean_squared_error(y_test, y_pred_linreg))
print(linreg_error)


# ## Lof_Ridge_Regression

# In[ ]:


alphas = [0.01, 0.001, 0.1, 1, 2, 1.1]
tuned_ridge = RidgeCV(alphas=alphas).fit(X_train, y_train)
y_pred_tuned_ridge = tuned_ridge.predict(X_test)
tuned_ridge_error = np.sqrt(mean_squared_error(y_test, y_pred_tuned_ridge))
print(tuned_ridge_error)


# ## Lof_Lasso_Regression

# In[ ]:


alphas = [0.01, 0.001, 0.1, 1, 2, 1.1]
tuned_lasso = LassoCV(alphas=alphas).fit(X_train, y_train)
y_pred_tuned_lasso = tuned_lasso.predict(X_test)
tuned_lasso_error = np.sqrt(mean_squared_error(y_test, y_pred_tuned_lasso))
print(tuned_lasso_error)


# ## Lof_ElasticNet_Regression

# In[ ]:


alphas = [0.01, 0.001, 0.1, 1, 2, 1.1]
ratios = [.1, .5, .7, .9, .95, .99, 1]
tuned_enet = ElasticNetCV(alphas=alphas, l1_ratio= ratios).fit(X_train, y_train)
y_pred_tuned_enet = tuned_enet.predict(X_test)
tuned_enet_error = np.sqrt(mean_squared_error(y_test, y_pred_tuned_enet))
print(tuned_enet_error)


# ## LOF_PLOT

# In[ ]:


# plot
scores = [linreg_error, tuned_ridge_error, tuned_lasso_error, tuned_enet_error]
plt.plot(scores, marker="o" ) 
plt.show()


# **After applied LOF we decrease error from 322 to 295.**

# # Keras

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


model = Sequential()

model.add(Dense(4, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'tanh'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))


model.add(Dense(1))

model.compile(optimizer = 'rmsprop', loss='mse')


# In[ ]:


model.fit(x = X_train, y = y_train, epochs=250)


# In[ ]:


loss_df = pd.DataFrame(model.history.history)
loss_df.plot();


# In[ ]:


model.evaluate(X_test, y_test, verbose=0)


# In[ ]:


test_predictions = model.predict(X_test)


# In[ ]:


np.sqrt(mean_squared_error(y_test, test_predictions))


# In[ ]:




