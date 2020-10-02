#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[18]:


import pandas as pd
from matplotlib import pyplot


# In[19]:


df = pd.read_csv('../input/gpx-tracks-from-hikr.org.csv')
df.head(n=2)


# ## Adding additional features
# The difficulty rating can be changed to a numeric value for easier processing.
# Many estimators and models won't work with text values. We can simply extract the second letter which results in an ordinal encoding. Our values For categorical data which cannot be transformed that easily you may want to look into some builtin helpers like http://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.CategoricalEncoder.html. Keras also has a util for one-hot-encoding https://keras.io/utils/#to_categorical

# In[20]:


df['avg_speed'] = df['length_3d']/df['moving_time']
df['difficulty_num'] = df['difficulty'].map(lambda x: int(x[1])).astype('int32')


# ## Removing Outliers

# In[21]:


df.describe()


# ### Suspicious speed values
# Looking at min and max values it is apparent that there are some tracks which we want to exclude from our data set. An infinite average speed, or a min elevation of more than 30km below see level just don't seem right. We can remove the extremes at both sides and remove all rows where there are null values.

# In[22]:


# drop na values
df.dropna()
df = df[df['avg_speed'] < 2.5] # an avg of > 2.5m/s is probably not a hiking activity


# ### Min elevation
# A min elevation of -32km doesn't seem right.

# In[23]:


def retain_values(df, column, min_quartile, max_quartile):
    q_min, q_max = df[column].quantile([min_quartile, max_quartile])
    print("Keeping values between {} and {} of column {}".format(q_min, q_max, column))
    return df[(df[column] > q_min) & (df[column] < q_max)]

# drop elevation outliers
df = retain_values(df, 'min_elevation', 0.01, 1)


# ## Correlations
# We expect altitude and distance to be highly correlated with the moving time as these two features are used in most estimation formulas in use [citation needed].

# In[25]:


df.corr()


# As expected, changes in altitude and the distance have the highest correlations with the moving time. Max elevation also shows low correlation as the terrain in higher altitudes can be more challenging than in lower altitudes. Interestingly the difficulty score doesn't seem to correlate as much with the moving time. This might be due to several reasons: The difficulty score of a whole tour is based on the most difficult section, it is set by users and thus varies due to subjectivity, a difficult track may be exposed and only for experienced hikers, but it is not automatically terrain which slows one down.
# 
# ## Building the models
# 
# ### A strong baseline
# Before putting too much time into a sophisticated model it is important to develop a simple baseline which serves as an anchor point to benchmark any other model against it. For many problems these simple baselines are already hard to beat and allow to identify approaches which can be discarded early. Given the nature of the problem, we will use a linear regression model to predict the moving time based on the most correlated fields (`length_3d`, `uphill`, `downhill` and `max_elevation`)

# In[26]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

y = df.reset_index()['moving_time']
x = df.reset_index()[['downhill', 'uphill', 'length_3d', 'max_elevation']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

lasso = Lasso()
lasso.fit(x_train, y_train)
print("Coefficients: {}".format(lasso.coef_))


# In[27]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

y_pred_lasso = lasso.predict(x_test)


# In[28]:


r2 = r2_score(y_test, y_pred_lasso)
mse = mean_squared_error(y_test, y_pred_lasso)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))


# ### GradientBoostingRegressor

# In[30]:


from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
y_pred_gbr = gbr.predict(x_test)


# In[33]:


r2 = r2_score(y_test, y_pred_gbr)
mse = mean_squared_error(y_test, y_pred_gbr)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))


# ### Regression with Keras

# In[52]:


from keras import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

model = Sequential()
model.add(Dense(12, input_shape=(4,)))
model.add(Dense(5, input_shape=(4,)))
model.add(Dense(1))
model.compile(optimizer=Adam(0.001), loss='mse')


# In[53]:


hist = model.fit(x_train, y_train, epochs=50, batch_size=10, validation_split=0.15, 
          callbacks=[
            ModelCheckpoint(filepath='./keras-model.h5', save_best_only=True),
            EarlyStopping(patience=2),
            ReduceLROnPlateau()
          ],
          verbose=1
)


# In[54]:


model.load_weights(filepath='./keras-model.h5')
y_pred_keras = model.predict(x_test)

r2 = r2_score(y_test, y_pred_keras)
mse = mean_squared_error(y_test, y_pred_keras)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))


# ## Ensemble results

# In[62]:


import numpy as np

combined = (y_pred_keras[:,0] + y_pred_gbr * 2) / 3.0
r2 = r2_score(y_test, combined)
mse = mean_squared_error(y_test, combined)

print("r2:\t{}\nMSE: \t{}".format(r2, mse))


# In[63]:


c = pd.DataFrame([combined, y_pred_keras[:,0], y_pred_lasso, y_pred_gbr, y_test]).transpose()
c.columns = ['combined', 'keras', 'lasso', 'tree', 'test']
c['diff_minutes'] = (c['test'] - c['combined']) / 60
c.describe()

