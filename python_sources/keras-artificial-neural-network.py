#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns


# # Explore Data
# ## Import Test Data

# In[ ]:


df = pd.read_csv('/kaggle/input/ihsmarkit-hackathon-june2020/train_data.csv')


# In[ ]:


print(df.shape)
df.head()


# In[ ]:


df.set_index('vehicle_id', inplace=True)
df.sample(10)


# In[ ]:


df.isnull().values.any()


# In[ ]:


df.describe()


# In[ ]:


df.columns.tolist()


# In[ ]:


df.columns


# In[ ]:


df.dtypes


# ## Plot Histograms of Numeric Columns

# In[ ]:


def draw_histograms(df, variables, n_rows, n_cols):
    fig=plt.figure(figsize=(10,10))
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(ax=ax)
        plt.axvline(df[var_name].mean(), color='r', linestyle='dashed', linewidth=2)
        ax.set_title(var_name+" Distribution")
    fig.tight_layout()
    plt.show()


# In[ ]:


draw_histograms(df,
                df.select_dtypes(include=[np.number]).columns.tolist(),
                int(len(df.select_dtypes(include=[np.number]).columns.tolist())/2)+1, 2)


# ## Plot Numeric Columns Against `PriceUSD` Using Scatter Plots

# In[ ]:


# Draw scatter plots of numerical columns
def draw_scatters(df, variables, target, n_rows, n_cols):
    fig=plt.figure(figsize=(10,10))
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        sns.regplot(x=var_name,y=target,data=df,fit_reg=False)
        ax.set_title(var_name + " vs. " + target)
    fig.tight_layout()
    plt.show()


# In[ ]:


draw_scatters(df,
              df.select_dtypes(include=[np.number]),
              'Price_USD',
              int(len(df.select_dtypes(include=[np.number]).columns.tolist()[:-1])/2)+1, 2)


# ## Use Bar Plots to Plot Categorical Features Against `Price_USD`

# In[ ]:


def draw_bars(df, variables, target, n_rows, n_cols):
    fig=plt.figure(figsize=(10,30))
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        sns.barplot(x=var_name,y=target,data=df,ci='sd')
        ax.set_title(var_name)
    fig.tight_layout()
    plt.show()


# In[ ]:


draw_bars(df,
          df.select_dtypes(include=[np.object]).columns.tolist(),
          'Price_USD', int(len(df.select_dtypes(include=[np.object]).columns.tolist())/2)+1, 2)


# ## Import Test Data

# In[ ]:


oos_df = pd.read_csv('/kaggle/input/ihsmarkit-hackathon-june2020/oos_data.csv')
oos_df.head()


# In[ ]:


oos_df.set_index('vehicle_id', inplace=True)
oos_df.head()


# In[ ]:


oos_df.isnull().values.any()


# In[ ]:


oos_df.dtypes


# ### Visualize OOS Data

# In[ ]:


draw_histograms(oos_df,
                oos_df.select_dtypes(include=[np.number]).columns.tolist(),
                int(len(oos_df.select_dtypes(include=[np.number]).columns.tolist())/2)+1, 2)


# ## Preprocessing Train Data

# ### Make New Columns

# In[ ]:


# Make a new column based on how old the vehicle was when it was sold
df['years_old'] = df['year'].astype(float) - df['Generation_Year']
df.head()


# In[ ]:


df['pw_ratio'] = df['Engine_KW'] / df['Curb_Weight']
df.head()


# ### Drop Columns

# In[ ]:


df.drop(['date', 'year'], axis=1, inplace=True)


# ### Make Categorical Columns

# In[ ]:


df[['Generation_Year', 'No_of_Gears', 'years_old']] = df[['Generation_Year', 'No_of_Gears', 'years_old']].astype(str)
df.dtypes


# ### Make Dummies

# In[ ]:


cat_columns = df.select_dtypes(include=[np.object]).columns.tolist()
dummies_df = pd.get_dummies(df[cat_columns], prefix_sep='_', drop_first=False)
print(dummies_df.shape)
dummies_df.head(10)


# In[ ]:


df = pd.concat([df, dummies_df], axis=1)
print(df.shape)
df.head()


# In[ ]:


df.drop(cat_columns,axis=1,inplace=True)
df.head()


# In[ ]:


print(df.shape)
df.sample()


# ## Preprocess OOS Data

# In[ ]:


# Make New Columns
oos_df['years_old'] = oos_df['year'].astype(float) - oos_df['Generation_Year']
oos_df['pw_ratio'] = oos_df['Engine_KW'] / oos_df['Curb_Weight']
# Drop columns
oos_df.drop(['date', 'year'], axis=1, inplace=True)
# Change to categories
oos_df[['Generation_Year', 'No_of_Gears', 'years_old']] = oos_df[['Generation_Year', 'No_of_Gears', 'years_old']].astype(str)
# Make Dummies
cat_columns = oos_df.select_dtypes(include=[np.object]).columns.tolist()
dummies_df = pd.get_dummies(oos_df[cat_columns], prefix_sep='_', drop_first=False)
oos_df = pd.concat([oos_df, dummies_df], axis=1)
oos_df.drop(cat_columns,axis=1,inplace=True)
# Print and validate
print(oos_df.shape)
oos_df.head()


# ### Find Intersection (For Modeling)

# In[ ]:


intersection = df.columns & oos_df.columns
intersection.tolist()


# # Test Train Split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = df.drop('Price_USD', axis=1)
X = df[intersection]
X.head()


# In[ ]:


y = df['Price_USD']
y[:10]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:


for dataset in [X_train, X_test, y_train, y_test]:
    print(dataset.shape)


# In[ ]:


oos_df[intersection].shape


# ## Model Definition

# In[ ]:


# Make Keras work for OSX Catalina
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.losses import MeanAbsolutePercentageError


# In[ ]:


NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])
NN_model.summary()


# In[ ]:


early_stopping_monitor = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5, 
                                       verbose=1, 
                                       mode='auto', 
                                       baseline=None, 
                                       restore_best_weights=True)


# In[ ]:


checkpoint = ModelCheckpoint('best_model.h5', monitor='val_mean_absolute_percentage_error', mode='max', verbose=0, save_best_only=True)


# In[ ]:


NN_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=256, verbose=1, callbacks=[early_stopping_monitor, checkpoint])
_, train_acc = NN_model.evaluate(X_train, y_train, verbose=0)
_, test_acc = NN_model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# In[ ]:


plt.plot(NN_model.history.history['mean_absolute_percentage_error'], label='train')
plt.plot(NN_model.history.history['val_mean_absolute_percentage_error'], label='test')
plt.legend()
plt.show()


# ### Apply Model to OOS Data

# In[ ]:


NN_model.load_weights('best_model.h5')


# In[ ]:


y_oos_pred = NN_model.predict(oos_df[intersection])


# In[ ]:


oos_df['Price_USD'] = y_oos_pred


# In[ ]:


out_df = pd.DataFrame(oos_df['Price_USD'], oos_df.index).reset_index()


# In[ ]:


out_df.columns[out_df.isnull().any()].tolist()


# In[ ]:


print(out_df.shape)
out_df.head()


# In[ ]:


out_df.to_csv('submission-XX.csv', index=False)


# In[ ]:




