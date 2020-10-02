#!/usr/bin/env python
# coding: utf-8

# ### Overfitting can be a difficult concept to wrap one's head around. Below is a visual demonstration of how a XGBoost regressor model output changes for the same input with more training iterations.
# ![](png_to_gif.gif)
# 
# ### The rest of the kernel showcases how to get this visualisation.
# #### We are going to be using a regressor model on the Energy Consumption dataset, trying to predict nominal energy consumption levels.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from PIL import Image, ImageDraw
from itertools import chain

plt.style.use('fivethirtyeight')


# In[ ]:


pjme = pd.read_csv('../input/PJME_hourly.csv', index_col=[0], parse_dates=[0])


# First of all, it helps to visualize the data we are working with.
# 
# PJME dataset is an hourly account of energy consumption on the East Coast of the US, in the years 2002-2019.

# In[ ]:


print(pjme.head())

color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
_ = pjme.plot(style='.', figsize=(15,5), color=color_pal[0], title='PJM East')


# Train/test split:
# 
# * Train: 2002-2014
# * Test: 2015-2019

# In[ ]:


split_date = '01-Jan-2015'
pjme_train = pjme.loc[pjme.index <= split_date].copy()
pjme_test = pjme.loc[pjme.index > split_date].copy()


# In[ ]:


_ = pjme_test     .rename(columns={'PJME_MW': 'TEST SET'})     .join(pjme_train.rename(columns={'PJME_MW': 'TRAINING SET'}), how='outer')     .plot(figsize=(15,5), title='PJM East', style='.')


# The following function adds time features to the data based on the **Datetime** column.

# In[ ]:


def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


# In[ ]:


X_train, y_train = create_features(pjme_train, label='PJME_MW')
X_test, y_test = create_features(pjme_test, label='PJME_MW')


# The following loop:
# * Trains the model, 1 iteration at a time
# * Saves the predicted set and the MAE (mean absolute error) of both the training set and the testing set
# * Saves a snapshot of a graph of the predicted vs actual values for 1 week of data to later make a GIF

# In[ ]:


frames = []

x1=0

for n in chain(range(1,50+1),range(60,300+1,10),range(325,1000+1,25)):    
    y=n-x1
    x1=n
    reg = xgb.XGBRegressor(n_estimators=y)
    if n==1:
        model = reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False)
        model.save_model('model.model')
        
        pjme_test['MW_Prediction'] = model.predict(X_test)
        pjme_all = pd.concat([pjme_test, pjme_train], sort=False)
        
        mae_test = mean_absolute_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test['MW_Prediction'])
        mae_train = model.evals_result()['validation_0']['rmse'][0]
        
        mae_plot = pd.DataFrame({'mae_test': [mae_test], 
                                 'mae_train': [mae_train],
                                 'n': [n]})
        
        plot1 = pjme_all[['MW_Prediction','PJME_MW']].plot(style=['.','-'],figsize=(14,6))
        plot1.set_ylim(0, 60000)
        plot1.set_xbound(lower='06-01-2016', upper='06-08-2016')
        plt.suptitle('First week of August 2016, Actual vs Predicted')
        plt.annotate(s='# of rounds: '+str(n), xy=(0.6, 0.2), xycoords='axes fraction',
            fontsize=30)
        plt.annotate(s='MAE train: '+str(round(mae_train,1)), xy=(0.4, 0.9), xycoords='axes fraction',
            fontsize=20)
        plt.annotate(s='MAE test: '+str(round(mae_test,1)), xy=(0.4, 0.8), xycoords='axes fraction',
            fontsize=20)
        plt.savefig('plot'+str(n))
        new_frame = Image.open('plot'+str(n)+'.png')
        frames.append(new_frame)
        
    else:
        model = reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False, xgb_model='model.model')
        model.save_model('model.model')
        
        pjme_test['MW_Prediction'] = model.predict(X_test)
        pjme_all = pd.concat([pjme_test, pjme_train], sort=False)
        
        mae_test = mean_absolute_error(y_true=pjme_test['PJME_MW'],
                   y_pred=pjme_test['MW_Prediction'])
        mae_train = model.evals_result()['validation_0']['rmse'][0]
        
        mae_plot = mae_plot.append(pd.DataFrame({'mae_test': [mae_test], 
                                                 'mae_train': [mae_train],
                                                 'n': [n]}))
        plot1 = pjme_all[['MW_Prediction','PJME_MW']].plot(style=['.','-'],figsize=(14,6))
        plot1.set_ylim(0, 60000)
        plot1.set_xbound(lower='06-01-2016', upper='06-08-2016')
        plt.suptitle('First week of August 2016, Actual vs Predicted')
        plt.annotate(s='# of rounds: '+str(n), xy=(0.6, 0.2), xycoords='axes fraction',
            fontsize=30)
        plt.annotate(s='MAE train: '+str(round(mae_train,1)), xy=(0.4, 0.9), xycoords='axes fraction',
            fontsize=20)
        plt.annotate(s='MAE test: '+str(round(mae_test,1)), xy=(0.4, 0.8), xycoords='axes fraction',
            fontsize=20)
        
        plt.savefig('plot'+str(n))
        new_frame = Image.open('plot'+str(n)+'.png')
        frames.append(new_frame)
    print(n)
("")


# Making the GIF animation:

# In[ ]:


frames[0].save('png_to_gif.gif', format = 'GIF',
              append_images = frames[1:],
              save_all=True,
              duration=100,loop=0)


# ![](png_to_gif.gif)

# **As can be observed in the animation, the model fits the data better up to a certain point (measured by Test MAE), but gradually deteriorates after.**
# 
# **This can be observed in the oddly specific shapes that the dotted line starts morphing into after approximately 300 rounds. The model begins to learn shapes that, although fit the training data better, are too specific for predicting future data.**

# In[ ]:


plot3 = mae_plot.plot(x='n',figsize=(12,6))
plot3.set_ylim(2000,4000)
plt.suptitle('MAE of train vs test set over the number of iterations')


# **The above graph showcases the gradual disparity between the train and test MAE. In this specific case, the prediction accuracy peaks at around 150 iterations.**
