#!/usr/bin/env python
# coding: utf-8

# Although cross validation is a common technique used to improve the general performance,   it is sometimes used in In case of series data,  
# you should be careful.
# shuffle of time series data during cross validation. I think this is typical.  
# **By shuffling past and future data, the learner learns the future that it is not supposed to know.**  
# As a result, the cross validation scores will be very good,
# but when the time comes to use the learner to try to predict the real future, the results will be very bad.  
# **This is what is called leakage.**  
# Therefore, Except for non-chronological data, I can't recommend the following examples  
# 
# from sklearn.model_selection import KFold  
# KFold(n_splits=3, shuffle=True)  
# StratifiedKFold(n_splits=2, shuffle=True)
# 
# Also, the train_test_split function provided in scikit-learn is The default setting is shuffle=True, so be careful when using it.  
# To prevent leakage, It is better to set the old data as training set and the new data as validation set.  
# There is a TimeSeriesSplit function in scikit-learn that does all of the above for you, and I'd like to introduce it in this article.  
# Look at the image below to see what kind of validation you can do.
# The image is borrowed from the link below.  
# https://www.r-bloggers.com/time-series-cross-validation-using-crossval/

# ![image.png](attachment:image.png)

# Let's use it for the crude oil data we are given.

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit  

df=pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv')  

fig, axes = plt.subplots(5, 1, figsize=(15, 20))  
folds = TimeSeriesSplit(n_splits=5)  
for i, (train_index, test_index) in enumerate(folds.split(df)):  
    sns.lineplot(data=df, x='Date', y='Price', ax=axes[i], label='no_use',color="0.8")  
    sns.lineplot(data=df.iloc[train_index], x='Date', y='Price', ax=axes[i], label='train',color="b")  
    sns.lineplot(data=df.iloc[test_index], x='Date', y='Price', ax=axes[i], label='Validation',color="r")  

plt.legend()  
plt.show()  


# No future data is mixed in during training. And you can see that Validation data from another point in time is used.  
# This allows us to create a robust validation schema.
# 
# However, as time goes by, the amount of training data is increasing, and the amount of data is unbalanced between folds.  
# In such a case, you can remove old data every time and equalize the amount of data between folds, as shown in the image below.

# ![image.png](attachment:image.png)

# Let's use the validation shema above for a given crude oil data.

# In[ ]:


def rolling_time_series_split(df,splits):
    n_samples = len(df)
    folds = n_samples // splits
    indices = np.arange(n_samples)

    margin = 0
    for i in range(splits): 
        start = i * folds  
        stop = start + folds  
        temp = int(0.8 * (stop - start)) + start #If you want to change the data ratio of train/Validation, change the 0.8 part.
        yield indices[start: temp], indices[temp + margin: stop]  

fig, axes = plt.subplots(5, 1, figsize=(15, 20))  
for i, (train_index, test_index) in enumerate(rolling_time_series_split(df,5)):  
    sns.lineplot(data=df, x='Date', y='Price', ax=axes[i], label='no_use',color="0.8")  
    sns.lineplot(data=df.iloc[train_index], x='Date', y='Price', ax=axes[i], label='train',color="b")  
    sns.lineplot(data=df.iloc[test_index], x='Date', y='Price', ax=axes[i], label='Validation',color="r")  

plt.legend()  
plt.show()  


# You can see that the amount of data is equal between folds.
# 
# 
# This article has shown you a simple way to prevent leakage and increase generalization performance.  
# if you want to make it even more robust, you can use nested cv (double cross validation).
# It is mainly used when you want to do further model evaluation after hyperparameter tuning.
# A little bit of writing on how to do it would be as follows.
# You can set aside a dataset that will not be added to the fold, and only fold the other data and cross-validate the model by dividing it into The trained model is then asked to predict a preallocated dataset. The difference between the predictions and the actual values is used to evaluate the hyperparameters as a CV error.
# 
# 
# ![image.png](attachment:image.png)
# 
# 
# The following article on nested cv may be helpful.  
# https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9
# 
# https://www.angioi.com/time-nested-cv-with-sklearn
# 
# https://www.mikulskibartosz.name/nested-cross-validation-in-time-series-forecasting-using-scikit-learn-and-statsmodels/
