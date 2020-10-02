#!/usr/bin/env python
# coding: utf-8

# Credits to Jason
# https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/    

# Regression Model uses the data from same input variable at previous time steps - **Auto Regression**
# 
# Input variables taken as observations at previous time step - **Lag variables**
# 
# Relationship between variables is **correlation**,  variables change in same direction - **Postivie correlation** and if goes in opposite direction -** Negative Correlation**
# 
# Correlation calculated between the variable and itself at previous time steps - **Auto Correlation** (Serial Correlation)
# 
# Based on the correlation between the Lag variable and output variable the model will be built
# 
# Dataset - minimum daily temperatures over 10 years (1981-1990) in the city Melbourne, Australia.

# In[ ]:


import os
import numpy as np
import pandas as pd
from pandas.tools.plotting import lag_plot,autocorrelation_plot

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns


# In[ ]:


os.listdir("../input")


# In[ ]:


temp_dic = pd.read_excel('../input/daily-minimum-temperatures-in-me.xlsx',sheet_name=['Temperature'],parse_dates=['Date'])


# In[ ]:


df = pd.DataFrame(temp_dic['Temperature'])
df.set_index(['Date'],inplace=True)
df.sample(5)


# In[ ]:


df.info()


# In[ ]:


df.plot(figsize=(15,5))


# Visual check of correlation by plotting the scatter plot against the previous and next time step
# Pandas built in plot - Lagplot - to do visual check

# In[ ]:


lag_plot(df)
#Clearly show some correlation


# #### Correlation Coefficents

# In[ ]:


df_corr = pd.concat([df.shift(1),df],axis=1)


# In[ ]:


df_corr.columns=['t-1','t+1']


# In[ ]:


df_corr.corr(method="pearson")


# In[ ]:


sns.heatmap(df_corr.corr(method="pearson"),cmap="Blues",annot=True)
#Shows strong positive Correlation


# **Auto Correlation Plot**
# 
# X Axis - Lag Number
# 
# Y Axis - Correlation Coefficient - -1 to +1
# 
# Dashed Lines- 95% Confidence
# 
# Solid Lines- 99% Confident
# 
# More significant are the ones above the lines than those below

# In[ ]:


plt.figure(figsize=(10,10))
autocorrelation_plot(df)


# **Auto Regression Model**
# Predict the Weather of the Last 7 Days
# 
# AR modeling from statsmodels
# Number of Coefficients learned by the model used for manual prediction 
# By the count of the coefficients the history observations retrived to make prediction

# In[ ]:


from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error


# In[ ]:


X = df.values


# In[ ]:


train,test = X[1:len(X)-7],X[len(X)-7:]


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


model = AR(train)
model_fit= model.fit()


# In[ ]:


window = model_fit.k_ar #Variables
coeff = model_fit.params # Coefficients
#Linear Regression - y= bX1 + bX2 ... + bXn


# In[ ]:


history = train[len(train) - window:]


# In[ ]:


len(train) - 29


# In[ ]:


history


# In[ ]:


history = [history[i] for i in range(len(history))]


# In[ ]:


history


# In[ ]:


predictions=[]

for t in test:
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    y = coeff[0]
    for d in range(window):       
        y += coeff[d + 1] * lag[window - d - 1]
        #print(coeff[d + 1] * lag[window - d - 1])
    predictions.append(y)
    history.append(t)
    print(f"Predicted :{y} and expected value:{t}")


# In[ ]:


mean_squared_error(test,predictions)


# In[ ]:


plt.plot(test,label='actual')
plt.plot(predictions,label='predicted')
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




