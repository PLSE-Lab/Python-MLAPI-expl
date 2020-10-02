#!/usr/bin/env python
# coding: utf-8

# ## Auto Regressive Model
# Let's try and predict the future of crypto! We're going to fit an Auto Regressive (AR) model on the XLM ([Stellar Lumens](https://www.coingecko.com/en/coins/stellar)) currency. Further information on these models [can be found here.](https://www.quantstart.com/articles/Autoregressive-Moving-Average-ARMA-p-q-Models-for-Time-Series-Analysis-Part-1)
# 
# The code here is adapted from [this excellent resource.](https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/)
# 
# As we will see, the AR model finds that a 23-day lag best explains the behaviour of this particular cryptocoin .

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
pd.options.mode.chained_assignment = None


# The following class will be used to generate all the subsequent code.

# In[ ]:


class crypto:
    def __init__(self, df, currency = 'XLM'):
        self.currency = currency
        df = df.rename(columns={'Unnamed: 0': 'Day'})
        self.df = df[(df['Symbol']==self.currency)]
        
        del df
    
    def fit(self, target='High', type_='AR', ARIMA_order=(1,1,1)):
        """
        Fit the model on the train set
        There is a choice between AR and ARIMA models.
        The ARIMA_order parameter is only used in the latter case.
        """
       # self.df['%s_shift' % (target)] = self.df[target].shift()
        #self.df.dropna(inplace=True)
        divide=len(self.df)-200
        self.train = self.df[:divide]
        self.test = self.df[divide:]
        self.target = target
        
        if type_=='AR':
            model = AR(self.train[target])
            
        elif type_=='ARIMA':
            model = ARIMA(self.train[target], order=ARIMA_order)
        self.model_fit = model.fit()
        print('Lag: %s' % self.model_fit.k_ar)
        print('Coefficients: %s' % self.model_fit.params)
        self.window = self.model_fit.k_ar
        self.coef = self.model_fit.params
        sns.lineplot(x = 'Day', y = 'High', data=self.train, label = 'train', color='b')
        sns.lineplot(self.train['Day'], y=self.model_fit.fittedvalues,  label = 'Fitted Values', color='r')
        plt.title('Fitted Model')
        plt.show()
        
    def fit_analysis(self):
        self.train['%s_predicted' % (self.target)] = self.model_fit.fittedvalues
        return self.train
    
    def plot_rolling_mean(self, window=7):
        sns.set(rc={'figure.figsize':(20,20)})
        sns.set(style="whitegrid", font_scale=1.5)
        sns.set_palette("Paired")
        sns.lineplot(x = self.train['Day'], y = self.train[self.target].rolling(window).mean(), data=self.train, label = 'Train rolling mean', color='b')
        sns.lineplot(x=self.train['Day'], y=self.model_fit.fittedvalues.rolling(window).mean(),  label = 'Fitted Values rolling mean', color='r')
        plt.title('Rolling Mean Window = %s' % (window))
        plt.show()
        
    def ups_and_downs(self, day_shift=1, type_='train'):
        """
        Shift the target by 1 day, and calculate the difference between today and yesterday.
        Drop NaNs and assign 0 to a down-shift and 1 to an up-shift in the training set.
        Repeat this process for the predicted values.
        """
        if type_=='train':
            train = self.train.copy()
            train.dropna(inplace=True)
            train['%s_shift' % (self.target)] = train[self.target].shift(day_shift)
            train.dropna(inplace=True)
            train['%s_shift_predicted' % (self.target)] = train['%s_predicted' % (self.target)].shift(day_shift)
            train.dropna(inplace=True)
            train['Diff_%s' % (self.target)] = train['%s_shift' % (self.target)] - train[self.target]
            train['Diff_%s_predicted' % (self.target)] = train['%s_shift_predicted' % (self.target)] - train['%s_predicted' % (self.target)]
            train['Ups_Downs_Real'] = np.where(train['Diff_%s' % (self.target)]<=0, 0, 1)
            train['Ups_Downs_Predicted'] = np.where(train['Diff_%s_predicted' % (self.target)]<=0, 0, 1)
            return train
        elif type_=='test':
            test = self.test.copy()
            test.dropna(inplace=True)
            test['%s_shift' % (self.target)] = test[self.target].shift(day_shift)
            test.dropna(inplace=True)
            test['%s_shift_predicted' % (self.target)] = test['pred'].shift(day_shift)
            test.dropna(inplace=True)
            test['Diff_%s' % (self.target)] = test['%s_shift' % (self.target)] - test[self.target]
            test['Diff_%s_predicted' % (self.target)] = test['%s_shift_predicted' % (self.target)] - test['pred']
            test['Ups_Downs_Real'] = np.where(test['Diff_%s' % (self.target)]<=0, 0, 1)
            test['Ups_Downs_Predicted'] = np.where(test['Diff_%s_predicted' % (self.target)]<=0, 0, 1)
            return test
        
    def predict(self, target='High'):
        """
        Predict the future!
        """
        history = self.train[target].iloc[len(self.train)-self.window:]
        history = [history.iloc[i] for i in range(len(history))]
        predictions = list()
        for t in range(len(self.test)):
            length = len(history)
            lag = [history[i] for i in range(length-self.window,length)]
            yhat = self.coef[0]
            for d in range(self.window):
                yhat += self.coef[d+1] * lag[self.window-d-1]
            obs = self.test[target].iloc[t]
            predictions.append(yhat)
            history.append(obs)
        self.predictions = predictions
        self.test['pred'] = self.predictions
        sns.set(rc={'figure.figsize':(10,10)})
        sns.set(style="whitegrid", font_scale=1.5)
        sns.set_palette("Paired")
        sns.lineplot(x = 'Day', y = 'High', data=self.test, label = 'test', color='b')
        sns.lineplot(self.test['Day'], y=self.predictions,  label = 'predictions', color='r')
        plt.title('%s' % (self.currency))
        plt.show()
        


# In[ ]:


df = pd.read_csv("../input/all_currencies.csv")


# Let us fit the model on the target 'High'

# In[ ]:


xlm = crypto(df=df, currency='XLM')
xlm.fit(target='High')


# It's difficult to gauge how well the model is fitting. It *looks* to be accurate, but of course we need to perform some analysis on this. Let us now call the *fit_analysis()* function:

# In[ ]:


tmp = xlm.fit_analysis()
tmp.head(25)


# The first 23 days are predicted as NaN. This is, of course, what we expect: the AR model has deemed the most accurate model to consist of a 23-day window. How accurate is our model at predicting the daily High value of XLM?

# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
tmp.dropna(inplace=True)
print('Mean Absolute Error: ', mean_absolute_error(tmp['High'], tmp['High_predicted']),'\n',
      'Mean Squared Error: ', mean_squared_error(tmp['High'], tmp['High_predicted']))
print('Mean Value of XLM: ', np.mean(tmp['High']))


# The average error compared to the average price seems to be rather low, meaning that we should be able to see quite a nice rolling mean

# In[ ]:


xlm.plot_rolling_mean(window=7)


# ## Ups-and-Downs
# 
# A rolling mean works well, but what about predicting the individual ups-and-downs?
# 
# We will drop the NaN rows and analyse how well our model can predict the ups-and-downs of the XLM behaviour.
# 
# Let's first look at the shift per day

# In[ ]:


ups_downs = xlm.ups_and_downs(day_shift=1)[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
ups_downs.head(15)


# Quite a few of these values are clearly wrong. Let's quantify this.

# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']])


# Not a good value at all! We are basically flipping a coin to decide whether the value of Stellar Lumens will increase or decrease. 
# 
# What about predicting the ups-and-downs over a long time frame, e.g. a week?

# In[ ]:


ups_downs = xlm.ups_and_downs(day_shift=7)[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
print('7 days: ', roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']]))

ups_downs = xlm.ups_and_downs(day_shift=14)[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
print('14 days: ',roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']]))


# The predictive power of our model over longer time periods is much stronger - not a surprise given:
# 1.  The noticeable trend of our fitted model to have an acceptable rolling mean
# 2. The volatility of cryptocurrency markets
# 
# **However** it's very important to remember that so far we are looking at the fitted model, and **not** the test set. 
# 
# Is the test score comparable to the train score?

# In[ ]:


xlm.predict()


# Again we see that the rolling mean will be an accurate value, and we should expect an accurate MAE / MSE. The up-and-down prediction we expect to be a little more spurious

# In[ ]:


print('Mean Absolute Error: ', mean_absolute_error(xlm.test['High'], xlm.predictions),'\n',
      'Mean Squared Error: ', mean_squared_error(xlm.test['High'], xlm.predictions))
print('Mean Value of XLM: ', np.mean(xlm.test['High']))


# In[ ]:


ups_downs = xlm.ups_and_downs(day_shift=1, type_='test')[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
ups_downs.head(15)


# In[ ]:


roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']])


# In[ ]:


ups_downs = xlm.ups_and_downs(day_shift=7, type_='test')[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
print('7 days: ', roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']]))

ups_downs = xlm.ups_and_downs(day_shift=14, type_='test')[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
print('14 days: ',roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']]))


# The model is not overfitting, as we see approximately the same metric scores on train and test. The one-day shift prediction is slightly better, but still not very accurate

# ## ARIMA
# 
# Let's go ahead and fit a more complex model - the [ARIMA model.](http://www.statsmodels.org/devel/generated/statsmodels.tsa.arima_model.ARIMA.html)
# There is an extra parameter that must be fed into this model - ARIMA_order=(p,d,q)
# 
# If we set d = q = 0 and p=23, we should retrieve the AR model that we had above.
# Let's check this
# 
# **Note: This fitting can be quite slow compared to a standard AR model**

# In[ ]:


xlm.fit(target='High', type_='ARIMA', ARIMA_order=(23,0,0))


# In[ ]:


xlm.fit(target='High', type_='ARIMA', ARIMA_order=(23,0,1))


# In[ ]:


xlm.fit_analysis()
ups_downs = xlm.ups_and_downs(day_shift=7)[['Ups_Downs_Real', 'Ups_Downs_Predicted']]
roc_auc_score(ups_downs[['Ups_Downs_Real']], ups_downs[['Ups_Downs_Predicted']])


# In[ ]:




