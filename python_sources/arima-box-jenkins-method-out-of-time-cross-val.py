#!/usr/bin/env python
# coding: utf-8

# **Beginners notes on selecting ARIMA parameters using the Box-Jenkins Method**

# 
# We need to make forecasts for some time series data.  
# 
# **The raw data and its description can be downloaded [here](https://www.itl.nist.gov/div898/handbook/pmc/section6/pmc621.htm)**  
# 
# To make predictions we need a model. One such model is called **ARIMA** which stands for "**A**utoRegressive **I**ntegrated **M**oving **A**verage".  
# The model is written as ARIMA(p,d,q) where   
#                         ${\:\:\:\:p\:}$ is the number of lags from the time series used for autoregression(**AR**),  
#                                           ${\:\:\:\:d\:}$ is the order of differencing (**I**) (explained below) and   
#                                            ${\:\:\:\:q\:}$ is the moving average window(**MA**)   ${\:\:\:}$i.e number of terms  to consider for performing the moving average operation.  
# 
# **Modeling**  
# Now we move onto the modeling  i.e. We need to find decent values for parameters ${p}$,${\:d}$ and ${q}$.  
# One of the methods  is the **Box-Jenkins Method** which is what this post is about. :)  
# 
# **Out-Of-Time Cross Validation** means that we split the data into training and test series.  
# Forecasting requires the order of the series to stay intact. Therefore we **don't shuffle or randomize the order** like we do usually with other ML tasks.  
# We build the ARIMA models on the training data and see how they fare on the test series using some metrics.
# 
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
import pandas as pd
import math
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm

### CONSTANTS USED 
TRAIN_PERC=0.90
PLT_WIDTH=5
PLT_HEIGHT=9
PLT_DPI=120
###
data=[]
file=None
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        file=os.path.join(dirname,filename)

#print(file)



#load the data and generate the time series
#interested only in the third column
df=pd.read_csv(file,delim_whitespace=True,header=None,names=['J1','J2','Y','J3','J4'])
df.head()
time_series=df.iloc[:,2]
#remove Nan and non-digits
time_series.dropna(inplace=True)
time_series=time_series.drop([0,1])


time_series=time_series.astype(float)
time_series=time_series.reset_index(drop=True)
##CREATE TRAIN AND TEST DATA 
###
trainlen=math.floor(time_series.size*TRAIN_PERC)
test_series=time_series[trainlen:]
train_series=time_series[:trainlen]

###


#display the time series
#https://machinelearningmastery.com/time-series-data-visualization-with-python/
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(PLT_HEIGHT,PLT_WIDTH), 'figure.dpi':PLT_DPI})
plt.rcParams.update({'font.size':9})
plt.rc('figure', titlesize=9)
#Line Plot


fig,ax=plt.subplots(2,2)
train_series.plot(ax=ax[0,0])
train_series.hist(ax=ax[0,1])
plot_acf(train_series,ax=ax[1,0],title='Auto')
plot_pacf(train_series,ax=ax[1,1],title='')

plt.show()



# 
# 
# There's trend in the data i.e. the time series is non-stationary as is visible from the autocorrelation plot.   
# Box Jenkings assumes a stationary time series where the mean and the standard deviation don't change with time.  
# Such series shows no trend.  
# 
# **Differencing**  
# Convertion of a non-stationary series to a stationary can be achieved using "differencing".   
# 
# There are orders to differencing.  
# First order  is ${\:\:\Delta_{1} y_{t}=Y_{t}-Y_{t-1}}$.   
# Second order is ${\:\:\Delta_{2} y_{t}=\Delta_{1} y_{t} - \Delta_{1} y_{t-1}=Y_{t}-2Y_{t-1}+Y_{t-2}}$.  
# ${\vdots}$  
# N-th order is ${\:\:\Delta_{i} y_{t}=\Delta_{i-1} y_{t} - \Delta_{i-1} y_{t-1}}$
# 
# where ${Y_{t}}$ is the ${t}$-th sample in the time series.   
# 
# The task is to find the order that looks like white noise. White noise is random noise and therefore it's distribution is Gaussian.  
# 
# 
# We can write a recursive function for getting the i-th order difference  
# 
#  
# 
# 

# In[ ]:


def difference(series,order):
    """
       Parameters:
                    Input:
                        series = input time series
                        order= order of differencing
                        
                    Returns:
                        returns a series consisting difference values of order=order
        
      
    """
    
    if order == 0:
        return series
    else:
        diff=series.diff(-1)
        order_=order-1
        return difference(series=diff,order=order_)


# In[ ]:



zod=difference(train_series,0)   #zeroth-order differencing = original series
zod=zod.rename('0')
fod=difference(train_series,1)   #first-order
fod=fod.rename('1')
sod=difference(train_series,2)   #second-order
sod=sod.rename('2')
tod=difference(train_series,3)   #third-order
tod=tod.rename('3')
#print(type(zod))
ll=[zod,fod,sod,tod]

#keys=[s.name for s in fod]
#print(zod.name)
df=pd.concat([zod,fod,sod,tod],axis=1,keys=[s.name for s in ll])
print(df.head())


# The order of difference with the lowest standard deviation is a good choice usually.
# 

# In[ ]:


df.describe()


# The first order difference looks like the one with the least std dev.  
# First impression would be select ${d=1}$  
# 

#  We define a function to create [4 plots](https://www.itl.nist.gov/div898/handbook/eda/section3/4plot.htm) which are a set of the following plots that are useful while evaluating a model.
#  1. run sequence plot  (time indices versus response variables)
#  2. lag plot (${y_t}$ versus ${y_{t-1}}$ . useful to check if there are any autocorrelations of lag-1)
#  3. normal probability distribution plot -(q-q-plot) (check if data can be approximated by the normal distribution [please read this ](https://www.itl.nist.gov/div898/handbook/eda/section3/normprpl.htm)) . 
#  4. histogram (freq distribution )  
# 
# 
#  In addition to the above I also plot the ACF and PACF.  
#  

# In[ ]:


from pandas.plotting import lag_plot

def six_plots(sr):
    
    sr=sr.dropna()
    plt.rcParams.update({'figure.figsize':(PLT_HEIGHT,PLT_WIDTH), 'figure.dpi':PLT_DPI})
    fontdict={'fontsize':9,'verticalalignment':'bottom'}
    fig,ax=plt.subplots(2,3)
    sr.plot(ax=ax[0,0])  #plot the series
    sr.hist(ax=ax[0,1]) #must be gaussian like
    sm.qqplot(sr,ax=ax[0,2],line='45') # how close does the series fit the normal distribution
    lag_plot(sr,ax=ax[1,0]) #lag-1 plot to see autocorrelations   
    plot_acf(sr,ax=ax[1,1],title='') #acf plot
    plot_pacf(sr,ax=ax[1,2],title='') #pacf plot
    
    #set the titles in the correct place. 
    #https://matplotlib.org/gallery/pyplots/text_layout.html#sphx-glr-gallery-pyplots-text-layout-py
    left = 0.45
    bottom = -0.5
    top = 1.2
    
    #for the top 3 plots, titles are on the top
    ax[0,0].text(left, top, 'run sequence',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax[0,0].transAxes)
    ax[0,1].text(left, top, 'hist',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax[0,1].transAxes)
    ax[0,2].text(left, top, 'Q-Q',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax[0,2].transAxes)
    ax[0,2].set_xlabel('')
    ax[0,2].set_ylabel('')
    
    #for the bottom 3 plots , titles are at the bottom
    ax[1,0].text(left, bottom, 'Lag-plot',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax[1,0].transAxes)
    ax[1,1].text(left, bottom, 'ACF',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax[1,1].transAxes)    
    ax[1,2].text(left, bottom, 'PACF',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax[1,2].transAxes)
    
    fig.tight_layout()
    fig.suptitle('')
    plt.show()
    


# In[ ]:


import matplotlib.gridspec as gridspec

six_plots(df['0'])
six_plots(df['1'])
six_plots(df['2'])


# The residuals appear to have Gaussian distribution (almost). The aim is to make the residuals follow a white noise type distribution. Completely random i.e. the more the residuals appear like a Gaussian distribution, the better.  
# In conjuction with the fact that the first order difference has the least std dev , we may justify a choice of ${d=1}$.  
# However let's run a test to check this.  
# 
# Let's see if the series after differencing is stationary using the **Augmented Dickey Fuller (ADF) Test**.  
# The ADF lets us know if a series is stationary.  
# **The null hypothesis here is that the series is non-stationary**. 
# If the ADF statistic is greater than it's p-value (p-value is the probability in favor of the null-hypothesis, larger p-values imply that we cannot reject the null hypothesis) , then we accept the null hypothesis else reject it with an alternate i.e the series is stationary.    
# We carry out the ADF for various orders of difference and then select the value of ${d}$ for which the series shows stationary behaviour
# 

# In[ ]:


from statsmodels.tsa.stattools import adfuller

def ADF(sr):
    """
    Augmented Dickey Fuller Test.
    """
    sr=sr.dropna() #remove any invalids
    results=adfuller(sr)
    print('ADF statistic: ',results[0])
    print('p-value: ',results[1]) #probability that the null hypothesis is true
    print('Critical vals: ')
    for i,j in results[4].items():
        print(i,j)

print('difference order = 1')        
ADF(df['1'])
print()
print('difference order = 2')      
ADF(df['2'])
#print('Aresults)


# we select ${d=1}$ based on the ADF result.  Note we can run the same tests for ${\:d=2,\:d=3}$. But I will using the principle of parsimony (Occams Razor).  
# 
# The statistic is lower than the critical value at 1%, so we reject the null hypothesis. Our differenced series is stationary with ${d=1}$  

# 
# Now we move on to parameters ${p}$ and ${q}$
# The MA term ${q}$ is associated with the ACF plot.   
# Wherever the ACF drops to within the confidence interval is candidate. Therefore ${q=1}$ seems like a good choice.  
#   
# ${p}$ is the corresponding AR term and it is associated with the PACF plot.  
# The value where the PACF value cuts to within the confidence interval is a reasonable choice. Hence ${p=2}$.   
# 
# The models ARIMA(2,1,0), ARIMA(0,1,1) and ARIMA(2,1,1) are models we will consider further for analysis
# 
# 

# **Evaluation of the models**  
# For evaluating a model we need a test. The **Ljunge-Box Test** provides us with such a measure .  
# The test is done using the residuals after the fit of the model and **NOT** the original time series.  
# 
# The test results in a statistic ${Q}$ defined ${Q=n(n+2){\Huge\sum}_{k=1}^{m}\large\frac{\hat{r}_{k}^{2}}{n-k}}$  where  ${\hat{r}_{k}}$ is the estimated autocorrelation of the series at lag ${k}$ and ${m}$ is the number of lags being tested and ${n}$ is the length of the time series.  
# 
# 
# 
# 
# **The null hypothesis ${H_{0}}$ is that our model ARIMA(${p,\:d,\:q}$) is a good fit**. i.e the residuals are random white noise. 
# 
# We get to reject ${H_{0}}$ in the critical region which is the area of the distribution determined by      
# 
#  ${\:\: Q\:>\:\:{\chi^{2}_{1-\alpha,h}}}\:$   
#  
#  where ${\:{\chi^{2}_{1-\alpha,h}}\:\:}$ is the chi-square distribution    
#  ${\:\:\:\:\:\:\:\:\:\:\:h}$ is the degrees of freedom  
#  ${\:\:\:\:\:\:\:\:\:\:\:\alpha}$ is the level of significance  
#  
#  Extreme values of Q have less chances of occurence. We need to find the probability associated value of Q for this series. A high ${P}$-value indicates failure to reject the null hypothesis i.e. a good fit. A low ${P}$ value (${\:P<0.05\:@\: 5\%\: significance\:level\:}$) is indicative of the alternate hypothesis i.e our fit is bad.     
#  
# 

# In[ ]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox as ljungbox

from scipy.stats import chi2


def chi_square_table(p,dof):
    """
    https://stackoverflow.com/questions/32301698/how-to-build-a-chi-square-distribution-table
    
    Parameters:
            Input:
                p= p-value 
                dof = degree of freedom
            Returns:
                chi-sq critical value corresponding to (p,dof)
    
    """
    return chi2.isf(p,dof)


def chi_sq_critical_val(alpha,dof):
    """
    return the critical val (c) for chi-sq distrib parameterized by 
    probability(pr)=1-alpha and degrees of freedom=dof 
    c is the value at and below which pr% of data exists
    
    """
    pr=1-alpha
    val=chi2.ppf(pr,dof)
    return val

    


def eval_arima(series,order,lags,dynamic=False,alpha=0.05):
    """
    1.fit the model 
    2.get the residuals
    3.plot the residuals
    4.does it look like white noise? mean=0, normally distributed?
    5.calculate Q on the residuals for number of lags
    6.choose a level of significance
    7.choose degrees of freedom
    8.calculate the critical value of the chi-sq statistic
    9.accept or reject null hypothesis

    Parameters:
            Input:
                series          = time series to be fit by the ARIMA model
                order           = 3-tuple of form (p,d,q) where p=AR terms. d=order of differencing, q= MA terms of an ARIMA(p,d,q) model
                dynamic         = True ==> out-of-sample (predict unseen (test) data),
                                  False ==> in-sample  (predict on the data trained on)
                alpha           = significance level
                lags            = number of lags used to calculate the Ljung-Box Q statistic
            Return:
                    fitted model
    """


    plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

    #fit the model
    model=ARIMA(series,order=order)   #ARIMA(0,1,1) model
    model_fit=model.fit(disp=-1)

    #print(type(model_fit))
    print(model_fit.summary())
    
    #display the fit of the model
    model_fit.plot_predict(dynamic=dynamic).suptitle("model fit on training data")
    plt.show()
    
   
    #get the residuals
    residuals=model_fit.resid
    #plot the residuals
    fig,ax=plt.subplots(1,2)

    residuals.plot(title='Residuals',ax=ax[0])
    residuals.plot(kind='kde',title='probability distribution of residuals',ax=ax[1])
    #print(model_fit.)
    plt.show()
    
    #are the residuals random?
    print(residuals.describe())
    #autocorrelation plots of residuals
    six_plots(residuals)
   
    #Significance Level at 5%
    #alpha=0.05

    #The Ljung-Box Test 
    Q,p=ljungbox(residuals,range(1,lags),boxpierce=False)
    c=[]
    for i in range(len(Q)):
        dof=i+1                
        c.append(chi_sq_critical_val(alpha,dof))
        #print('Chi-statistic(Q) :',Q[i],'  p-value:',p[i],'   critical value: ',c," KEEP H0" if Q[i]<c else "DNT KEEP H0")
    
    #plot Q versus c
    #accept if Q stays below the 45 deg line i.e Q<c
    arstr="ARIMA"+str(order)+""
    plt.plot(c,Q,label=arstr)
    plt.plot(c,c,label='c=Q')
    plt.xlabel('Q values')
    plt.ylabel('critical values')
    plt.title('Ljung - Box Test')
    plt.legend()
    plt.show()
    return model_fit
    


# We now evaluate each of our 3 models.

# In[ ]:



arima_011=eval_arima(train_series,order=(0,1,1),lags=25)
arima_210=eval_arima(train_series,order=(2,1,0),lags=25)
arima_211=eval_arima(train_series,order=(2,1,1),lags=25)


# how well do the models forecast the test data?  
# 

# In[ ]:


def arima_forecast(model,test_sr,train_sr):
    """
    Forecast arima models on the test series (test_sr)
    Parameters:
        Input:
            model= arima model used for forecasting
            test_sr = test series for forecasting
            train_sr= training data used to build model
        Returns:
            dictionary containing metric values
            
    """
    fc,se,cf= model.forecast(test_sr.size,alpha=0.05)
    #Convert to series

    fc_series=pd.Series(fc,index=test_sr.index)
    lower_cf=pd.Series(cf[:,0],index=test_sr.index)
    upper_cf=pd.Series(cf[:,1],index=test_sr.index)

    #plotting
    plt.plot(train_sr,label='training')
    plt.plot(test_sr,label='test')
    plt.plot(fc_series,label='forecast')
    plt.fill_between(lower_cf.index,lower_cf,upper_cf,
                    color='k',alpha=0.15)
    plt.legend()
    plt.show()
    
    #forecast accuracies
    actual=test_sr.values
    forecast=fc_series
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse,
            'corr':corr, 'minmax':minmax})
    
    


# In[ ]:


arima011_metrics=arima_forecast(arima_011,test_sr=test_series,train_sr=train_series)
arima210_metrics=arima_forecast(arima_210,test_sr=test_series,train_sr=train_series)
arima211_metrics=arima_forecast(arima_211,test_series,train_series)


# We now have some metrics for each of the models which we wish to compare.  
# I will put them in a dataframe and plot each column and pick the model which looks the best. 

# In[ ]:


lm=[arima011_metrics,arima210_metrics,arima211_metrics]
dlmdf=pd.DataFrame(lm)
dlmdf.head()


# In[ ]:



f,axx=plt.subplots(3,3)    
dlmdf['mape'].plot(ax=axx[0,0])
dlmdf['me'].plot(ax=axx[0,1])
dlmdf['mae'].plot(ax=axx[0,2])
dlmdf['mpe'].plot(ax=axx[1,0])
dlmdf['rmse'].plot(ax=axx[1,1])
dlmdf['corr'].plot(ax=axx[1,2])
dlmdf['minmax'].plot(ax=axx[2,0])
#axx[2,1].setvisible(False)
f.delaxes(axx[2,1])
f.delaxes(axx[2,2])
f.tight_layout()


# from the plots of the metrics above, **ARIMA(2,1,0)** looks like a good choice.
# 
# In Summary:  
# We chose ${d}$ by differencing the original series and then by using the ADF test.  
# We chose ${p}$ by looking at the PACF plots of the stationary residuals.  
# We chose ${q}$ by looking at the ACF plots of the stationary residuals.
# 
# I will be covering automated arima modeling for the same dataset in the coming posts which will cover a wider range of the parameters ${\:d\:,p\:,q\:}$. 
# 
# 
# 
# 
# 
# 
# 
# 
# 
