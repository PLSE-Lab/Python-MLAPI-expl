#!/usr/bin/env python
# coding: utf-8

# In this notebook, as in last weeks (week 2), we are going to fit logitic curves to each intersection and save the estimated parameters for prediction on test data.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
import scipy.optimize as opt
import os


# In[ ]:


for dirname, _, filenames in os.walk('../'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:



train = pd.read_csv("/kaggle/input/coronascraper/timeseries-tidy.csv")
train.head()


# In[ ]:


train[train["country"] == 'South Africa'].head()


# In[ ]:


train = train[train["level"] == 'country']
train = train.drop(columns = ['name','city','county','state','population','lat','long','aggregate','tz'])
train = train.pivot_table(values='value',index=['country','date'],columns='type').reset_index()
train.index.name = train.columns.name = None
train.head()


# Need to replace all NULLs with country:

# In[ ]:


#train_=train
#EMPTY_VAL = "EMPTY_VAL"

#def fillState(state, country):
#    if state == EMPTY_VAL: return country
#    return state

#train_['Province_State'].fillna(EMPTY_VAL, inplace=True)
#train_['Province_State'] = train_.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
#test['Province_State'].fillna(EMPTY_VAL, inplace=True)
#test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)
#test.head()


# In[ ]:


train['deaths'] = train['deaths'].fillna(0)
train['cases'] = train['cases'].fillna(0)
train = train[train["cases"] > 0]
train.head(20)


# Run a graph and see how the logistic curve fits:

# In[ ]:


train['row_number'] = train.groupby(['country']).cumcount()
x = train[train["country"] == 'United Kingdom']['row_number']
y = train[train["country"] == 'United Kingdom']['cases']
y_ = train[train["country"] == 'United Kingdom']['deaths']
#print(x,y,y_)


# In[ ]:


y_.head(10)


# In[ ]:




def f(x, L, b, k, x_0):
    return L / (1. + np.exp(-k * (x - x_0))) + b


def logistic(xs, L, k, x_0):
    result = []
    for x in xs:
        xp = k*(x-x_0)
        if xp >= 0:
            result.append(L / ( 1. + np.exp(-xp) ) )
        else:
            result.append(L * np.exp(xp) / ( 1. + np.exp(xp) ) )
    return result

p0 = [max(y), 0.0,max(x)]
p0_ = [max(y_), 0.0,max(x)]
x_ = np.arange(0, 150, 1).tolist()
try:
    popt, pcov = opt.curve_fit(logistic, x, y,p0)
    yfit = logistic(x_, *popt)
    popt_, pcov_ = opt.curve_fit(logistic, x, y_,p0_)
    yfit_ = logistic(x_, *popt_)
except:
    popt, pcov = opt.curve_fit(f, x, y, method="lm", maxfev=10000)
    yfit = f(x_, *popt)
    popt_, pcov_ = opt.curve_fit(f, x, y_, method="lm", maxfev=10000)
    yfit_ = f(x_, *popt_)
    #print("problem")


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(x, y, 'o', label ='Actual Cases')
ax.plot(x_, yfit, '-', label ='Fitted Cases')

ax.plot(x, y_, 'o', label ='Actual Fatalities')
ax.plot(x_, yfit_, '-', label ='Fitted fatalities')
ax.title.set_text('South Africa')
plt.legend(loc="center right")
plt.show()


# Create a dimension of the country/states:

# In[ ]:


unique = pd.DataFrame(train_.groupby(['Country_Region', 'Province_State'],as_index=False).count())
unique.head()


# Fit Logistic curves to data:

# In[ ]:


import datetime as dt

def date_day_diff(d1, d2):
    delta = dt.datetime.strptime(d1, "%Y-%m-%d") - dt.datetime.strptime(d2, "%Y-%m-%d")
    return delta.days

log_regions = []

for index, region in unique.iterrows():
    st = region['Province_State']
    co = region['Country_Region']
    
    rdata = train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]

    t = rdata['Date'].values
    t = [float(date_day_diff(d, t[0])) for d in t]
    y = rdata['ConfirmedCases'].values
    y_ = rdata['Fatalities'].values

    p0 = [max(y), 0.0, max(t)]
    p0_ = [max(y_), 0.0, max(t)]
    try:
        popt, pcov = opt.curve_fit(logistic, t, y, p0, maxfev=10000)
        try:
            popt_, pcov_ = opt.curve_fit(logistic, t, y_, p0_, maxfev=10000)
        except:
            popt_, pcov_ = opt.curve_fit(f, t, y_,method="trf", maxfev=10000)
        log_regions.append((co,st,popt,popt_))
    except:
        popt, pcov = opt.curve_fit(f, t, y,method="trf", maxfev=10000)
        popt_, pcov_ = opt.curve_fit(f, t, y_,method="trf", maxfev=10000)
        log_regions.append((co,st,popt,popt_))

print("All done!")


# Give dimension column headers:

# In[ ]:


log_regions = pd.DataFrame(log_regions)
log_regions.columns = ['Country_Region','Province_State','ConfirmedCases','Fatalities']
log_regions.head(1)


# Test one country:

# In[ ]:


T = np.arange(0, 100, 1).tolist()
popt = list(log_regions[log_regions["Country_Region"] == 'Italy'][log_regions["Province_State"] == 'Italy']['ConfirmedCases'])[0]
popt_ = list(log_regions[log_regions["Country_Region"] == 'Italy'][log_regions["Province_State"] == 'Italy']['Fatalities'])[0]

try:
    yfit = logistic(T, *popt)
    yfit_ = logistic(T, *popt_)
except:
    yfit = f(T, *popt)
    yfit_ = f(T, *popt_)
    

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.plot(T, yfit, label="Fitted ConfirmedCases")
ax.plot(T, yfit_, label="Fitted Fatalities")
ax.title.set_text('Italy fitted params')
plt.legend(loc="upper left")
plt.show()


# In[ ]:


for index, rt in log_regions.iterrows():
    st = rt['Province_State']
    co = rt['Country_Region']
    popt = list(['ConfirmedCases'])
    popt_ = list(rt['Fatalities'])
    print(co,st,popt,popt_)


# So there are a lot of countries that dont have fatalities just yet, but we are going to go with the data that says that meaningful cases lead inevitably to fatalities. To do that we need to estimate fatality parameters for each curve, im going to looking to the medians across the populaiton of countries to infer the params.

# In[ ]:


data0 = log_regions['Fatalities'].str[0]/log_regions['ConfirmedCases'].str[0]
data1 = log_regions['Fatalities'].str[1]/log_regions['ConfirmedCases'].str[1]
data2 = log_regions['Fatalities'].str[2]/log_regions['ConfirmedCases'].str[2]
bins = np.arange(0, 3, 0.01)
plt.hist(data1,bins=bins, alpha=0.5)
plt.xlim([0,3])
plt.ylabel('count')
plt.ylim([0,5])
plt.show()
fp = np.array([data0.median(),data1.median(),data2.median()])
fp


# Check that we can infer:

# In[ ]:


for index, rt in log_regions.iterrows():
    st = rt['Province_State']
    co = rt['Country_Region']
    popt = list(rt['ConfirmedCases'])
    popt_ = list(rt['Fatalities'])
    
    if popt_ == [0.0,0.0,70.0]:
        popt_ = np.multiply(fp,popt)
        print(co,st,popt,popt_)


# Apply to test data:
# edit: gonna try estimate based in case curve, say 3.5%:

# In[ ]:


submission = []

for index, rt in log_regions.iterrows():
    st = rt['Province_State']
    co = rt['Country_Region']
    popt = list(rt['ConfirmedCases'])
    popt_ = list(rt['Fatalities'])
    if popt_ == [0.0,0.0,70.0]:
        #popt_ = np.multiply(fp,popt)
        popt_ = np.array([popt[0]*0.035,popt[1],popt[2]])
    print(co,st,popt,popt_)
    rtest = test[(test['Province_State']==st) & (test['Country_Region']==co)]
    for index, rt in rtest.iterrows():
        try:
            tdate = rt['Date']
            ca = logistic([date_day_diff(tdate, min(train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]['Date'].values))], *popt)
            try:
                fa = logistic([date_day_diff(tdate, min(train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]['Date'].values))], *popt_)
            except:
                fa = f([date_day_diff(tdate, min(train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]['Date'].values))], *popt_)
            submission.append((rt['ForecastId'], int(ca[0]), int(fa[0])))
        except:
            tdate = rt['Date']
            ca = f([date_day_diff(tdate, min(train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]['Date'].values))], *popt)
            fa = f([date_day_diff(tdate, min(train_[(train_['Province_State']==st) & (train_['Country_Region']==co)]['Date'].values))], *popt_)
            submission.append((rt['ForecastId'], int(ca[0]), int(fa[0])))

print("All done!") 


# Submit predictions:

# In[ ]:


submission = pd.DataFrame(submission)
submission.columns = ['ForecastId','ConfirmedCases','Fatalities']
submission.to_csv('./submission.csv', index = False)
print("submission ready!")

