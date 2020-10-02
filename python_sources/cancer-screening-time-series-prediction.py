#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#commit version 10 
#this sho


# In[ ]:


import pandas as pd
import numpy as np
from datetime import datetime
import warnings  
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')

import time
import lightgbm as lgb
import xgboost as xgb
import seaborn as sns


from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold
from scipy import stats


import statsmodels.api as sm

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




locations = ["shingyi", "nanggang", "daan","nehoo" ]


# In[ ]:


#todo: 
1. how to deal with test train split /
2. dealing with missing vals /
3. predition using only trend /
4.establish baseline prediction/ 
5. calculate smape score /


#data visualizations 
1.time serties of each item at each store /
2.time series of every item in each store /
3.time series of each item at every stoer combines / 
4.all items time series /
5. decompostion of trend . seasonaty , noise /



#model predicitons 
1.arima/
2.exponetial smoothing / 
3.stationaty . naive /
moving average
4.neural network. pending 
5.fb prophet / 
6. mean/median of past vsits /
7. xgboost 
8. light gbm / pending 


#chanllenges 
1.not enough data points for good train/valid split 
2.no seasonality and however trend is possible 


# In[ ]:


#Load the datasets
df = pd.read_csv('../input/fm.csv')
df_og = df.copy()
df.date = pd.to_datetime(df.date, format='%Y')
df = df.set_index('date')
#show dimenion 
print("train shape: ", df.shape)
# Set of features we have are: date	location	item	visits
display(df.sample(10))


# In[ ]:


print(df.head(5))


# In[ ]:


#change column names to make it match sales prediction dataset, this makes our life easer 
print(df.columns)
df.columns = ['store','item','sales']
print(df.head(5))


# In[ ]:


locations = df.store.unique()
print(locations)
print("number of locations: ", len(locations))


# In[ ]:





# * <h1>data exploration</h1>[](http://)

# 

# In[ ]:


df_sns = df.copy()
df_sns["date"] = df_sns.index
print(df_sns.head()) 


# In[ ]:


print("all exams in all location combines (all exam)")
ax = sns.lineplot(x = "date", y = "sales",data = df_sns)
#     legend = ax.legend()
#     legend.texts[0].set_text("Whatever else")


# In[ ]:


print("all exams in every location")
locations = df_sns['store'].unique()
ax = sns.lineplot(x="date", y="sales", hue="store",data=df_sns)


# In[ ]:


print("all exams in every location")
locations = df_sns['store'].unique()
ax = sns.lineplot(x="date", y="sales", hue="store",data=df_sns[df_sns['store'].isin(locations[:4])])


# In[ ]:


print("all exams in every location")
locations = df_sns['store'].unique()
ax = sns.lineplot(x="date", y="sales", hue="store",data=df_sns[df_sns['store'].isin(locations[4:8])])


# In[ ]:


print("all exams in every location")
locations = df_sns['store'].unique()
ax = sns.lineplot(x="date", y="sales", hue="store",data=df_sns[df_sns['store'].isin(locations[8:12])])


# In[ ]:


print("all exams in every location")
locations = df_sns['store'].unique()
ax = sns.lineplot(x="date", y="sales", hue="store",data=df_sns[df_sns['store'].isin(locations[12:16])])


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt
locations = df_sns['store'].unique() 
for i, loc in enumerate(locations): 
    plt.figure(i)
    ax = sns.lineplot(x="date", y="sales", hue="item", style="store", data=df_sns[df_sns["store"]==loc])
    


# In[ ]:


import matplotlib.pyplot as plt
exams  = df_sns['item'].unique() 
for i, exam in enumerate(exams): 
    plt.figure(i)
    ax = sns.lineplot(x="date", y="sales", hue="store", style="item", data=df_sns[df_sns["item"]==exam])


# In[ ]:


#### Seasonality Check
# preparation: input should be float type
df_raw = df.copy()
df_raw['sales'] = df_raw['sales'] * 1.0


stores = []

for loc in locations: 
    stores.append(df_raw[df_raw.store == loc]['sales'].sort_index(ascending = True))

f, axs = plt.subplots(len(locations), figsize = (15, 52))
c = '#386B7F'

# store types
for i in range(len(locations)): 
    stores[i].resample('W').sum().plot(color = c, ax = axs[i],title = "                               "+locations[i])
        

#There seems to be no seasonality 
#daan showed increasinf trend 
#nehoo showed decreasing trend 


# In[ ]:





# In[ ]:


date_sales = df_raw.drop(['store','item'], axis=1).copy() #it's a temporary DataFrame.. Original is Still intact..

date_sales.get_ftype_counts()
y = date_sales['sales'].resample('YS').mean() 
print(y['2011':]) #sneak peak

import statsmodels.api as sm
#We can also visualize our data using a method called time-series decomposition that allows us to decompose our time series into three distinct components: 
#trend, seasonality, and noise.
decomposition = sm.tsa.seasonal_decompose(y, model='additive')

print("mean visits for every exam in every location")
decomposition.plot();
#The plot shows no seasonality which is to be expected. we should focus on trend prediction instead . 
#espceialy at locations 


# In[ ]:


#We can also visualize our data using a method called time-series decomposition that allows us to decompose our time series into three distinct components: 
#trend, seasonality, and noise.
decomposition = sm.tsa.seasonal_decompose(y, model='multiplicative')
decomposition.plot();
#The plot above clearly shows that the sales is unstable, along with its obvious seasonality.;


# In[ ]:


plt.ylim(0, 400)
sns.boxplot(x="date", y="visits", data=df_og)


# <h1>Does imputing missing values help? </h1> 

# In[ ]:


#imputing methods. 
#impute using mean 
#impute using median 
#impute with rolling mean 
df_imputed = df.copy()
df_imputed["mean"] = df_imputed["sales"]
df_imputed["median"] = df_imputed["sales"]

for i in  range(df_imputed.shape[0]): 
    row = df_imputed.iloc[i, :]
    if row[2] == 0:
       #find time series of particualt item and store 
#         print(row[3])
        historical_vals_df =  df_imputed[(df_imputed.store == row[0]) ]
        historical_vals_df =   historical_vals_df[( historical_vals_df.item == row[1]) ]
#         print(historical_vals_df)
        row[3] = int(np.mean(historical_vals_df.sales))
#         print(row[3])
        row[4] = int(np.median(historical_vals_df.sales))
#         print(row[4])
    df_imputed.iloc[i, :] = row
       


# In[ ]:





# <h1> Train valid split</h1>

# In[ ]:





# In[ ]:


#train valid split 
#use 2011 to 2017 data to predict 2018 data 
#theorectically we need  a trian/valid/test set but i fear we do not have enough data points 
train = df_imputed.loc['2011-01-01':'2017-02-01']
valid = df_imputed.loc['2017-02-01':'2018-02-01']

valid_x = valid.iloc[:,:-1]
valid_y = valid.iloc[:,-1]
print(train.head())
print(valid.head())


# <h1> prediciton section </h1>

# In[ ]:





# 

# In[ ]:


df = df_imputed


# In[ ]:


#error from 0 -100 %
def smape(y_true, y_pred): 
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 100.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)


def score(y_label,y_pred,verbose = 0,target = "sales"):
    if verbose == 1: 
        print("you predictied: ",y_pred)
        print("label: ",y_label)
    y_pred = [x if x > 0 else 0 for x in y_pred]
    print("smape score(",target,"):",smape(y_label,y_pred))
    #plot here 


# In[ ]:





# In[ ]:


y_true = np.array(3)
y_pred = np.ones(1)
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)


# train/valid split 
# 
# 

# In[ ]:





# <h1>mean prediction</h1> 
# 

# In[ ]:


def mean_prediction(valid_x, train,target = "sales"):
    #from valid_x find average of all item,store combo 
    preds = []
    for index, row in valid_x.iterrows():
        preds.append(mean_pred_single(row.values[0],row.values[1],train, target = target))
    return preds
        
    

def mean_pred_single(store, item ,train, target = "sales"):
    historical_vals_df =  train[(train.store == store) ]
    historical_vals_df =   historical_vals_df[( historical_vals_df.item == item) ]
    return np.mean(historical_vals_df[target])

pred = mean_prediction(valid_x,train)
score(valid_y,pred, verbose =0 )
    
pred = mean_prediction(valid_x,train, target  = "median")
score(valid_y,pred, verbose =0 ,target  = "median")

pred = mean_prediction(valid_x,train, target = "mean")
score(valid_y,pred, verbose =0 , target = "mean")


# <h1>median prediction: a baseline likely hard to beat </h1>

# In[ ]:


def median_prediction(valid_x, train, target = "sales"):
    #from valid_x find average of all item,store combo 
    preds = []
    for index, row in valid_x.iterrows():
        preds.append(median_pred_single(row.values[0],row.values[1],train, target = target ))
    return preds
        
    

def median_pred_single(store, item ,train, target = "sales"):
    historical_vals_df =  train[(train.store == store) ]
    historical_vals_df =   historical_vals_df[( historical_vals_df.item == item) ]
    return np.median(historical_vals_df[target])


pred = median_prediction(valid_x,train)
score(valid_y,pred, verbose =0 )
    
pred = median_prediction(valid_x,train, target  = "median")
score(valid_y,pred, verbose =0 ,target  = "median")

pred = median_prediction(valid_x,train, target = "mean")
score(valid_y,pred, verbose =0 , target = "mean")
    


# <h1>naive baseline : same as last year. </h1>

# In[ ]:


def last_prediction(valid_x, train):
    #from valid_x find average of all item,store combo 
    preds = []
    for index, row in valid_x.iterrows():
        preds.append(last_pred_single(row.values[0],row.values[1],train))
    return preds
        
    

def last_pred_single(store, item ,train):
    historical_vals_df =  train[(train.store == store) ]
    historical_vals_df =   historical_vals_df[( historical_vals_df.item == item) ]
    historical_vals_df=historical_vals_df.sort_index()
#     print(historical_vals_df)
#     print( historical_vals_df.iloc[-1,2])
    
    return historical_vals_df.iloc[-1,2]

pred = last_prediction(valid_x,train)

score(valid_y,pred,verbose = 0)
    


# <h1>exponential smoothing </h1>

# In[ ]:


def exponential_prediction(valid_x, train, target = 'sales'):
    #from valid_x find average of all item,store combo 
    preds = []
    for index, row in valid_x.iterrows():
        preds.append(exponential_pred_single(row.values[0],row.values[1],train, target = target ))
    return preds
        
    

def exponential_pred_single(store, item ,train, target = "sales"):
    historical_vals_df =  train[(train.store == store) ]
    historical_vals_df =   historical_vals_df[( historical_vals_df.item == item) ]
    historical_vals_df=historical_vals_df.sort_index()
    p = historical_vals_df[target].ewm(alpha = 0.9).mean()[-1]
    return p

pred = exponential_prediction(valid_x,train)

score(valid_y,pred,verbose = 0)

pred = exponential_prediction(valid_x,train, target  = "median")
score(valid_y,pred, verbose =0 ,target  = "median")

pred = exponential_prediction(valid_x,train, target = "mean")
score(valid_y,pred, verbose =0 , target = "mean")


# <h1>double exponential smoothing</h1> 

# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

train_exp = train.copy()
train_exp = train.replace(0,0)
import warnings  
warnings.filterwarnings('ignore')

def exponential_prediction(valid_x, train, target = "sales"):
    #from valid_x find average of all item,store combo 
    preds = []
    for index, row in valid_x.iterrows():
        preds.append(exponential_pred_single(row.values[0],row.values[1],train,target = target))
    return preds
        
    

def exponential_pred_single(store, item ,train, target = "sales"):
    historical_vals_df =  train[(train.store == store) ]
    historical_vals_df =   historical_vals_df[( historical_vals_df.item == item) ]
    historical_vals_df=historical_vals_df.sort_index()
    model = ExponentialSmoothing(historical_vals_df[target],trend = 'add',damped = True ,seasonal = None )
    model_fit = model.fit(smoothing_level = 1, smoothing_slope  = 1)
    p = model_fit.predict(8,8)
    return p.tolist()[0]

pred =exponential_prediction(valid_x,train)
score(valid_y,pred, verbose =0 )
    
pred = exponential_prediction(valid_x,train, target  = "median")
score(valid_y,pred, verbose =0 ,target  = "median")

pred = exponential_prediction(valid_x,train, target = "mean")
score(valid_y,pred, verbose =0 , target = "mean")


# In[ ]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing




def prediction(valid_x, train, target = 'sales'):
    #from valid_x find average of all item,store combo 
    preds = []
    for index, row in valid_x.iterrows():
        preds.append(pred_single(row.values[0],row.values[1],train, target = target
                                ))
    return preds
        
    

def pred_single(store, item ,train, target = 'sales'):
    historical_vals_df =  train[(train.store == store) ]
    historical_vals_df =   historical_vals_df[( historical_vals_df.item == item) ]
    historical_vals_df=historical_vals_df.sort_index()
    vals = historical_vals_df.sales.tolist() 
    return vals[-1] + (vals[-1]-vals[-2])

pred = prediction(valid_x,train)

score(valid_y,pred,verbose = 0)

pred = prediction(valid_x,train, target  = "median")
score(valid_y,pred, verbose =0 ,target  = "median")

pred =prediction(valid_x,train, target = "mean")
score(valid_y,pred, verbose =0 , target = "mean")


# In[ ]:


np.__version__


# <h1>FB prophet !!! </h1>

# In[ ]:


from fbprophet import Prophet
import matplotlib.pyplot as plt 



def fb_prediction(valid_x, train):
    #from valid_x find average of all item,store combo 
    preds = []
    for index, row in valid_x.iterrows():
        preds.append(fb_pred_single(row.values[0],row.values[1],train))
    return preds
        
    

def fb_pred_single(store, item ,train):
    historical_vals_df =  train[(train.store == store) ]
    historical_vals_df =   historical_vals_df[( historical_vals_df.item == item) ]
    historical_vals_df=historical_vals_df.sort_index()
    df_fb = pd.DataFrame()
    df_fb["Y"] = historical_vals_df["median"]
    df_fb["DS"] = df_fb.index
    df_fb = df_fb[['DS', 'Y']]
    df_fb.columns = ['ds','y']
    m = Prophet()
#     print(df_fb)
    m.fit(df_fb)
    future = m.make_future_dataframe(periods=1,freq='Y')
#     print(future.tail())
    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)

#     plt.show()
    print(forecast["yhat"].tolist()[-1])
    return forecast["yhat"].tolist()[-1]
    
pred = fb_prediction(valid_x,train)

score(valid_y,pred,verbose = 0)


# <h1> ARIMA</h1>

# #train a arima model for every time series. 
# #caveats : 
# 1.

# In[ ]:



np.seterr(divide='ignore', invalid='ignore')

stores = train.store.unique()
items = train.item.unique()
# print(stores)

train_arima = train.copy()
train_arima = train_arima.replace(0,1)
# print(train_arima)

counter = 0
success = 0
pred = []
for store in stores: 
    for item in items:
        print(store,item)
        counter += 1
        train_df = train_arima[(train_arima['store']==store) &(train_arima['item']==item)]
        try:
            arima = sm.tsa.ARIMA(train_df.sales, (1,1,0)).fit(disp=False)
#             print(arima.summary())
            p = arima.forecast(1)[0]
            pred.append(p[0])
            print(arima.forecast(1)[0])
            success += 1
        except Exception as e:
            print("shit")
            pred.append(median_pred_single(store, item, train))
            
print(counter, success)
print(pred)

score(valid_y,pred, verbose =0 )


# <h1>Does imputing missing values help? </h1> 

# In[ ]:





# In[ ]:






# In[ ]:


df_imputed


# In[ ]:




