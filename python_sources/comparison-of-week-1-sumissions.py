#!/usr/bin/env python
# coding: utf-8

# 10 days have passed since week 1 private LB period started.  
# Here I compared some submissions of week-1 which are in public.  
# The ranking is as of 2020-4-5.   
# - 2nd place [COVID19_LightGBM](https://www.kaggle.com/osciiart/covid19-lightgbm?scriptVersionId=30830623)
# - 4th place [COVID19: prediction with restcountries.eu API data](https://www.kaggle.com/nonserial/covid19-prediction-with-restcountries-eu-api-data)
# - 9th place [SARIMAX Baseline Starter Prediction](https://www.kaggle.com/skeller/sarimax-influenza-baseline?scriptVersionId=30638995)
# - 10th place [Fork of france_missing_value_and_postprocessing](https://www.kaggle.com/pietromarinelli/8th-place-at-day-1-with-lgb-with-few-features)
# - 19th place [COVID-19 Global Forecasts Using Probabilistic RNNs](https://www.kaggle.com/jlgleason/covid-19-global-forecasts-using-probabilistic-rnns?scriptVersionId=30839206)
# - 23rd place [COVID19 | Minimalist polynomial regressor proposal](https://www.kaggle.com/franlopezguzman/covid19-minimalist-polynomial-regressor-proposal?scriptVersionId=30837988)
# - 24th place [Logistic Curve Fitting - Global Covid-19](https://www.kaggle.com/diamondsnake/covid-19-logistic-curve-fitting-week-1?scriptVersionId=30844536)

# In[ ]:


import os, gc, pickle, copy, datetime, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import metrics
pd.set_option('display.max_columns', 100)
warnings.filterwarnings('ignore')


# # Preprocessing

# In[ ]:


df_test = pd.read_csv("../input/my-covid-pred/test_week1.csv")
df_test.head()


# In[ ]:


df_test['Province_State'] = df_test['Province/State'] 
df_test['Country_Region'] = df_test['Country/Region'] 


# In[ ]:


df_week3 = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")
df_week3.head()


# In[ ]:


df_test = pd.merge(df_test, df_week3, on=['Province_State', 'Country_Region', 'Date'], how='left')
df_test.head()


# In[ ]:


df_test['Date'] = pd.to_datetime(df_test['Date'])
df_test['day'] = df_test['Date'].apply(lambda x: x.dayofyear).astype(np.int16)
df_week3['Date'] = pd.to_datetime(df_week3['Date'])
df_week3['day'] = df_week3['Date'].apply(lambda x: x.dayofyear).astype(np.int16)
df_test.head()


# In[ ]:


# concat Country/Region and Province/State
def func(x):
    try:
        x_new = x['Country_Region'] + "/" + x['Province_State']
    except:
        x_new = x['Country_Region']
    return x_new
        
df_test['place_id'] = df_test.apply(lambda x: func(x), axis=1)
df_week3['place_id'] = df_week3.apply(lambda x: func(x), axis=1)
df_test.head()


# In[ ]:


tmp = df_test[pd.isna(df_test['ConfirmedCases'])==False]['Date'].max()
print("last day of existing true data: {}".format(tmp))


# In[ ]:


day_before_private = 85
df_tmp = df_test[df_test['day']<day_before_private]
df_tmp = df_tmp[pd.isna(df_tmp['ConfirmedCases'])]
tmp = np.sort(df_tmp['place_id'].unique())
print("differences of places between week-1 and week-3")
print(tmp)
# df_tmp.head()


# In[ ]:


df_sub2 = pd.read_csv("../input/my-covid-pred/submission_osciiart.csv") # my final submission with bug
df_sub4 = pd.read_csv("../input/my-covid-pred/submission_Nonserial.csv")
df_sub9 = pd.read_csv("../input/my-covid-pred/submission_Stephen Keller.csv")
df_sub10 = pd.read_csv("../input/my-covid-pred/submission_Pietro Marinelli.csv")
df_sub19 = pd.read_csv("../input/my-covid-pred/submission_Uncharted_UCB.csv")
df_sub23 = pd.read_csv("../input/my-covid-pred/submission_Fran Lpez Gumn.csv")
df_sub24 = pd.read_csv("../input/my-covid-pred/submission_ODI6s.csv")
df_sub2.head()


# # Comparison

# In[ ]:


# list of places
places_sort = df_test[['place_id', 'ConfirmedCases']][df_test['day']==day_before_private]
places_sort = places_sort.sort_values('ConfirmedCases', ascending=False).reset_index(drop=True)['place_id'].values
print(len(places_sort))
for i, place in enumerate(places_sort):
    print(i, place)


# In[ ]:


def show_graph(place):
    sns.set()
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize = (15,6)) 
    plt.subplot(1,2,1)
    x_pred = df_test[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Date'].values
    y_pred2 = df_sub2[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values
    y_pred4 = df_sub4[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values
    y_pred9 = df_sub9[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values
    y_pred10 = df_sub10[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values
    y_pred19 = df_sub19[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values
    y_pred23 = df_sub23[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values
    y_pred24 = df_sub24[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['ConfirmedCases'].values
    x_true= df_week3[(df_week3['place_id']==place)]['Date'].values
    y_true = df_week3[(df_week3['place_id']==place)]['ConfirmedCases'].values

    ax = sns.lineplot(x=x_pred, y=y_pred2, label='#2 LGB')
    #     ax.set(yscale="log")
    sns.lineplot(x=x_pred, y=y_pred4, label='#4 LGB')
    sns.lineplot(x=x_pred, y=y_pred9, label='#9 SARIMAX')
    sns.lineplot(x=x_pred, y=y_pred10, label='#10 LGB')
    sns.lineplot(x=x_pred, y=y_pred19, label='#19 RNN')
    sns.lineplot(x=x_pred, y=y_pred23, label='#23 PolyReg')
    sns.lineplot(x=x_pred, y=y_pred24, label='#24 LogFit')
    sns.lineplot(x=x_true, y=y_true, label='true')
    plt.ylim(-1, y_true.max()*2)
    plt.title("{}/ConfirmedCases".format(place))

    fig.autofmt_xdate()
    plt.subplot(1,2,2)
    x_pred = df_test[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Date'].values
    y_pred2 = df_sub2[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values
    y_pred4 = df_sub4[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values
    y_pred9 = df_sub9[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values
    y_pred10 = df_sub10[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values
    y_pred19 = df_sub19[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values
    y_pred23 = df_sub23[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values
    y_pred24 = df_sub24[(df_test['place_id']==place) & (df_test['day']>day_before_private)]['Fatalities'].values
    x_true= df_week3[(df_week3['place_id']==place)]['Date'].values
    y_true = df_week3[(df_week3['place_id']==place)]['Fatalities'].values
    ax = sns.lineplot(x=x_pred, y=y_pred2, label='#2 LGB')
    #     ax.set(yscale="log")
    #     sns.lineplot(x=df_interest3['day'], y=df_interest2['Fatalities'].values, label='pred2')
    sns.lineplot(x=x_pred, y=y_pred4, label='#4 LGB')
    sns.lineplot(x=x_pred, y=y_pred9, label='#9 SARIMAX')
    sns.lineplot(x=x_pred, y=y_pred10, label='#10 LGB')
    sns.lineplot(x=x_pred, y=y_pred19, label='#19 RNN')
    sns.lineplot(x=x_pred, y=y_pred23, label='#23 PolyReg')
    sns.lineplot(x=x_pred, y=y_pred24, label='#24 LogFit')
    sns.lineplot(x=x_true, y=y_true, label='true')
    plt.ylim(-1, y_true.max()*2)
    plt.title("{}/Fatalities".format(place))
    fig.autofmt_xdate()
    plt.show()


# In[ ]:


show_graph(places_sort[0])
show_graph(places_sort[1])
show_graph(places_sort[2])
show_graph(places_sort[3])
show_graph(places_sort[4])
show_graph(places_sort[10])
show_graph(places_sort[20])
show_graph(places_sort[50])
show_graph(places_sort[100])
show_graph(places_sort[150])
show_graph(places_sort[200])
show_graph("Japan")
show_graph("Korea, South")


# In[ ]:




