#!/usr/bin/env python
# coding: utf-8

# ### Task Details:
# 
# The outbreak of Covid-19 is developing into a major international crisis, and it's starting to influence important aspects of daily life. For example:
# 
# * Travel: Bans have been placed on hotspot countries, corporate travel has been reduced, and flight fares have dropped.
# * Supply chains: International manufacturing operations have often had to throttle back production and many goods solely produced in China have been halted altogether.
# * Grocery stores: In highly affected areas, people are starting to stock up on essential goods.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, Bidirectional, GRU
from keras.optimizers import SGD, Adadelta, RMSprop, Adam, Nadam
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping
from tqdm import tqdm

rcParams['figure.figsize'] = 12, 7
pd.options.display.max_columns = 50
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Let's take a look into the data avaliable

# In[ ]:


covid_open_line = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv')
#Drop empty columns
drop_list = [i for i in covid_open_line.columns if 'Unnamed' in i] 
covid_open_line = covid_open_line.drop(columns = drop_list)


# In[ ]:


covid_open_line.head()


# In[ ]:


covid_open_line.info()


# In[ ]:


covid_line = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
#Drop empty columns
drop_list = [i for i in covid_line.columns if 'Unnamed' in i] 
covid_line = covid_line.drop(columns = drop_list)


# In[ ]:


len(covid_line.symptom.unique())


# In[ ]:


covid_line.head()


# In[ ]:


covid_line.info()


# In[ ]:


covid = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')


# In[ ]:


covid.head()


# In[ ]:


covid.info()


# ### Let's see how the pandemic is growing in the world

# In[ ]:


df = covid.groupby('ObservationDate').agg({'Confirmed':'sum', 'Deaths':'sum', 'Recovered':'sum'}).reset_index()

x=df.ObservationDate.values
y=[df.Confirmed.values, df.Deaths.values, df.Recovered.values]
 
# Plot
pal = sns.color_palette("Set1")

plt.stackplot(x,y, labels=['Confirmed','Deaths','Recovered'], colors=pal, alpha=0.7)
plt.legend(loc='upper left')
plt.show()


# ### We can see the confirmed cases are still increasing. We can also see a increase in recovered cases as people from the beggining are healing and a increase in deaths as more people get infected. Let's see how is China (the starting point).

# In[ ]:


df_china = covid[covid['Country/Region'] == 'Mainland China'].groupby('ObservationDate').agg({'Confirmed':'sum', 'Deaths':'sum', 'Recovered':'sum'}).reset_index()

x=df_china.ObservationDate.values
y=[df_china.Confirmed.values, df_china.Deaths.values, df_china.Recovered.values]
 
# Plot
pal = sns.color_palette("Set1")

plt.stackplot(x,y, labels=['Confirmed','Deaths','Recovered'], colors=pal, alpha=0.7)
plt.legend(loc='upper left')
plt.show()


# ### We can see that in China there is almost no new cases and the recovered cases are almost the same as the confirmed. 

# ### Let's see how the recovery and death ratio are globally and in China

# In[ ]:


df['recovered_ratio'] = df['Recovered'] / df['Confirmed']
df['death_ratio'] = df['Deaths'] / df['Confirmed']

sns.lineplot(x="ObservationDate", y="recovered_ratio", data=df)
sns.lineplot(x="ObservationDate", y="death_ratio", data=df)


# In[ ]:


df_china['recovered_ratio'] = df_china['Recovered'] / df_china['Confirmed']
df_china['death_ratio'] = df_china['Deaths'] / df_china['Confirmed']

sns.lineplot(x="ObservationDate", y="recovered_ratio", data=df_china)
sns.lineplot(x="ObservationDate", y="death_ratio", data=df_china)


# ### We can see that China has a recovery ratio of 80% and increasing, meaning that they already have the spreading under control. Globally the recovery ratio was increasing but new countrys getting infected can be the reason behind this new decline in recovery. Deaths ratio are stable around 3%.
# ### Let's see if time since first case and recovery are correlated

# In[ ]:


covid['ObservationDate'] = pd.to_datetime(covid['ObservationDate'])
covid_start = covid.groupby(['Country/Region', 'Province/State'])['ObservationDate'].min().reset_index()
covid_start.columns = ['Country/Region', 'Province/State', 'FirstObservation']
covid = covid.merge(covid_start, on = ['Country/Region', 'Province/State'])


# In[ ]:


covid['TimeSinceFirst'] = (covid['ObservationDate'] - covid['FirstObservation']).dt.days
covid['RecoveredRatio'] = covid['Recovered'] / covid['Confirmed']
covid['DeathsRatio'] = covid['Deaths'] / covid['Confirmed']


# In[ ]:


covid[['Confirmed', 'TimeSinceFirst', 'RecoveredRatio', 'DeathsRatio']].corr()


# In[ ]:


(covid.loc[covid['ObservationDate'] == '2020-03-14', 'TimeSinceFirst']).hist()


# ### We can see that time since first case and recovery are correlated, and that many regions are less that a week away from first case. So spreading time is a important factor. Let's see if we have a pattern

# In[ ]:


covid['WeeksSinceFirst'] = round(covid['TimeSinceFirst'] / 7)
covid_week = covid.groupby(['Country/Region', 'Province/State', 'WeeksSinceFirst'])['Confirmed'].max().reset_index(name='LastValueWeek')
# covid_mean = covid_week.groupby(['WeeksSinceFirst']).agg({'max': [np.mean, np.std]}).reset_index()
# covid_mean.columns = ['WeeksSinceFirst', 'mean', 'std']
sns.lineplot(x="WeeksSinceFirst", y="LastValueWeek", data=covid_week)


# ### We can see that cases evolution are not equal in different places, mostly because of population size, govement measures, etc. that we can't measure with the data.

# ### Another important factor is age. The disease is worse in the elderly

# In[ ]:


def clean_age(x):
    try:
        x = int(x)
    except ValueError:
        x = np.nan
    
    return x


# In[ ]:


covid_open_line.age.apply(clean_age).hist()


# ### As we can see there are little cases between ages 0 and 20, as more from 40 to 60. Unfortunally we only have 14k cases in the open line dataset, so we cant use age for now.

# # Models
# 
# ### We have data until march, so we are going to leave this last month out for validation

# ### RNN 1 week

# In[ ]:


combinations = covid[['Province/State', 'Country/Region']].drop_duplicates().reset_index(drop=True)
len(combinations)


# In[ ]:


TARGETS = ['Confirmed', 'Deaths', 'Recovered']
cols = ['D-1', 'D-2', 'D-3', 'D-4', 'D-5', 'D-6', 'D-7']
cols_final = ['ObservationDate', 'Province/State', 'Country/Region'] + TARGETS
df_final = pd.DataFrame(columns=cols_final)

for j in tqdm(range(len(combinations))):

    PROVINCE = combinations['Province/State'][j]
    COUNTRY = combinations['Country/Region'][j]

    df_city = pd.DataFrame(columns=['ObservationDate', 'Province/State', 'Country/Region'])

    dates = []
    for i in range(1,32):
        if i < 10:
            date = '2020/03/0'+str(i)
        else:
            date = '2020/03/'+str(i)
        dates.append(date)

    df_city['ObservationDate'] = dates
    df_city['Province/State'] = PROVINCE
    df_city['Country/Region'] = COUNTRY
    
    covid_slice = covid[(covid['Province/State'] == PROVINCE) & (covid['Country/Region'] == COUNTRY)]
    
    covid_slice = covid_slice.sort_values('ObservationDate').reset_index(drop=True)
        
    if (len(covid_slice) < 15) & (len(covid_slice[covid_slice['ObservationDate'] >= '2020-03-01']) < 7):
        continue

    for TARGET in TARGETS:

        for i in range(1, len(covid_slice)):
            covid_slice.loc[i, 'D-1'] = covid_slice.loc[i-1, TARGET]
        for i in range(2, len(covid_slice)):
            covid_slice.loc[i, 'D-2'] = covid_slice.loc[i-2, TARGET]
        for i in range(3, len(covid_slice)):
            covid_slice.loc[i, 'D-3'] = covid_slice.loc[i-3, TARGET]
        for i in range(4, len(covid_slice)):
            covid_slice.loc[i, 'D-4'] = covid_slice.loc[i-4, TARGET]
        for i in range(5, len(covid_slice)):
            covid_slice.loc[i, 'D-5'] = covid_slice.loc[i-5, TARGET]
        for i in range(6, len(covid_slice)):
            covid_slice.loc[i, 'D-6'] = covid_slice.loc[i-6, TARGET]
        for i in range(7, len(covid_slice)):
            covid_slice.loc[i, 'D-7'] = covid_slice.loc[i-7, TARGET]

        covid_slice = covid_slice.fillna(0)

        train = covid_slice[covid_slice['ObservationDate'] < '2020-03-01']
        test = covid_slice[covid_slice['ObservationDate'] >= '2020-03-01']

        X_train = train[cols].values.reshape(len(train), 7, 1)
        y_train = train[TARGET].values

        X_test = test[cols].values.reshape(len(test), 7, 1)
        y_test = test[TARGET].values

        reg = l1_l2(l1=0.0015, l2=0.0)

        opt = Adam(lr=0.0015) 

        model = Sequential()
        model.add(Bidirectional(LSTM(140, activation='relu', return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg), 
                                input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Bidirectional(LSTM(140, activation='relu', return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg)))
        model.add(Bidirectional(LSTM(140, activation='relu', kernel_regularizer=reg, recurrent_regularizer=reg)))
        model.add(Dense(28))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.compile(loss='mean_absolute_error', optimizer=opt)

        history = model.fit(X_train, y_train, epochs=600, batch_size=256, validation_data=(X_test, y_test),
                            verbose=0, shuffle=False,callbacks=[EarlyStopping(patience=10)])

        df_target = pd.DataFrame(columns=['ObservationDate', 'Province/State', 'Country/Region', TARGET])

        for i in range(1,32):
            if i < 10:
                date = '2020-03-0'+str(i)
            else:
                date = '2020-03-'+str(i)

            if i <= len(X_test):
                df_slice = covid_slice[covid_slice['ObservationDate'] < date].tail(7).reset_index(drop=True)

            else:
                df_slice = df_target[df_target['ObservationDate'] < date].tail(7).reset_index(drop=True)

            to_predict = [df_slice[TARGET][6], df_slice[TARGET][5], df_slice[TARGET][4], df_slice[TARGET][3], 
                         df_slice[TARGET][2], df_slice[TARGET][1], df_slice[TARGET][0]]

            to_predict = np.array(to_predict).reshape(1,7,1)

            y_hat = model.predict(to_predict)

            df_target = df_target.append({'ObservationDate': date,'Province/State': PROVINCE,
                                        'Country/Region': COUNTRY, TARGET: round(y_hat[0][0])}, ignore_index=True)

        df_city = df_city.merge(df_target, on = ['ObservationDate', 'Province/State', 'Country/Region'])
    
    df_city = df_city[cols_final]
    df_final = df_final[cols_final]
        
    df_final = pd.concat([df_final, df_city])


# In[ ]:


df_final.head()

