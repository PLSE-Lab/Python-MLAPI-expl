#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")


# In[ ]:


df.head()


# In[ ]:


df


# In[ ]:


df = df.drop('Sno',axis=1)


# In[ ]:


df.head()


# In[ ]:


df=df.replace(to_replace ="China", 
                 value ="Mainland China")
df['Country'].value_counts().plot.bar()


# In[ ]:


df['Country'].value_counts().plot.pie()


# In[ ]:


print('Total Confirmed Cases:',df['Confirmed'].sum())
print('Total Deaths: ',df['Deaths'].sum())
print('Total Recovered Cases: ',df['Recovered'].sum())


# In[ ]:


affected = df['Confirmed'].sum()
died = df['Deaths'].sum()
recovered = df['Recovered'].sum()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


labels = ['affected','died','recovered']
sizes = [15, 30, 45, 10]
explode = (.1, .1, .1)  # only "explode" the 2nd slice (i.e. 'Hogs')

plt.pie([affected,died,recovered],explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.legend()


# In[ ]:


df1 = df.groupby(["Country"]).sum()


# In[ ]:


df1.plot.bar()


# In[ ]:


china_df = df.loc[df['Country'] == 'Mainland China']


# In[ ]:


china_df['Province/State'].value_counts().plot.pie()


# In[ ]:


china_df['Province/State'].value_counts().plot.bar()


# In[ ]:


print('Total Confirmed Cases in china:',china_df['Confirmed'].sum())
print('Total Deaths in china: ',china_df['Deaths'].sum())
print('Total Recovered Cases i china: ',china_df['Recovered'].sum())


# In[ ]:


affected = china_df['Confirmed'].sum()
died = china_df['Deaths'].sum()
recovered = china_df['Recovered'].sum()


# In[ ]:


labels = ['affected','died','recovered']
sizes = [15, 30, 45, 10]
explode = (.1, .1, .1)  # only "explode" the 2nd slice (i.e. 'Hogs')

plt.pie([affected,died,recovered],explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.legend()


# In[ ]:


## top ten countried
# top10_countries=df[['Country']].nlargest(10,['Confirmed']).reset_index(drop=True)

highest_affected = df[['Province/State','Confirmed','Last Update']].nlargest(10, 'Confirmed')


# In[ ]:


highest_affected.set_index('Last Update')[['Confirmed']].plot.bar()


# In[ ]:


new_df = df.set_index('Country')


# In[ ]:


highest_death = new_df[['Deaths','Last Update']].nlargest(10, 'Deaths').plot.bar()


# In[ ]:


df.groupby(["Province/State"]).sum().plot()


# In[ ]:


highest_affected = df[['Province/State','Confirmed','Last Update']]


# In[ ]:


china_df.groupby(["Province/State"]).sum()[['Confirmed']].plot.bar()


# In[ ]:


china_df.groupby(["Province/State"]).sum()[['Deaths']].plot.bar()


# In[ ]:


china_df.groupby(["Province/State"]).sum()[['Recovered']].plot.bar()


# In[ ]:


df.groupby(["Date"]).sum()['Confirmed'].plot()


# In[ ]:


df.groupby(["Date"]).sum()['Deaths'].plot()


# In[ ]:


df.groupby(["Date"]).sum()['Recovered'].plot()


# In[ ]:


df.isnull().sum()


# In[ ]:


df3 = df[['Last Update','Deaths']]


# In[ ]:


df3.set_index('Last Update').plot()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.corr()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


df3


# In[ ]:


df3.columns = ['ds', 'y']


# In[ ]:


df3


# In[ ]:


from fbprophet import Prophet


# In[ ]:


m = Prophet()
m.fit(df3)


# In[ ]:


future = m.make_future_dataframe(periods=365)


# In[ ]:


forecast = m.predict(future)


# In[ ]:


forecast


# In[ ]:


m.plot_components(forecast)


# In[ ]:


m.plot(forecast,uncertainty=True)


# In[ ]:


df4 = df[['Last Update','Confirmed']]


# In[ ]:


df4.columns = ['ds', 'y']


# In[ ]:


m = Prophet()
m.fit(df4)


# In[ ]:


future = m.make_future_dataframe(periods=365)


# In[ ]:


forecast = m.predict(future)


# In[ ]:


forecast


# In[ ]:


m.plot_components(forecast)


# In[ ]:


m.plot(forecast,uncertainty=True)


# In[ ]:


df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
df.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


df.Date = pd.to_datetime(df.Date)
#df = df.set_index("Month")
df.head()


# In[ ]:


df2 = df[['Date','Confirmed','Deaths','Recovered']]


# In[ ]:


df2.head()


# In[ ]:


X = df2[['Date']]
Y = df2[['Confirmed']]


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X,Y)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
scaled_xtrain_data = scaler.transform(x_train)
scaled_xtest_data = scaler.transform(x_test)


# In[ ]:


scaled_xtest_data


# In[ ]:


from keras.preprocessing.sequence import TimeseriesGenerator


# In[ ]:


n_input = scaled_xtrain_data.shape[1]
n_features= 1
generator = TimeseriesGenerator(scaled_xtrain_data, scaled_xtrain_data, length=n_input, batch_size=1)


# In[ ]:


generator


# In[ ]:


scaled_xtrain_data.shape


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(200, activation='relu', input_shape=(1,1)))
lstm_model.add(Dense(200,activation='relu'))
lstm_model.add(Dense(200,activation='relu'))
lstm_model.add(Dense(200,activation='relu'))
lstm_model.add(Dense(200,activation='relu'))
lstm_model.add(Dense(200,activation='relu'))
lstm_model.add(Dense(200,activation='relu'))
lstm_model.add(Dense(200,activation='relu'))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

lstm_model.summary()


# In[ ]:


lstm_model.fit_generator(generator,epochs=20)


# In[ ]:


import matplotlib.pyplot as plt
losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_lstm)),losses_lstm);

