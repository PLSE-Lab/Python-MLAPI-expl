#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd


# In[ ]:


data=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.shape


# In[ ]:


data.drop('Sno',axis=1,inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna().sum()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


numerical_features = [feature for feature in data.columns if data[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
data[numerical_features].head()


# In[ ]:


data[numerical_features].corr()


# In[ ]:


sns.heatmap(data[numerical_features].corr())


# In[ ]:


data.groupby('Date')['Confirmed'].plot()
plt.xlabel('Date')
plt.ylabel('Confirmed')
plt.title("Date_vs_Confirmed")


# In[ ]:


data.groupby('Date')['Deaths'].plot()
plt.xlabel('Date')
plt.ylabel('Deaths')
plt.title("Date_vs_Deaths")


# In[ ]:


data.groupby('Date')['Cured'].plot()
plt.xlabel('Date')
plt.ylabel('Cured')
plt.title("Date_vs_Cured")


# In[ ]:


data.groupby('State/UnionTerritory')['Confirmed'].plot()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Confirmed')
plt.title("State/UnionTerritory_vs_Confirmed")


# In[ ]:


data.groupby('State/UnionTerritory')['Deaths'].plot()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Deaths')
plt.title("State/UnionTerritory_vs_Confirmed")


# In[ ]:


data.groupby('State/UnionTerritory')['Cured'].plot()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Cured')
plt.title("State/UnionTerritory_vs_Confirmed")


# In[ ]:


data.groupby('State/UnionTerritory')['Cured'].median().plot()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Cured')
plt.title("State/UnionTerritory_vs_Cured")


# In[ ]:


data.groupby('State/UnionTerritory')['Cured'].mean().plot()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Cured')
plt.title("State/UnionTerritory_vs_Cured")
plt.figure(figsize=(15,8))
plt.show()


# In[ ]:


data.groupby('State/UnionTerritory')['Confirmed'].mean().plot()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Confirmed')
plt.title("State/UnionTerritory_vs_Confirmed")
plt.figure(figsize=(15,8))
plt.show()


# In[ ]:


data.groupby('State/UnionTerritory')['Confirmed'].median().plot()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Confirmed')
plt.title("State/UnionTerritory_vs_Confirmed")
plt.figure(figsize=(15,8))
plt.show()


# In[ ]:


data.groupby('State/UnionTerritory')['Deaths'].mean().plot()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Deaths')
plt.title("State/UnionTerritory_vs_Death")
plt.figure(figsize=(15,8))
plt.show()


# In[ ]:


data.groupby('State/UnionTerritory')['Deaths'].median().plot()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Deaths')
plt.title("State/UnionTerritory_vs_Death")
plt.figure(figsize=(15,8))
plt.show()


# In[ ]:


data.groupby('State/UnionTerritory')['Deaths'].median().plot.bar()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Deaths')
plt.title("State/UnionTerritory_vs_Death")
plt.figure(figsize=(15,8))
plt.show()


# In[ ]:


data.groupby('State/UnionTerritory')['Confirmed'].median().plot.bar()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Confirmed')
plt.title("State/UnionTerritory_vs_Confirmed")
plt.figure(figsize=(15,8))
plt.show()


# In[ ]:


data.groupby('State/UnionTerritory')['Cured'].median().plot.bar()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Cured')
plt.title("State/UnionTerritory_vs_Cured")
plt.figure(figsize=(15,8))
plt.show()


# In[ ]:


data.groupby('State/UnionTerritory')['Cured'].mean().plot.bar()
plt.xlabel('State/UnionTerritory')
plt.ylabel('Cured')
plt.title("State/UnionTerritory_vs_Cured")
plt.figure(figsize=(15,8))
plt.show()


# In[ ]:


categorical_features=[feature for feature in data.columns if data[feature].dtypes=='O']
categorical_features


# In[ ]:


data[categorical_features].head()


# In[ ]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(data[feature].unique())))


# In[ ]:


for feature in categorical_features:
    data=data.copy()
    data.groupby(feature)['Confirmed'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Confirmed')
    plt.title(feature)
    plt.show()


# In[ ]:


for feature in categorical_features:
    data=data.copy()
    data.groupby(feature)['Confirmed'].mean().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Confirmed')
    plt.title(feature)
    plt.show()


# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot(data)


# In[ ]:


sns.countplot('State/UnionTerritory',data=data)


# In[ ]:


state_cases = data.groupby('State/UnionTerritory')['Confirmed','Deaths','Cured'].max().reset_index()
import plotly.graph_objects as go


# In[ ]:


state_cases = state_cases.sort_values(by='Confirmed', ascending=True)

fig = go.Figure(data=[go.Bar(name='Confirmed', x = state_cases['Confirmed'], 
                             y = state_cases['State/UnionTerritory'],
                             orientation='h',marker_color='#5642C5'),
                      go.Bar(name='Cured', x=state_cases['Cured'], 
                             y=state_cases['State/UnionTerritory'],
                             orientation='h', marker_color='#00974E'),
                      go.Bar(name='Deaths', x=state_cases['Deaths'],
                             y=state_cases['State/UnionTerritory'],
                             orientation='h', marker_color='#EC2566')
                     ])

fig.update_layout(plot_bgcolor='white', 
                  barmode='stack', height=900)
fig.show()


# In[ ]:


state_cases['Death rate per 100'] = np.round((100*state_cases["Deaths"]/state_cases["Confirmed"]), 2)
state_cases['Active'] = state_cases['Confirmed'] - (state_cases['Cured'] + state_cases['Deaths'])
state_cases['Cure rate per 100'] = np.round((100*state_cases["Cured"]/state_cases["Confirmed"]), 2)
state_cases = state_cases.sort_values(by='Confirmed', ascending=False)
state_cases.style.background_gradient(cmap='Blues',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Cured"])                        .background_gradient(cmap='Blues', subset=["Active"])                        .background_gradient(cmap='Reds', subset=["Death rate per 100"])                        .background_gradient(cmap='Greens', subset=["Cure rate per 100"])


# In[ ]:





# In[ ]:


import plotly.express as px
state_cases['Death rate per 100'] = np.round((100*state_cases["Deaths"]/state_cases["Confirmed"]), 2)
fig = px.scatter(state_cases, y='State/UnionTerritory', x='Death rate per 100', color='State/UnionTerritory', size='Death rate per 100')
fig.update_layout(title={
                  'text': "Death rate per 100 in each state",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
                  plot_bgcolor='rgb(275, 275, 275)', 
                  height=650,
                  showlegend=False)


# In[ ]:


locations = {
    "Kerala" : [10.8505,76.2711],
    "Maharashtra" : [19.7515,75.7139],
    "Karnataka": [15.3173,75.7139],
    "Telangana": [18.1124,79.0193],
    "Uttar Pradesh": [26.8467,80.9462],
    "Rajasthan": [27.0238,74.2179],
    "Gujarat":[22.2587,71.1924],
    "Delhi" : [28.7041,77.1025],
    "Punjab":[31.1471,75.3412],
    "Tamil Nadu": [11.1271,78.6569],
    "Haryana": [29.0588,76.0856],
    "Madhya Pradesh":[22.9734,78.6569],
    "Jammu and Kashmir":[33.7782,76.5762],
    "Ladakh": [34.1526,77.5770],
    "Andhra Pradesh":[15.9129,79.7400],
    "West Bengal": [22.9868,87.8550],
    "Bihar": [25.0961,85.3131],
    "Chhattisgarh":[21.2787,81.8661],
    "Chandigarh":[30.7333,76.7794],
    "Uttarakhand":[30.0668,79.0193],
    "Himachal Pradesh":[31.1048,77.1734],
    "Goa": [15.2993,74.1240],
    "Odisha":[20.9517,85.0985],
    "Andaman and Nicobar Islands": [11.7401,92.6586],
    "Puducherry":[11.9416,79.8083],
    "Manipur":[24.6637,93.9063],
    "Mizoram":[23.1645,92.9376],
    "Assam":[26.2006,92.9376],
    "Meghalaya":[25.4670,91.3662],
    "Tripura":[23.9408,91.9882],
    "Arunachal Pradesh":[28.2180,94.7278],
    "Jharkhand" : [23.6102,85.2799],
    "Nagaland": [26.1584,94.5624],
    "Sikkim": [27.5330,88.5122],
    "Dadra and Nagar Haveli and Daman and Diu":[20.1809,73.0169],
    "Lakshadweep":[10.5667,72.6417],
    "Daman and Diu":[20.4283,72.8397] , 
    'State Unassigned':[0,0]
}


# In[ ]:


import requests 
india_json = requests.get('https://api.rootnet.in/covid19-in/unofficial/covid19india.org/statewise').json()
loc_india = pd.io.json.json_normalize(india_json['data']['statewise'])
loc_india = loc_india.set_index("state")
for index in loc_india.index :
    loc_india.loc[loc_india.index == index,"Lat"] = locations[index][0]
    loc_india.loc[loc_india.index == index,"Long"] = locations[index][1]


# In[ ]:


import folium 
from folium.plugins import HeatMap, HeatMapWithTime


covid_area = folium.Map(location=[20.5937, 78.9629], zoom_start=15,max_zoom=4,min_zoom=3,
                          tiles='CartoDB positron',height = 500,width = '70%')

HeatMap(data=loc_india[['Lat','Long','confirmed']].groupby(['Lat','Long']).sum().reset_index().values.tolist(),
        radius=18, max_zoom=14).add_to(covid_area)

covid_area


# In[ ]:


data


# In[ ]:


data['Date'].unique()


# In[ ]:


s= data.groupby('Date')['Confirmed'].sum()


# In[ ]:


s


# In[ ]:


NEW=pd.DataFrame(s,index=data['Date'].unique())


# In[ ]:


NEW


# In[ ]:


NEW.reset_index(inplace=True)


# In[ ]:


confirmed_data=NEW['Confirmed']


# In[ ]:


df1=np.array(confirmed_data).reshape(-1,1)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(confirmed_data).reshape(-1,1))


# In[ ]:


training_size=int(len(df1)*0.80)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[ ]:


training_size,test_size


# In[ ]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)


# In[ ]:


time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[ ]:


print(X_train.shape), print(y_train.shape)


# In[ ]:


print(X_test.shape), print(ytest.shape)


# In[ ]:


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
import tensorflow


# In[ ]:


model=Sequential()
model.add(LSTM(25,return_sequences=True,input_shape=(10,1)))
model.add(LSTM(25))               
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[ ]:


model.summary()


# In[ ]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=32,verbose=1)


# In[ ]:


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
ytest = np.asarray(ytest)


# In[ ]:


X_train.shape


# In[ ]:


train_predict=model.predict(X_train)


# In[ ]:


train_predict


# In[ ]:


train_predict=scaler.inverse_transform(train_predict)


# In[ ]:


train_predict


# In[ ]:


y_train.shape


# In[ ]:


Y=scaler.inverse_transform(y_train.reshape((104,1)))


# In[ ]:


Y


# In[ ]:


import math
from sklearn.metrics import mean_squared_error,mean_absolute_error
math.sqrt(mean_squared_error(Y,train_predict))


# In[ ]:


plt.plot(Y)
plt.plot(train_predict)


# In[ ]:


test_predict=model.predict(X_test)


# In[ ]:


test_predict=scaler.inverse_transform(test_predict)


# In[ ]:


test_predict.shape


# In[ ]:


T=scaler.inverse_transform(ytest.reshape(18,1))


# In[ ]:


plt.plot(T)
plt.plot(test_predict)


# In[ ]:


import math
from sklearn.metrics import mean_squared_error,mean_absolute_error
math.sqrt(mean_squared_error(T,test_predict))


# In[ ]:


look_back=10
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.figure(figsize=(6.4,4.8))
plt.show()


# In[ ]:


s= data.groupby('Date')['Cured'].sum()


# In[ ]:


NEW=pd.DataFrame(s,index=data['Date'].unique())


# In[ ]:


NEW.reset_index(inplace=True)


# In[ ]:


confirmed_data=NEW['Cured']


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(confirmed_data).reshape(-1,1))


# In[ ]:


training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[ ]:


def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)


# In[ ]:


time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[ ]:


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[ ]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=32,verbose=1)


# In[ ]:


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
ytest = np.asarray(ytest)


# In[ ]:


train_predict=model.predict(X_train)


# In[ ]:


train_predict=scaler.inverse_transform(train_predict)


# In[ ]:


train_predict.shape


# In[ ]:


Y=scaler.inverse_transform(y_train.reshape(89,1))


# In[ ]:


plt.plot(Y)
plt.plot(train_predict)


# In[ ]:


math.sqrt(mean_squared_error(Y,train_predict))


# In[ ]:


test_predict=model.predict(X_test)


# In[ ]:


test_predict=scaler.inverse_transform(test_predict)


# In[ ]:


test_predict.shape


# In[ ]:


Y=scaler.inverse_transform(ytest.reshape(33,1))


# In[ ]:


plt.plot(Y)
plt.plot(test_predict)


# In[ ]:


math.sqrt(mean_squared_error(Y,test_predict))


# In[ ]:


look_back=10
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.figure(figsize=(6.4,4.8))
plt.show()


# In[ ]:


data.columns


# In[ ]:


s= data.groupby('Date')['Deaths'].sum()


# In[ ]:


NEW=pd.DataFrame(s,index=data['Date'].unique())


# In[ ]:


NEW.reset_index(inplace=True)


# In[ ]:


D=NEW['Deaths']


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(D).reshape(-1,1))


# In[ ]:


training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[ ]:


import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return numpy.array(dataX), numpy.array(dataY)


# In[ ]:


time_step = 10
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[ ]:


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[ ]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=128,verbose=1)


# In[ ]:


X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
ytest = np.asarray(ytest)


# In[ ]:


train_predict=model.predict(X_train)


# In[ ]:


train_predict=scaler.inverse_transform(train_predict)


# In[ ]:


train_predict.shape


# In[ ]:


Y=scaler.inverse_transform(y_train.reshape(89,1))


# In[ ]:


plt.plot(Y)
plt.plot(train_predict)


# In[ ]:


math.sqrt(mean_squared_error(Y,train_predict))


# In[ ]:


test_predict=model.predict(X_test)


# In[ ]:


test_predict=scaler.inverse_transform(test_predict)


# In[ ]:


test_predict.shape


# In[ ]:


Y=scaler.inverse_transform(ytest.reshape(33,1))


# In[ ]:


plt.plot(Y)
plt.plot(test_predict)


# In[ ]:


math.sqrt(mean_squared_error(Y,test_predict))


# In[ ]:


look_back=10
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.figure(figsize=(6.4,4.8))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




