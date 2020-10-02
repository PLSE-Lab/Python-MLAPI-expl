# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
import matplotlib.pyplot as plt
from sklearn import metrics,preprocessing
import folium
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as pyplot
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/housing.csv")
df.head()
from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
df['ocean_proximity']=number.fit_transform(df['ocean_proximity'].astype('str'))
data = df.columns
predictors = df[data[data != 'median_house_value']]
target = df['median_house_value']
predictors.fillna(predictors.mean(), inplace=True)
predictors.info()
predictors.describe()

from sklearn.model_selection import train_test_split
df_train,df_test=train_test_split(df,test_size=0.20)

x_norm = (predictors-predictors.mean())/predictors.std()
y_norm = (target-target.mean())/target.std()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_norm,y_norm,test_size=0.20,random_state=0)
n_cols = x_train.shape[1]

def my_model():
    model = Sequential()
    model.add(Dense(100,activation='relu',input_shape=(n_cols,)))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

model = my_model()
history = model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=2,epochs=30)
    
pyplot.title('Loss / Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

y_pred= model.predict(x_test)
print("Root Mean Squared Error is {}".format(np.sqrt(metrics.mean_squared_error(y_test,y_pred))))

latmean=df['latitude'].mean() 
lonmean=df['longitude'].mean() 
    
map5 = folium.Map(location=[latmean,lonmean], 
        zoom_start=6,tiles = 'Mapbox bright') 
          
def color(value): 
    if value in range(0,149999): 
        col = 'green'
    elif value in range(150000,249999): 
        col = 'yellow'
    elif value in range(250000,349999): 
        col = 'orange'
    else: 
        col='red'
    return col 
      
for lat,lan,value in zip(df_test['latitude'],df_test['longitude'],df_test['median_house_value']): 
    folium.Marker(location=[lat,lan],icon= folium.Icon(color=color(value),icon_color='black',icon = 'home')).add_to(map5) 
                    

print(map5.save('finalnn.html'))
