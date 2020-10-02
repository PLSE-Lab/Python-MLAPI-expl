#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


a=pd.read_csv('../input/time-series-covid19/time_series_covid19_confirmed_global.csv')


# In[ ]:


a


# In[ ]:


b=a.transpose()


# In[ ]:


b


# In[ ]:


train=b[[131]].values


# In[ ]:


train_x=train[4:]


# In[ ]:


y=[]
for i in train_x:
    y.append(i[0])


# In[ ]:


y.append(3374)


# In[ ]:


#infected in india in the following days
y


# In[ ]:


x=list(range(len(y)))


# In[ ]:


#nth day
x


# In[ ]:


import seaborn as sns


# In[ ]:


#infection in the  nth day in india
sns.relplot(data=pd.DataFrame(y))
#x axis is days and y axis is infected people


# In[ ]:


x=np.array(x)
y=np.array(y)


# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# In[ ]:


import tensorflow as tf


# In[ ]:


predict=Sequential([Dense(74,activation='relu',input_shape=(1,)),
                    Dense(74*2,activation='relu'),
                    Dense(74*2*2,activation='relu'),
                    Dense(74*2*2*2,activation='relu'),
                    Dense(74*2*2*2,activation='relu'),
                    Dense(1)])


# In[ ]:


predict.compile(optimizer='adam',loss='mse',metrics=['mse','mae','accuracy'])


# In[ ]:


'''You can uncomment to start training from begining or you can use weights of pre trained model'''
#predict.fit(x,y,batch_size=75,epochs=50000)


# In[ ]:


predict.load_weights('/kaggle/input/corona-india-future-predictions-weights/corona_future_prediction_india_weights.h5')


# In[ ]:


predict.summary()


# In[ ]:


'''You can uncomment the following things if you have started to train from the begining'''
#loss=predict.history.history
#loss_pd=pd.DataFrame(loss)
#loss_pd.plot()
#loss_pd.to_csv('loss.csv')


# In[ ]:


loss_pd=pd.read_csv('/kaggle/input/model-loss/loss.csv')


# In[ ]:


loss_pd=loss_pd.drop('Unnamed: 0',axis=1)


# In[ ]:


loss_pd


# In[ ]:


loss_pd.plot()


# In[ ]:


predict.predict([[74]]) #75th day is August 5 2020


# In[ ]:


predicted_values=[]
print('||',end='')
for i in range(1000):
    if i%9 is 0:
        print('=',end='')
    predicted_values.append(predict.predict([[i]]))
print('||')    


# In[ ]:


predicted_values


# In[ ]:


future=[]
for i in predicted_values:
    if round(i[0][0])<=0:
        future.append(0)
    else:
        future.append(round(i[0][0]))


# In[ ]:


future_df=pd.DataFrame({'Infected_people':future})


# In[ ]:


future_df


# In[ ]:


print("Actual_graph")
sns.relplot(data=pd.DataFrame(y))


# In[ ]:


print("predicted_graph")
sns.relplot(data=future_df[:75])


# In[ ]:


print("Prediction on future days")
future_df


# In[ ]:


print("Future predictions graph")
sns.relplot(data=future_df[:100])


# In[ ]:


def preprocess(day):
    return round(day)


# In[ ]:


#prediction on the nth day 
'''74th  day is 5th april 2020'''
'''change the day number to your own number to predict'''
day=74
if preprocess(predict.predict([[day]])[0][0] <= 0):
    infected=0
else:
    infected=(preprocess(predict.predict([[day]])[0][0]))
print("The Predicted infected people on the day",day,"in India are :",infected)    
sns.relplot(data=future_df[:day])


# In[ ]:


print("Day 89 is August 20")
day=89
if preprocess(predict.predict([[day]])[0][0] <= 0):
    infected=0
else:
    infected=(preprocess(predict.predict([[day]])[0][0]))
print("The Predicted infected people on the day",day,"in India are :",infected)    
sns.relplot(data=future_df[:day])


# In[ ]:


'''If it goes on like this in india then approximately 8426 people will get infected on Auguest 20 2020 so please keep masks and wash your hands to save yourself and others'''


# In[ ]:


'''Change the day number as your wish , set your reference to August 5th as 74th day'''
'''Enter any day number to predict the infection rate on the specific day'''

day=100


if preprocess(predict.predict([[day]])[0][0] <= 0):
    infected=0
else:
    infected=(preprocess(predict.predict([[day]])[0][0]))
print("The Predicted infected people on the day",day,"in India are :",infected)    
sns.relplot(data=future_df[:day])


# In[ ]:


future_df.to_csv('submission.csv')


# In[ ]:




