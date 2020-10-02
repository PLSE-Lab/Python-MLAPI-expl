#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
import datetime
import math
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.layers import Input,Dense
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint


# ### Let's load and see the data
# First of all we need to need the data and get an understanding about it.

# In[ ]:


df = pd.read_csv('../input/renfe.csv')
df.head()


# We need to remove the Nan (None or data not available entries) from the data, before that we need to see if there is any ?!
# 

# In[ ]:


df.isna().sum() #Number of null values in different columns


# In[ ]:


df.dropna(inplace=True) #Remove all the rows containing nan


# For prediciting the price of a ticket it doesn't depend on the insert_date (time of scrapping) and that unamed column at first,so let's remove unwanted noise from our data

# In[ ]:


df.drop(["Unnamed: 0"], axis = 1, inplace = True)
df.drop(["insert_date"], axis = 1, inplace = True) #These values doesn't effect
df.head()


# ### Plotting Time!!!

# Nice!!!! now the data is cleaned of unwanted values and nan values, but still we can't pass it to our neural network model as it still has lots of categorical strings like origin,destination,train_type etc which must be converted to numerical values.
# 
# Before that let's plot some graphs and understand how the data is distributed and which category is having more data and as such.
# 
# Let's do that!

# In[ ]:


df["origin"].value_counts().plot(kind = "bar") #Origin places distribution


# In[ ]:


df["destination"].value_counts().plot(kind = "bar")


# It seems like both origin and destination of trains are pretty much distributed the same and trains along MADRID is highest in our dataset.
# 
# hmm... Okay let's encode these columns to numerical values

# ### Encoding Categorical Values 

# In[ ]:


df['destination'].replace(["MADRID","BARCELONA",'SEVILLA','VALENCIA','PONFERRADA'],[0,1,2,3,4], inplace = True) #Maps to numerical values
df['origin'].replace(["MADRID","BARCELONA",'SEVILLA','VALENCIA','PONFERRADA'],[0,1,2,3,4], inplace = True)


# Let's do the same with train_class,train_type and fare

# In[ ]:


df["train_type"].value_counts().plot(kind = "bar")
k = df["train_type"].unique()
l = [x for x in range(len(k))]
print("Numbers used to encode different train types",l) #numbers used to encode different trains
df['train_type'].replace(k,l, inplace = True)


# In[ ]:


df["train_class"].value_counts().plot(kind = "bar")#Plotting for train classes
k = df["train_class"].unique()
l = [x for x in range(len(k))]
print("Numbers used to encode different train classes:",l)
df['train_class'].replace(k,l, inplace = True)


# In[ ]:


df["fare"].value_counts().plot(kind = "bar") #Plotting for fare types
k = df["fare"].unique()
l = [x for x in range(len(k))]
print("Numbers used to encode different fare classes:",l)
df['fare'].replace(k,l, inplace = True)


# ### Visualization
# Okay, enough of graphs! now let's see how are different features related to each other using the pandas inbuilt function corr() and seaborn to plot it as a heatmap

# In[ ]:


f,ax = plt.subplots(figsize=(6, 6))
sns.heatmap(df.corr(), annot=True, cmap = "Blues", linewidths=.5, fmt= '.2f',ax = ax)
plt.show()


# It seems the heatmap is not as dense as we require we can infact squeeze out more information out of the dataset to create new features and refine existing features!

# ### Feature Engineering 
# We have start_date and end_date. Think about it ,If we are able to find the duration of the journey then it should have better relation to price that we are trying to predict! Also the tickets booked from Monday to Friday (working days) and Tickets booked in (Saturdays and Sundays) can have different influence on the price too. Now the departure and arrival times are given in string and if we convert it to cyclic features that have the knowledge that hour 23 is nearer to hour 1 and similary with minutes.
# 
# Let's create all these extract features.

# #### 1.Extracting Duration of Journey
# We have start_date and end_date so just subtracting them after converting into python datetime format will just do it!

# In[ ]:


start = df['start_date'].values
end = df['end_date'].values
datetimeFormat = '%Y-%m-%d %H:%M:%S'
duration = []
for i in range(len(start)):
    diff = datetime.datetime.strptime(end[i], datetimeFormat)- datetime.datetime.strptime(start[i], datetimeFormat)
    duration.append(diff.seconds)
df['duration'] = duration


# #### 2.Extracting Weekdays information
# The python datetime module also gives you support to get which day of the week in the following way!

# In[ ]:


start_weekdays = []
for i in start:
    start_weekdays.append(datetime.datetime.strptime(i,datetimeFormat).weekday())
end_weekdays = []
for i in end:
    end_weekdays.append(datetime.datetime.strptime(i,datetimeFormat).weekday())
df['start_weekday'] = start_weekdays
df['end_weekday'] = end_weekdays


# #### 3.Extracting hour and minutes in cyclic forms 
# We have date and time given here in the string format but before passing it into Neural Network(NN) we have to encode it , also we need to factor in the idea that 23rd hour is closer to 1 hour (the cyclic nature of time).I have used a unit cirle of radius 1 unit to encode the time in it's points on the circumference!
# 
# So a particular time is encoded as the x and y coordinates of that time in the circle.  
# ***
# The formulas used to convert are
# 
#   
#   y = sin(hr*pi/24)  
#   
#   x = cos(hr*pi/24)
# ***
# 
# The features will look similar to this picture
# ![Cyclic Time Representation](https://i.stack.imgur.com/MkVNg.png)
# Similary we can convert for day,date and month and even year.
# 
# We do it for both departure and arrival times.

# In[ ]:


#Converting datetime to cyclic features for departure times
hr_cos = [] #hr_cos,hr_sin,min_cos,min_sin
hr_sin = []
min_cos = []
min_sin = []
data = df['start_date'].values
for i in range(len(data)):
    time_obj = datetime.datetime.strptime(data[i],'%Y-%m-%d %H:%M:%S')
    hr = time_obj.hour
    minute = time_obj.minute
    sample_hr_sin = math.sin(hr*(2.*math.pi/24))
    sample_hr_cos = math.cos(hr*(2.*math.pi/24))
    sample_min_sin = math.sin(minute*(2.*math.pi/60))
    sample_min_cos = math.cos(minute*(2.*math.pi/60))
    hr_cos.append(sample_hr_cos)
    hr_sin.append(sample_hr_sin)
    min_cos.append(sample_min_cos)
    min_sin.append(sample_min_sin)
df['depart_time_hr_sin'] = hr_sin
df['depart_time_hr_cos'] = hr_cos
df['depart_time_min_sin'] = min_sin
df['depart_time_min_cos'] = min_cos
#Converting datetime to cyclic features for arrival times
hr_cos = [] #hr_cos,hr_sin,min_cos,min_sin
hr_sin = []
min_cos = []
min_sin = []
data = df['end_date'].values
for i in range(len(data)):
    time_obj = datetime.datetime.strptime(data[i],'%Y-%m-%d %H:%M:%S')
    hr = time_obj.hour
    minute = time_obj.minute
    sample_hr_sin = math.sin(hr*(2.*math.pi/24))
    sample_hr_cos = math.cos(hr*(2.*math.pi/24))
    sample_min_sin = math.sin(minute*(2.*math.pi/60))
    sample_min_cos = math.cos(minute*(2.*math.pi/60))
    hr_cos.append(sample_hr_cos)
    hr_sin.append(sample_hr_sin)
    min_cos.append(sample_min_cos)
    min_sin.append(sample_min_sin)
df['arrival_time_hr_sin'] = hr_sin
df['arrival_time_hr_cos'] = hr_cos
df['arrival_time_min_sin'] = min_sin
df['arrival_time_min_cos'] = min_cos


# ### Remove Redundant Features
# As we already extracted required information from the start_date and end_date, which currently only has the month and year of the journey which is actually doesn't matter for the price so we remove these features from data.Also no need in having information about year and month in start_date and end_date as the dataset doesn't contain all the months and different years.

# In[ ]:


df.drop(["start_date"], axis = 1, inplace = True)
df.drop(["end_date"], axis = 1, inplace = True)


# ### Visualise Again!
# Now we have structured and squeezed out the important features from the data,Let's see how does it correlate now

# In[ ]:


f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True, cmap = "Greens", linewidths=.5, fmt= '.2f',ax = ax)
plt.show()


# As we can see now the heatmap show better correlations and it's comparetively more dense than the previous one,it's also crowded towards the top left corner.

# In[ ]:


df.head()#Just checking to see if all worked as per our logic


# ### Scaling features 
# Before sending in the input if we scale it then it will be easier for the model and simpler model can capture the pattern in the data quickly,I am using MinMaxScaler provided by sklearn here (It showed more faster and accurate convergence than StandardScaler)

# In[ ]:


places_sc = MinMaxScaler(copy=False)
train_type_sc = MinMaxScaler(copy=False)
train_class_sc = MinMaxScaler(copy=False)
fare_sc = MinMaxScaler(copy=False)
weekday_sc = MinMaxScaler(copy=False)
duration_sc = MinMaxScaler(copy=False)
price_sc = MinMaxScaler(copy=False)
df['origin'] = places_sc.fit_transform(df['origin'].values.reshape(-1,1))
df['destination'] = places_sc.fit_transform(df['destination'].values.reshape(-1,1))
df['train_type'] = train_type_sc.fit_transform(df['train_type'].values.reshape(-1,1))
df['train_class'] = train_class_sc.fit_transform(df['train_class'].values.reshape(-1,1))
df['fare'] = fare_sc.fit_transform(df['fare'].values.reshape(-1,1))
df['start_weekday'] = weekday_sc.fit_transform(df['start_weekday'].values.reshape(-1,1))
df['end_weekday'] = weekday_sc.fit_transform(df['end_weekday'].values.reshape(-1,1))
df['duration'] = duration_sc.fit_transform(df['duration'].values.reshape(-1,1))
df['price'] = price_sc.fit_transform(df['price'].values.reshape(-1,1))


# In[ ]:


df.head()


# ### Model
# Now let's convert the pandas datafram into numpy arrays and let's create the model using keras and train it !

# In[ ]:


data = df.values
Y = data[:,3]
X = np.delete(data,3,1)
#Creating different data splits for training,validation and testing!
x_train = X[:2223708]
y_train = Y[:2223708]
x_validation = X[2223708:2246398]
y_validation = Y[2223708:2246398]
x_test = X[2246398:]
y_test = Y[2246398:]


# #### Structure
# My model has 3 hidden layers and 1 output layer, all layers are initialized Xavier initialization.
# I have tried out different layer structures and activation functions as a result I find the below configuration to be the best!
# I am using Nestrov Adam Optimizer and loss function is Mean Squared Error  
# 
# * [Nestrov + Adam](http://cs229.stanford.edu/proj2015/054_report.pdf)
# * [Xavier and He Normal (He-et-al) Initialization](https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528) 
# 
# 

# In[ ]:


input_layer = Input((X.shape[1],))
y = Dense(64,kernel_initializer='he_normal',activation='tanh')(input_layer)
y = Dense(8,kernel_initializer='he_normal',activation='sigmoid')(y)
y = Dense(1,kernel_initializer='he_normal',activation='sigmoid')(y)
y = Dense(1,kernel_initializer='he_normal',activation='tanh')(y)
model = Model(inputs=input_layer,outputs=y)
model.compile(Nadam(),loss='mse')
model.summary()


# ### Let the training begins!

# In[ ]:


history = model.fit(x_train,y_train,validation_data=(x_validation,y_validation),epochs = 100,batch_size=2048,callbacks=[ModelCheckpoint('best_model.hdf5',monitor='val_loss',mode='min')])


# ### Plot the training results

# In[ ]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()


# ### Conclusion and Results
# We have come to the final part of this story!  
# 
# 
# Now we will load the best model that we achieved during training that is the one with lowest error on validation set of our data and use that to predict the prices for test test then we see how much root mean squared error we make on price after we inverse scale the prediction to euros.  

# In[ ]:


model.load_weights('best_model.hdf5') #this will load the best model that we saved earlier
scores = model.evaluate(x_test,y_test)
print("Test Set  RMSE(before scaling ):",scores)
pred = model.predict(x_test)
y_test = y_test.reshape(22692,1)
k = y_test-pred
k = price_sc.inverse_transform(k)
rmse = np.sqrt(np.mean(np.square((k))))
print('Test Set RMSE(after scaling) :',rmse)


# As I conclude our story ,I should say from that little model to get this much low error is much of an awe to me! 
# This is the best I am able to come up with for now! but who knows maybe you guys can get inspirations and ideas from my work and create a better model or better features and beat my score!!! I am happy to help if you have any doubts in my code !
# 
# 
# Please do comment your thoughts and ideas and if you feel this was upto the mark then do Up vote this kernel on top right corner and let other's know about this !
# 
# Thanks for your time! :-)

# In[ ]:




