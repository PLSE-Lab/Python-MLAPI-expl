#!/usr/bin/env python
# coding: utf-8

# Hi! today we will do some EDA for basic-Computer-data-set and then we will predict prices of of computers.
# 
# Now first let's import the required libraries and the data.

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
from tensorflow import keras

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# to display beautiful plotting
plt.style.use('fivethirtyeight')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('../input/basic-computer-data-set/Computers.csv')
print(df.head())
# Any results you write to the current directory are saved as output.


# let's clean the data 

# In[ ]:


# drop useless features
df.drop(['Unnamed: 0'], axis=1, inplace=True)

# check for missing values
nulls = pd.DataFrame(df.isnull().sum() / len(df) * 100)
print(nulls)


# Phew! there is no missing values.
# 
# Now let's do some data analysis
# 
# we will start with 'RAM'
# let's see if it affect the speed of ocomputers

# In[ ]:


ramgroup = df.groupby('ram').mean()
ram = [ram for ram, df in df.groupby('ram')]  # to extract the unique values 'names'

plt.barh(ram, ramgroup['speed'], color='lightcoral') # barh is short for horizontal bar
plt.title('The Speed of Computers By Quantity of RAMs', color='lightcoral')
plt.ylabel('RAM')
plt.xlabel('Speed', color='lightcoral')
plt.show()


# WTF! what happened ?
# 
# I will tell you don't lower your spirits. because 'ram' is numeric matplotlib will plot the quantity of 'ram'
# as 'int' but a quantity in this case needs lineplot 'plt.plot()'
# 
# but in our case 'plt.barh' we need the quantity as 'srting'. So how should we do this!
# we will  type just one line to fix this problem which is as below

# In[ ]:


df['ram'] = df['ram'].map({2: '2', 24: '24', 32: '32', 4: '4', 8: '8', 16: '16'})
# we converted the digit from int to object so now it's represent the quantity of rams as 'string'


# And now let's plot the ram quantity.

# In[ ]:


ramgroup = df.groupby('ram').mean()
ram = [ram for ram, df in df.groupby('ram')]  # to extract the unique values 'names'

plt.barh(ram, ramgroup['speed'], color='lightcoral') # barh is short for horizontal bar
plt.title('The Speed of Computers By Quantity of RAMs', color='lightcoral')
plt.ylabel('RAM')
plt.xlabel('Speed', color='lightcoral')
plt.show()


# That's cool! As you can see the speed  of computers increase with increasing of RAM quantity.
# But there is when you look 2 and 4  the speed didn't increase and the same happned with 8 and 16. This is interesting !

# Now let's see if the price of computers increase with increasing of RAMs:

# In[ ]:


plt.barh(ram, ramgroup['price'], color='tomato')
plt.title('The Price of Computers By Quantity of RAMs', color='tomato')
plt.xlabel('Price', color='tomato')
plt.ylabel('RAM')


# as we expected, the more RAMs the more the price will be this makes sense.
# 
# but let's combine the speed and price of computers by the amount of RAMs
# this will be more interesting :}

# In[ ]:


fig, ax = plt.subplots(nrows=2, ncols=1)
plt.figure(figsize=(16, 12))

plt.subplot(2, 1, 1)

# plotting the speed of computer by quantity of rams
ramgroup = df.groupby('ram').mean()
ram = [ram for ram, df in df.groupby('ram')]

plt.barh(ram, ramgroup['speed'], color='lightcoral')
plt.title('The Speed of Computers By Quantity of RAMs', color='lightcoral')
plt.ylabel('RAM')
plt.xlabel('Speed', color='lightcoral')

plt.subplot(2, 1, 2)
plt.barh(ram, ramgroup['price'], color='tomato')
plt.title('The Price of Computers By Quantity of RAMs', color='tomato')
plt.xlabel('Price', color='tomato')


# that's awesome!
# 
# Now let's see if the cd affect the price or the speed
# 

# In[ ]:


cdgroup = df.groupby('cd').mean()
cd = [cd for cd, df in df.groupby('cd')] # to extract the unique values ex: 'yes', 'no'


plt.barh(cd, cdgroup['speed'], color='lightslategray')
plt.title('Speed of Computers With CD', color='lightslategray')
plt.xlabel('Speed', color='lightslategray')
plt.ylabel('CD')


# WoW this is so interesting, i didn't know that 'CD' affect the speed.
# 
# bu6t what about the price?

# In[ ]:


plt.barh(cd, cdgroup['price'], color='teal')
plt.title('Price of Computers With CD', color='teal')
plt.xlabel('Price', color='teal')
plt.ylabel('CD')


# this makes sense.
# 
# But let's combine  speed and price of computers by 'CD' to make it more interesting

# In[ ]:


cdgroup = df.groupby('cd').mean()
cd = [cd for cd, df in df.groupby('cd')]

fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True)
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.barh(cd, cdgroup['speed'], color='lightslategray')
plt.xlabel('Speed', color='lightslategray')
plt.title('Speed and Price of Computers With CD')
plt.ylabel('CD')

plt.subplot(2, 1, 2)
plt.barh(cd, cdgroup['price'], color='teal')
plt.xlabel('Price', color='teal')
plt.ylabel('CD')


# that's cool
# 
# Now let's see speed, trend and price by Hard-Desk

# In[ ]:



hdgroup = df.groupby('hd').mean()
hd = [hd for hd, df in df.groupby('hd')]

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
plt.subplot(2, 1, 1)
sns.barplot()
sns.barplot(hd, hdgroup['speed'], label='Speed', color='crimson', alpha=0.9)
sns.barplot(hd, hdgroup['trend'], label='Trend', color='lightgreen', alpha=0.9)
plt.title("Computer's Speed and Trend by Hard-Desk Space")
plt.legend()

plt.subplot(2, 1, 2)
sns.barplot(hd, hdgroup['price'], label='Price', color='forestgreen', alpha=0.9)
plt.legend()
plt.title("Computer's Price by Hard-Desk Space")
plt.xlabel('Hard-Desk space')
plt.show()


# **whaaat!** why it looks like this?
# 
# Because the Hard-Desk feature contains alot of unique values as a result the plot is huge and even you can't take any info from this plot.
# 
# So let's clean this data and conveted to a special unique values like (100, 200, 300 insteed of 120, 234, 302) 
# we first need to u6nderstand the data and know what is the maximum and minimum values.
# 
# We will do this by the foloowing code:

# In[ ]:


print(df.hd.describe())


# The **minimum** value is 80 ad the **maximum** is 2100, after we know this we will do some feature engineering to apply our idea.

# In[ ]:


df.loc[df['hd'] <= 100, 'hd'] = 100
df.loc[(df['hd'] > 100) & (df['hd'] <= 200), 'hd'] = 200
df.loc[(df['hd'] > 200) & (df['hd'] <= 300), 'hd'] = 300
df.loc[(df['hd'] > 300) & (df['hd'] <= 400), 'hd'] = 400
df.loc[(df['hd'] > 400) & (df['hd'] <= 500), 'hd'] = 500
df.loc[(df['hd'] > 500) & (df['hd'] <= 600), 'hd'] = 600
df.loc[(df['hd'] > 600) & (df['hd'] <= 700), 'hd'] = 700
df.loc[(df['hd'] > 700) & (df['hd'] <= 800), 'hd'] = 800
df.loc[(df['hd'] > 800) & (df['hd'] <= 900), 'hd'] = 900
df.loc[(df['hd'] > 900) & (df['hd'] <= 1000), 'hd'] = 1000
df.loc[(df['hd'] > 1000) & (df['hd'] <= 1100), 'hd'] = 1100
df.loc[(df['hd'] > 1100) & (df['hd'] <= 1200), 'hd'] = 1200
df.loc[(df['hd'] > 1200) & (df['hd'] <= 1300), 'hd'] = 1300
df.loc[(df['hd'] > 1300) & (df['hd'] <= 1400), 'hd'] = 1400
# the data doesn't contains any computer with 1500 GB so there is no value with 1500
df.loc[(df['hd'] > 1400) & (df['hd'] <= 1600), 'hd'] = 1600
df.loc[(df['hd'] > 1700) & (df['hd'] <= 2100), 'hd'] = 2100


# Now let's run our code again 

# In[ ]:



hdgroup = df.groupby('hd').mean()
hd = [hd for hd, df in df.groupby('hd')]


fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
plt.figure(figsize=(13, 12))


plt.subplot(2, 1, 1)
sns.barplot()
sns.barplot(hd, hdgroup['speed'], label='Speed', color='crimson', alpha=0.9)
sns.barplot(hd, hdgroup['trend'], label='Trend', color='lightgreen', alpha=0.9)
plt.title("Computer's Speed and Trend by Hard-Desk Space")
plt.legend()

plt.subplot(2, 1, 2)
sns.barplot(hd, hdgroup['price'], label='Price', color='forestgreen', alpha=0.9)
plt.legend()
plt.title("Computer's Price by Hard-Desk Space")
plt.xlabel('Hard-Desk space')
plt.show()


# Most computers with a Higher Hard-Desk have trend speed and price higher then the computers that don't have high Hard-Desk.
# 
# fially let's see if the premium computers affect speed, price and Hard-Desk

# In[ ]:


pregroup = df.groupby('premium').mean()
pre = [premium for premium, df in df.groupby('premium')]
plt.subplots(nrows=1, ncols=3)
plt.figure(figsize=(12, 13))

plt.subplot(1, 3, 1)
sns.barplot(pre, pregroup['speed'], label='Speed', color='crimson', alpha=0.9)
plt.legend()

plt.subplot(1, 3, 2)
sns.barplot(pre, pregroup['hd'], label='Hard-Desk', color='lightblue', alpha=0.9)
plt.legend()
plt.title('Speed & Price & Hard-Desk by Premium')
plt.xlabel('Premium')

plt.subplot(1, 3, 3)
sns.barplot(pre, pregroup['price'], label='Price', color='darkgreen', alpha=0.9)
plt.legend()
plt.show()


# As you can see, the computers with premium have a higher Hard-Desk and speed but lowe price.
# 
# Now let's do some prediction:
# 
# * First we will preprocesss the data.
# 
# * Second we will train LinearSVR.
# 
# * Third we will train a basic neural network.

# In[ ]:


# PRE-PROCESSING THE DATA
mapping = {'yes': 1, 'no': 0}
for col in df:
    if df[col].dtypes == object:
        df[col] = df[col].map(mapping)

df.dropna(axis=1, inplace=True)

y = df.price
X = df.drop(['price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

# TRAIN LINEARSVR
model_1 = LinearSVR(C=11, epsilon=3)
model_1.fit(X_train, y_train)

y_pred = model_1.predict(X_test)
print(mean_squared_error(y_test, y_pred))


# In[ ]:


# TRAINING A NEURAL NETWORK
n_feature = X_train.shape[1]
model_2 = keras.Sequential()
model_2.add(keras.layers.Dense(100, keras.activations.relu, kernel_initializer=keras.initializers.RandomNormal(),
                             kernel_regularizer='l2', input_dim=n_feature))
model_2.add(keras.layers.GaussianNoise(0.2))
model_2.add(keras.layers.Dense(90, keras.activations.relu, kernel_initializer=keras.initializers.RandomNormal(),
                             kernel_regularizer='l2'))
model_2.add(keras.layers.GaussianNoise(0.2))
model_2.add(keras.layers.Dense(90, keras.activations.relu, kernel_initializer=keras.initializers.RandomNormal(),
                             kernel_regularizer='l2'))
model_2.add(keras.layers.GaussianNoise(0.2))
model_2.add(keras.layers.Dense(90, keras.activations.relu, kernel_initializer=keras.initializers.RandomNormal(),
                             kernel_regularizer='l2'))
model_2.add(keras.layers.GaussianNoise(0.2))
model_2.add(keras.layers.Dense(90, keras.activations.relu, kernel_initializer=keras.initializers.RandomNormal(),
                             kernel_regularizer='l2'))
model_2.add(keras.layers.GaussianNoise(0.2))
model_2.add(keras.layers.Dense(1, 'linear', kernel_regularizer='l2'))
model_2.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
print(model_2.summary())
model_2.fit(X_train, y_train, batch_size=22, epochs=60, validation_data=(X_test, y_test))
model_2.predict(X_test, batch_size=32)


# As you can see, neural network(mse: 70991.898) is better then LinearSVR(mse: 244671.318).
# 
# I hope this notebook was useful for you and you learned some thing pleas upvote this notebook if you see it important or useful good luck ;)
