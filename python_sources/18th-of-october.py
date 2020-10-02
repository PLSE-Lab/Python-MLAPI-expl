#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os #operating system 
for dirname, _, filenames in os.walk('/kaggle/input'):   
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


for a in os.walk('/kaggle/input'):
    print(a)  #extracts a tuple 


# In[ ]:


for a, b, c in os.walk('/kaggle/input'):
    print(a,b,c) #extracts what is inside the tuple. 


# In[ ]:


for a, b, c in os.walk('/kaggle/input'):
    print(c)


# In[ ]:


for a,b,c in os.walk('/kaggle/input'):
    for d in c: #for loop in a for loop 
        print(d)


# In[ ]:


#profversion
for a, b, c in os.walk('/kaggle/input'):
    print(a,b,c)
    for d in c:
        print(d)


# 1. **/kaggle/input ['nfl-big-data-bowl-2020'] [] **: *first directory is /kaggle/input and its name is ['nfl-big-data-bowl-2020']  and there is no file in that directory. * 
# 2. **/kaggle/input/nfl-big-data-bowl-2020 ['kaggle'] ['train.csv']**: *what is inside the previous directory ['nfl-big-data-bowl-2020'] in this directory path it has directory named ['kaggle'] and it has  ['train.csv'] in it* 
# train.csv
# 3. **/kaggle/input/nfl-big-data-bowl-2020/kaggle ['competitions'] []** :*again, in the previous directory ['kaggle'] , there is  ['competitions'] directory with no file in it. * 
# 4. **/kaggle/input/nfl-big-data-bowl-2020/kaggle/competitions ['nflrush'] []** : *in the previous ['competitions'] directory there is ['nflrush'] directory with no file in it* 
# 5. **/kaggle/input/nfl-big-data-bowl-2020/kaggle/competitions/nflrush [] ['test.csv.encrypted', 'sample_submission.csv.encrypted', '__init__.py', 'competition.cpython-36m-x86_64-linux-gnu.so'] **: * in the previous ['nflrush'] no there is no more directory but there exists 4 files in it*

# In[ ]:


df=pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv',low_memory=False)


# In[ ]:


df
#various PlayId in one GameId. 


# In[ ]:


df.head()


# In[ ]:


df.tail(10)


# In[ ]:


df.info()


# cf StadiumType               476828 non-null object <--empty data what should we do with this
# 
# PlayerHeight              509762 non-null object<-- height a string?? probably we will have to convert to a number(integer)
# 
# 
# PlayerWeight              509762 non-null int64

# In[ ]:


df['PlayerHeight']


# In[ ]:


df[['PlayerHeight','PlayerWeight']]


# In[ ]:


df.iloc[0:20]


# In[ ]:


df.loc[0] #the first row 


# In[ ]:


df.isnull()


# In[ ]:


df.isnull().sum()


# In[ ]:


#grouping
g=df.groupby('GameId')


# In[ ]:


g.size() #we have 512 games and we have number of plays in each game.


# In[ ]:


g['PlayId']


# In[ ]:


import matplotlib.pylab as plt


# In[ ]:


df.groupby('PlayId').first()['Yards'].plot(kind='hist',
            figsize=(15,5),
            bins=100,
            title ='Distributions of Yards Gained (Target)')
plt.show()


# In[ ]:


train=df.select_dtypes(include='number') #only the numeric data


# In[ ]:


train #now we only have 25 columns 


# In[ ]:


Y=train.pop('Yards') #popping the 'Yards' column from the train data


# In[ ]:


Y


# In[ ]:


train #'Yards'is popped out so now we have 24 columns this will be the shape of the input later on. 


# In[ ]:


import tensorflow as tf


# In[ ]:


my_model=tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[24])
])
#we made a very/extremely simple model with a dense layer with one node


# In[ ]:


my_model.compile(
    loss='mse',
    optimizer='adam'
)


# In[ ]:


my_model.fit(train,Y,epochs=1) 
#loss:nan there is something wrong! 
#we had a lot of missing data it is simply impossible to fit a model with missing data


# In[ ]:


train=train.dropna()


# In[ ]:


Y = train.pop('Yards')
df


# In[ ]:


train = df.select_dtypes(include='number')


# In[ ]:


train.info()


# In[ ]:


train=train.dropna()


# In[ ]:


Y = train.pop('Yards')


# In[ ]:


train.info()


# In[ ]:


my_model.fit(train, Y, epochs=1)


# In[ ]:


train.pop('GameId')


# In[ ]:


train.pop('PlayId')


# In[ ]:


my_model.fit(train, Y, epochs = 1)
#we popped out 2 columns input shape must be edited


# In[ ]:


my_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[22])
])


# In[ ]:


my_model.compile(
    loss = 'mse',
    optimizer = 'adam'
)


# In[ ]:


my_model.fit(train, Y, epochs = 1)


# In[ ]:


my_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(512, input_shape=[22], activation='relu'),
    tf.keras.layers.Dense(1)
])

my_model_2.compile(
    loss = 'mse',
    optimizer = 'adam'
)


# In[ ]:


my_model_2.fit(train, Y, epochs = 10)


# In[ ]:




