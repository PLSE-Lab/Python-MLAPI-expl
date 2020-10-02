#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# This is simple excersice applying neural networks to the titanic data and predicting the survival
# Kudos to: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import itertools

testinput = pd.read_csv('/kaggle/input/test.csv')
train = pd.read_csv('/kaggle/input/train.csv')
gender = pd.read_csv('/kaggle/input/gender_submission.csv')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Data clean-up and converting textual information into numbers

def mapsex(sex):
    if sex == 'male': return int(1)
    else: return int(2)

def maptitle(n,sex):
    if 'Mr.'in n: return int(1)
    if ('Master.' in n) or ('Don.' in n) or ('Major' in n) or ('Capt' in n) or ('Jonkheer' in n) or ('Rev' in n) or ('Col' in n) or ('Sir.' in n): 
        return int(2)
    if ('Countess' in n) or ('Mme' in n) or ('Dona.' in n) or ('Lady.' in n): return int(3)
    if ('Mrs.' in n): return int(4)
    if ('Mlle' in n) or ('Ms.' in n) or ('Miss.' in n): return int(5)
    if ('Dr.' in n):
        if sex == 0: return int(2)
        else: return int(3)
    
def number_of_cabins(c):
    if pd.isnull(c): return 1
    return (c.count(' ') + 1.)*10.
 
def cabin_level(c):
    if pd.isnull(c): return 1
    elif c[0] == 'T': return 9
    elif c[0] == 'A': return 8
    elif c[0] == 'B': return 7
    elif c[0] == 'C': return 6
    elif c[0] == 'D': return 5
    elif c[0] == 'E': return 4
    elif c[0] == 'F': return 3
    elif c[0] == 'G': return 2
    else:
        print ('cabin:' , c)

def cabin_number(c):
    if pd.isnull(c): return 1.
    if c.count(' ') == 1:
        if len(c) == 1: return 1.
        return int(c[1:])
    else:
        i = c.rfind(' ')
        return int(c[i+2:])

def port_number(c):
    if c == "C": return 1
    if c == "S": return 2
    if c == "Q": return 3
    return 0

def ticket_number(c):
    if c == "LINE": return 0
    i = c.rfind(' ')
    return int(c[i+1:])

def fix_age(c, average):
    if pd.isnull(c): return average
    return c

def add_one(c):
    return c + 1

def prepareData(data):    
#Create numeric sex indicator 'binsex'
    data['Binsex'] = data.Sex.apply(lambda x: mapsex(x))
    data['Title'] = data[['Name','Sex']].apply(lambda x: maptitle(*x), axis = 1)
    data['cabins'] = data.Cabin.apply(number_of_cabins)
    data['cabinLevel'] = data.Cabin.apply(cabin_level)
    data['port'] = data.Embarked.apply(port_number)
    data['ticketNumber'] = data.Ticket.apply(ticket_number)
    aveAge = data[data.Age.notnull()].Age.mean()
    data['fixedAge'] = data.Age.apply(lambda x: fix_age(x, aveAge))
    data[data.Age.isnull()].Age = aveAge 
    data['siblings'] = data.SibSp.apply(add_one)
    data['parch1'] = data.Parch.apply(add_one)
    return data


# In[ ]:


#get the actual feature vectors
t = prepareData(train)
te = prepareData(testinput)
tt = tnum = t[['PassengerId', 'Pclass', 'fixedAge', 'siblings', 'parch1', 'Fare', 'Binsex', 'Title','cabins','cabinLevel', 'port','Survived']]
tnum = t[['PassengerId', 'Pclass', 'fixedAge', 'siblings', 'parch1', 'Fare', 'Binsex', 'Title','cabins','cabinLevel', 'port','Survived']].values
tenum= te[['PassengerId', 'Pclass', 'fixedAge', 'siblings', 'parch1', 'Fare', 'Binsex', 'Title','cabins','cabinLevel', 'port']].values


# In[ ]:


# organize imports for the neural network libraries
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import LeakyReLU
import numpy as np

# seed for reproducing same results
seed = 9
np.random.seed(seed)

# split into input and output variables
X = tnum[:,0:11]
Y = tnum[:,11]


# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)
if (np.isnan(X_train).sum()) > 0: print ("X_train contains nan")
if (np.isnan(X_test).sum()) > 0: print ("X_test contains nan")
if (np.isnan(Y_train).sum()) > 0: print ("Y_train contains nan")
if (np.isnan(Y_test).sum()) > 0: print ("Y_test contains nan")


# In[ ]:


# Build the neural network model
model = Sequential()
model.add(Dense(11, input_dim=11, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


# In[ ]:


# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=500, batch_size=10)


# In[ ]:


#Check against the test sample
scores = model.evaluate(X_test, Y_test)
print ("Accuracy: %.2f%%" %(scores[1]*100))


# In[ ]:


#Calculate the prediction and output to result.csv file
prediction = model.predict(tenum).round().astype(int)
result = pd.DataFrame(testinput.PassengerId)
result['Survived'] = pd.DataFrame(prediction)
result.to_csv('result.csv',index = False)

