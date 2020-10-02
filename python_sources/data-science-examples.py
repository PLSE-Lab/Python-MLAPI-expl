# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from subprocess import check_output
print(check_output(["ls","../input/pokemon-challenge"]).decode("utf8"))

data=pd.read_csv('../input/pokemon-challenge/pokemon.csv')

data.info()
data.corr()

f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidth=.5, fmt= '.1f',ax=ax)
plt.show()

data.head(10)
data.columns

#Line Plot
data.Speed.plot(kind='line',color='g',label='Speed',linewidth=1,alpha=0.5,grid=True,linestyle=':')
data.Defense.plot(color='r',label='Defense',linewidth=1,alpha=0.5,grid=True,linestyle='-.')
plt.legend(loc='upper right')
plt.xlabel=('x axis')
plt.ylabel=('y axis')
plt.title("Line Plot")
plt.show()


#Scatter Plot
#data.plot(kind='scatter',x='Attack',y='Defense',alpha=0.5,color='red')
#plt.xlabel('Attack')
#plt.ylabel('Defence')
#plt.title('Attack Defense Scatter Plot')

data.Speed.plot(kind='hist',bins=50,figsize=(12,12))
plt.show()

data.Speed.plot(kind='hist',bins=50)
plt.clf()


dictionary={'spain':'barcelona','usa':'vegas'}
print(dictionary.keys())
print(dictionary.values())

dictionary['spain']='madrid'
print(dictionary)
dictionary['france']='paris'
print(dictionary)
del dictionary['spain']
print(dictionary)
print('france' in dictionary)
dictionary.clear()
print(dictionary)
dictionary={'spain':'barcelona','usa':'vegas','turkey':'Trabzon','turkey':'istanbul'}
print(dictionary)
dictionary={'spain':'barcelona','usa':'vegas','turkey':'Trabzon','turkey1':'istanbul','turkey2':'adana'}
print(dictionary)



