# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

 # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

numberofiterations=1000
lrate=1


data=pd.read_csv("../input/voice.csv")

data.label=[1 if each =="male" else 0 for each in data.label]

x=data.drop(["label"],axis=1)
y=data.label.values

x=((x-np.min(x))/(np.max(x)-np.min(x))).values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T

dimension=x_train.shape[0]
w=np.full((dimension,1),0.01)
b=0.0

z=np.dot(w.T,x_train)+b


def sigmoid(z):
    
    y_head=1/(1+np.exp(-z))
    return y_head
 

    
cost_list=[]
cost_list2=[]
index=[]

for i in range(numberofiterations):
    z=np.dot(w.T,x_train)+b
    y_head=sigmoid(z)
    loss=y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost=(np.sum(loss))/x_train.shape[1]
    
    
    #backward propagation
    der_w=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    der_b=np.sum(y_head-y_train) / x_train.shape[1] 
    gradients={"derivative_weight":der_w,"derivative_bias":der_b}
    cost_list.append(cost)
    w=w-lrate*der_w
    b=b-lrate*der_b
    
    if i%10 ==0:
        cost_list2.append(cost)
        index.append(i)
        print("Cost after iteration %i\ncost:%f"%(i,cost))
        
    parameters={"weight":w,"bias":b}
    plt.plot(index,cost_list2)
    plt.xlabel("number of iteration")
    plt.ylabel("Cost")
    plt.grid()
    
z=np.dot(w.T,x_test)+b    
y_prediction=np.zeros((1,x_test.shape[1]))


for i in range(z.shape[1]):
    if z[0,i]<=0.5:
        y_prediction[0,i]=0
    else:
        y_prediction[0,i]=1
        
        
y_prediction_test=y_prediction
print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))     