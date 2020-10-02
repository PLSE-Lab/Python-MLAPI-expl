# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
iris = pd.read_csv('../input/Iris.csv')
#loads dataset
iris = ((iris[iris["Species"] != "Iris-virginica"]).drop("SepalWidthCm", axis=1)).drop("PetalWidthCm", axis=1)
#trim uneccecary infromation
print(iris)
#prints part of the dataset for refrence
lr = .01
#speceifes the learning rate
w = np.array([0,0,0])
#creates the weights matrix

#creates a reset for the weights
def resetweights():
    global w
    w = np.array([0,0,0])
    print("Reset successful")



#creates a testing function
def testing():
    samp = iris.sample()
    l = np.array([1,samp.iloc[0,1],samp.iloc[0,2]])
    true_labelt = samp.iloc[0,3]
    zed = np.matmul(w,l)
    if zed <= 0:
        str_zed ="Iris-setosa"
    else:
        str_zed ="Iris-versicolor"
    if true_labelt == "Iris-versicolor":
        int_label = 1
    else: 
        int_label = -1
    print(true_labelt+"("+str(int_label)+")"+", "+str_zed+"("+str(zed)+")")
    return((str_zed == true_labelt))
def runepoch(epoch):
    global w
    i = 1
    #creates counter varible for weights
    while i <= epoch:
        #runs code for "epoch" times
        print("--------------------------------------------")
        #prints seperator 
        samp = iris.sample()
        #takes a random sample from the dataset 
        x = np.array([1,samp.iloc[0,1],samp.iloc[0,2]])
        #creates the "infromation" matrix from infromation in sample
        true_label = np.where(samp.iloc[0,3] == 'Iris-versicolor',1,-1)
        #stores the species label of sample for later
        z = np.matmul(w,x)
        #multiples the transpose of weights and infromation given to create prediction
        delta_w = lr*(true_label-z)*x 
        #runs the perceptron formula with above info
        w = w+delta_w
       #adjusts weights as neccecary 
        print(w)
        #prints new weights (for refrence)
        i = i+1
        #counts up

    #defines another test function
def percentege():
    xhat = iris    
    xhat[['Id']] = 1
    xx = xhat[['Id','SepalLengthCm','PetalLengthCm']]
    pred = xx.dot(w)
    iris_new = iris
    iris_new['prediction'] = np.where(pred > 0,"Iris-versicolor","Iris-setosa")
    predection_true = iris_new["Species"] == iris_new["prediction"]
    #predection_true = iris["Species"].map(lambda p: print(p.iloc[0,0]))
    
    return(str(predection_true.sum())+"% Correctly Predicted")
    
    
    

#prints percent correct in dataset