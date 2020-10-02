# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

dataset = pd.read_csv('../input/Iris.csv')
x = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

for i in range(len(y)):
    if(y[i]=='Iris-virginica'):
        y[i] = 3
    elif(y[i]=='Iris-versicolor'):
        y[i] = 2
    else:
        y[i] = 1
         
    
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

for i in range(len(y_pred)):
    y_pred[i] = float(round(y_pred[i]))
    
t = []
for i in range(len(y_pred)):
    if(y_pred[i]==3.0):
        t.append("Iris-virginica")
    elif(y_pred[i]==2.0):
        t.append("Iris-versicolor")
    else:
        t.append("Iris-setosa")
        

yr = []
for i in range(len(y_test)):
    if(y_test[i]==3):
        yr.append("Iris-virginica")
    elif(y_test[i]==2):
        yr.append("Iris-versicolor")
    else:
        yr.append("Iris-setosa")
        
print(len(yr))
print(len(t))
        
output = open("multipleLinearReg.csv","w+")
output = open("multipleLinearReg.csv","a")
output.write("prediction,real,status \n")
for i in range(len(yr)):
    p = str(t[i])+","+str(yr[i])+","
    if(yr[i]==t[i]):
        p = p + "matched"
    else:
        p = p + "unmatched"
    p = p + "\n"
    output.write(p)





# Any results you write to the current directory are saved as output.