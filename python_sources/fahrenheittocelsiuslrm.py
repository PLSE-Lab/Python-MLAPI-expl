# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#prepare dataset to
df=pd.DataFrame(columns=["Fahrenheit","Celsius"])
df.set_index('Fahrenheit')
for i in range(0,150):
   df.loc[i] = [int(i),float(((i - 32) * 5)/9)]

X = df['Fahrenheit'].values    
y = df['Celsius'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
X_train= X_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

reg = LinearRegression()
cvscores_10 = cross_val_score(reg,X.reshape(-1, 1),y.reshape(-1, 1),cv=20)
print(np.mean(cvscores_10))



reg.fit(X_train,y_train)
pred = reg.predict(X_test)
result = {}
with open('predict_f_c.csv','w') as file:
     file.write("f,c\n")
     for i in range(len(X_test)):
         file.write("{},{}\n".format(X_test[i][0],pred[i][0]))









    

