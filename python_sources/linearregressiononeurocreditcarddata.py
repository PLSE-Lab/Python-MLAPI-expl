import os
#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
# Split the independent and dependent variables
os.chdir('../input')
dataset=pd.read_csv('creditcard.csv',engine='python')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1]

# Split the training and testing data
 

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# Apply linear regression to the dataset
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predict from the testing dataset 
y_pred=regressor.predict(x_test)

y_test=list(y_test)

#round off the negative and positive values closer to 0

y_pred=[max(0,n) for n in y_pred]
y_pred=[int(round(n,0)) for n in y_pred]
print(y_pred)
flag=0

#check for wrong predictions

for i in range(len(y_pred)):
    if y_pred[i]!=y_test[i]:
        flag+=1
print(len(y_pred))
print('No. of wrong predictions:',flag)
accuracy=((len(y_pred)-flag)/len(y_pred)*100)
print('Accuracy:',accuracy)
