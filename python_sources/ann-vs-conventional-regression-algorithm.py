import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#Lets first preprocess the data info test and train dataset
dataset = pd.read_csv('../input/random-salary-data-of-employes-age-wise/Salary_Data.csv')
X = dataset.iloc[:,0:1].values 
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)





#Lets see how a Convensional Algorithm (Linear Regression) give the output
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred_reg = regressor.predict(X_test)




#Now lets look at the ANN approach
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(32,activation='relu',input_dim=1))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))
opt  = keras.optimizers.RMSprop(learning_rate = 0.0099)
model.compile(optimizer=opt,loss='mean_squared_error')
model.fit(X_train,y_train,epochs=500)

y_pred_ann = model.predict(X_test)


#lets visualize the result on training dataset (yellow is ann; blue is LinReg)
plt.scatter(X_train,y_train, color='red')#point real
plt.plot(X_train,regressor.predict(X_train), color='blue') #line
plt.plot(X_train,model.predict(X_train), color='yellow') #line
plt.title('Salary vs Exp (training set)')
plt.xlabel('years of exp')
plt.ylabel('Salary')
plt.show()


#lets visualize the result on test dataset (yellow is ann; blue is LinReg)
plt.scatter(X_test,y_test, color='red')#point real
plt.plot(X_train,regressor.predict(X_train), color='blue') #line
plt.plot(X_train,model.predict(X_train), color='yellow') #line
plt.title('Salary vs Exp (test set)')
plt.xlabel('years of exp')
plt.ylabel('Salary')
plt.show()


#Now finally lets examine the R-Squeared Error in both models
from sklearn.metrics import r2_score
print("Conventional Linear Regression: ", r2_score(y_test, y_pred_reg))
print("Artificial Neural Network: ", r2_score(y_test, y_pred_ann))


""" CONCLUSION:
    
In Both Cases we got around 98.8% Accuracy, 
But in ANN we can see the accuracy is a tiny bit more"""




