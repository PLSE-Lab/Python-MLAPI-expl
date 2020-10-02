#Importing the libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import StandardScaler

#Reading the data
#Breaking the dataset by the # of Bedrooms
data = pd.read_excel('../input/House_Data.xlsx')
oneBHK = data[data.Bedrooms==1]
twoBHK = data[data.Bedrooms==2]
threeBHK = data[data.Bedrooms==3]
fourBHK = data[data.Bedrooms==4]
fiveBHK = data[data.Bedrooms==5]
X1 = oneBHK.iloc[:,0]
X2 = twoBHK.iloc[:,0]
X3 = threeBHK.iloc[:,0]
X4 = fourBHK.iloc[:,0]
X5 = fiveBHK.iloc[:,0]
X = data.iloc[:,:2]
Y1 = oneBHK.iloc[:,-1]
Y2 = twoBHK.iloc[:,-1]
Y3 = threeBHK.iloc[:,-1]
Y4 = fourBHK.iloc[:,-1]
Y5 = fiveBHK.iloc[:,-1]
Y = data.iloc[:,-1]

#Correlation between SqFt & Price
plt.scatter(X['SqFt'], Y, c ='red')
plt.xlabel('SqFt')
plt.ylabel('Price')

plt.scatter(X['Bedrooms'], Y, c='teal')
plt.xlabel('Bedrooms')
plt.ylabel('Price')

#Correlation between Bedrooms & Price
plt.scatter(X1, Y1, c = 'red', label = '1BHK')
plt.scatter(X2, Y2, c = 'blue', label = '2BHK')
plt.scatter(X3, Y3, c = 'black', label = '3BHK')
plt.scatter(X4, Y4, c = 'teal', label = '4BHK')
plt.scatter(X5, Y5, c = 'orange', label = '5BHK')
leg = plt.legend(fancybox=True, loc = 'lower right')

#We see that there is a positive correlation between both the independent variables and the dependent variable

#Fitting the model
reg = LinearRegression()
reg.fit(X, Y)
print ("Intercept:", reg.intercept_)
print ("Coefficient:", reg.coef_)

#Prediction
new_SqFt = 1600
new_Bedrooms = 3
print ("Predicted price of the new house is:",reg.predict([[new_SqFt, new_Bedrooms]]))