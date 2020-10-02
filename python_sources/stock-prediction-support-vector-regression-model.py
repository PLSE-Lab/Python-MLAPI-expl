# ML Model Based on Siraj Raval's Stock Price Prediction Tutorial
# Support Vector Regression Models (Linear, Polynomial, RBF) for predicting future stock high prices

# In[14]:

import csv
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

#Reads in data from online source, updated daily
#Replace MFST with other stock symbol per desire
data = pd.read_csv('../input/daily_adjusted_MSFT (2).csv')
data = data.head(20)

dfdates  = []
dfprices = []
# Preparing Data



###Use this function for pandas dataframes
def gd(dataframe):
    i = 1.0
    while i < len(dataframe):
        a = dataframe.xs(i)
        #print(a)
        dfdates.append(i)
        #print(dates)
        dfprices.append(a[1])
        i += 1
    return


#Use this function for raw local csv files
def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)	# skipping column names
        i = 1.0
        for row in csvFileReader:
            csvdates.append(i)
            #print(dates)
            csvprices.append(float(row[1]))
            i += 1
    return

#input: dates, prices of length n for n days
#trains models, plots models, returns each model's prediction on day #x
def predict_prices(dates, prices, x):
    dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1

    svr_lin = SVR(kernel= 'linear', C= 1e3)
    svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
    svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
    svr_rbf.fit(dates, prices) # fitting the data points in the models
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)

    plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints
    plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
    plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
    plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]



#parses in dataframe, creates dates and prices arrays, dates is just 1-n for calculation purposes
gd(data)

#reverses prices list, which was fed in reverse chronological order originally
dfprices.reverse()

#trains models, so far model not fast enough for >20 day inputs
predicted_price = predict_prices(dfdates, dfprices, 20)

#prints out predictions from prediction function
print("RBF:", predicted_price[0],"LIN:", predicted_price[1],"POLY:", predicted_price[2])