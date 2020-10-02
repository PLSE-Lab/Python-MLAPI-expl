# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder #for data preprocessing: converting categorical values into numerical
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None # to disable the warning signs when replacing the original columns values with new values 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Function to convert the "Yes" and "No" into 1 and 0 in the feature variable and dependent Y variable
def conv(x):
    if x == 'No':
        return(0)
    else:
        return(1)

#Reading the data
dat = pd.read_csv("../input/weatherAUS.csv")

#Checking the data types of all features here
feat_var = dat.columns #Extracting th column names
print(" The number of variables are ", len(feat_var))

print(" The data types of different variables are:")
print(dat.dtypes)

print("Total number of observations are: ", len(dat))
# Some variables have "NaN" values, so I decided to remove any observation if any one of the variable has NaN values
data_final = dat.dropna(how = 'any')
print("Total number of observations after deleting NaN values are: ", len(data_final))

#Convertin the categorical vaiable into numeric digits
lab_enc = LabelEncoder()

data_final['WindGustDir'] = lab_enc.fit_transform(data_final['WindGustDir'])
data_final['WindDir9am'] = lab_enc.fit_transform(data_final['WindDir9am'])
data_final['WindDir3pm'] = lab_enc.fit_transform(data_final['WindDir3pm'])

#Converting yes and no into 1 and 0
data_final['RainToday'] = data_final['RainToday'].apply(conv)

#I thought data and location variable might not contribute anything to the model, so decided not to use it
data_final = data_final.drop(['Date','Location'], axis = 1)

#INput and Output variables
X = data_final[data_final.columns[:-1]]
Y = data_final[data_final.columns[-1:]]

#Varibles used as inputs
print("The {0} variables used as inputs are :".format(len(X.columns)))
print(X.columns.values)
#Converting yes and no into 1 and 0
Y = Y.RainTomorrow.apply(conv) 

#splitting the data for testing and trainning 
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.7)

#As the removing of observations containing NaN reduced the usable observation, I decided to use
#Gaussian Naive Bayes, as it is better when the sample is sample
#
gaus_model = GaussianNB()
gaus_model.fit(X_train, y_train)

#Predicting the test sample
pred = gaus_model.predict(X_test)

#Getting the accuracy of the model
accu = gaus_model.score(X_test, y_test)
print("The accuracy of the model is: {0}".format((accu * 100).round(2))) 

#To predict the probability of raining 
pred_prb = gaus_model.predict_proba(X_test)

