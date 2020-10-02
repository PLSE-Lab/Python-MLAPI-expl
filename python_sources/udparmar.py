import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing DataSets------------------------------------------------------------------------------------------------------

dataset = pd.read_csv('../input/50-startups/50_Startups.csv')
#print(dataset) 

x = dataset.iloc[:,: -1].values
y = dataset.iloc[:,4].values



# Handling missing Data(Replacing missing values with respective Mean value)--------------------------------------------------

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values = 0, strategy= 'mean')
imp_mean.fit(x[:,0:3])
x[:,0:3] = imp_mean.transform(x[:,0:3])



# Dealing with(Encoding) Categorical Data---------------------------------------------------------------------------------------------

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_x = LabelEncoder()                                     # Making Object of LabelEncoder,
x[:, 3] = labelencoder_x.fit_transform(x[:,3])                      # fitting in it.

# State column

ct = ColumnTransformer( [ ('State', OneHotEncoder(), [3]) ], remainder ='passthrough')
x = ct.fit_transform(x)

x = x[:,1:]                                                         # Avoiding Dummy Variable Trap



# Splitting the dataSet into the training set and test set -----------------------------------------------------------------------

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 1/3, 
                                                         random_state = 0)


# Fitting the MLR Model to the training dataSet -----------------------------------------------------------------------------------

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()                              # Making object of LinearRegression,
regressor.fit(x_train,y_train)                              # fitting in it.


# Predicting the test,train set result --------------------------------------------------------------------------


y_pred = regressor.predict(x_test)



# Printing Coefficients -------------------------------------------------------------------------------------

print('Intercept =', regressor.intercept_)
print('Coefficient =', regressor.coef_)
accuracy = (regressor.score(x_test,y_test))
print(accuracy)

