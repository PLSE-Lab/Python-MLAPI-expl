# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
os.system('pip install reverse_geocoder')

from sklearn import linear_model
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from scipy.stats import norm
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import math
import reverse_geocoder as rgc
import warnings

warnings.simplefilter('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

filePath = ''
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filePath = os.path.join(dirname, filename)

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv(filePath, sep = ",", header = 0)

#Start PreProcessing
#Changing the date format to get the year 
train_data['dateInFormat'] = train_data['date'].apply(lambda x: x[0:8])

train_data['dateInFormat'] = train_data['dateInFormat'].apply(lambda x: datetime.strptime(x, '%Y%m%d'))


#The year in which the propert was sold
train_data['YearSold'] = train_data['dateInFormat'].apply(lambda x: int(x.year))

#Check the correlation matrix with price (the below command only works with jupyter notebook or other software having HTML support)
#train_data.corr().loc[:,['Log_price', 'price']].style.background_gradient(cmap='coolwarm', axis=None)

#We derive the Time since its built
train_data['HomeAgeinYear'] = train_data['YearSold'] - train_data['yr_built']

#We derive the Time after which it was renovated
train_data['RenovatedafterYears'] = train_data['yr_renovated'] - train_data['yr_built']

#Lot of variables are not specifically normally distributed. Therefore transforming them using log(Base e) transform
train_data['Log_sqftlot'] = train_data['sqft_lot'].apply(lambda x: np.log(x))
train_data['Log_price'] = train_data['price'].apply(lambda x: np.log(x))
train_data['Log_sqftlot15'] = train_data['sqft_lot15'].apply(lambda x: np.log(x))
train_data['Log_sqftLiving'] = train_data['sqft_living'].apply(lambda x: np.log(x))


#Visualize different data
#mu, std = norm.fit(train_data['price'])
mu, std = norm.fit(train_data['Log_price'])
#plt.hist(train_data['price'], color = "green", normed = True)
#xmin, xmax = plt.xlim(train_data['price'].min(), train_data['price'].max())
plt.hist(train_data['Log_price'], color = "blue", normed = True)
xmin, xmax = plt.xlim(train_data['Log_price'].min(), train_data['Log_price'].max())


pdf_x = np.linspace(xmin, xmax, 100)
pdf_y = norm.pdf(pdf_x, mu, std)
plt.plot(pdf_x, pdf_y, 'k--')
plt.xlabel('Price Range')
plt.ylabel('# Homes in Price Range')
plt.title('Histogram for Price')
plt.show()


#mu_living, std_living = norm.fit(train_data['sqft_living'])
mu_living, std_living = norm.fit(train_data['Log_sqftLiving'])

#plt.hist(train_data['sqft_living'], color = "red", normed = True)
plt.hist(train_data['Log_sqftLiving'], color = "green",  normed = True)
#xminsq, xmaxsq = plt.xlim(train_data['sqft_living'].min(), train_data['sqft_living'].max())
xminsq, xmaxsq = plt.xlim(train_data['Log_sqftLiving'].min(), train_data['Log_sqftLiving'].max())

pdf_sqx = np.linspace(xminsq, xmaxsq, 100)
pdf_sqy = norm.pdf(pdf_sqx, mu_living, std_living)
plt.plot(pdf_sqx,pdf_sqy, 'k--')
plt.xlabel('Square Feet Range')
plt.ylabel('# Homes in Square Feet Range')
plt.title('Histogram for Square Feet')
plt.show()


#GeoLocation to be added into the picture. Try to convert Latitude and Longitude to names of places in Kings County Region
for i in range(0, len(train_data['lat'])):
	coordiantes = (train_data.loc[i,'lat'], train_data.loc[i,'long'])
	train_data.loc[i, 'Location'] = rgc.search(coordiantes, mode = 1)[0].get('name')


mappingDictionary = {}

#Mapping the location to the mean of prices associated with the region
for k in train_data.Location.unique():
	mappingDictionary[k] = train_data[train_data['Location'] == k].price.mean()

#Using the above made dictionary to map each row of dataset with the mean value of price for that location
train_data['LocationMapping'] = train_data['Location'].apply(lambda x: mappingDictionary.get(x))

#Normalizing the values to make the scale the value
meanOfLocation = train_data['LocationMapping'].mean()
standardDeviationLocation = train_data['LocationMapping'].std()

#Normalizing the price to check whether the VIF changes or not
meanOfPrice = train_data['Log_price'].mean()
standardDeviationOfPrice = train_data['Log_price'].std()
train_data['ScaledLog_Price'] = train_data['Log_price'].apply(lambda x: ((x-meanOfPrice)/standardDeviationOfPrice))

#Scaling the values to reduce the range of columns 
train_data['LocationMapping'] = train_data['LocationMapping'].apply(lambda x: ((x-meanOfLocation)/standardDeviationLocation))

#The living space is a part of the total lot space therefore finding the ratio of Living Space available
train_data['LivingSpaceAvailable'] = train_data['sqft_living']/train_data['sqft_lot']

#The living space ratio of neighbouring 15 houses
train_data['NeighbourSpace'] = train_data['sqft_living15']/train_data['sqft_lot15']

#variables are not specifically normally distributed. Therefore transforming them using log(Base e) transform
train_data['Log_LivingSpaceAvailable'] = train_data['LivingSpaceAvailable'].apply(lambda x: np.log(x))
train_data['Log_NeighbourSpace'] = train_data['NeighbourSpace'].apply(lambda x: np.log(x))


#The Value of VIF tells that there is a collinearity between the Living space and above space. Assuming that both will be same if basement is not there.
#Therefore removing basement values and converting it into a variable to express the presence or absence of it

scaler = StandardScaler(with_mean = True, with_std = True)

#Training Vector based on correlation and VIF
columnsToTrain = ['Log_sqftLiving','waterfront','floors','view', 'grade', 'HomeAgeinYear','LocationMapping', 'Log_LivingSpaceAvailable', 'Log_NeighbourSpace', 'RenovatedafterYears']

#Can be used to check the VIF values with and without constant
scaledData = scaler.fit(train_data.loc[:, columnsToTrain])
scaledArray = scaler.transform(train_data.loc[:, columnsToTrain])
scaledDataFrame = pd.DataFrame(scaledArray, columns = ['Log_sqftLiving','waterfront','floors','view', 'grade', 'HomeAgeinYear','LocationMapping', 'Log_LivingSpaceAvailable', 'Log_NeighbourSpace', 'RenovatedafterYears'])

#Predicting the Log Price 
X, y = train_data.loc[:, columnsToTrain], train_data.loc[:, 'Log_price']

#Generating VIF Data
#Anything less than with vif < 5 can be considered to have less colinearity 
vif = pd.DataFrame()
New_X = add_constant(X)
vif['VIF Factors']  = [variance_inflation_factor(New_X.values, i) for i in range(New_X.shape[1])]
vif['Columns'] = New_X.columns

#Splitting the values into training and validation set
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size = 0.30)

#Polynomial regression since columns like floors, view had polynomial relation with the log of price 
#Using Degree = 3, because anything greater than 3 leads to overfitting and less than 3 leads to underfitting
polynomialVariable = PolynomialFeatures(degree = 3)
polynomialCurveFitting = polynomialVariable.fit_transform(X_train)
polynomialCurveFittingTest = polynomialVariable.fit_transform(X_test)

#Generating the test results by taking exponential of log values to get the actual price again
Y_train_Exponential = [math.exp(x) for x in Y_train]
Y_test_Exponential = [math.exp(x) for x in Y_test]


#Ridge Regression Model
RidgeModel = Ridge(alpha = 500.0, solver = 'auto', fit_intercept = True)
fittedRidgeModel = RidgeModel.fit(polynomialCurveFitting, Y_train)
scoreRidge = model_selection.cross_val_score(fittedRidgeModel, polynomialCurveFittingTest, Y_test, cv = 10) 

print('Ridge Regression Average Score', np.mean(scoreRidge))
print('\n')

PredictedRidgeDataTrain = fittedRidgeModel.predict(polynomialCurveFitting)
PredictedRidgeDataTrainExponential = [math.exp(x) for x in PredictedRidgeDataTrain]
PredictedRidgeData = fittedRidgeModel.predict(polynomialCurveFittingTest)
PredictedRidgeDataExponential = [math.exp(x) for x in PredictedRidgeData]

print('Root mean square Value Train Ridge Regression', np.sqrt(mean_squared_error(Y_train_Exponential, PredictedRidgeDataTrainExponential)))
print('Root mean square Value Test Ridge Regression', np.sqrt(mean_squared_error(Y_test_Exponential, PredictedRidgeDataExponential)))
print('\n')

#Fitting a linear model on lienar features
model = LinearRegression()
fittedModel = model.fit(X_train, Y_train)

#Fitting a Linear model on polynomial feaures
model2 = LinearRegression()
fittedModel2 = model2.fit(polynomialCurveFitting, Y_train)

#Predict the values using a Polynomial model
PredictedTrainData = fittedModel2.predict(polynomialCurveFitting)

#Convert predicted log price to actual Log price by taking exponential 
PredictedTrainDataExponential = [math.exp(x) for x in PredictedTrainData]

#Mean Squared Training Error
#print('RootMeanSquare Training', np.sqrt(mean_squared_error(Y_train_Exponential, PredictedTrainDataExponential)))

#Predict the values using a Polynomial model
PredictedTestData = fittedModel2.predict(polynomialCurveFittingTest)

#Convert predicted log price to actual Log price by taking exponential
PredictedTestDataExponential = [math.exp(x) for x in PredictedTestData]

print('Root mean square Value Least Square Polynomial Regression', np.sqrt(mean_squared_error(Y_train_Exponential, PredictedTrainDataExponential)))
print('Root mean square Value Least Square Polynomial Regression', np.sqrt(mean_squared_error(Y_test_Exponential, PredictedTestDataExponential)))
print('\n')
#scores = model_selection.cross_val_score(model2, polynomialCurveFitting, Y_train, cv = 10)
scoresTest = model_selection.cross_val_score(model2, polynomialCurveFittingTest, Y_test, cv = 10)

#print(fittedModel.coef_)
#print(fittedModel.score(X_train, Y_train))

#print('Polynomial Regression Score', fittedModel2.score(polynomialCurveFitting, Y_train))

#Using a statsmodel package to fit the linear regression model
#The package allows us to view the summary of regression in a very R like format
#It also provides the vaule of Adjusted-R^2, a valuable term to assess the fit of the model
formuala = 'Log_price ~ Log_sqftLiving+waterfront+floors+view+grade+HomeAgeinYear+LocationMapping+Log_LivingSpaceAvailable+Log_NeighbourSpace+RenovatedafterYears'
statisticalModel = sm.ols(formuala, data = train_data)
statsfitted = statisticalModel.fit()
predictedStats = statsfitted.predict(X_test)

#Convert predicted log price to actual Log price by taking exponential
predictedStatsExponential = [math.exp(x) for x in predictedStats]


print('Testing Root Mean Square Linear Model', np.sqrt(mean_squared_error(Y_test_Exponential, predictedStatsExponential)))
print('\n')
print('Statistical Summary of Least Square Regression Model with an adjusted R^2 = 80% \n', statsfitted.summary())
print('\n')

