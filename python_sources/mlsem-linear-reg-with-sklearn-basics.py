import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))

class HousingDataHeaderElement:
    def __init__(self, shortName, description):
        self.shortName = shortName
        self.description = description

    def __str__(self):
        return self.shortName + " (" + self.description + ")"

def boston_housing_simple():
    housingDataHeader = [['CRIM', 'per capita crime rate by town'], 
                     ['ZN', 'proportion of residential land zoned for lots over 25,000 sq.ft.'], 
                     ['INDUS', 'proportion of non-retail business acres per town'],
                     ['CHAS', 'Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)'],
                     ['NOX', 'nitric oxides concentration (parts per 10 million)'],
                     ['RM', 'average number of rooms per dwelling'],
                     ['AGE', 'proportion of owner-occupied units built prior to 1940'],
                     ['DIS', 'weighted distances to five Boston employment centres'],
                     ['RAD', 'index of accessibility to radial highways'],
                     ['TAX', 'full-value property-tax rate per $10,000'],
                     ['PTRATIO', 'pupil-teacher ratio by town'],
                     ['B', '1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town'],
                     ['LSTAT', '% lower status of the population'],
                     ['MEDV', 'Median value of owner-occupied homes in $1000s']]

    housingDataHeader = list(HousingDataHeaderElement(element[0], element[1]) for element in housingDataHeader)
    print("\n".join([str(element) for element in housingDataHeader]))
    
    fullHousingData = pd.read_csv("../input/housingdata.csv", header=None)
    
    fullHousingData.columns = [element.shortName for element in housingDataHeader]
    print('data head\n' + str(fullHousingData.head(3)))
    
    trainingSetPerAllDataFraction = 0.6
    housingTrainingSet = fullHousingData.sample(frac=trainingSetPerAllDataFraction)
    housingTestSet = fullHousingData.loc[~fullHousingData.index.isin(housingTrainingSet.index)]
    
    housePropertyTrainingSet = housingTrainingSet.iloc[:, 0:13]
    housePricesTrainingSet = housingTrainingSet.iloc[:, 13]
    
    print('housePropertyTrainingSet head\n' + str(housePropertyTrainingSet.head(3)))
    print('housePricesTrainingSet head\n' + str(housePricesTrainingSet.head(3)))
    
    print('housePropertyTrainingSet shape: ' + str(housePropertyTrainingSet.shape))
    print('housePricesTrainingSet shape: ' + str(housePricesTrainingSet.shape))
    
    housePropertyTrainingSet = housePropertyTrainingSet.values
    housePricesTrainingSet = housePricesTrainingSet.values
    
    housePricesTrainingSet = housePricesTrainingSet.reshape(housePricesTrainingSet.shape[0], 1)
    
    print('new (desired) housePricesTrainingSet shape: ' + str(housePricesTrainingSet.shape))
    
    housePropertyTestSet = housingTestSet.iloc[:, 0:13].values
    housePricesTestSet = housingTestSet.iloc[:, 13].values
    
    housePricesTestSet = housePricesTestSet.reshape(housePricesTestSet.shape[0], 1)
    
    print('housePropertyTestSet shape: ' + str(housePropertyTestSet.shape))
    print('housePricesTestSet shape: ' + str(housePricesTestSet.shape))
    
    regr = linear_model.LinearRegression()
    regr.fit(housePropertyTrainingSet, housePricesTrainingSet)
    
    print("coefficients: " + ", ".join('%.3f' % coeff for coeff in regr.coef_[0]))
    print("intercept: " + "%.3f" % regr.intercept_[0])
    
    pricePredictions = regr.predict(housePropertyTestSet)
    fig1 = plt.figure(1)
    plt.scatter(housePricesTestSet, pricePredictions)
    plt.xlabel("Prices")
    plt.ylabel("Predicted Prices")
    
    fig2 = plt.figure(2)
    predictionOnTrainingSet = regr.predict(housePropertyTrainingSet)
    trainingPriceResidual = predictionOnTrainingSet - housePricesTrainingSet
    testPriceResidual = pricePredictions - housePricesTestSet
    plt.scatter(predictionOnTrainingSet, trainingPriceResidual, c='b', alpha=0.5)
    plt.scatter(pricePredictions, testPriceResidual, c='g')
    plt.ylabel("Residuals")
    
    plt.show()

boston_housing_simple()