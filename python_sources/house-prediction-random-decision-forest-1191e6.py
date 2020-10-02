#!/usr/bin/env python
# coding: utf-8

# # **House Price Prediction (Random Decision Forest)**

# ## Models Trained with Few Selected Features
# Codes below will train 3 models using some features and print the MAE for every models.

# 

# 

# In[ ]:


# Import Libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import joblib
from xgboost import XGBRegressor


# In[ ]:


skripsiku = '../input/skripsiku/datapanas.csv'
homeData = pd.read_csv(skripsiku)
homeData.describe()

y = homeData.CurahHujan

features = ['Provinsi', 'Kota', 'Long', 'Lat',
            'Bulan', 'Hari', 'FFMC', 'DMC', 
            'DC', 'ISI', 'Temperatur', 'KecepatanAngin', 'CurahHujan', 
            'Luas Area']

x = homeData[features]
x = pd.get_dummies(x)
imputer = SimpleImputer()
x = imputer.fit_transform(x)

trainX, valX, trainY, valY = train_test_split(
    x, y, shuffle = False)

#print data train dari index 0 sampai 198
print("Sumber data sebelum train_test_split (index 0=198)")
print(x[:299])
print("Data Training")
print(trainX)

#print data test dari index 198 sampe terakhir buat test
print("Sumber data sebelum train_test_split (index 199-399)")
print(x[299:])
print("Data Testing")
print(valX)
hotspotPredictorModel = DecisionTreeRegressor(random_state=1)

hotspotPredictorModel.fit(trainX, trainY)
joblib.dump(hotspotPredictorModel, 'my_model.pkl', compress=9)
valPredictions = hotspotPredictorModel.predict(valX)
valMae = mean_absolute_error(valPredictions, valY)
print("Validation MAE when not " +
      "specifying max_leaf_nodes: {:,.0f}".format(valMae))

hotspotPredictorModel = DecisionTreeRegressor(
    max_leaf_nodes=100, random_state=1)
hotspotPredictorModel.fit(trainX, trainY)
valPredictions = hotspotPredictorModel.predict(valX)
print(valX[0:5],valPredictions[0:5])
valMae = mean_absolute_error(valPredictions, valY)
print("Validation MAE for best " +
      "value of max_leaf_nodes: {:,.0f}".format(valMae))

rfModel = RandomForestRegressor(
    random_state=1, n_estimators=100)
rfModel.fit(trainX, trainY)
rfValPredictions = rfModel.predict(valX)
rfValMae = mean_absolute_error(rfValPredictions, valY)

print("Validation MAE for Random " +
      "Forest Model: {:,.0f}".format(rfValMae))
prediction = pd.DataFrame(valPredictions, columns=['HASIL_PREDIKSI']).to_csv('prediction.csv')


# ## Models Trained with All Features

# ### Handling Missing Values
# Codes below will train all models with all features. Columns with missing values are handled by using 3 methods. Columns with non-numerical values are dropped.

# In[ ]:


# This is a function to calculate MAE score using Random Forest model
def maeScore(trainX, trainY, testX, testY):
    rfModel = RandomForestRegressor(
            random_state=1, n_estimators=100)
    rfModel.fit(trainX, trainY)
    rfValPredictions = rfModel.predict(testX)
    return mean_absolute_error(rfValPredictions, testY)


# In[ ]:


# Preparation
homeTarget = homeData.SalePrice
homePredictors = homeData.drop(['SalePrice'], axis=1)
homePredictors = homePredictors.select_dtypes(exclude=['object'])
trainX, testX, trainY, testY = train_test_split(
        homePredictors, homeTarget, random_state=1)


# In[ ]:


# Handling missing values by dropping columns
colsWithMissing = [col for col in trainX.columns
                   if trainX[col].isnull().any()]
reducedTrainX = trainX.drop(colsWithMissing, axis=1)
reducedTestX = testX.drop(colsWithMissing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(maeScore(reducedTrainX, trainY, reducedTestX, testY))


# In[ ]:


# Handling missing values by imputation
imputer = SimpleImputer()
imputedTrainX = imputer.fit_transform(trainX)
imputedTestX = imputer.fit_transform(testX)
print("Mean Absolute Error from Imputation:")
print(maeScore(imputedTrainX, trainY, imputedTestX, testY))


# In[ ]:


# Handling missing values by imputation and telling which values are imputed
imputedTrainXPlus = trainX.copy()
imputedTestXPlus = testX.copy()
for col in colsWithMissing:
    imputedTrainXPlus[col + 'was_missing'] = imputedTrainXPlus[col].isnull()
    imputedTestXPlus[col + 'was_missing'] = imputedTestXPlus[col].isnull()

imputerPlus = SimpleImputer()
imputedTrainXPlus = imputerPlus.fit_transform(imputedTrainXPlus)
imputedTestXPlus = imputerPlus.fit_transform(imputedTestXPlus)
print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(maeScore(imputedTrainXPlus, trainY, imputedTestXPlus, testY))


# ## Using One Hot Encoding for Categorical Data

# In[ ]:


houseTarget = homeData.SalePrice
housePredictors = homeData.drop(['SalePrice'], axis=1)
encodedPredictors = pd.get_dummies(housePredictors)
trainX, testX, trainTarget, testTarget = train_test_split(
        encodedPredictors, houseTarget, random_state=1)
imputedEncodedTrainX = imputer.fit_transform(trainX)
imputedEncodedTestX = imputer.fit_transform(testX)
print("Mean Absolute Error from Imputation & One Hot Encoding:")
print(maeScore(imputedEncodedTrainX, trainTarget, imputedEncodedTestX, testTarget))

