#IMDB- Meta Score Calculation


#Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Predict_Scaled, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print (regressor_OLS.summary())
    return x

#Loading Dataset 
IMDB=pd.read_csv('IMDB-Movie-Data.csv')
IMDB.describe()

#Selecting needed columns from Dataframe
Features_List=["Rank", "Year", "Runtime", "Rating", "Votes", "Revenue"]
Predict_List=["Metascore"]
Features=IMDB.loc[:, Features_List].values
Predict=IMDB.loc[:, Predict_List].values

#Adding Missing Values to the Columns
 
from sklearn.preprocessing import Imputer
imputer_features=Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer_predict=Imputer(missing_values="NaN", strategy="mean", axis=0)

imputer_features=imputer_features.fit(Features)
Features=imputer_features.transform(Features)

imputer_predict=imputer_predict.fit(Predict)
Predict=imputer_predict.transform(Predict)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
Scale=StandardScaler()
Features_Scaled=Scale.fit_transform(Features)
Predict_Scaled=Scale.fit_transform(Predict)
 

#Applying Backward Elimination
import statsmodels.formula.api as sm

SL = 0.05
#Features_Modeled = backwardElimination(Features, SL)

#Spliting the Dataset for Training and Testing
from sklearn.model_selection import train_test_split
Features_Train, Features_Test, Predict_Train, Predict_Test=train_test_split(Features_Scaled, Predict_Scaled, test_size=0.2, random_state=0)

#Creating Multiple Linear Regression Model
from sklearn.linear_model import LinearRegression
Metascore_Predictor=LinearRegression()
Metascore_Predictor.fit(Features_Train, Predict_Train)
Metascore_Predicted=Metascore_Predictor.predict(Features_Test)

#Regression Analysis

from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix


print ("R Score of the Model: " + str(mean_squared_error(Predict_Test, Metascore_Predicted)))
print ("R^2 Score of the Model: " + str(r2_score(Predict_Test, Metascore_Predicted)))
#print (Metascore_Predictor.coef_)
#print (Metascore_Predictor.intercept_)
#print (Metascore_Predictor.rank_)
















