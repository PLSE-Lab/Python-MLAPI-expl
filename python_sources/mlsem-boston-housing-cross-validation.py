import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from sklearn import linear_model
from sklearn.model_selection import cross_val_score

def boston_housing_with_cross_validation():
    fullHousingData = pd.read_csv("../input/housingdata.csv", header=None)
    housingProperties = fullHousingData.iloc[:, 0:13]
    housingPrices = fullHousingData.iloc[:, 13]
    
    #regressor = linear_model.RidgeCV(alphas=[0.1, 0.5, 1, 10])
    regressor = linear_model.Ridge(alpha=0.5)
    scores = cross_val_score(regressor, housingProperties, housingPrices, cv = 5)
    print("scores")
    print(scores.mean())
    print(scores)
