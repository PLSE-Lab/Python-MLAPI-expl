import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


train_set = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

def clearn_data(train):
    col = train.columns
    number_col = []
    obj_col = []
    for i in col:
        if train_set[i].dtype == "int64" or train_set[i].dtype == "float64":
            number_col.append(i)
        elif train_set[i].dtype == "object":
            obj_col.append(i)
            
    train["GarageYrBlt"] = train["GarageYrBlt"].fillna(0)
    train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
    train[obj_col] = train[obj_col].fillna("None")
    train[number_col] = train[number_col].fillna(0)
    return train

def model(train, test, category_col, goal):
    train = clearn_data(train)
    test = clearn_data(test)
    category = list(np.unique(train[category_col]))
    category.append("None")
    
    col = train.columns
    number_col = []
    obj_col = []
    for i in col:
        if train[i].dtype == "int64" or train[i].dtype == "float64":
            number_col.append(i)
        elif train[i].dtype == "object":
            obj_col.append(i)
            
    high_interest = {}
    for i in category:
        a = train[train[category_col] == i]
        b = []
        for j in number_col:
            if j != "Id" and j != goal:
                if a[[goal,j]].corr().iloc[0, 1] >= 0.5:
                    b.append(j)
        high_interest[i] = b

    interest_col = []
    for i in number_col:
        if i != "Id" and i != goal:
            if np.abs(train[[i, goal]].corr().iloc[0, 1]) >= 0.5:
                interest_col.append(i)

    predicted = []
    k = 0
    for i in range(len(category)):
        table = train[train[category_col] == category[i]]
        col = high_interest[category[i]]
        test_1 = test[test[category_col] == category[i]]
        test_table = test[test[category_col] == category[i]][col]
        if category[i] != "None" and test_1.shape[0] > 0:            
            if i == 0:
                model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                          ('linear', LinearRegression(fit_intercept=False))])
                model.fit(table[col], table[goal])
                i = pd.DataFrame(model.predict(test_table), index=test_1["Id"])
                predicted.append(i)

                
            elif i == 1:
                model = Pipeline([('poly', PolynomialFeatures(degree=1)),
                          ('linear', LinearRegression(fit_intercept=False))])
                model.fit(table[col], table[goal])
                i = pd.DataFrame(model.predict(test_table), index=test_1["Id"])
                predicted.append(i)

                
            elif i == 2:
                model = Pipeline([('poly', PolynomialFeatures(degree=1)),
                          ('linear', LinearRegression(fit_intercept=False))])
                model.fit(table[col], table[goal])
                i = pd.DataFrame(model.predict(test_table), index=test_1["Id"])
                predicted.append(i)

                
            elif i == 3:
                model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                          ('linear', LinearRegression(fit_intercept=False))])
                model.fit(table[col], table[goal]) 
                i = pd.DataFrame(model.predict(test_table), index=test_1["Id"])
                predicted.append(i)
                
            elif i == 4:
                model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                          ('linear', LinearRegression(fit_intercept=False))])
                model.fit(table[col], table[goal])
                i = pd.DataFrame(model.predict(test_table), index=test_1["Id"])
                predicted.append(i)
            
            

        elif category[i] == "None" and test_1.shape[0] > 0:
            model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                      ('linear', LinearRegression(fit_intercept=False))])
            model.fit(train[interest_col], train[goal])
            Non = pd.DataFrame(model.predict(test_1[interest_col]),
                               index = test_1["Id"])
            predicted.append(Non)

    predict = pd.concat(predicted).rename(index = str,
                                          columns = {"Id":"Id", 0:"SalePrice"})
    predict = predict.sort_index()

    return predict

def rmsle(train, test):
    a = ((np.log(train + 1) - np.log(test + 1)) ** 2).sum() / train.shape[0]
    return np.sqrt(a)

category_col = "MSZoning"
goal = "SalePrice"
predict = model(train_set, test, category_col, goal)
predict.to_csv("submission.csv")
'''Any advices are welcome'''

