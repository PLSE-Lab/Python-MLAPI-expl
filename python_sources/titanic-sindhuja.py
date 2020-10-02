import numpy as np
import pandas as pd
from sklearn import linear_model

def prepare_data(data, Trained ="False"):
    pass_result = []
    pass_class = []
    pass_sex = []
    pass_sib = []
    pass_parch = []
    pass_fare = []
    pass_emb = []
    for index, rows in data.iterrows():
        if pd.isnull(rows["Pclass"]) or pd.isnull(rows["Sex"]) or pd.isnull(rows["SibSp"]) \
        or pd.isnull(rows["Parch"]) or pd.isnull(rows["Fare"]) or pd.isnull(rows["Embarked"]):
            continue
        
        if rows["Sex"] == "female":
            pass_sex = np.append(pass_sex, 0)
        if rows["Sex"] == "male":
            pass_sex = np.append(pass_sex, 1)
        
        if rows["Embarked"] == "C":
            pass_emb = np.append(pass_emb, 0)
        if rows["Embarked"] == "Q":
            pass_emb = np.append(pass_emb, 1)
        if rows["Embarked"] == "S":
            pass_emb = np.append(pass_emb, 2)
            
        if Trained=="True":
            pass_result = np.append(pass_result,rows["Survived"])
            
        pass_class = np.append(pass_class, rows["Pclass"])
        pass_sib = np.append(pass_sib, rows["SibSp"])
        pass_parch = np.append(pass_parch, rows["Parch"])
        pass_fare = np.append(pass_fare, rows["Fare"])
    
    return np.c_[pass_sex, pass_emb, pass_class, pass_sib, pass_parch, pass_fare], pass_result

train = pd.read_csv ("../input/train.csv")
test = pd.read_csv("../input/test.csv")

X_train, Y_train = prepare_data(train, "True")
X_test, Y_test = prepare_data(test)
    
regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
print('Coefficients: \n', regr.coef_)

Y_test = regr.predict(X_test)
Y_test_final = [1 if x>=0.5 else 0 for x in Y_test]

a = np.asarray(Y_test_final)
print (a)
np.savetxt("Kaggle_titanic_2.csv", a, delimiter=",")