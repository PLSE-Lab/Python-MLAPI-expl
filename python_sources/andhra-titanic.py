import numpy as np
import pandas as pd
from sklearn import  linear_model
import math

def prep_data(data,training=False):
    pass_sex=[]
    pass_EmbIdx = []
    pass_class =[]
    pass_fare=[]
    pass_sibling=[]
    pass_parch=[]
    pass_age=[]
    pass_survived =[]
    for idx, row in data.iterrows():
        #Skip if any of the entries is NaN
        if pd.isnull(row["Embarked"]) or pd.isnull(row["Sex"]) or pd.isnull(row["Pclass"]) or pd.isnull(row["Fare"]) \
            or pd.isnull(row["SibSp"]) or pd.isnull(row["Parch"]):
            continue
        
        if row["Embarked"] == "S":
            pass_EmbIdx= np.append(pass_EmbIdx,1)
        elif row["Embarked"] == "C":
            pass_EmbIdx= np.append(pass_EmbIdx,2)
        elif row["Embarked"] == "Q":
            pass_EmbIdx= np.append(pass_EmbIdx,3)
        
            
        
        if row["Sex"] == 'male':
            pass_sex= np.append(pass_sex,0)
        else:
            pass_sex = np.append(pass_sex,1)
        pass_class = np.append(pass_class,row["Pclass"])
        pass_age = np.append(pass_age,row["Age"])
        pass_fare = np.append(pass_fare,row["Fare"])
        pass_sibling = np.append(pass_sibling,row["SibSp"])
        pass_parch = np.append(pass_parch,row["Parch"])
        if training:
            pass_survived = np.append(pass_survived,row["Survived"])
    return np.c_[pass_sex,pass_class,pass_fare,pass_sibling,pass_parch,pass_EmbIdx], (pass_survived)
#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64, "Fare" : np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

#Prepare training data
X_train, Y_train = prep_data(train,training=True)
print (X_train.shape)
X_test, Y_test = prep_data(test)
print (X_test.shape)
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)
#Print the model coeffs
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_train) - Y_train) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_train, Y_train))

#Test your model
Y_test=regr.predict(X_test)
Y_test_predict = [1 if z>=0.5 else 0 for z in Y_test]
#Write to a csv
a = np.asarray(Y_test_predict)
print (a)
np.savetxt("Kaggle_titanic_1.csv", a, delimiter=",")
#print(Y_test_predict)