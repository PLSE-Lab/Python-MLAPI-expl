import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
###################################
'''Read training data'''
df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
###################################
passengerId = df['PassengerId']
y = df['Survived']
###################################
'''Transform 'Sex' column to represent males with 1 and females with 0'''
Sex = df['Sex'].copy()
Sex[Sex=='male'] = 1
Sex[Sex!=1] = 0
df['Sex'] = Sex
###################################
'''Fill 'Nan' values in the 'Age' column with the average of the column'''
m = df['Age'].mean()
df['Age'].fillna(value=m, inplace=True)
###################################
'''Define the Dataframe X, as follows & split it into train and testing datasets'''
X = df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
###################################
'''Standardize the columns of X_train in order to improve performance of the Logistic Regression Model & apply the same standardization parameters to X_test for consistency'''
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
###################################
'''Create Logistic Regression model with a low regularization parameter, fit it and make predictions'''
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)
###################################
'''Print the percentage of correctly predicted test values'''
print ("Logistic Regression Train Test Accuracy: {}".format(sum(y_pred==y_test)/len(y_test)*100.0))
###################################