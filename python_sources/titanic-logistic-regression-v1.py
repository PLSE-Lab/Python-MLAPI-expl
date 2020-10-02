import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from numpy import loadtxt, where, zeros, e, array, log, ones, append, linspace
import scipy.optimize as op




train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64})
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64},)

print(train.head())

print(train.describe())
print(train.info())

train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2

test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1

train['Fare'] = train['Fare'].fillna(train["Fare"].median())
train['Fare'] = train['Fare'].astype(int)

test['Fare'] = test['Fare'].fillna(test["Fare"].median())
test['Fare'] = test['Fare'].astype(int)

train["Age"] = train["Age"].fillna(train["Age"].median())
train['Age'] = train['Age'].astype(int)

test["Age"] = test["Age"].fillna(test["Age"].median())
test['Age'] = test['Age'].astype(int)

train["Family"] = train["SibSp"] + train["Parch"]
train['Family'].loc[train['Family'] > 0] = 1
train['Family'].loc[train['Family'] == 0] = 0

test['Family'] =  test["Parch"] + test["SibSp"]
test['Family'].loc[test['Family'] > 0] = 1
test['Family'].loc[test['Family'] == 0] = 0
pid=test["PassengerId"]

family_mean = train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_mean, order=[1,0], palette="Set3")

train = train.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
test = test.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)

predictors = ["Age", "Fare", "Embarked", "Family", "Sex", "Pclass", "SibSp", "Parch"]

selector = SelectKBest(f_classif, k=5)
selector.fit(train[predictors], train["Survived"])

scores = -np.log10(selector.pvalues_)
scores, predictors = zip(*sorted(zip(scores, predictors)))

def sigmoid(z):
# Computes the Sigmoid function of z

    g = 1.0 / (1.0 + e ** (-1.0 * z))

    return g 

def compute_cost(theta,X,y): 
# Computes cost function
   
    #h = sigmoid(theta.T.dot(X.T))
    h = sigmoid(X.dot(theta).T)
    
    m = len(y);

    J = (1.0/m)*sum(-y.T.dot(np.log(h.T)) - (1-y).T.dot(np.log(1-h.T)))
    
    return  J

def compute_grad(theta, X, y):
# Computes gradient of cost function 

    grad = zeros(len(theta))
    
    h = sigmoid(theta.T.dot(X.T))
    
    m = len(y);

    for j in range(len(theta)):
       #grad(j)=(1/m)*sum((h'-y).*X(:,j));

       grad[j]=(1.0/m)*sum((h.T-y).T.dot(X[:,j]))

    grad.shape=(len(grad),1)

    return grad

def prediction(theta, X):
# Predicts the survival (0 or 1) based on learned logistic regression data

    a, b = X.shape
    pred = zeros(shape=(a, 1))

    h = sigmoid(X.dot(theta.T))

    for i in range(0, h.shape[0]):
        if h[i] > 0.5:
            pred[i] = 1
        else:
            pred[i] = 0

    return pred

y_t = train['Survived']
X_t = train.drop("Survived",axis=1)

y = y_t.values
X_temp = X_t.values

X_temp_test = test.values

m, n = X_temp.shape

m_test,n_test = X_temp_test.shape

X=np.ones((m,n+1))

X_test=np.ones((m_test,n_test+1))  

X[:,-n:] = X_temp

X_test[:,-n:] = X_temp_test

y.shape = (m, 1)

y_test = np.ones((m_test, 1))

initial_theta = zeros(shape=(n+1, 1))

cost = compute_cost(initial_theta, X, y)
grad = compute_grad(initial_theta, X, y)

def decorated_cost(theta):
    return compute_cost(theta, X, y)
    
Result = op.minimize(fun = compute_cost, x0 = initial_theta, args = (X, y), method = 'TNC');
optimal_theta = Result.x;

p = prediction(array(optimal_theta), X)

t = (p == y)

y_test = prediction(optimal_theta, X_test) 

y_test = y_test.astype(int)

submission = pd.DataFrame({
        "PassengerId": pid,
        "Survived": y_test.reshape(-1)
        })

submission.to_csv('titanic.csv', index=False)