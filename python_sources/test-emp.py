# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

x =np.array([1,2,3,4,5,6])
y =np.array([2,3,4,5,6,7])
print(1 - (np.cos(x) - np.cos(y)).sum())

def fn1(arr, k): return arr[arr.argsort()[:-k]]
def fn2(arr, k): return arr[arr.argsort()[:k][::-1]]
def fn3(arr, k): return arr[arr.argsort()[::-k]]
def fn4(arr, k): return arr[arr.argsort()[::-1][:k]]

print(fn1(np.array([5,3,2,1,4]), 2))
print(fn2(np.array([5,3,2,1,4]), 2))
print(fn3(np.array([5,3,2,1,4]), 2))
print(fn4(np.array([5,3,2,1,4]), 2))

x = np.array([[1,1], [1,1]])
y = np.array([[1,1], [1,1]])
z = x.dot(y)
print(z.shape)

x = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
y = x[:2, 1:3]
print(y)

x = np.ones((3,4))
y = np.random.random((5,1,4))
z = np.matmul(y, x.T)
print(z.shape)


# Any results you write to the current directory are saved as output.


def normalize(df, column = []):
    result = df.copy()
    col_list = df.columns
    if(len(column) > 0):
        col_list = column
    for feature_name in col_list:
        std = df[feature_name].std()
        avg = df[feature_name].mean()
        result[feature_name] = (result[feature_name] - avg) / std
        """
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        """
    return result
    
def pull(df, pull_column = ''):
    df2 = df.copy()
    df2 = df2.drop(pull_column, 1)
    return [df2, df[pull_column]]
"""
df = pd.read_csv('../input/HR_comma_sep.csv');
df = normalize(df, ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company'])
df = pd.get_dummies(df, columns=['sales', 'salary'])
df = df.sample(frac=1).reset_index(drop=True)
X, Y = pull(df, 'left')


X = np.array(X.values)
Y = np.array(Y.values)

Xtrain = X[:7500]
Ytrain = Y[:7500]

N, D = Xtrain.shape

print(N, D)

ones = np.ones((N, 1))
Xtrain_b = np.concatenate((ones, Xtrain), axis=1)

w = np.random.randn(D + 1)
z = Xtrain_b.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

predict = sigmoid(z)

print(predict)

def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

learning_rate = 0.1
for i in range(100):
    if i % 10 == 0:
        print(cross_entropy(predict, Y))

    # gradient descent weight udpate with regularization
    # w += learning_rate * ( np.dot((T - Y).T, Xb) - 0.1*w ) # old
    w += learning_rate * ( Xtrain_b.T.dot(predict - Y) - 0.1*w )

    # recalculate Y
    predict = sigmoid(Xtrain_b.dot(w))


print("Final w:", w)

"""
"""


X = np.random.randn(N,D)

# center the first 50 points at (-2,-2)
X[:50,:] = X[:50,:] - 2*np.ones((50,D))

# center the last 50 points at (2, 2)
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

# labels: first 50 are 0, last 50 are 1
T = np.array([0]*50 + [1]*50)

# add a column of ones
# ones = np.array([[1]*N]).T
ones = np.ones((N, 1))
Xb = np.concatenate((ones, X), axis=1)

# randomly initialize the weights
w = np.random.randn(D + 1)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


Y = sigmoid(z)

# calculate the cross-entropy error
def cross_entropy(T, Y):
    E = 0
    for i in xrange(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


# let's do gradient descent 100 times
learning_rate = 0.1
for i in xrange(100):
    if i % 10 == 0:
        print cross_entropy(T, Y)

    # gradient descent weight udpate with regularization
    # w += learning_rate * ( np.dot((T - Y).T, Xb) - 0.1*w ) # old
    w += learning_rate * ( Xb.T.dot(T - Y) - 0.1*w )

    # recalculate Y
    Y = sigmoid(Xb.dot(w))


print "Final w:", w
"""



