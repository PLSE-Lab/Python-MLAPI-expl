import pandas as pd
import numpy as np

import scipy.optimize as op

from numpy import loadtxt, where, zeros, e, array, log, ones, append, linspace

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

num_labels = 10
lambd = 0.1

y_t = train['label']
X_t = train.drop("label",axis=1)

y = y_t.values

print(y)

X_temp = np.multiply(X_t.values, 1.0/255.0)
Xtest_temp = np.multiply(test.values, 1.0/255.0)

m, n = X_temp.shape
m_test, n_test = Xtest_temp.shape

X=np.ones((m,n+1))
Xtest=np.ones((m_test,n_test+1))

X[:,-n:] = X_temp
Xtest[:,-n_test:] = Xtest_temp

y.shape = (m, 1)
ytest = np.zeros(shape=(m_test,1))

def sigmoid(z):
# Computes the Sigmoid function of z
    g = 1.0 / (1.0 + e ** (-1.0 * z))

    return g 
    
def compute_cost(theta,X,y,lambd): 
# Computes cost function
    theta=theta.reshape(len(theta),1)
    h = sigmoid(X.dot(theta))
    m = len(y)
    h = h.reshape(len(h),1)
    y = y.reshape(len(y),1)
    theta[0] = 0
    
    J = (1.0/m)*np.sum(-y*(np.log(h)) - (1-y)*(np.log(1-h))) + (lambd/(2*m))*np.sum(theta.T*theta.T)

    return  J
    
def compute_grad(theta, X, y, lambd):
# Computes gradient of cost function 
    theta = theta.reshape(len(theta),1)
    h = sigmoid(X.dot(theta))
    h = h.reshape(len(h),1)
    y = y.reshape(len(y),1)
    m = len(y)
    theta[0] = 0
    
    grad = (1.0/m)*(X.T.dot((h-y)).T) + (lambd/m)*theta.T
    grad = np.ndarray.flatten(grad)

    return grad

def prediction(theta, X):
# Predicts the number [0:9] based on learned logistic regression data
    a, b = X.shape
    pred = np.zeros(shape=(a, 1))
    h = sigmoid(X.dot(theta.T))
    
    pred=np.argmax(h, axis=1)  
    
    return pred

def oneVsAll(X, y, num_labels, lambd):
    
    m, n = X.shape
    all_theta = np.zeros(shape=(num_labels, n))
    initial_theta = np.zeros(shape=(n, 1))
    y_t = np.ones(shape=(len(y),num_labels))
    y = y.reshape(len(y))
    
    for i in range(0, num_labels):
        y_t[:,i] = y
        y_t[y != i, i] = 0
        y_t[y == i, i] = 1
    
    for i in range(0, num_labels):
        #optimal_theta = op.fmin_cg(f = compute_cost, x0 = initial_theta, fprime = compute_grad, args = (X, y_t[:,i], lambd), maxiter=300)
        Result = op.minimize(fun = compute_cost, x0 = initial_theta, args = (X, y_t[:,i],lambd), jac = compute_grad, method = 'TNC')
        optimal_theta = Result.x;
        all_theta[i,:] = optimal_theta
    return all_theta
    
all_theta = oneVsAll(X, y, num_labels, lambd)
p=prediction(all_theta[:,:],X)

t = (p == y.T)

print ("Train Accuracy is: ", np.mean(t)*100)

p2=prediction(all_theta[:,:],Xtest)

#print (p2.shape, range(1,m_test).shape)

np.savetxt('sample_submission.csv', np.c_[range(1,m_test+1),p2.reshape(len(p2),1)], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')





