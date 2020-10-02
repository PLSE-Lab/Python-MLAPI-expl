# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# My code :

###########################################################################################################################################################3

# Reading input from train.csv file

df = pd.read_csv('../input/train.csv')

# Gain valuable insight from the data 

df.info()
df.head()
for c in df.columns:
    print(c, len(df[c].unique()), sum(df[c].isna())/len(df[c])*100,"%")

### From the data we can see that a few Embarked and Age data is missing which will be required. Since 77% of Cabin data is missing we will ignore it for the time being. We will also ignore 
### Name, PassengerID and Ticket because they have high cardinality and are mostly unique. 

X = pd.DataFrame(df.loc[:,['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Fare']])
y = df['Survived']

# Feature Processing

X.Sex.replace('male', 0, inplace=True)
X.Sex.replace('female', 1, inplace=True)
X.Embarked.replace('C', 1, inplace=True)
X.Embarked.replace('S', 2, inplace=True)
X.Embarked.replace('Q', 3, inplace=True)
X['Embarked'].describe()
X['Embarked'].fillna(2, inplace=True)
X['Age'].describe()
X['Age'].fillna(30, inplace=True)

# Data Visualization 

# Converting into NumPy Feature Matrix
m,n = X.shape
X = X.values
y = y.values

# Feature Normalization

def norm(X, axis=1):
    m,n = X.shape
    if(axis):
        l=n
    else:
        l=m
        X=X.T
    for i in range(l):
        mu = np.mean(X[:,i])
        sigma = np.std(X[:,i])
        if(sigma!=0):
            X[:,i] = (X[:,i]-mu)/sigma
        else:
            X[:,i] = (X[:,i]-mu)/(max(X[:,i]) - min(X[:,i]))
    if(axis == 0):
        X=X.T
    return X
    
# Creating separate training and validation set

mt = round(m*0.7)
mv = m-mt
Xv = X[mt:m, : ] 
X = X[0:mt, : ]
yv = y[mt:m]
y = y[0:mt]

# Adding Ones and Normalizing

X = norm(X)
X = np.insert(X, 0, 1, axis=1)
Xv = norm(Xv)
Xv = np.insert(Xv, 0, 1, axis=1)

# Neural Network Architecture 

no_hid_layers = 1
hid = 3
no_out = 1

# Xavier Ininitialization of weights w

w1 = np.random.randn(hid, n+1)*np.sqrt(2/(hid+n+1))
w2 = np.random.randn(no_out, hid+1)*np.sqrt(2/(no_out+hid+1))

# Sigmoid Activation Function
def g(x):
    sig = 1/(1+np.exp(-x))
    return sig
    
# Forward Propagation

def frwrd_prop(X, w1, w2):
    z2 = w1 @ X.T
    z2 = norm(z2, axis=0)
    a2 = np.insert(g(z2), 0, 1, axis=0)
    h = g(w2@a2)
    return (h,a2)
    
# Calculating Cost and Gradient

def Cost(X, y, w1, w2, lmbda=0):
    # Initializing Cost J and Gradients dw
    J = 0
    dw1 = np.zeros(w1.shape)
    dw2 = np.zeros(w2.shape)
    # Forward Propagation to calculate the value of the output
    h, a2 = frwrd_prop(X, w1, w2)
    #print(h)
    # Calculate the Cost Function J 
    J = -(np.sum(y.T*np.log(h) + (1-y).T*np.log(1-h)) - lmbda/2*(np.sum(np.sum(w1[:,1:].T@w1[:,1:])) + np.sum(w2[:,1:].T@w2[:,1:])))/mt
    # Applying Back Propagation to calculate the Gradients dw
    D3 = h-y
    D2 = (w2.T@D3)*a2*(1-a2)
    dw1[:,0] = (D2[1:]@X)[:,0]/m
    dw2[:,0] = (D3@a2.T)[:,0]/m
    dw1[:, 1:] = ((D2[1:]@X)[:,1:] + lmbda*w1[:,1:])/mt
    dw2[:, 1:] = ((D3@a2.T)[:,1:] + lmbda*w2[:,1:])/mt
    # Gradient clipping
    if(abs(np.linalg.norm(dw1))>1):
        dw1 = dw1*1/(np.linalg.norm(dw1))
    if(abs(np.linalg.norm(dw2))>1):
        dw1 = dw1*1/(np.linalg.norm(dw2))
    #print(dw1,"\n\n",dw2)
    return (J, dw1, dw2)
    
# Adam's Optimization technique for training w 

def Train(w1, w2, maxIter=500):
    # Algorithm
    a = 0.3
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    for i in range(maxIter):
        J, dw1, dw2 = Cost(X, y, w1, w2)
        plt.plot(i, J, 'r.')
        w1 = w1 - a*dw1
        w2 = w2 - a*dw2
        print("\t\t\tIteration : ", i+1, " \tCost : ", J)
    plt.show()
    return (w1, w2)

# Training Neural Network     

w1, w2 = Train(w1,w2)

# Calculating Training Accuracy

h, a2 = frwrd_prop(X, w1, w2)
Jtr = -np.sum(y.T*np.log(h) + (1-y).T*np.log(1-h))/mt
h = np.where(h>=0.5, 1, 0)
train_acc = np.sum(1 - np.absolute(h - y))*100/mt
print("Training Accuracy : ", train_acc,"% \tTraining Error : ", Jtr)

# Calculating Cross-Validation Accuracy

h,a2 = frwrd_prop(Xv, w1, w2)
Jcv = -np.sum(yv.T*np.log(h) + (1-yv).T*np.log(1-h))/mv
h = np.where(h>=0.5, 1, 0)
crossVal_acc = np.sum(1 - np.absolute(h-yv))*100/mv
print("Cross-Validation Accuracy : ", crossVal_acc,"%\tCross-Validation Error : ", Jcv)


# Reading the data from the test.csv file

df_test = pd.read_csv('../input/test.csv')

# Gain insight about Test data

df_test.info()
df_test.head()
for c in df_test.columns:
    print(c, len(df_test[c].unique()), sum(df_test[c].isna())/len(df_test[c])*100)
### From the data we can see that a few Fare and Age data is missing which will be required. Since 77% of Cabin data is missing we will ignore it for the time being. We will also ignore 
### Name, PassengerID and Ticket because they have high cardinality and are mostly unique. 
    


# Processing Test Data

test = pd.DataFrame(df_test.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Fare']])
test.Sex.replace('male', 0, inplace=True)
test.Sex.replace('female', 1, inplace=True)
test.Embarked.replace('C', 1, inplace=True)
test.Embarked.replace('S', 2, inplace=True)
test.Embarked.replace('Q', 3, inplace=True)
test.Age.describe()
test.Age.fillna(24, inplace=True)
test.Fare.describe()
test.Fare.fillna(35.542, inplace=True)

X_test = test.values
X_test = np.insert(X_test, 0, 1, axis=1)

# Evaluating the value of test output 

h,a2 = frwrd_prop(X_test, w1, w2)
h = np.where(h>=0.5, 1, 0)

# Convert into an output csv file
df_test['Survived'] = np.transpose(h)
test_out = df_test.loc[:,['PassengerId', 'Survived']]
test_out.to_csv('test_out2.csv', index=False)
