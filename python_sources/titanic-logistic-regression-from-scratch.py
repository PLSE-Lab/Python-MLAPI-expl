import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def normalize(X):
    X_Norm=((X-X.min(0))/(X.max(0)-X.min(0)))
    return X_Norm

def initialize_with_zeros(dim):
    w = np.zeros((dim,1)) 
    b=0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[0]
    A = sigmoid(np.dot(X,w)+b)
    cost = -1/m*((np.dot(Y.T,np.log(A))+(np.dot((1-Y).T,np.log(1-A)))))
    dw = 1/m*(np.dot(X.T,((A-Y))))
    db = 1/m*(np.sum(np.sum(A-Y,axis=0)))
    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X_train, Y_train)
        dw = grads["dw"]
        db = grads["db"]
        w = w-(learning_rate*dw)
        b = b-(learning_rate*db)
        if i % 100 == 0: 
            costs.append(cost)
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads

def predict(w, b, X):
    m = X.shape[0]
    Y_prediction = np.zeros((m,1))
    A = sigmoid(np.dot(X,w)+b)
    
    for i in range(A.shape[0]):
        if A[i,0]>0.5:
            Y_prediction[i,0]=1 
        else:
            Y_prediction[i,0]=0
    assert(Y_prediction.shape == (m,1))
    return Y_prediction

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df['Sex Category'] = np.where(train_df['Sex']=="female", 1, 0)
test_df['Sex Category'] = np.where(test_df['Sex']=="female", 1, 0)

Y_train=train_df.drop(['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Sex Category'],axis=1)
X_train=train_df.drop(['PassengerId','Survived','Ticket','Fare','Name','Age','Cabin','Embarked','Sex'],axis=1)
X_test = test_df.drop(['PassengerId','Name','Sex','Age','Ticket','Fare','Cabin','Embarked'],axis=1)

X_train=normalize(X_train)
X_test=normalize(X_test)

w, b = initialize_with_zeros(X_train.shape[1])
params, grads = optimize(w,b,X_train,Y_train,num_iterations=200000,learning_rate=0.005)

Y_predict = predict(params['w'], params['b'], X_test)
Y_predict=Y_predict.astype(int)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": np.reshape(Y_predict,test_df["PassengerId"].shape)
    })
submission.to_csv('submission.csv', index=False)