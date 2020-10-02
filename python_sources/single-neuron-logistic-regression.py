# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the csv file
raw_df = pd.read_csv("/kaggle/input/iris/Iris.csv")
new_df = raw_df.drop('Id', axis=1)
print(new_df.Species.unique())
print(new_df.head())

# Converting the species coloumn into the given three species
h = new_df['Species'].str.get_dummies("EOL")
new_df = new_df.merge(h, left_index=True, right_index=True)

print(new_df.head(),new_df.tail())

# Splitting the dataset in train and test sets

def setosa():# Classifier: Iris-setosa
    train_x, test_x, train_y, test_y = train_test_split(new_df.drop(['Species','Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],axis=1), new_df['Iris-setosa'], test_size=0.2, random_state=2)
    train_x = train_x.T
    test_x = test_x.T
    # To prevent dimensions of the form (m,)
    # instead we want of the form (m,n)
    train_y = pd.DataFrame(train_y).T
    test_y = pd.DataFrame(test_y).T
    # To obtain array
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    return train_x, test_x, train_y, test_y

def versicolor():# Classifier: Iris-versicolor
    train_x, test_x, train_y, test_y = train_test_split(new_df.drop(['Species','Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],axis=1), new_df['Iris-versicolor'], test_size=0.2, random_state=2)
    train_x = train_x.T
    test_x = test_x.T
    # To prevent dimensions of the form (m,)
    # instead we want of the form (m,n)
    train_y = pd.DataFrame(train_y).T
    test_y = pd.DataFrame(test_y).T
    # To obtain array
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    return train_x, test_x, train_y, test_y

def virginica():# Classifier: Iris-virginica
    train_x, test_x, train_y, test_y = train_test_split(new_df.drop(['Species','Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],axis=1), new_df['Iris-virginica'], test_size=0.2, random_state=2)
    train_x = train_x.T
    test_x = test_x.T
    # To prevent dimensions of the form (m,)
    # instead we want of the form (m,n)
    train_y = pd.DataFrame(train_y).T
    test_y = pd.DataFrame(test_y).T
    # To obtain array
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    return train_x, test_x, train_y, test_y

# Implementing single layer logistic regression
def initialize_parameters(df_dims):
    W = np.zeros((df_dims,1))
    b = 0
    
    assert(W.shape == (df_dims, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return W, b
def activation(z):
    a = 1/(1+np.exp(-z))
    return a
def propagate(W, b, X, Y):
    m = X.shape[1]
    
    # Forward propagation
    z = np.dot(W.T,X)+b
    a = activation(z)
    cost = (-1/m)*np.sum(Y*np.log(a)+(1-Y)*np.log(1-a))
    
    # Backward propagation
    dw = (1/m)*np.dot(X,(a-Y).T)
    db = (1/m)*np.sum(a-Y)
    
    grads = {"dw": dw,
             "db": db}
    
    
    assert(dw.shape == W.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return grads, cost

def optimize(W, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(W, b , X, Y)
        dw = grads['dw']
        db = grads['db']
        
        W = W - learning_rate*dw
        b = b - learning_rate*db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": W,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, cost

def predict(W, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    W = W.reshape(X.shape[0], 1)

    A = activation(np.dot(W.T,X)+b)
    
    for i in range(A.shape[0]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
    
        if (A[0,i] < 0.5):
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    assert(Y_prediction.shape == (1, m))
    return Y_prediction

def model(train_x, train_y, test_x, test_y, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    w, b = initialize_parameters(train_x.shape[0])
    parameters, grads, costs = optimize(w, b, train_x, train_y, num_iterations, learning_rate, print_cost = False)
    Y_pred_train = predict(w, b, train_x)
    Y_pred_test = predict(w, b, test_x)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - train_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test - test_y)) * 100))

    
    d = {"Y_prediction_test": Y_pred_test, 
         "Y_prediction_train" : Y_pred_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

iris_list = [setosa(), versicolor(), virginica()] # list of iris classifier functions
iris_list_names = [setosa, versicolor, virginica] # list of iris names

for flower_type in range(len(iris_list)):
    print('\nIris-',iris_list_names[flower_type])
    train_x, test_x, train_y, test_y = iris_list[flower_type]
    d = model(train_x, train_y, test_x, test_y,  learning_rate = 0.005, print_cost = True)

