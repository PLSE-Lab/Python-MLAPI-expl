# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv') # Load Training data
Test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv') # Load Test data
print(data.shape)

def reshape(data): # Reshape data if needed
    return np.reshape(np.array(data),(-1,1))

def classify(num_of_class,_class=0): # Classify labels these function going to be applied in Pandas DataFrame.
    return int(num_of_class==_class)

def isTrue(Y): # Final Prediction of sigmoid function for each label
    if Y.any()>0.5:
        return 1
    else:
        return 0


def classify_one_vs_all(Y,n=0): # apply classification for each label and return it as Pandas DataFrame.
    _Y = pd.DataFrame()
    for i in range(0,10):
        _Y['class {}'.format(i)] = (Y.apply(classify,args=[i])).values
    return _Y

def initialization(data): # Initialize training set for One-vs-All Classification (theta is a weight ).
    m = data.shape[0]
    X_Train = data.loc[:,'pixel0':]
    X_Train.insert(0, 'X0', np.ones(X_Train.shape[0]))

    Y_Train = data.loc[:,'label']
    Y_Train = classify_one_vs_all(Y_Train)


    theta = (np.zeros((X_Train.shape[1],1)))

    return X_Train,Y_Train,theta


def sigmoid(z): # sigmoid function return output as probability terms from 0 to 1 (0<=sigmoid(z)<=1)
    return 1.0/(1+(np.exp(-z)))

def hypothesis(X,theta): # Return hypothesis for input variable and theta each iteration of gradient descent
    return np.dot(X,theta)

def forward_propagation(X,y,h,lamda,m,theta): #  Return Forward Propagation (cost) each iteration
    l2 = lamda*sum(np.dot(theta.T,theta)) # L2 Regularization Parameter
    s1 = np.dot(y.T,np.log(h)) # if y = 1 
    s2 = np.dot((1-y).T,np.log(1-h)) # if y = 0
    return (-1/m)*sum(s1 + s2 + l2)

def backward_propagation (X,Y,h,m): # Return Backward Propagation each iteration  
    # (b = theta[0] is bias ,w = theta[1:] all data except first element is weight). for updating parameters dw and db
    dw_b = (1/m)*np.dot(X.T,(h-Y))
    dw  = dw_b[1:]
    db  = dw_b[0]
    return dw,db

def propagate(theta,X,Y,lamda): # Return the gradient and cost each iteration 
    
    m = X.shape[0] # length of input variable (number of rows)
    
    h = sigmoid(hypothesis(X,theta)) # sigmoid of hypothesis

    cost = forward_propagation(X,Y,h,lamda,m,theta) # cost 
    
    dw,db = backward_propagation(X,Y,h,m) #weight error & bias error
    
    
    gradient = {"dw": dw,"db": db}
    
    return gradient, cost 

def optimize(theta,X, Y,num_iterations,learning_rate,sv_cost = False,cost_itr=20): # Apply gradient descent and return Pandas DataFrame of predictions
 
    costs = []
    for i in range(num_iterations):
        gradient, cost =  propagate(theta, X, Y,lamda=10000)
        
        dw = gradient["dw"] # weight error
        db = gradient["db"] # bias error
        
        w = theta[1:]
        b = theta[0]
        
        w = w - learning_rate*dw # Update weight 
        b = b - learning_rate*db # Update bias
        
        
        costs.append(float(cost))
        
        if sv_cost and i % cost_itr == 0 :
            print ("iteration %i/{} : loss  %.3f".format(num_iterations) %(i,cost))
        
        theta[1:] = w
        theta[0]  = b
    
    print('\n\n--------------------------\n\n')
    
    new_theta = np.concatenate((b, w), axis=None)
    
    gradient = {"dw": dw,
             "db": db}
    
    return new_theta, gradient, costs

def predict(theta,X):
    m = X.shape[1]
    
    h = sigmoid(np.dot(X,theta))

    
    Y_prediction = pd.DataFrame({'Y_Pred':h.squeeze().tolist()})
    Y_prediction.loc[Y_prediction['Y_Pred'] > 0.5, 'Y_Pred'] = 1
    Y_prediction.loc[Y_prediction['Y_Pred'] <= 0.5, 'Y_Pred'] = 0   
    return Y_prediction


def model(data,lamda=10000,learning_rate=0.0001,num_of_iter=100,_class=1,cost_itr=20):
    

    X_Train,Y_Train,theta = initialization(data) # call initialization function

    Y_Train = reshape(Y_Train['class {}'.format(_class)])
      
    new_theta, gradient, costs = optimize(theta,X_Train,Y_Train,
                                        num_iterations= num_of_iter, 
                                        learning_rate = learning_rate,
                                        sv_cost = True,cost_itr=cost_itr) # call optimize function apply gradient descent
    w = new_theta[1:]
    b = new_theta[0]
    
    
    Y_prediction_train = predict(theta,X_Train)
    
    print('--------------------------------------\n\n')
    model_data = {"costs": costs,
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_of_iter,
                 "X":X_Train,
                 "Y":Y_Train,
                 "theta":new_theta,
                 "gradient":gradient}

    return model_data

def one_vs_all_predict(X,theta):
    temp = sigmoid(np.dot(X,theta))
    return temp

def Test_data_prediction(trained_model_data,X): # Return Prediction of given set (Test set) as Pandas DataFrame
    predection_coff = pd.DataFrame()
    
    Test_dt = X.copy()
    Test_dt.insert(loc=0,column='X0',value=np.ones(X.shape[0]))

    for i in range(10):
        temp = trained_model_data['class {}'.format(i)]['theta']
    
        predection_coff.insert(loc=i,column='class {}'.format(i)
        ,value=temp)

    Test_Prediction = one_vs_all_predict(Test_dt,predection_coff)
    Test_Prediction = Test_Prediction.argmax(axis=1)
    
    Test_Prediction = pd.DataFrame({'ImageId':np.arange(1,Test_Prediction.shape[0]+1),
                                   'Label':Test_Prediction})
    return Test_Prediction

def train_all(_class=9): # Train each class from 0 to 9 
    assert(_class>=0 and _class<=9)
    model_classes = {}
    for i in range(_class+1):
        print('The {}/{}th logistic classifier training...\n'.format(i,_class))
        print('-------------------------------\n\n')
        m = model(data,10000,0.00001,100,i,10)
        model_classes['class {}'.format(i)] = m
    return model_classes

trained_model_data = train_all(9) # Train all classes from 0 to 9


_Predicted = Test_data_prediction(trained_model_data,Test_data)
_Predicted.head()

_Predicted.to_csv('./submission.csv',index=False,encoding='utf8')

submission = pd.read_csv('submission.csv')
submission.head()