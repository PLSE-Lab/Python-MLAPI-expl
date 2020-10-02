import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import minimize
from toolz import *

##### load and cleanse data #######################################################################

#Print you can execute arbitrary python code
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

full_data = [train, test]

#Print to standard output, and see the results in the "log" section below after running your script
#print("\n\nTop of the training data:")
#print(train.head())

#print("\n\nSummary statistics of training data")
#print(train.describe())

# Convert the male and female groups to integer form
train.loc[(train["Sex"] == "male"),"Sex"] = 0
train.loc[(train["Sex"] == "female"), "Sex"] = 1

test.loc[(test["Sex"] == "male"),"Sex"] = 0
test.loc[(test["Sex"] == "female"), "Sex"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# Convert the Embarked classes to integer form
train.loc[(train["Embarked"] == "S"),"Embarked"] = 0
train.loc[(train["Embarked"] == "C"),"Embarked"] = 1
train.loc[(train["Embarked"] == "Q"),"Embarked"] = 2

test.loc[(test["Embarked"] == "S"),"Embarked"] = 0
test.loc[(test["Embarked"] == "C"),"Embarked"] = 1
test.loc[(test["Embarked"] == "Q"),"Embarked"] = 2

# Impute the missing value with the median
test.loc[152,"Fare"] = test["Fare"].median()

# norm data
train.loc[:,"Age"] /= 10.0
test.loc[:,"Age"] /= 10.0
train.loc[:,"Fare"] /= 10.0
test.loc[:,"Fare"] /= 10.0

# new features
train["Title"] = 0
test["Title"] = 0

for r in range(len(train)):
    if("Capt." in train.loc[r, "Name"]): train.loc[r,"Title"] = 2
    if("Rev." in train.loc[r, "Name"]): train.loc[r,"Title"] = 1    
    if("Col." in train.loc[r, "Name"]): train.loc[r,"Title"] = 1  
    if("Dr." in train.loc[r, "Name"]): train.loc[r,"Title"] = 2  
    if("Lady" in train.loc[r, "Name"]): train.loc[r,"Title"] = 1  
    if("Major." in train.loc[r, "Name"]): train.loc[r,"Title"] = 1  
    if("Sir." in train.loc[r, "Name"]): train.loc[r,"Title"] = 1      
    if("Countess." in train.loc[r, "Name"]): train.loc[r,"Title"] = 1 
    if("Col." in train.loc[r, "Name"]): train.loc[r,"Title"] = 1 
    if("Don." in train.loc[r, "Name"]): train.loc[r,"Title"] = 1 
    if("Dona." in train.loc[r, "Name"]): train.loc[r,"Title"] = 1 
    if("Jonkheer." in train.loc[r, "Name"]): train.loc[r,"Title"] = 1 

            
for r in range(len(test)):
    if("Capt." in test.loc[r, "Name"]): test.loc[r,"Title"] = 2
    if("Rev." in test.loc[r, "Name"]): test.loc[r,"Title"] = 1    
    if("Col." in test.loc[r, "Name"]): test.loc[r,"Title"] = 1  
    if("Dr." in test.loc[r, "Name"]): test.loc[r,"Title"] = 2  
    if("Lady" in test.loc[r, "Name"]): test.loc[r,"Title"] = 1  
    if("Major." in test.loc[r, "Name"]): test.loc[r,"Title"] = 1  
    if("Sir." in test.loc[r, "Name"]): test.loc[r,"Title"] = 1  
    if("Countess." in test.loc[r, "Name"]): test.loc[r,"Title"] = 1 
    if("Col." in test.loc[r, "Name"]): test.loc[r,"Title"] = 1 
    if("Don." in test.loc[r, "Name"]): test.loc[r,"Title"] = 1 
    if("Dona." in test.loc[r, "Name"]): test.loc[r,"Title"] = 1 
    if("Jonkheer." in test.loc[r, "Name"]): test.loc[r,"Title"] = 1 

train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

for dataset in full_data:
    dataset['FamSize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamSize'] == 1, 'IsAlone'] = 1
    
# Create the target and features numpy arrays: target, features_forest
y = train["Survived"].values
mt = int(.80 * len(y))
#mv = len(y) - mt
yt = np.concatenate((y[:(len(y)//2)], y[(len(y)//2)+mt:]),axis=0) 
yv = y[(len(y)//2):(len(y)//2)+mt]

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
X = (train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked", "FamSize", "Title", "IsAlone"]].values).astype(np.float64)
Xt = np.concatenate((X[:(len(y)//2)], X[(len(y)//2)+mt:]),axis=0)      # X[:mt,:]
Xv = X[(len(y)//2):(len(y)//2)+mt]           # X[mt:,:]
TestX = (test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked", "FamSize", "Title", "IsAlone"]].values).astype(np.float64)

##### functions ###################################################################################

# define the sigmoid function
def sigmoid(x,deriv=False):
    if(deriv==True):
        return sigmoid(x)*(1.0-sigmoid(x))
    #print(x)
    return 1.0/(1.0+np.exp(-x.astype(float)))

def h_theta(t0, t1, x):
    m = len(x)
    temp_h0 = sigmoid(np.dot(t0, x.T))
    temp_h0plus = np.concatenate((np.ones((1,m), dtype=np.float), temp_h0), axis = 0)
    temp_h1 = sigmoid(np.dot(t1, temp_h0plus))
    return (temp_h1.T)

def init_theta(s0,s1,s2):
    # randomly initialize our weights with mean 0 for weights  
    epsilon_init = 0.12
    np.random.seed(1)
    theta_0 = 2*epsilon_init*np.random.random((s1,s0+1)) - epsilon_init  # 8 x 8 - seven inputs
    theta_1 = 2*epsilon_init*np.random.random((s2,s1+1)) - epsilon_init  # 1 x 9 - one output, 8 hidden
    return flatten(theta_0,theta_1)

def fold(t_flat,s0,s1,s2,n=0):
    if(n==1):
        t1 = t_flat[((s0+1)*s1):]
        t1.shape = (s2,(s1+1))
        return t1
    t0 = t_flat[:((s0+1)*s1)]
    t0.shape = (s1,(s0+1))
    return t0
        
def flatten(t0, t1):
    t0.shape = (t0.size)
    t1.shape = (t1.size)
    return (np.hstack((t0, t1)))
    
def nnCost(theta, X, y, bias_lambda, s0, s1, s2):
    J = 0.0
    m = len(X)
    theta_0 = fold(theta,s0,s1,s2,0)                    # 8 x 8
    theta_1 = fold(theta,s0,s1,s2,1)                    # 1 x 9
    H = h_theta(theta_0, theta_1, X)
    one_H = np.ones(H.shape)
    one_y = np.ones(y.shape)
    
    J += (1.0/m) * sum(np.dot(-y.T,np.log(H)) - np.dot((one_y-y).T,np.log(one_H-H)))
    J += (bias_lambda/(2.0*m)) * (np.sum((theta_0*theta_0)[:,1:]) + np.sum((theta_1*theta_1)[:,1:]))

    a_0 = np.zeros((s0,1), dtype=np.float)              # 7 x 1
    a_1 = np.zeros((s1,1), dtype=np.float)              # 8 x 1
    a_2 = np.zeros((s2,1), dtype=np.float)              # 1 x 1
    delta_1 = np.zeros((s1+1,1), dtype=np.float)        # 9 x 1
    delta_1_trimmed = np.zeros((s1,1), dtype=np.float)  # 8 x 1
    delta_2 = np.zeros((s2,1), dtype=np.float)          # 1 x 1
    theta_0 = fold(theta,s0,s1,s2,0)                    # 8 x 8
    theta_1 = fold(theta,s0,s1,s2,1)                    # 1 x 9
    D_0 = np.zeros(theta_0.shape, dtype=np.float)       # 8 x 8
    D_1 = np.zeros(theta_1.shape, dtype=np.float)       # 1 x 9
    
    # training the model
    for t in range(m): 
        
        # Feed forward through layers 0, 1, and 2
        
        a_0 = (X[t,:].T).astype(float)                  # 8 x 1
        a_0.shape = (s0+1,1)
        a_1 = sigmoid(np.dot(theta_0,a_0))
        a_1 = np.vstack((1.0, a_1))                     # 9 x 1
        a_2 = sigmoid(np.dot(theta_1,a_1))              # 1 x 1
        
        # Back propogate
        
        delta_2 = a_2 - y[t].T                          # 1 x 1
        delta_2.shape = (s2,1)

        delta_1 = sigmoid( np.vstack((1.0, np.dot(theta_0, a_0) )), True)
        delta_1 = delta_1 * np.dot(theta_1.T,delta_2)
        delta_1.shape = (s1+1,1)                        # 9 x 1
        delta_1_trimmed = delta_1[1:,:]                 # 8 x 1

        D_1 += np.dot(delta_2,a_1.T)                    # 1 x 9
        D_0 += np.dot(delta_1_trimmed,a_0.T)            # 8 x 8
        
    theta_0_clear_bias = theta_0.copy()
    theta_0_clear_bias[:,0] = 0.0
    theta_1_clear_bias = theta_1.copy()
    theta_1_clear_bias[:,0] = 0.0
    
    theta_0_grad = (1.0/m) * D_0 + (bias_lambda/m) * theta_0_clear_bias
    theta_1_grad = (1.0/m) * D_1 + (bias_lambda/m) * theta_1_clear_bias

    return(J, flatten(theta_0_grad,theta_1_grad))


##### variables ##################################################################################

learning_rate = .2
bias_lambda = 2
maxiter = 20
totaliter = 5000
s0 = len(Xt.T)  # 7
s1 = 8
s2 = 1
theta = init_theta(s0,s1,s2)
theta_grad = np.zeros(theta.shape, dtype=np.float)

# add bias column to X and TestX
Xt = np.concatenate((np.ones((len(Xt), 1)), Xt), axis = 1)
Xv = np.concatenate((np.ones((len(Xv), 1)), Xv), axis = 1)
TestX = np.concatenate((np.ones((len(TestX), 1)), TestX), axis = 1)

##### loop regression #############################################################################

result = nnCost(theta, Xt, yt, bias_lambda, s0, s1, s2)
validate = nnCost(theta, Xv, yv, bias_lambda, s0, s1, s2)
print("Iter: %i, lda: %.1f, lrn: %.1f" % (totaliter, bias_lambda, learning_rate))
print("")
print("Iter 0: %f, %f" % (result[0], validate[0]))
for l in range(totaliter):
    theta -= learning_rate * result[1]     #/(np.exp(l/1000))
    result = nnCost(theta, Xt, yt, bias_lambda, s0, s1, s2)
    validate = nnCost(theta, Xv, yv, bias_lambda, s0, s1, s2) 
    if((l+1)%200 == 0): print("Iter %d: %f, %f" % (l+1, result[0], validate[0]))
    if(l > (totaliter - 20)): print("Iter %d: %f, %f" % (l+1, result[0], validate[0]))

##### cr4oss validation ###########################################################################

cv_cost, _ = nnCost(theta, Xv, yv, bias_lambda, s0, s1, s2) 
print("Train cost = %f" % (result[0]))
print("CV cost = %f" % (cv_cost))

##### minimize function ###########################################################################

#myargs = (X, y, bias_lambda, s0, s1, s2)
#min_result = minimize(nnCost, x0=theta, args=myargs, options={'disp':True, 'maxiter':maxiter}, method="L-BFGS-B", jac=True)
#thetaM = min_result["x"]
#J = min_result[0]
#J, _ = nnCost(thetaM, X, y, bias_lambda, s0, s1, s2)
#print(min_result["fun"])

##### predict test data ##########################################################################

Prediction = h_theta(fold(theta,s0,s1,s2,0),fold(theta,s0,s1,s2,1),TestX)
PredictionB = (Prediction >= 0.5).astype(int)

##### prepare file ###############################################################################

submit_file = pd.DataFrame(test, columns=["PassengerId","Survived"])
submit_file["Survived"] = PredictionB
submit_file.to_csv("pred_nn_lam0_1_iter800_no_title.csv", index = False)





