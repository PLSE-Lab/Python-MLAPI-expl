# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import scipy.optimize as opt
import math

lambda_value = 0

#Function to preprocess the data
def preprocessTrainingData(unprocessedData):
    X = np.array(unprocessedData.drop(['PassengerId','Name','Ticket','Cabin','Survived'],axis=1))
    y = unprocessedData.loc[:,'Survived'].values
    y = y.reshape(y.shape[0],1)

    #define male as 1 and female as 0
    i=0
    for values in X[:,1]:
        if values.upper() == 'MALE':
            X[i,1] = 1
        else:
            X[i,1] = 0
        i = i+1
        
    #define embarkment values : C as 3,Q as 17 and S as 19, this is just the alphabet position
    i=0
    for values in X[:,6]:
        if values == 'C':
            X[i,6] = 1
        elif values == 'Q':
            X[i,6] = 2
        elif values == 'S':
            X[i,6] = 3
        else:
            X[i,6] = 0
            
        i = i+1

    #Fill up the cells without any value(or nan) with the mean value of that column for all the cells
    mean = np.nanmean(X,axis=0)
    for i in range(X.shape[1]):
        for j in range(X.shape[0]):
            if(np.isnan(X[j,i])):
                X[j,i] = mean[i]

    X = np.array(X,dtype=float)
    return X,y

#Feature normalization
def feature_normalize(X):
    X_norm = X
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    for i in range(X_norm.shape[1]-1):
        X_norm[:,i] = (X_norm[:,i]-mu[i])/sigma[i]

    return X_norm,mu,sigma

#Function to initialize the theta values randomly
def randInitialization(l_in,l_out):
    epsilon_init = (math.sqrt(6))/(math.sqrt(l_in+l_out))
    Theta = np.zeros([l_out,l_in+1])
    Theta = np.random.rand(l_out,l_in+1)*(2*epsilon_init)-epsilon_init;
    return Theta

#Function to compute the cost
def computeCost(nn_parameters,X,y,input_layer_size,hidden_layer_size,num_labels,lambda_value):

    Theta1 = nn_parameters[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_parameters[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,hidden_layer_size+1)
    
    m = X.shape[0]
    J = 0

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    layer1Activation = np.c_[np.ones(X.shape[0]),X]
    z2 = layer1Activation@Theta1.transpose()
    layer2Activation = sigmoid(z2)
    layer2Activation = np.c_[np.ones(m),layer2Activation]
    z3 = layer2Activation@Theta2.transpose()
    hypothesis = sigmoid(z3)

    squaredSumTheta = np.sum(np.square(Theta1[:,1:]))+np.sum(np.square(Theta2[:,1:]))
    
    regularizationTerm = (lambda_value/(2*m))*(squaredSumTheta)
    
    J = 0
    J = np.sum((y*np.log(hypothesis))+((1-y)*(np.log(1-hypothesis))))
    J = -(J/m) + regularizationTerm

    return J

#Function to compute the gradient
def computeGradient(nn_parameters,X,y,input_layer_size,hidden_layer_size,num_labels,lambda_value):

    Theta1 = nn_parameters[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_parameters[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,hidden_layer_size+1)

    m = X.shape[0]
    J = 0

    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    layer1Activation = np.c_[np.ones(X.shape[0]),X]
    z2 = layer1Activation@Theta1.transpose()
    layer2Activation = sigmoid(z2)
    layer2Activation = np.c_[np.ones(m),layer2Activation]
    z3 = layer2Activation@Theta2.transpose()
    hypothesis = sigmoid(z3)

    delta3 = hypothesis - y

    gprimez2 = np.c_[np.ones(z2.shape[0]),sigmoidGradient(z2)]
    gprimez2 = sigmoidGradient(z2)

    delta2 = (delta3@Theta2[:,1:])*gprimez2

    gradient1 = delta2.transpose()@layer1Activation

    gradient2 = delta3.transpose()@layer2Activation

    Theta1_gradient = (1/m)*gradient1
    Theta2_gradient = (1/m)*gradient2

    regularizationTermForGradient1 = (lambda_value/m)*(np.sum(Theta1_gradient[:,1:]))

    regularizationTermForGradient2 = (lambda_value/m)*(np.sum(Theta2_gradient[:,1:]))

    Theta1_gradient = Theta1_gradient + regularizationTermForGradient1
    Theta2_gradient = Theta2_gradient + regularizationTermForGradient2

    Theta1_gradient = Theta1_gradient.reshape(Theta1_gradient.shape[0]*Theta1_gradient.shape[1],1)
    Theta2_gradient = Theta2_gradient.reshape(Theta2_gradient.shape[0]*Theta2_gradient.shape[1],1)

    gradient = np.r_[Theta1_gradient,Theta2_gradient]
    gradient = gradient.reshape(gradient.shape[0],1)

    return gradient.flatten()

#Function to calculate sigmoid
def sigmoid(z):
    g=(1/(1+np.exp(-z)))
    return g

#Function to calculate sigmoid gradient
def sigmoidGradient(z):
    g = np.zeros(z.shape)
    g = sigmoid(z)*(1-sigmoid(z))
    return g

#function to predict
def predict(X,Theta1,Theta2):
    X = np.c_[np.ones(X.shape[0]),X]
    predictions = np.zeros((X.shape[0],1))
    hiddenLayerOutput = sigmoid(X@Theta1.transpose())
    hiddenLayerOutput = np.c_[np.ones(hiddenLayerOutput.shape[0]),hiddenLayerOutput]
    hypothesis = sigmoid(hiddenLayerOutput@Theta2.transpose())
    predictions[np.where(hypothesis >= 0.5)] = 1
    return predictions

#Function to calculate Accuracy
def calculateAccuracy(predictedValues,actualValues):
    m = len(actualValues)
    accuracy = 0
    True_positive = 0
    True_negative = 0
    for i in range(m):
        if (predictedValues[i] == 0 and actualValues[i] == 0):
            True_negative+= 1
        elif(predictedValues[i] == 1 and actualValues[i] == 1):
            True_positive+= 1
    accuracy = (True_positive + True_negative)*(100/m)
    return accuracy


#Main program beggining 
if (__name__ == '__main__'):
    
    unprocessedData = pd.read_csv('../input/train.csv',index_col=False)
    X,y = preprocessTrainingData(unprocessedData)
        
    X = np.ceil(X).astype('int')

    X,mu,sigma = feature_normalize(X)

    input_layer_size = X.shape[1]
    hidden_layer_size = 5
    num_labels = 1
    
    initial_Theta1 = randInitialization(input_layer_size,hidden_layer_size)
    initial_Theta2 = randInitialization(hidden_layer_size,num_labels)

    newinitial_Theta1 = initial_Theta1.reshape(initial_Theta1.shape[0]*initial_Theta1.shape[1],1)
    newinitial_Theta2 = initial_Theta2.reshape(initial_Theta2.shape[0]*initial_Theta2.shape[1],1)
    newneuralNetworkParams = np.r_[newinitial_Theta1,newinitial_Theta2]
    
    all_theta = opt.fmin_cg(computeCost,fprime = computeGradient,x0 = newneuralNetworkParams,args=(X,y,input_layer_size,hidden_layer_size,num_labels,lambda_value))

    new_Theta1 = all_theta[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    new_Theta2 = all_theta[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,hidden_layer_size+1)


    predictions = predict(X,new_Theta1,new_Theta2)
    accuracyPercent = calculateAccuracy(predictions,y)
    print("\nAccuracy of the model on test set is: ", accuracyPercent,"%")
    print("Theta values found after running fmin_cg are: \n","Theta1 = ",new_Theta1,"\nTheta2 = ",new_Theta2)
    