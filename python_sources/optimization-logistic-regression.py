import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import time


#Method to find the sigmoid of the function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Function to add an Interncept
def addintercept(X):
    intercept = np.ones((X.shape[0], 1))

    return np.concatenate((intercept, X), axis=1)

#Function to find the loss function
def loss(h, y):
    return np.sum((y * np.log(h) + (1 - y) * np.log(1 - h)))

#Function to find the Hessian Matrix
def hessian(X,h):
    return  np.dot(X.T,(np.diag(h*(1-h)).dot(X)))


def fit_hessian(X,y):
    w = np.zeros(X.shape[1])
    lr = 0.01
    lossvalue=0
    for i in range(1000):

        z = np.dot(X, w)
        h = sigmoid(z)
        gradient = np.dot(X.T, (y-h)) #Get the gradient
        hessian_value=np.linalg.inv(hessian(X,h))#Inverse the hessian
        w+=hessian_value.dot(gradient) #Update Weights
        previousloss = lossvalue
        lossvalue=loss(h,y) #Update loss function value



        if lossvalue-previousloss==0: #CHeck if Converged
            print(i)
            break;

    return w

def fit(X,y):

    theta = np.zeros(X.shape[1])
    lr = 0.01
    lossvalue=0;
    for i in range(1000):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (y-h)) #Get the gradient
        previousloss = lossvalue  #Save old Loss function value
        lossvalue=loss(h,y) #Update the loss function value
        theta += lr * gradient  #Update the weigths
        if lossvalue-previousloss==0: #CHeck if Converged
            print(i)
            break;


    return theta



def plot_gradient(X,y,theta):
    x_class0 = []
    x_class1 = []
    for i in range(len(y)):
        # Seperating Classes
        if (y[i] == 0):
            x_class0.append(X[i, :])
        else:
            x_class1.append(X[i, :])

    x_class0 = np.array(x_class0)
    x_class1 = np.array(x_class1)

    plt.scatter(x_class0[:, 1], x_class0[:, 2], c='b', label='y = 0')
    plt.scatter(x_class1[:, 1], x_class1[:, 2], c='r', label='y = 1')
    x1 = np.linspace(0, 8, 4)

    x2 = -(theta[0] + theta[1] * x1) / theta[2]
    plt.plot(x1, x2, c='k', label='reg line')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

def plot_hessian(X,y,w):
    x_class0 = []
    x_class1 = []
    for i in range(len(y)):
        # Seperating Classes
        if (y[i] == 0):
            x_class0.append(X[i, :])
        else:
            x_class1.append(X[i, :])

    x_class0 = np.array(x_class0)
    x_class1 = np.array(x_class1)

    plt.scatter(x_class0[:, 1], x_class0[:, 2], c='b', label='y = 0')
    plt.scatter(x_class1[:, 1], x_class1[:, 2], c='r', label='y = 1')
    x1 = np.linspace(0, 8, 4)

    x2 = -(w[0] + w[1] * x1) / w[2]
    plt.plot(x1, x2, c='k', label='reg line')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #Load the data into X and y
   X= np.genfromtxt("../input/q1x.dat")
   y=np.genfromtxt("../input/q1y.dat")


    #Add an Interncept in X
   X=addintercept(X)

    #Get weights using Gradient method
   start_time = time.time()
   theta=fit(X,y)
   print("--- %s seconds ---" % (time.time() - start_time))


    #Get weights using Newton
   start_time2 = time.time()
   w=fit_hessian(X,y)
   print("--- %s seconds ---" % (time.time() - start_time2))


    #Plot for Gradient Desecent method
   plot_gradient(X,y,theta)
    #Plot for Newton Method
   plot_hessian(X,y,w)
