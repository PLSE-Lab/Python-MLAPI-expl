# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import pandas as pd
import numpy as np
from numba import jit
# Input data files are available in the "../input/" directory.


# For NN calculations and visualizaing progress:
import matplotlib.pyplot as plt
from shutil import copyfile
from datetime import datetime


#400 28 10 0.00001 1 True

def main(num_iters=40, nneurons=28, nlabels=10, alpha=0.00001, lam=1, reg=True):

        results = pd.DataFrame()

        print("Running labeler for {0} iterations and learning rate = {1}".format(num_iters,alpha))

        results['Label'] = nnLabeler(num_iters = int(num_iters),
                            nneurons  = int(nneurons),
                            nlabels   = int(nlabels),
                            alpha     = float(alpha),
                            lam       = float(lam),
                            reg       = bool(lam))
                            
        results['ImageId'] = np.arange(1,28001)
                            
        # Print results to csv:
        results.to_csv('results.csv')

        return results

def nnLabeler(nneurons=28, nlabels=10, alpha=1e-5, num_iters=400, lam=1, reg=True):

    '''Function which actually runs the process start-to-finish:
    nneurons : Number of hidden units in the model. Default is the image pixel width, 28
    nlabels : Number of classes to be trained. Default = 10 (digits)
    alpha : The learning rate.
    num_iters : Number of gradient descent iterations.
    lam : The regularization parameter, lambda.
    reg : Whether or not regularization should be used (to avoid overfitting)'''

    # Get the starting time for labeling output files:
    time = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')

    X, y = readInMnistCSV(dataset='train',scale=True)

    ## Randomly initialize theta for symmetry-breaking:
    ### Initial value by this method seemed to be too big..
    ### Thus I arbitrarily divided them by 10, which improves training and test results
    ### However I admit I do not have a good explanation for doing so... it just works!
    print("Initializing input layer weights")
    std_X = X.std()
    print("STD of input data = {}".format(X.std()))
    theta1_size = (nneurons,np.size(X[0])+1) 
    print("Size of theta1 weight array: {}".format(theta1_size))
    
    theta1 = np.random.normal(0, 
                                    std_X, 
                                    size=theta1_size
                                    )/10
                                
    print("Initializing hidden layer weights")
    theta2 = np.random.normal(0, theta1.std(), size=(nlabels,nneurons+1))/10

    # Intialize the cost plot for visualizing convergence/non-convergence:
    plt.axis()
    plt.ion()
    plt.xlim([0,num_iters])

    Jplot = []
    iplot = []

    #Begin the iterations of gradient descent- Using simple 'batch' gradient decsent
    ## SGD would perhaps be faster, but I prefer to be able to clearly see whether or not
    ## the cost is decreasing, as this is my first attempt to implenent a NN in python

    for i in range(0,num_iters):

        # Forward pass (apply activations and get the cost):
        J, a1, a2, a3 = costFunction(X,y,theta1, theta2, lam=lam, reg=reg)

        # Reverse pass (get the gradients):
        grad1, grad2 = backProp(a1, a2, a3, theta1,theta2, y, lam=lam, reg=reg)

        # Take a learning-rate-sized step along the gradients:
        ## These updates need to be simultaneous!
        ###FUTURE WORK: Add an SGD option.
        theta1_ = theta1 - grad1*alpha
        theta2_ = theta2 - grad2*alpha

        theta1 = theta1_
        theta2 = theta2_

        print("Iteration #"+str(i)+" of "+str(num_iters))

        Jplot.append(J)
        iplot.append(i)

        plt.scatter(i,J)
        plt.pause(0.05)

    # Save an image of the training progress
    plt.savefig("J_progress.pdf")
    #plt.savefig("../input/J_progress.pdf")

    # Show test the results against the "true" labels:
    # print("Getting training set score..")
    # output, output_label, result, score = outputMapper(a3,y)
    # Will need to modify 'outputMapper' to just return labels, rather than a score


    # Show the final weights-images:
    showWeightImgs(time, theta1,theta2)

    # Show distributions of the expected vs. predicted labels for diagnosis:
    #showHist(time, output_label, y)

    # Now use the model trained above on the test data, to check if it generalizes..
    print("Loading test data")
    X_test = readInMnistCSV(dataset='test')
    print("Running forward-prop on test data")
    a1_test, a2_test, a3_test = forwardProp(X_test,theta1,theta2)
    results_test = outputMapper(a3_test)

    # Copy the source code for the current run and:
    #copyfile('nnLabeler.py', './history/source_'+time+'.py')

    # Arrange the results into a dictionary:
    # results =  {'theta1':theta1,
    #             'theta2':theta2,
    #             'output_label':output_label,
    #             #'score': score,
    #             'outout_label_test': output_label_test,
    #             'results': result_test}
    #             #'score_test' : score_test}


    # Pickle the dictionary for later inspection:
    #with open('../input/result_'+time+'.p', 'w') as f:
    #    pickle.dump(results, f)

    return results_test
    
# We don't need all these raw-data input functions for this Kaggle Challenge.
# The Kaggle data is already reformed into .CSV files, so we can just use pandas to read the test and train sets

def readInMnistCSV(dataset='train',scale=True):

    # Reads in MNIST data using standard python packages:

    # Given the 'raw' idx MNIST files, return numpy arrays ready for training:

    if dataset == 'train':
    #### Training data:
        print("Loading in training data")
        train = pd.read_csv('../input/train.csv')
        images = train.drop(['label'],axis=1).values
        labels = train.label.values
        print("Training images and labels loaded")
        
        # Standardize the data to improve performance:
        if scale==True:
            print("Scaling data")
            images = scaleData(images)
    
        return images, labels

    elif dataset == 'test':
        #### Testing data:
        print("Loading in test data")
        test = pd.read_csv('../input/test.csv')
        images = test.values
        #labels = test.label.values
        print("Testing images and labels loaded")
        
        if scale==True:
            print("Scaling data")
            images = scaleData(images)
            
        return images

    else:
        sys.exit("Please specify either 'train' or 'test' data!")

    #print("Data successfully loaded!")

    
    
    
def scaleData(X):
    std_x = X.std()
    print("STD of input data = {}".format(std_x))
    
    mean_x = X.mean()
    print("Mean of input data = {}".format(mean_x))

    return ( X - X.mean() ) / X.std()

def sigmoid(z):

    return 1 / (1 + np.exp(-z))

def sigmoidGrad(z):

    return sigmoid(z)*(1-sigmoid(z))

def binaryMapper(y, nlabels=10):

    m = np.size(y)

    y = np.reshape(y,(m,1))

    # Map labels to binary label vectors
    y_temp    = np.arange(0,nlabels)
    y_broad   = np.ones((np.size(y),nlabels))

    return np.array(y*y_broad==y_temp, dtype=int)

def forwardProp(X,theta1, theta2):
    a1 = np.insert(X,0,1, axis=1)

    a2 = np.insert(
        sigmoid(theta1.dot(a1.T)),
                    0,1, axis=0)

    a3 = sigmoid(theta2.dot(a2))

    return a1, a2, a3

def costFunction(X, y,theta1, theta2, lam=None, reg=False):

    # Get the number of training examples:
    m = np.size(y)
    # Map labels to binary vectors:
    y = binaryMapper(y).T
    # Feed it forward:
    a1, a2, a3 = forwardProp(X, theta1, theta2)

    # Get the cost without regularization:
    J = np.sum(
        -(np.log(a3)*y)-(np.log(1-a3)*(1-y))
                                    ) /m

    # Add-regularization penalties to the cost (excluding the bias ):
    if reg==True:

        J += ( ( np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2) )*(lam/(2.0*m) ) )

        print("Regularized Cost: "+str(J))
    else:
        print("Unregularized Cost: "+str(J))

    return J, a1, a2, a3


def backProp(a1, a2, a3, theta1, theta2, y, reg=True, lam=1):

    # Get the 'error' for the third layer (aka first step of back-propagation):

    # Number of training examples
    m  = np.size(y)

    # The difference between expected and output values:
    ### (Input labels are mapped to binary vectors)
    s3 = a3 - binaryMapper(y).T

    # The second backprop layer: Applying the activation function's derivative:
    s2 = theta2.T.dot(s3)[1:]*sigmoidGrad(theta1.dot(a1.T))

    # Gradients along Theta 1 (should be the same dim as theta1!):
    d1 = s2.dot(a1)

    # Gradients along Theta 2 (should be the same dim as theta1!):
    d2 = s3.dot(a2.T)

    # Apply regularization if needed:
    ## Essentially just scale by the ratio of lambda to n-examples.
    if reg==True:

        d1 += (lam/m)*theta1
        d2 += (lam/m)*theta2

    return d1, d2


def showWeightImgs(time, theta1, theta2):

    nlabels = np.size(theta2[:,0])

    filters = theta2[:,1:].dot(theta1[:,1:])

    fig = plt.figure(figsize=(11.69,8.27))

    for lbl in  range(0, nlabels):

        ax = fig.add_subplot(2,5,lbl+1)

        ax.imshow(filters.reshape(10,28,28)[lbl])

        aspect = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) / abs(ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.set_aspect(aspect)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.05)
    plt.show()
    fig.savefig("weights.pdf")

def showHist(time, output_label,y):

        # Plot the 'true' labels dist. against taht of the output labels:
        ## This helps diagnose systematic errors- if score is low but distributions
        ## have much the same shape, something is probably going wrong!

        fig = plt.figure()
        plt.hist(output_label,alpha=0.3, label="Predicted distribution")
        plt.hist(y, alpha=0.3, label="Actual distribution")
        plt.legend()
        plt.show()
        fig.savefig("resHist.pdf")

    
def outputMapper(output, m = 28000):

    # Maps the output layer back to the labels
    # Revised for kaggle submission

    print("Number of samples checked: "+str(m))
    # Take just the elements on each row with the highest value:
    ## In other words, the highest probability det. by the NN

    output_maxprob = np.max(output, axis=0)

    # Convert output into a simple list that gives the label for each example
    ## that corresponds to the highest probability output

    output_label = (output_maxprob == output) # 'where' gives the full coordinates. we just need the "x" component.
    tmp_broad    = np.arange(0,10)*np.ones((m,10))
    output_label = output_label.T*tmp_broad
    
    results = np.sum(output_label, axis=1)
    print("return results")
    return results

if __name__ == '__main__':

    results = main()
