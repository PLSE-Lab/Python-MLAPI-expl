# For NN calculations and visualizaing progress:
import numpy as np
import matplotlib.pyplot as plt
# For reading in the MNIST raw data:
from array import array
import struct
import os
# For logging the results:
import pickle
from shutil import copyfile
from datetime import datetime
# For running from the command line
import sys
# For GPU-enabled processing:
import gnumpy as gpu
import pandas as pd


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

    X, y = readInMnistKaggle(dataset='train',scale=True)

    ## Randomly initialize theta for symmetry-breaking:
    ### Initial value by this method seemed to be too big..
    ### Thus I arbitrarily divided them by 10, which improves training and test results
    ### However I admit I do not have a good explanation for doing so... it just works!
    theta1 = np.random.normal(0, X.std(), size=(nneurons,np.size(X[0])+1))/10
    theta2 = np.random.normal(0, theta1.std(), size=(nlabels,nneurons+1))/10

    # Put intput matrices into gnumpy arrays for GPU-enabled calculation:
    X      = gpu.garray(X)
    theta1 = gpu.garray(theta1)
    theta2 = gpu.garray(theta2)

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

        print "Iteration #"+str(i)+" of "+str(num_iters)

        Jplot.append(J)
        iplot.append(i)

        plt.scatter(i,J)
        plt.pause(0.05)

    # Save an image of the training progress
    plt.savefig("./plots/GPU_J_progress_"+time+".pdf")

    # Show test the results against the "true" labels:
    print "Getting training set score.."
    output, output_label, result, score = outputMapper(a3,y)

    # Show the final weights-images:
    showWeightImgs(time, theta1,theta2)

    # Show distributions of the expected vs. predicted labels for diagnosis:
    showHist(time, output_label, y)

    # Now use the model trained above on the test data, to check if it generalizes..
    print "Calculating test-set score..."
    X_test, y_test = readInMnistRaw(dataset='test')
    a1_test, a2_test, a3_test = forwardProp(X_test,theta1,theta2)
    output_test, output_label_test, result_test, score_test = outputMapper(a3_test, y_test)

    # Copy the source code for the current run and:
    copyfile('nnLabeler.py', './history/GPU_source_'+time+'.py')

    # Arrange the results into a dictionary:
    results =  {'theta1':theta1,
                'theta2':theta2,
                'output_label':output_label,
                'score': score,
                'outout_label_test': output_label_test,
                'result': result_test,
                'score_test' : score_test}


    # Pickle the dictionary for later inspection:
    with open('./history/GPU_result_'+time+'.p', 'w') as f:
        pickle.dump(results, f)

    return results

def readInMnistRaw(dataset='train',scale=True):

    # Reads in MNIST data using standard python packages:

    dpath  = './data/'

    fnames = ['train-images.idx3-ubyte',
              'train-labels.idx1-ubyte',
              't10k-images.idx3-ubyte',
              't10k-labels.idx1-ubyte']

    # Given the 'raw' idx MNIST files, return numpy arrays ready for training:

    if dataset == 'train':
    #### Training data:
        print "Loading in training data"
        file_img =  open(dpath+fnames[0],'rb')
        file_lbl = open(dpath+fnames[1], 'rb')

    elif dataset == 'test':
        #### Testing data:
        print "Loading in testing data"
        file_img =  open(dpath+fnames[2],'rb')
        file_lbl =  open(dpath+fnames[3], 'rb')

    else:
        sys.exit("Please specify either 'train' or 'test' data!")

    # Load-in the image data the file objects and process them into numpy arrays:
    ## Note:: This will load each image as a single 'flattened' row, not matrix!!!
    index, n_imgs, n_rows, n_cols = struct.unpack(">IIII", file_img.read(16))

    image_data = array("B", file_img.read())

    images = []

    for i in range(n_imgs):
        images.append([0] * n_rows * n_cols)

    for i in range(n_imgs):
        images[i][:] = image_data[i * n_rows * n_cols:(i + 1) * n_rows * n_cols]

    images = np.array(images)

    print "Images succesfully loaded! (As a "+str(np.shape(images))+" numpy array)"

    # Load the labels in and convert to a numpy array:
    index, n_imgs = struct.unpack(">II", file_lbl.read(8))
    labels = array("B", file_lbl.read())
    labels = np.array(labels)
    print "Labels succesfully loaded! (As a "+str(np.shape(labels))+" numpy array)"

    # Standardize the data to improve performance:
    if scale==True:
        images = scaleData(images)

    return images, labels
    
    
    
def readInMnistKaggle(dataset='train',scale=True, nrows=28, ncols=28):

    # Reads in MNIST data using standard python packages:

    if dataset == 'train':
        #### Training data:
        print "Loading in traning data"
        data = pd.read_csv('../input/train.csv') 

    elif dataset == 'test':
        #### Testing data:
        print "Loading in testing data"
       data = pd.read_csv('../input/test.csv')
       

    else:
        sys.exit("Please specify either 'train' or 'test' data!")

    # Load-in the image data the file objects and process them into numpy arrays:
    ## Note:: This will load each image as a single 'flattened' row, not matrix!!!

    image_data = data.drop('label',axis=1).values.copy()    

    images = []

    for i in range(n_imgs):
        images.append([0] * n_rows * n_cols)

    for i in range(n_imgs):
        images[i][:] = image_data[i * n_rows * n_cols:(i + 1) * n_rows * n_cols]

    images = np.array(images)

    print "Images succesfully loaded! (As a "+str(np.shape(images))+" numpy array)"

    # Load the labels in and convert to a numpy array:
    labels = data.label.values.copy()
    print "Labels succesfully loaded! (As a "+str(np.shape(labels))+" numpy array)"

    # Standardize the data to improve performance:
    if scale==True:
        images = scaleData(images)

    return images, labels

def scaleData(X):

    return ( X - X.mean() ) / X.std()

def sigmoid(z):

    return 1 / (1 + gpu.exp(-z))

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

    a1 = gpu.concatenate( (X, gpu.ones((np.size(X[:,0]),1)) ), axis=1)

    a2 = sigmoid(theta1.dot(a1.T))
    a2 = gpu.concatenate( (a2, gpu.ones((1, np.size(a2[0,:])) ) ), axis=0)

    a3 = sigmoid(theta2.dot(a2))

    return a1, a2, a3

def costFunction(X, y,theta1, theta2, lam=None, reg=False):

    # Get the number of training examples:
    m = np.size(y)
    # Map labels to binary vectors:
    y = gpu.garray(binaryMapper(y)).T
    # Feed it forward:
    a1, a2, a3 = forwardProp(X, theta1, theta2)

    # Get the cost without regularization:
    J = gpu.sum(
        -(gpu.log(a3)*y)-(gpu.log(1-a3)*(1-y))
                                    ) /m

    # Add-regularization penalties to the cost (excluding the bias ):
    if reg==True:

        J += ( ( gpu.sum(theta1[:,1:]**2) + gpu.sum(theta2[:,1:]**2) )*(lam/(2.0*m) ) )

        print "Regularized Cost: "+str(J)
    else:
        print "Unregularized Cost: "+str(J)

    return J, a1, a2, a3


def backProp(a1, a2, a3, theta1, theta2, y, reg=True, lam=1):

    # Get the 'error' for the third layer (aka first step of back-propagation):

    # Number of training examples
    m  = np.size(y)

    # The difference between expected and output values:
    ### (Input labels are mapped to binary vectors)
    s3 = a3 - gpu.garray(binaryMapper(y).T)

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

        ax.imshow(filters.as_numpy_array().reshape(10,28,28)[lbl])

        aspect = abs(ax.get_xlim()[1] - ax.get_xlim()[0]) / abs(ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.set_aspect(aspect)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.05)
    plt.show()
    fig.savefig("./plots/GPU_weights_"+time+".pdf")

def showHist(time, output_label,y):

        # Plot the 'true' labels dist. against taht of the output labels:
        ## This helps diagnose systematic errors- if score is low but distributions
        ## have much the same shape, something is probably going wrong!

        fig = plt.figure()
        plt.hist(output_label,alpha=0.3, label="Predicted distribution")
        plt.hist(y, alpha=0.3, label="Actual distribution")
        plt.legend()
        plt.show()
        fig.savefig("./plots/GPU_resHist_"+time+".pdf")

def outputMapper(output, expected):

    # Maps the output layer back to the labels and gives the F1 score:
    m = np.size(expected)
    print "Number of samples checked: "+str(m)
    # Take just the elements on each row with the highest value:
    ## In other words, the highest probability det. by the NN

    output_maxprob = np.max(output.as_numpy_array(), axis=0)
    print "Output maxprob: "+str(output_maxprob)

    # Convert output into a simple list that gives the label for each example
    ## that corresponds to the highest probability output

    output_label = (output_maxprob == output.as_numpy_array()) # 'where' gives the full coordinates. we just need the "x" component.
    tmp_broad = np.arange(0,10)*np.ones((m,10))
    output_label = output_label.T*tmp_broad
    output_label = np.sum(output_label, axis=1)

    result = expected.flatten()==output_label

    n_correct = np.count_nonzero(result)

    score = (float(n_correct)/m)*100

    print "Test: Confirm output label range = "+str(np.min(output_label))+" "+str(np.max(output_label))
    print "Test: Confirm expected label range = "+str(np.min(expected))+" "+str(np.max(expected))
    print "Score: "+str(score)+"% correct labels"


    return output, output_label, result, score

def main(num_iters=sys.argv[1], nneurons=sys.argv[2], nlabels=sys.argv[3], alpha=sys.argv[4], lam=sys.argv[5], reg=sys.argv[6]):

        results = []

        print "Running labeler for "+num_iters+" iterations and learning rate = "+alpha

        results = nnLabeler(num_iters = int(num_iters),
                            nneurons  = int(nneurons),
                            nlabels   = int(nlabels),
                            alpha     = float(alpha),
                            lam       = float(lam),
                            reg       = bool(lam))

        return results

if __name__ == '__main__':

    results = main()
