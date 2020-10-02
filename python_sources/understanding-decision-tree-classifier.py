######################################## PARAMETERS ###########################################

# The default values for the parameters controlling the size of the trees (e.g. max_depth, min_samples_leaf, etc.) 
# lead to fully grown and unpruned trees which can potentially be very large on some data sets. To reduce memory consumption, 
# the complexity and size of the trees should be controlled by setting those parameter values.

# The features are always randomly permuted at each split. Therefore, the best found split may vary, 
# even with the same training data and max_features=n_features, if the improvement of the criterion is identical 
# for several splits enumerated during the search of the best split. To obtain a deterministic behaviour during 
# fitting, random_state has to be fixed.

# min_samples_split = 50. In this case iif we have 100 data points in previous branch it will split into 50, 50 each.
# min_samples_leaf = 1. In this case for above example we will get 50 leaves in each branch that is splitted. If the values is 2 the we will get 25 leaves (data points) in each branch

# entropy. Very important parameters whcih mesures the impurities. Helps in understanding at what point the Decision Tree should split. ranges from [0,1]
# Information gain. It helps in understanding which features we have to use to split the decision trees. IG = (entropy of parent)-[weighted average](entropy of children)

# Instead of entropy and Information gain we can use the other parameter called criterion which is the default parameter in scikit learn.

############# Generating Sample Data CONSIDERING A TERRIAN SURFACE WHICH HAS STEEPNESS AND BUMPINESS AS FEATURES ##########

import random

def makeTerrainData(n_points=1000): ### Preparing a sample data set
    random.seed(42)
    grade = [random.random() for ii in range(0,n_points)]
    bumpy = [random.random() for ii in range(0,n_points)]
    error = [random.random() for ii in range(0,n_points)]
    y = [round(grade[ii]*bumpy[ii]+0.3+0.1*error[ii]) for ii in range(0,n_points)]
    for ii in range(0, len(y)):
        if grade[ii]>0.8 or bumpy[ii]>0.8:
            y[ii] = 1.0

### split into train/test sets
    X = [[gg, ss] for gg, ss in zip(grade, bumpy)]
    split = int(0.75*n_points)
    X_train = X[0:split]
    X_test  = X[split:]
    y_train = y[0:split]
    y_test  = y[split:]

    grade_sig = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==0]
    bumpy_sig = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==0]
    grade_bkg = [X_train[ii][0] for ii in range(0, len(X_train)) if y_train[ii]==1]
    bumpy_bkg = [X_train[ii][1] for ii in range(0, len(X_train)) if y_train[ii]==1]

#    training_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
#            , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}


    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    test_data = {"fast":{"grade":grade_sig, "bumpiness":bumpy_sig}
            , "slow":{"grade":grade_bkg, "bumpiness":bumpy_bkg}}

    return X_train, y_train, X_test, y_test
#    return training_data, test_data

############## visualising the points and decision boundary ##############


import warnings
warnings.filterwarnings("ignore")

import matplotlib 
matplotlib.use('agg')

import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

def prettyPicture(clf, X_test, y_test):
    x_min = 0.0; x_max = 1.0
    y_min = 0.0; y_max = 1.0

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)

    # Plot also the test points
    grade_sig = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==0]
    bumpy_sig = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==0]
    grade_bkg = [X_test[ii][0] for ii in range(0, len(X_test)) if y_test[ii]==1]
    bumpy_bkg = [X_test[ii][1] for ii in range(0, len(X_test)) if y_test[ii]==1]

    plt.scatter(grade_sig, bumpy_sig, color = "b", label="fast")
    plt.scatter(grade_bkg, bumpy_bkg, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")

    plt.savefig("test.png")
    
import base64
import json
import subprocess

def output_image(name, format, bytes):
    image_start = "BEGIN_IMAGE_f9825uweof8jw9fj4r8"
    image_end = "END_IMAGE_0238jfw08fjsiufhw8frs"
    data = {}
    data['name'] = name
    data['format'] = format
    #data['bytes'] = base64.encodestring(bytes) # Only In python 2X byte codes can only be Json serialized. 
    print(image_start+json.dumps(data)+image_end)
    
######################### Utilising Algorithm Code ###########################

from sklearn.tree import DecisionTreeClassifier
    
def classify(features_train, labels_train):   
    clf = DecisionTreeClassifier(random_state=0) 
    clf.fit(features_train,labels_train)
    return clf
    
########################## CALCULATING ACCURACY ##############################
    
from sklearn.metrics import accuracy_score
def accuracy():
    score = accuracy_score(labels_test, clf.predict(features_test), normalize=True) # True by default. Gives the fraction of correct classifications
    return score*100
    
#################### Main Code to Combine All ################################

import numpy as np
import pylab as pl


features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow" points mixed
### in together--separate them so we can give them different colors in the scatterplot,
### and visually identify them
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.
clf = classify(features_train, labels_train)


### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())

print("Total correct classification percent = ", accuracy())

###############################################################################