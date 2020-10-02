import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt

###############################################################################
# This script detects potential outliers :
# - focusing on "accommodates" and "price" columns
# - using two different classifiers, robust covariance and one class SVM
# - displaying results in simple charts
###############################################################################

FILE_PATH = "../input/listings.csv"

###############################################################################
# Display results
###############################################################################
def displayResults(inliers, outliers, classifier, outputTitle, outputName):
    
    plt.figure()        
    
    # Defining grid
    gridX, gridY = np.meshgrid(np.linspace(-0.5, 1.5, 1000), np.linspace(-0.5, 1.5, 1000))
    
    # Computing decision for each point of the grid
    gridDecisions = classifier.decision_function(np.c_[gridX.ravel(), gridY.ravel()])
    
    # Plotting decision boundary (each point of the grid whose decision value is 0)
    gridDecisions = gridDecisions.reshape(gridX.shape)
    plotBoundary = plt.contour(gridX, gridY, gridDecisions, levels=[0], linewidths=2, colors='blue')
    plt.clabel(plotBoundary, inline=1, fontsize=12)

    # Plotting inliers and outliers
    plt.scatter(inliers.loc[:, 'accommodates'], inliers.loc[:, 'price'], label="Inliers", color='green', alpha=0.2)
    plt.scatter(outliers.loc[:, 'accommodates'], outliers.loc[:, 'price'], label="Outliers", color='red', alpha=1.0)
    
    plt.xlabel("Accomodates (normalized)")
    plt.ylabel("Price (normalized)")
    plt.title(outputTitle)
    plt.legend()    
    
    plt.savefig(outputName + ".png")
    plt.clf()

###############################################################################
# Get classifier robust covariance
###############################################################################

def getClassifierRobustCovariance(data):

    #------------------------------------------------------------------------------
    # Checking prerequisites
    #------------------------------------------------------------------------------

    numberOfSamples = data.shape[0]
    numberOfFeatures = data.shape[1]
    
    if (numberOfSamples > numberOfFeatures ** 2):
        
        #------------------------------------------------------------------------------
        # Preparing and fitting model
        #------------------------------------------------------------------------------
        
        # Initializing classifier
        classifier = EllipticEnvelope(contamination=0.001)
        
        # Fitting classifier
        classifier.fit(data)
        
        return classifier
       
    return None

###############################################################################
# Get classifier oone class SVM
###############################################################################

def getClassifierOneClassSVM(data):

    #------------------------------------------------------------------------------
    # Preparing and fitting model
    #------------------------------------------------------------------------------    
    
    # Initializing classifier
    classifier = svm.OneClassSVM(nu=0.003, gamma=2.0)
    
    # Fitting classifier
    classifier.fit(data)
    
    return classifier

###############################################################################
# Main
###############################################################################

#------------------------------------------------------------------------------
# Importing data
#------------------------------------------------------------------------------

# Importing CSV file
listings = pd.read_csv(FILE_PATH)

#------------------------------------------------------------------------------
# Preparing data
#------------------------------------------------------------------------------

# Selecting features
listings_features = listings.loc[:, ['accommodates', 'price']]

# Fixing price column (removing unit, removing commas, converting to float)
listings_features.loc[:, 'price'] = listings_features.loc[:, 'price'].apply(lambda x: x.replace('$','')).apply(lambda x: x.replace(',','')).astype(np.float)

# Normalizing features
listings_features = (listings_features - listings_features.mean()) / (listings_features.max() - listings_features.min())    

#------------------------------------------------------------------------------
# Detecting outliers with robust covariance
#------------------------------------------------------------------------------

# Getting classifier
classifierRobustCovariance = getClassifierRobustCovariance(listings_features)

# Classifying inliers/outliers
decisionsRobustCovariance = classifierRobustCovariance.decision_function(listings_features)

# Displaying results
displayResults(inliers=listings_features[decisionsRobustCovariance >= 0],
               outliers=listings_features[decisionsRobustCovariance < 0],
               classifier=classifierRobustCovariance,
               outputTitle = "Detecting potential outliers using robust covariance",
               outputName="outliers_robust_covariance")

#------------------------------------------------------------------------------
# Detecting outliers with one class SVM
#------------------------------------------------------------------------------

# Getting classifier
classifierOneClassSVM = getClassifierOneClassSVM(listings_features)

# Classifying inliers/outliers
decisionsOneClassSVM = classifierOneClassSVM.decision_function(listings_features)

# Displaying results
displayResults(inliers=listings_features[decisionsOneClassSVM >= 0],
               outliers=listings_features[decisionsOneClassSVM < 0],
               classifier=classifierOneClassSVM,
               outputTitle = "Detecting potential outliers using one class SVM",
               outputName="outliers_one_class_svm")
