#import pandas as pd


# The competition datafiles are in the directory ../input
# Read competition data files:
#train = pd.read_csv("../input/train.csv")
#test  = pd.read_csv("../input/test.csv")

# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

import numpy as np
import matplotlib.pylab as plt
import csv
from sklearn.ensemble import RandomForestClassifier

# CSV Loading Data.   
def loadCsv(filename,n_samples):
    lines = csv.reader(open(filename,"r"))
    dataset = list(lines)
    return [line for line in dataset[1:n_samples]]

#print "...Loading CSV Training file..." 
filename="../input/train.csv"
M=loadCsv(filename,42000)

#print "...Preparing Training Data..."
# TRAINING DATA
n_train_samples=42000
Mf=[[float(line[x]) for x in range(0,len(line))] for line in M[0:n_train_samples]] 
matrixTrain=np.array(Mf)
X=matrixTrain[:,1:]
Y=matrixTrain[:,0]

#print "...Fitting Model with Training Data..."
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)

#print "...Loading CSV Test file..." 
filename="../input/test.csv"
Mtest=loadCsv(filename,28000)

#print "...Preparing Test Data..."
# TEST DATA
n_test_samples=28000
Mftest=[[float(line[x]) for x in range(0,len(line))] for line in Mtest[0:n_test_samples]] 
matrixTest=np.array(Mftest)
Ypredict=clf.predict(matrixTest)




