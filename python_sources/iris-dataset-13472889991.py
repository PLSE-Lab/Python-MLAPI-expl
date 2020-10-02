import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/Iris.csv")
#Read in the data
Y = pd.get_dummies(df['Species'])
#One hot encode the Species data to be our output data
X = df.drop(['Species'], axis = 1)
#Set up our input data
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
#Normalize all the data in X
model = RandomForestClassifier(n_estimators = 10000)
#Use RandomForestClassifier model to classify Irises. N_estimators increases model accuracy 
kfold = KFold(n_splits = 10)
#Kfold for cross validation, we iterate over our data multiple times, used when little data is available
results = cross_val_score(model,X,Y,cv=kfold)
#results returns the accuracies of each iteration, we call results.mean() toget a mean of 98.0% accuracy
