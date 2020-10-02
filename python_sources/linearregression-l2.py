#Frequentist Machine Learning Final Project

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import Data:
trainFeatures = pd.read_csv("../input/trainFeatures.csv")
trainLabels = pd.read_csv("../input/trainLabels.csv")
testFeatures = pd.read_csv("../input/testFeatures.csv")

# Save ids of test features for later export of results
testid = testFeatures['ids']

# Convert to Numpy Style Arrays
trainFeatures_array = trainFeatures.values
trainLabels_array = trainLabels.values
testFeatures_array = testFeatures.values
testid_array = testid.values

# Remove id info(first 2 columns in features, first column in labels)
X = trainFeatures_array[:,3:63]
Y = trainLabels_array[:,2]
KaggleFeatures = testFeatures_array[:,3:63]

# Remove from training data observations with no label
Y_nan_index = np.where(np.isnan(Y))
Y = np.delete(Y,Y_nan_index,0)
X = np.delete(X,Y_nan_index,0)

# Check size of arrays
print('Training Features Shape:', X.shape)
print('Training Labels Shape:', Y.shape)
print('Testing Features Shape:', KaggleFeatures.shape)

# Processing Pipeline
#### Impute data - remove Nan,infinity and replace with mean
from sklearn.impute import SimpleImputer

#Set imputer to replace nan with mean
imp = SimpleImputer(strategy='mean') #copy=False will do an inplace imputation
X_imp = imp.fit_transform(X)

#notice 10 features removed because all nans
print(X)
print(X_imp)

#Apply imputer to kaggle test data
KaggleFeatures_imp = imp.transform(KaggleFeatures)

#ML Models

from sklearn import linear_model
#data_dmatrix = xgb.DMatrix(data=X_imp,label=Y)
model = linear_model.RidgeCV(cv=5)
model.fit(X_imp,Y)
y_pred = model.predict(KaggleFeatures_imp)
print(y_pred)

#Convert solution into correct output
testid = testFeatures['ids']
testid = pd.DataFrame(testFeatures,columns=['ids'])
testid.insert(1, 'OverallScore', y_pred)
testid.rename(index=str, columns={"ids": "Id"})


testid.to_csv('LinReg_L2.csv',index=False)
