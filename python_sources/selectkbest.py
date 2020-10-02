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
print(np.isnan(Y).any())

Y_nan_index = np.where(np.isnan(Y))
print(Y[Y_nan_index])

Y = np.delete(Y,Y_nan_index,0)

print(Y[Y_nan_index])
print(np.isnan(Y).any())

X = np.delete(X,Y_nan_index,0)
    
print(X)
print(Y)
print(KaggleFeatures)

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

print(KaggleFeatures)
print(X.shape)
print(X_imp.shape)
print(KaggleFeatures_imp.shape)

#Scale Data
#from sklearn.preprocessing import StandardScaler
#X_imp = StandardScaler().fit_transform(X_imp)
#KaggleFeatures_imp = StandardScaler().fit_transform(KaggleFeatures_imp)

#Absolute Value of data bc no neg allowed
X_imp = np.absolute(X_imp)
KaggleFeatures_imp = np.absolute(KaggleFeatures_imp)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# feature extraction
Y=Y.astype('int')
X_new = SelectKBest(chi2, k=25).fit_transform(X_imp,Y)

SKB = SelectKBest(chi2, k=25)
X_new = SKB.fit_transform(X_imp,Y)
Kag = SKB.transform(KaggleFeatures_imp)

#ML Models
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 1000, random_state=42)
model.fit(X_new, Y);
K = model.predict(Kag)

K = np.transpose(K)

###THIS BLOCK WORKS BELOW
testid = testFeatures['ids']
testid = pd.DataFrame(testFeatures,columns=['ids'])
testid.insert(1, 'OverallScore', K)
testid.rename(index=str, columns={"ids": "Id"})


testid.to_csv('SelectKBest.csv',index=False)
