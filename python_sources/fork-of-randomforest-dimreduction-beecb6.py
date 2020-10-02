#Frequentist Machine Learning Final Project

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import Data:
trainFeatures = pd.read_csv("../input/trainFeatures.csv")
trainLabels = pd.read_csv("../input/trainLabels.csv")
testFeatures = pd.read_csv("../input/testFeatures.csv")

# Save ids of test features for later export of results
testid = testFeatures['ids']

trainFeatures.corr()

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

#Scale Data (Bad)
#from sklearn.preprocessing import StandardScaler
#X_imp = StandardScaler().fit_transform(X_imp)
#KaggleFeatures_imp = StandardScaler().fit_transform(KaggleFeatures_imp)

#DIMENSIONALITY REDUCTION USING PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_imp)
KaggleFeatures_pca = pca.transform(KaggleFeatures_imp)

#corr
All= np.random.rand(9999,52)
All[:,0:50] = X_imp
print(All)
print(All.shape)
All[:,51] = Y
print(All)
print(All.shape)

coff = np.corrcoef(All)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(coff,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
plt.show()

#ML Models

###Random Forest
#Import model
from sklearn.ensemble import RandomForestRegressor
#Instantiate with 1000 trees
model = RandomForestRegressor(n_estimators = 1000, random_state=42)
#Train
model.fit(X_pca, Y);
#Generate labels for kaggle data
K = model.predict(KaggleFeatures_pca)
#print(K)
#print(testid_array)

#print(K)
K = np.transpose(K)
#print(K)
#testid_array = np.transpose(testid_array)

#K2 = np.append(testid_array,K,axis=1)
#print(K2)

#RandomForest = pd.DataFrame(K, columns=['OverallScore'])
#print(RandomForest)
#R = RandomForest['id'] = testid.values
#print(R)
#R = testid.append(RandomForest)
#df['e'] = e.values

#Neural Net
#from sklearn import neural_network
#model = neural_network.MLPRegressor()
#model.fit(X_imp,Y)
#K = model.predict(KaggleFeatures_imp)

###THIS BLOCK WORKS BELOW
testid = testFeatures['ids']
testid = pd.DataFrame(testFeatures,columns=['ids'])
testid.insert(1, 'OverallScore', K)
testid.rename(index=str, columns={"ids": "Id"})


testid.to_csv('RFCorr.csv',index=False)