#!/usr/bin/env python
# coding: utf-8

# I am going to apply various regression machine learning models to the dataset to predict the price per square feet of housing flats in Singapore. In different models, I will use absolute percentage error difference of test data to assess their predictive accuracy. 
# 
# To make the notebook entry easier to read, the data contains the following features of a HDB flat in Singapore - district of flat, furnishing status, price, floor area, number of bedrooms, number of bathrooms, floor plan available(?), latitude, longitude, with one final predicted value - price per square feet. The data was taken in March 2017.
# 
# **In this notebook, I will be covering 3 basic ML algorithms with some analysis on their parameters, followed by some feature selection and feature extraction methods:**
# - ** General Preprocessing **
# - ** K Nearest Neighbours **
# - ** Single Decision Tree **
# - ** Random Forest **
# - ** Feature Selection using correlation matrix and VIF **
# - ** Feature extraction using PCA **
# 

# **Import panda library and read dataset for preprocessing:**

# In[ ]:


import pandas as pd
HDB_Data = pd.read_csv('../input/Housing_Data.csv')


# ## **The general preprocessing**
# <br>
# **Briefly eyeing the csv file tells us that there are data rows with wrong entries; we will remove all rows with wrong datatype for the rows and duplicate rows; duplicate rows can be identified as rows with the same price per square feet.**
# 
# 
# 
# 

# In[ ]:


HDB_Data.floorArea = pd.to_numeric(HDB_Data.floorArea, errors='coerce')
HDB_Data.pricePerSqFt = pd.to_numeric(HDB_Data.pricePerSqFt, errors='coerce')
HDB_Data.bedrooms = pd.to_numeric(HDB_Data.bedrooms, downcast = 'integer', errors='coerce')
HDB_Data.latitude = pd.to_numeric(HDB_Data.latitude, downcast = 'integer', errors='coerce')
HDB_Data.longitude = pd.to_numeric(HDB_Data.longitude, downcast = 'integer', errors='coerce')

HDB_Data.dropna(subset=['latitude'], inplace=True)
HDB_Data.dropna(subset=['longitude'], inplace=True)
HDB_Data.dropna(subset=['floorArea'], inplace=True)
HDB_Data.dropna(subset=['pricePerSqFt'], inplace=True)
HDB_Data.dropna(subset=['bedrooms'], inplace=True)

HDB_Data.drop_duplicates(subset=['pricePerSqFt'], keep='last', inplace=True)
del HDB_Data['Unnamed: 0']


# **Defining the categorical and continuous columns, subsequently converting categorical features to dummy variables**

# In[ ]:


continous_Col = ['value', 'floorArea', 'pricePerSqFt']
categorical_Col = ['furnishingCode', 'bedrooms', 'hasFloorplans']
HDB_Data = pd.get_dummies(HDB_Data, columns = categorical_Col)

HDB_Data_Backup = HDB_Data #create a backup dataframe


# ## K Nearest Neighbours
#  **Why?**
#  The K nearest neighbours algorithm looks at the 'nearest' K training datapoints of a test datapoint to produce the predicted result. Intuitively, I hope to predict the price per square feet of a housing flat based on a few flats which most closely resemble it. In fact, this is still the method used by property agents in real life to predict housing prices.
# 
# **How?**
# I decide to cluster the flats according to their location - in real life, no property agent will predict the price of a flat in location A using knowledge of a flat in location B. It makes sense to follow this train of  thought in a ML model as well.
# 
# Moreover, because both floor area and flat value are features, we need to remove one of them. This is because price per square feet = flat value / floor area. **I remove flat value because it doesn't make sense to have flat value as a feature.**
# 
# I also remove the number of bathrooms as a feature because some of the data points have missing values. However, this may be added later for future analysis.
# 
# Lastly, I standardized the data using a Scaler package, which is essential for KNN algorithm.
# 
# <br>
# **Preprocessing using the* Boon Lay/Jurong/Tuas district* as a sample location:**

# In[ ]:


data_Set = HDB_Data.loc[HDB_Data['districtText'] == ' Boon  Lay  /  Jurong  /  Tuas ']
del data_Set['districtText']
del data_Set['bathrooms']
del data_Set['value']

#reindex columns to make prediction column the first column
col = data_Set.columns.tolist()
col[0], col[1] = col[1], col[0]
data_Set = data_Set.reindex(columns = col)

#import KNN, scaling libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing.data import StandardScaler


# **1. Defining 30% of flats in Boon Lay/Jurong/Tuas district as test data and 70% of the flats as training data (chosen randomly) **
# 
# **2. Scaling the features in both subsets using only the training data:**

# In[ ]:


Y = data_Set.values[:,0]
X = data_Set.values[:, 1:len(col)]    
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 45)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# **Setting nearest neighbours K as 4 (approximately equals square root of number of features; [see here for more analysis on value of K](http://www.cs.haifa.ac.il/~rita/ml_course/lectures/KNN.pdf)), we initiate the KNN model with distance inversed as a weight measure.**   

# In[ ]:


num_Neighbours = 4
KNN = KNeighborsRegressor(n_neighbors=num_Neighbours, weights='distance', p=1)
KNN.fit(X_train, y_train) 
y_pred = KNN.predict(X_test)


# **With a list of predicted price per square feet of the test data, we will derive the difference between the actual price per square feet and the predicted price per square feet. This difference will be used constantly thoughout this notebook.**

# In[ ]:


#define as a function
def findErrorDifference(y_test, y_pred):
    errorPercentageDiff = []
    numberOfTestData = len(y_test)
    for x in range(numberOfTestData):
        percentageDiff = (y_pred[x] - y_test[x])/y_test[x]
        errorPercentageDiff.append(percentageDiff)
    return errorPercentageDiff

errorPercentageDiff = findErrorDifference(y_test, y_pred)
#The error percentages between prediction and actual values
print(errorPercentageDiff)
#Mean error percentage
print(sum(errorPercentageDiff)/len(errorPercentageDiff))
        


# **Plotting the percentage differences as a histogram:**

# In[ ]:


import matplotlib.pyplot as plt
n, bins, patches = plt.hist(errorPercentageDiff)
plt.show()


# **Visual inspection tells us that the percentage difference is centred around 0 (not perfectly though); this is gives us some indication that the model is fitting to some extent. Now I will formalise this by deriving the median absolute deviation and root mean square error of the percentage difference for all predictions. **
# 
# **Furthermore, we can also find out how many predicted price per square feet's lie within 20% of the actual price per square feet**

# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics

rms = sqrt(mean_squared_error(y_pred, y_test))
print('For district of Boon Lay/Jurong/Tuas:')
print('root mean square error: ' + str(rms))

median = statistics.median(errorPercentageDiff)
median_Deviation = [(x - median) for x in errorPercentageDiff]
Median_AD = statistics.median(median_Deviation)
print('median absolute deviation: ' + str(Median_AD))
Mean_AD = statistics.mean(median_Deviation)
print('mean absolute deviation: ' + str(Mean_AD))

threshold = 0.20
numTest = len(errorPercentageDiff)
withinThreshold = [x for x in errorPercentageDiff if abs(x) < threshold]
ratioWithinThreshold = len(withinThreshold)/numTest
print('ratio of predictions within threshold of 20%: ' + str(ratioWithinThreshold))


# **Now, let's generalise the KNN model to all districts with more than 10 listings. I simply identify all districts from the dataset and iterate through all of them to generate the overall error percentage.**

# In[ ]:


import numpy as np

districts = np.unique(HDB_Data_Backup['districtText'].values)
index_1 = np.argwhere(districts == 'D11')
index_2 = np.argwhere(districts == 'D12')
index_3 = np.argwhere(districts == 'D21')
districts = np.delete(districts, index_1)
districts = np.delete(districts, index_2)
districts = np.delete(districts, index_3)
data = HDB_Data_Backup

print('All the districts:')
print(districts)


# In[ ]:


errorPercentageList = []
for district in districts:
    data_Set = data.loc[data['districtText'] == district]
    del data_Set['districtText']
    del data_Set['bathrooms']
    del data_Set['value']
    col = data_Set.columns.tolist()
    col[0], col[1] = col[1], col[0]
    data_Set = data_Set.reindex(columns = col)

    Y = data_Set.values[:,0]
    X = data_Set.values[:, 1:len(col)]

    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3)
    
    #skip districts with less than 10 total datapoints
    if (len(X_train) < 10):
        continue
    
    #scale training data and test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    num_Neighbours = 3
    KNN = KNeighborsRegressor(n_neighbors=num_Neighbours, weights='distance', p=1)
    KNN.fit(X_train, y_train)
    
    y_pred = KNN.predict(X_test)
    
    #derive errorPercentage
    numberOfTestData = len(y_test)
    for x in range(numberOfTestData):
        percentageDiff = (y_pred[x] - y_test[x])/y_test[x]
        errorPercentageList.append(percentageDiff)
        
#print histogram
bins = np.linspace(-1, 1, 50)
n, bins, patches = plt.hist(errorPercentageList,bins)
plt.show()

#find predictions within error percentage threshold
threshold = 0.20
numTest = len(errorPercentageDiff)
withinThreshold = [x for x in errorPercentageDiff if abs(x) < threshold]
ratioWithinThreshold = len(withinThreshold)/numTest
print('ratio of predictions within threshold of 20%: ' + str(ratioWithinThreshold))


# **Before we analyse all these complicated values we derived from our KNN algorithm and figure out how to fine tune the model with PCA and removal of highly correlated features, I will introduce another popular ML algorithm - Decision Trees and Random forest (which is essentially an extension of the Decision Tree to reduce variance and increase bias)**
# 
# ## Simple Decision Tree
#  **Why?**
# Decision Tree is a branch of commonly used supervised ML algorithm in both classifier and regressor models. It performs implicit feature selection by selecting features that allow us to predict on the Price per square feet of a HDB flat. In particular, it partitions each continuous feature into two distinct intervals and decides which feature partitioning gives us the most information.
# 
# **How?**
# What defines the most information? There are many criterions to decide which feature should the split be based on; these include:
# - [information gain](https://en.wikipedia.org/wiki/Information_gain_ratio)
# - [gini impurity](https://datascience.stackexchange.com/questions/10228/gini-impurity-vs-entropy)
# 
# for classifier trees and
# - [minisation of mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) 
# 
# for decision tree regressors.
# 
# *Intuitively,  if the training data's price per square feet (what we want to predict) is uniformly distributed about the floor area size, then it is useless to split the training data into floor area > 1000, floor area < 1000 because we cannot gain any insightful information from this. On the other hand, it will be more intuitive to split the training data into unfurnished flats and furnished flats because it can be shown in the data that furnished flats tend to yield a higher price per square feet - this can be formalised by calculating the split which results in the lowest MSE. As such, splitting the training data into unfurnished and furnished flats will give us more insightful information.

# **Using 'Boon Lay/Jurong/Tuas' as the district region again and scale the data using only the training data:**

# In[ ]:


data_Set = HDB_Data.loc[HDB_Data['districtText'] == ' Boon  Lay  /  Jurong  /  Tuas ']
del data_Set['districtText']
del data_Set['bathrooms']
del data_Set['value']

#reindex columns to make prediction column the first column
col = data_Set.columns.tolist()
col[0], col[1] = col[1], col[0]
data_Set = data_Set.reindex(columns = col)

Y = data_Set.values[:,0]
X = data_Set.values[:, 1:len(col)] 

#partition training and test data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

#scale data using only training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# **Now, apart from deciding how to split the decision tree, another crucial decision in creating single decision trees is to know when to terminate the tree - a tree that includes all feature splits will be huge and causes overfitting for future data (too much biasness) while too small of a tree may lead to higher variance (leading to inaccurate predictions)**
# 
# **For our single tree, we explore two stopping criterion: **
# - Setting the minimum number of datapoints a leaf can contain to 50 (around 15% of training set)
# - Setting the max depth of tree to log(number of features) = 4
# 
# **Using the first stopping criterion (leaf size):**
# 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor

min_samples_leaf = 50
#setting minimum samples in a leaf to 50
regTree = DecisionTreeRegressor(min_samples_leaf=min_samples_leaf) 
regTree.fit(X_train, y_train)
y_pred = regTree.predict(X_test)

errorPercentageDiff = []
errorPercentageDiff = findErrorDifference(y_test, y_pred)

n, bins, patches = plt.hist(errorPercentageDiff)
plt.show()


# **Using the second stopping criterion (max tree depth):**

# In[ ]:


max_depth = 4
#setting max depth as 4
regTree = DecisionTreeRegressor(max_depth=max_depth) 
regTree.fit(X_train, y_train)
y_pred = regTree.predict(X_test)

errorPercentageDiff = findErrorDifference(y_test, y_pred)
n, bins, patches = plt.hist(errorPercentageDiff)
plt.show()


# ## Random Forest
# **Why?**
#  While decision trees give us considerable prediction results, we have not done any explicit feature selection to any of our training data. In particular, correlation and collinearity between features may lead to various problems; a principal danger of such data redundancy is that of overfitting in regression models. Furthermore, decision trees greedily chooses the best features to split - this leads to highly correlated features to be omitted in lower levels of the tree ([See here for a more in-depth explanation](https://stats.stackexchange.com/questions/1292/what-is-the-weak-side-of-decision-trees)).
#  
#  The best regression models are those in which the predictor variables each correlate highly with the dependent (outcome) variable but correlate at most only minimally with each other. Such a model is often called "low noise" and will be statistically robust.
#  
#  **How?**
#  what can we do to minimise the effects of highly correlated features? Apart from Principle Component Analysis and other general methods in reducing or extracting the number of features (explained in last part of this notebook), we can use [Random Forest](https://en.wikipedia.org/wiki/Random_forest) to create multiple overfitting decision trees with high bias and low variance using a random subset of features and training data. The large number of decision trees, trained from different subsets of the same training data, each produces a certain result which is then aggregated to create a model with much less variance.
#  
#  **Let's first start with some standard parameters usually adopted for random forests:**
#  - **n_estimator = 128 (number of trees on forest = 128)**
#  - **max_depth=4 (for each tree)**
#  - **max_features = 'sqrt' (only sqrt(max features) number of features are considered when splitting)**
# 
# **First, I remove irrelevant districts and import the random forest regressor library.**
# 

# In[ ]:


#import library
from sklearn.ensemble import RandomForestRegressor

#take back up dataframe and remove irrelevant districts (D11, D12, D21 somehow made it into a list of proper address names)
data_Set = HDB_Data_Backup
districts = np.unique(data_Set['districtText'].values)
index_1 = np.argwhere(districts == 'D11')
index_2 = np.argwhere(districts == 'D12')
index_3 = np.argwhere(districts == 'D21')
districtIndex = np.concatenate((index_1, np.concatenate((index_2, index_3))))
districtIndex = np.ndarray.flatten(districtIndex).tolist()
districts = np.delete(districts, districtIndex)
data = data_Set


# **Now, I iterate through all districts in Singapore and measure the algorithm's regressor prediction accuracy for each distrct using some standard measures such as error percentage variance and R^2 prediction score (I know this is not the best measure in decision trees but hey, it doesn't hurt right?).**
# 
# **Moreover, random forests are known to give better results because of its ability to suppress noisy data. Hence, I lower the error percentage threshold to 15% and derive how much test data is giving us acceptable predictions (within the threshold).**

# In[ ]:


errorPercentageList = []
#iterate through districts
for district in districts:
    data_Set = data.loc[data['districtText'] == district]
    #remove irrelevant features similar to before
    del data_Set['districtText']
    del data_Set['bathrooms']
    del data_Set['value']
    
    #reorder columns
    col = data_Set.columns.tolist()
    col[0], col[1] = col[1], col[0]
    data_Set = data_Set.reindex(columns = col)

    Y = data_Set.values[:,0]
    X = data_Set.values[:, 1:len(col)]
    
    #define training and test data
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 50)

    #ignore districts with less than 20 listings
    if (len(X_train) < 20):
        continue
        
    #initialise random forest with the following parameters
    n_estimators = 128
    max_depth = 4
    v = 'sqrt'
    
    regTree = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_depth)
    regTree.fit(X_train, y_train)
    
    #obtain R^2 prediction score
    print(district)
    print('R^2 prediction score: ' + str(regTree.score(X_test, y_test)))

    #derive percentage error difference, total variance and mean
    y_pred = regTree.predict(X_test)
    numberOfTestData = len(y_test)
    districtError = []
    for x in range(numberOfTestData):
        percentageDiff = (y_pred[x] - y_test[x])/y_test[x]
        errorPercentageList.append(percentageDiff)
        districtError.append(percentageDiff)
    print('error percentage difference mean: ' + str(np.mean(districtError)))
    print('error percentage difference variance: ' + str(np.var(districtError)))
    threshold = 0.15
    numTest = len(districtError)
    withinThreshold = [x for x in districtError if abs(x) < threshold]
    ratioWithinThreshold = len(withinThreshold)/numTest
    print('ratio of predictions within threshold of 15%: ' + str(ratioWithinThreshold))


# **Finally, I plot the union of all district's percentage error difference as a histogram: **

# In[ ]:


bins = np.linspace(-1, 1, 50)
n, bins, patches = plt.hist(errorPercentageList,bins)
plt.show()
threshold = 0.15
numTest = len(errorPercentageList)
withinThreshold = [x for x in errorPercentageList if abs(x) < threshold]
ratioWithinThreshold = len(withinThreshold)/numTest
print('ratio of predictions within threshold of 15% for all districts combined: ' + str(ratioWithinThreshold))


# ## What now?
# 
# Now, the intuitive step for any aspiring data scientist or machine learning enthusiast now is to try to tune the hyper parameters of their machine learning model. Perhaps we can increase the number of trees in the forest! Or maybe we can adjust the depth of each tree to increase the accuracy of our predictions? How can we improve the accuracy even further?
# 
# I will not cover the tedious process of tuning hyper parameters for both KNN and random forest in this notebook because although there are systematic and measured ways to tune parameters, in reality, people tune hyperparameters through a repetitive process of iterating through all possible parameters to obtain the highest accuracy score. 
# 
# In fact, my Machine Learning professor in college literally said this on the first day of class, "Many data scientists literally spend so much time adjusting the hyperparameters until the best result is obtained. However, the improvement in accuracy sometimes is so negligible compared to the time taken to obtain them." If you are interested in how parameters can be adjusted in random forests, please take a look at the **VERY GENERAL** guide in [this website](https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/).
# 
# ### What do we do if I find the model unsatisfactory?
# 
# Perhaps it is time to look at our features - are they strongly correlated and subsequently weaken the model? are there too many features which lead to the [curse of dimensionality in KNN](https://math.stackexchange.com/questions/346775/confusion-related-to-curse-of-dimensionality-in-k-nearest-neighbor)? In the following section, I will do a brief runthrough on the methods we can use to perform** feature selection** (selecting relevant features) and** feature extraction** (creating new, lower dimensional features from existing ones).
# 

# ## Feature Selection with correlation matrix
# 
# ### How?
# Simple identify closely correlated features and remove a a subset of them to get rid of correlation between features.

# In[ ]:


data_Set = HDB_Data.loc[HDB_Data['districtText'] == ' Boon  Lay  /  Jurong  /  Tuas ']
del data_Set['districtText']
del data_Set['bathrooms']
del data_Set['value']
del data_Set['bedrooms_6']
del data_Set['pricePerSqFt']
del data_Set['latitude']
del data_Set['longitude']

plt.matshow(data_Set.corr())


# ## Feature Selection with Variance Inflation Factor
# 
# ### How?
# Derive the [VIF](https://en.wikipedia.org/wiki/Variance_inflation_factor) of each feature and identify the presence of [multi-collinearity](https://en.wikipedia.org/wiki/Multicollinearity). Remove features until linear independence between features.

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

data_Set = HDB_Data.loc[HDB_Data['districtText'] == ' Boon  Lay  /  Jurong  /  Tuas ']
del data_Set['districtText']
del data_Set['bathrooms']
del data_Set['value']
del data_Set['bedrooms_6']
del data_Set['pricePerSqFt']
del data_Set['latitude']
del data_Set['longitude']

# For each X, calculate VIF and save in dataframe. credits to https://etav.github.io/python/vif_factor_python.html,
# where I took reference the following 4 lines of code
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(data_Set.values, i) for i in range(data_Set.shape[1])]
vif["features"] = data_Set.columns
print(vif.round(1))


# **Look at all those huge values for VIF!** Clearly our features are linearly dependent. *Intuitively, why do we need to have both hasFloorplans_False and hasFloorplans_True as features since a HDB flat could only fall into either one of these categories?*Let's remove one feature from each group of similar features and see if the VIF values decrease.

# In[ ]:


#remove one features from each group of similar features to destroy linear dependence
del data_Set['furnishingCode_UNFUR']
del data_Set['bedrooms_4']
del data_Set['hasFloorplans_False']

# get VIF
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(data_Set.values, i) for i in range(data_Set.shape[1])]
vif["features"] = data_Set.columns
print(vif.round(1))


# Much better! This seems like a better set of features with no multi-collinearity.

# ## Feature Extraction with [Principle Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis)
# 
# ### How?
# Feature selection, as explained in previous sections, selects a subset of the original features via a certain criterion. *Feature extraction*, on the other hand, creates a new set of features which are linearly uncorrelated with each other. The transformation makes use of eigenvectors and eigenvalues to linearly transform the data points (Remember the chapter on linear transformation in your Linear Algebra classes?) such that they lie on a new set of axis.
# 
# To explain the meaning of the new set of features with layman terms is difficult - the new features have no intuitive meaning and do not represent anything physical. They are merely mathematical abstraction formed via [linear transformation of matrices](https://mathcs.clarku.edu/~ma130/lintrans2.pdf). If you have some prior knowledge on linear algebra, then I think the following snippet will help - *'Mathematically, if your data is captured by the matrix X, PCA is done by performing singular value decomposition (SVD) on X, or eigendecomposition (i.e. diagonalization) on the covariance matrix XTX. Both methods are equivalent. Since XTX is real and symmetric, this will give you an eigenbasis that is orthonormal. Then you select the eigenvectors with the strongest eigenvalues to be your principal components, and project all data points into the subspace spanned by these chosen eigenvectors.'*
# 
# Usually, we want PCA to reduce the dimension of features in the dataset. The final number of dimensions is one of the parameters of PCA and can be decided manually or via a variance threshold i.e how many final features is good enough. Here, we set the variance threshold to 90% i.e we want the number of final features to explain 90% of variance in data.
# 

# In[ ]:


#import PCA as decomposition library
from sklearn.decomposition.pca import PCA 
from numpy import around

#set variance threshold to 90%
var_Threshold = 90

# taking boon lay district as a sample dataset and remove longitude, Latitude
data_Set = HDB_Data.loc[HDB_Data['districtText'] == ' Boon  Lay  /  Jurong  /  Tuas ']
del data_Set['districtText']
del data_Set['bathrooms']
del data_Set['value']
del data_Set['bedrooms_6']
del data_Set['latitude']
del data_Set['longitude']
col = data_Set.columns.tolist()
col[0], col[1] = col[1], col[0]
data_Set = data_Set.reindex(columns = col)

Y = data_Set.values[:,0]
X = data_Set.values[:, 1:len(col)]

# use PCA to transform all n features to another set of n features
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 50)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA()
pca.fit(X_train)

#find which features among the new n features are the most important by looking at the explained variance
explain_Variance = around(pca.explained_variance_ratio_, decimals = 4)
explain_Variance = explain_Variance.tolist()
explain_Variance = [x * 100 for x in explain_Variance] #convert to percentage for easier viewing
print('These are the explained variance of each component (feature) created by PCA for the training data:')
print(explain_Variance)

#calculate cumulative variance and thus how many components to keep
temp=0
for x in range(len(explain_Variance)):
    explain_Variance[x] = temp + explain_Variance[x]
    temp = explain_Variance[x]
explain_VarianceCumulative = [x for x in explain_Variance if (x < 90)]
n_components = len(explain_VarianceCumulative)
print('\nNumber of components needed to reach the 90% threshold: ' + str(n_components))

#transforming the data according to the components needed
print('\nHow our data looks like after transformation:')
pca = PCA(n_components=n_components)
pca.fit_transform(X_train)
pca.transform(X_test)


# Hence, we can see that after PCA transformation, only 7 features are present in the final dataset. However, notice the values of the data cannot be explained physically because they are mathematical abstractions. **An important thing to note is that we should always transform the test data using PCA fitted from the training data.**
# 
# With the new features, we can then use normal ML methods on the new data.
