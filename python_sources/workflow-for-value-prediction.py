#!/usr/bin/env python
# coding: utf-8

# # HOUSE PRICING COMPETITION

# # Reference
# 
# * Code for finding skewed features and taking np.log1p of skewed features and SalePrice taken from Alexandru Papiu's notebook: https://www.kaggle.com/apapiu/regularized-linear-models

# In[ ]:


import os, sys
import itertools, time
import numpy as np 
import pandas as pd

# preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from pandas_profiling import ProfileReport

# postprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, explained_variance_score, mean_squared_log_error


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# # Workflow
# 
# * [Analyse data](#data)
# * [Feature engineering](#feature)
#     * Relevant and new features
#     * Clean given features
# * [Train - CV - test split](#split)
# * [Trial models](#models)

# ---
# <a name="data"></a>
# # 1) Analyse data

# * PassengerId (Numerical, Discrete) - Set as member identifier

# In[ ]:


test = pd.read_csv('../input/test.csv')
test.set_index('Id', inplace=True, drop=True)
data = pd.read_csv('../input/train.csv')
data.set_index('Id', inplace=True, drop=True)

y = data[['SalePrice']]
X = data.drop('SalePrice', axis=1)


# In[ ]:


ProfileReport(data)


# 
# *    SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# *    MSSubClass: The building class
# *    MSZoning: The general zoning classification
# *    LotFrontage: Linear feet of street connected to property
# *    LotArea: Lot size in square feet
# *    Street: Type of road access
# *    Alley: Type of alley access
# *    LotShape: General shape of property
# *    LandContour: Flatness of the property
# *    Utilities: Type of utilities available
# *    LotConfig: Lot configuration
# *    LandSlope: Slope of property
# *    Neighborhood: Physical locations within Ames city limits
# *    Condition1: Proximity to main road or railroad
# *    Condition2: Proximity to main road or railroad (if a second is present)
# *    BldgType: Type of dwelling
# *    HouseStyle: Style of dwelling
# *    OverallQual: Overall material and finish quality
# *    OverallCond: Overall condition rating
# *    YearBuilt: Original construction date
# *    YearRemodAdd: Remodel date
# *    RoofStyle: Type of roof
# *    RoofMatl: Roof material
# *    Exterior1st: Exterior covering on house
# *    Exterior2nd: Exterior covering on house (if more than one material)
# *    MasVnrType: Masonry veneer type
# *    MasVnrArea: Masonry veneer area in square feet
# *    ExterQual: Exterior material quality
# *    ExterCond: Present condition of the material on the exterior
# *    Foundation: Type of foundation
# *    BsmtQual: Height of the basement
# *    BsmtCond: General condition of the basement
# *    BsmtExposure: Walkout or garden level basement walls
# *    BsmtFinType1: Quality of basement finished area
# *    BsmtFinSF1: Type 1 finished square feet
# *    BsmtFinType2: Quality of second finished area (if present)
# *    BsmtFinSF2: Type 2 finished square feet
# *    BsmtUnfSF: Unfinished square feet of basement area
# *    TotalBsmtSF: Total square feet of basement area
# *    Heating: Type of heating
# *    HeatingQC: Heating quality and condition
# *    CentralAir: Central air conditioning
# *    Electrical: Electrical system
# *    1stFlrSF: First Floor square feet
# *    2ndFlrSF: Second floor square feet
# *    LowQualFinSF: Low quality finished square feet (all floors)
# *    GrLivArea: Above grade (ground) living area square feet
# *    BsmtFullBath: Basement full bathrooms
# *    BsmtHalfBath: Basement half bathrooms
# *    FullBath: Full bathrooms above grade
# *    HalfBath: Half baths above grade
# *    Bedroom: Number of bedrooms above basement level
# *    Kitchen: Number of kitchens
# *    KitchenQual: Kitchen quality
# *    TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# *    Functional: Home functionality rating
# *    Fireplaces: Number of fireplaces
# *    FireplaceQu: Fireplace quality
# *    GarageType: Garage location
# *    GarageYrBlt: Year garage was built
# *    GarageFinish: Interior finish of the garage
# *    GarageCars: Size of garage in car capacity
# *    GarageArea: Size of garage in square feet
# *    GarageQual: Garage quality
# *    GarageCond: Garage condition
# *    PavedDrive: Paved driveway
# *    WoodDeckSF: Wood deck area in square feet
# *    OpenPorchSF: Open porch area in square feet
# *    EnclosedPorch: Enclosed porch area in square feet
# *    3SsnPorch: Three season porch area in square feet
# *    ScreenPorch: Screen porch area in square feet
# *    PoolArea: Pool area in square feet
# *    PoolQC: Pool quality
# *    Fence: Fence quality
# *    MiscFeature: Miscellaneous feature not covered in other categories
# *    MiscVal: Value of miscellaneous feature
# *    MoSold: Month Sold
# *    YrSold: Year Sold
# *    SaleType: Type of sale
# *    SaleCondition: Condition of sale
# 

# In[ ]:


numerical = X._get_numeric_data().columns
categorical = X.columns.drop(numerical)


# ---
# <a name="feature"></a>
# # 2) Feature engineering

# In[ ]:


# Transformations must be applied to both training and testing set.
Xtot = pd.concat((X, test))
len(X), len(test), len(Xtot)


# In[ ]:


# Empty dataframe for building features
X_eng = pd.DataFrame()


# ## Numerical
# * Include all numerical columns in dataframe
# * Replace missing values with mean
# * Feature scaling
# * Take log of all values to manage skewness

# In[ ]:


# Find skewed features
from scipy.stats import skew

skewed_feats = Xtot[numerical].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index


# In[ ]:


for header in numerical:
    # Set nan values to mean
    column = Xtot[header].copy()
    colmean = np.mean(column)
    colnan = np.isnan(column)
    column[colnan] = colmean
    
    # Take log1p of skewed features
    if header in skewed_feats:
        column = np.log1p(column)
        header = header+'_log1p'
    
    # Feature scaling
    colstd = np.std(column)
    column = (column - colmean)/colstd
    
    # Set column in engineered dataframe
    X_eng[header] = column


# ## Categorical
# * Collect rare values
# * One Hot encode

# In[ ]:


def ohe_cols(column, label='', index='index'):
    
    # Combine rare values (values with counts less than threshold)
    threshold = 0.025
    unique_values = column.unique()
    for value in unique_values:
        if (np.sum(column==value)/len(column))<threshold:
            column[column==value] = 'rare'

    # Encode values into integers
    label_encoder = LabelEncoder()
    try:integer_encoded = label_encoder.fit_transform(column)
    except TypeError:
        integer_encoded = label_encoder.fit_transform(column.astype(str))
    headers = label_encoder.classes_.astype(str)
    
    if len(headers)>2:
        # Encode integers into onehot
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        onehot_encoded

        headers = [str(label)+"_"+x for x in headers]
    
        df = pd.DataFrame(onehot_encoded, columns=headers, index=column.index)
        
    else:
        df = pd.DataFrame(integer_encoded, columns=[str(label)+'_binary'], index=column.index)
    
    return df


# In[ ]:


for header in categorical:
    # One hot encode data
    column  = Xtot[header].copy()
    ohe_column = ohe_cols(column, label=header)
    
    # Merge data into dataframe
    X_eng = pd.merge(X_eng, ohe_column, left_index=True, right_index=True)


# # Target
# * log transform the target

# In[ ]:


log1p_y = np.log1p(y)


# ---
# <a name='split'></a>
# # 3) Train - CV - Test split

# In[ ]:


# Recover input data and test data
Xinput = X_eng[:len(X)].copy()
test = X_eng[len(X):].copy()

# Train test split - 20% testing
Xtrain, Xtest, ytrain, ytest = train_test_split(Xinput, y, test_size=0.2, random_state=15)

# Train cross-validation split - overall 20% cross-validation
Xtrain, Xcv, ytrain, ycv = train_test_split(Xtrain, ytrain, test_size=(0.2/0.8), random_state=15)

# So we have a 60-20-20 train-cv-test split
len(Xtrain), len(Xcv), len(Xtest), len(test)


# In[ ]:





# ------
# ------
# <a name='models'></a>
# # 4) Run models

# In[ ]:


# machine learning
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor


# In[ ]:


models = {'svm-SVR': SVR,
          'svm-LinearSVR': LinearSVR, 
          'svm-NuSVR': NuSVR,
          'RandomForest': RandomForestRegressor,
          'KNeighbors': KNeighborsRegressor,
          'DecisionTree': DecisionTreeRegressor, 
          'ExtraTree': ExtraTreeRegressor,
          'GaussianProcess': GaussianProcessRegressor}


# In[ ]:


def RUN(X_train, y_train, X_test, y_test, model, cm=False, label='', target=''):

    """
    Fits model to X_train and y_train
    Predicts targets for X_test
    Provides metrics for prediction success of y_test
    """
    
    start = time.time()
    
    print(label)
    
    model.fit(X_train, np.array(y_train[target]))
    
    pred = model.predict(X_test)
    y_test = np.array(y_test[target])
    
    now = time.time()
    timetaken = now-start
    
    acc = explained_variance_score(y_test, pred)
    error = np.sqrt(mean_squared_log_error(y_test, pred))
    print("Accuracy: %f, Error: %f, time: %.3f" % (acc, error, timetaken))
    
    if cm:
        cm = confusion_matrix(y_test, pred)
        plot_confusion_matrix(cm, np.arange(1), normalize=True)
        


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# # Find most successful models for log1p SalePrice
# * 'Error' is the root mean square log error for log 1p
# * svm-NuSVR generates the lowest error

# In[ ]:


for value in models:
    _=RUN(Xtrain, np.log1p(ytrain), Xcv, np.log1p(ycv), models[value](), label=value, target='SalePrice')


# In[ ]:


# Use random forest to find the most important features

forest = RandomForestRegressor()

forest.fit(Xtrain, np.log1p(np.array(ytrain)[:,0]))
    
pred = forest.predict(Xcv)
y_test = np.array(np.log1p(ycv.SalePrice))
    
acc = explained_variance_score(y_test, pred)
error = np.sqrt(mean_squared_log_error(y_test, pred))
print("Accuracy: %f, Error: %f" % (acc, error))
    
importances = forest.feature_importances_

featureimportance = np.vstack((Xtrain.columns.values, importances))
featureimportance = pd.DataFrame(featureimportance.T, columns=['Feature', 'Importance'])

plt.figure(figsize=(20,10))
_=plt.bar(Xtrain.columns.values, importances)
_=plt.xticks(rotation=45, fontsize=10)


# In[ ]:


featureimportance.sort_values('Importance', ascending=False)


# # Optimise parameters for NuSVR which provide best performance on cross-validation set
# * 'Error' root mean square logarithmic error on SalePrice (not log SalePrice now)
# * Optimal parameters: nu=0.5, C=0.01, kernel='linear'

# In[ ]:


clf = NuSVR()

clf.fit(Xtrain, np.log1p(np.array(ytrain)[:,0]))
    
pred = clf.predict(Xcv)
y_test = np.log1p(np.array(ycv.SalePrice))
    
acc = explained_variance_score(y_test, pred)
error = np.sqrt(mean_squared_log_error(y_test, pred))

error = np.sqrt(mean_squared_log_error(np.exp(y_test)-1, np.exp(pred)-1))
print("Accuracy: %f, Error: %f" % (acc, error))


# In[ ]:


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in np.linspace(0.001, 0.1, 10):
    clf = NuSVR(nu=0.5, C=i, kernel='linear')

    clf.fit(Xtrain, np.log1p(np.array(ytrain)[:,0]))

    pred = clf.predict(Xcv)
    y_test = np.log1p(np.array(ycv.SalePrice))

    acc = explained_variance_score(y_test, pred)
    error = np.sqrt(mean_squared_log_error(np.exp(y_test)-1, np.exp(pred)-1))
    print("Accuracy: %f, Error: %f, Parameter: %s" % (acc, error, i))


# # Drop lowest importance features
# * NuSVR handles high dimensionality ok so we don't need to reduce the number of features
# * RandomForest Error doesn't reduce when removing features therefore we continue to calculate based on full feature set.

# In[ ]:


Xnew = Xtrain.copy()
Xcv_new = Xcv.copy()
Xtest_new = Xtest.copy()
test_new = test.copy()

threshold = 0.005
min_importance = 0.

while min_importance<threshold:

    forest = RandomForestRegressor()
    forest.fit(Xnew, np.array(ytrain)[:,0])

    pred = forest.predict(Xcv_new)
    y_test = np.array(ycv.SalePrice)

    acc = explained_variance_score(y_test, pred)
    error = np.sqrt(mean_squared_log_error(y_test, pred))
    print("Accuracy: %f, Error: %f" % (acc, error))

    importances = forest.feature_importances_

    min_importance = np.min(importances)
    
    if min_importance<threshold:
        col = Xnew.columns.values[importances == min_importance][0]
        print(col)
        
        Xnew.drop(col, inplace=True, axis=1)
        Xcv_new.drop(col, inplace=True, axis=1)
        Xtest_new.drop(col, inplace=True, axis=1)
        test_new.drop(col, inplace=True, axis=1)


# In[ ]:


forest = RandomForestRegressor()

forest.fit(Xnew, np.array(ytrain)[:,0])
    
pred = forest.predict(Xcv_new)
y_test = np.array(ycv.SalePrice)
    
acc = explained_variance_score(ycv, pred)
error = np.sqrt(mean_squared_log_error(ycv, pred))
print("Accuracy: %f, Error: %f" % (acc, error))
    
importances = forest.feature_importances_

plt.figure(figsize=(20,10))
_=plt.bar(Xnew.columns.values, importances)
_=plt.xticks(rotation=45, fontsize=15)


# In[ ]:


for value in models:
    _=RUN(Xnew, ytrain, Xcv_new, ycv, models[value](), label=value, target='SalePrice')


# In[ ]:





# # Test and submission
# * Submission1 - Random Forest - Expect a rms log error of ~0.15
# * Submission2 - NuSVR - Expect a rms log error of ~0.12

# ## Random Forest

# In[ ]:


# Running the Regressor on the test sample tells us what we expect the error/accuracy to be.

clf = RandomForestRegressor()

clf.fit(Xtrain, np.array(ytrain)[:,0])
    
pred = clf.predict(Xtest)
y_test = np.array(ytest.SalePrice)
    
acc = explained_variance_score(y_test, pred)
error = np.sqrt(mean_squared_log_error(y_test, pred))
print("Accuracy: %f, Error: %f" % (acc, error))


# In[ ]:


pred = clf.predict(test)
pi = test.index.values.astype(int)

prediction = pd.DataFrame(np.vstack((pi, pred)).T, columns=['Id', 'SalePrice'])
prediction.Id = prediction.Id.astype(int)
prediction


# ## NuSVR

# In[ ]:


clf = NuSVR(nu=0.5, C=0.01, kernel='linear')

clf.fit(Xtrain, np.log1p(np.array(ytrain)[:,0]))

pred = clf.predict(Xtest)
y_test = np.log1p(np.array(ytest.SalePrice))

acc = explained_variance_score(np.exp(y_test)-1, np.exp(pred)-1)
error = np.sqrt(mean_squared_log_error(np.exp(y_test)-1, np.exp(pred)-1))
print("Accuracy: %f, Error: %f, Parameter: %s" % (acc, error, i))


# In[ ]:


pred = clf.predict(test)
pred = np.exp(pred)-1
pi = test.index.values
prediction = pd.DataFrame(np.vstack((pi, pred)).T, columns=['Id', 'SalePrice'])
prediction.Id = prediction.Id.astype(int)
prediction

