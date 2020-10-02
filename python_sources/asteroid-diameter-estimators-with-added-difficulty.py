#!/usr/bin/env python
# coding: utf-8

# *  **Goal:** Getting the diameter of an asteroid from the other data given about that asteroid, in other words supervised regression with the target being $\log(diameter)$.   
# The metric used during testing will be the $R^2$ score.
# * **How:**
#     * Step 1: Cleaning and preparing the data
#         * a. I cleared the samples with nan diameter.
#         * b. I dropped the features with more than half nan values.
#         * c. The dataframe had a lot of nan values, I chose to substitute them with the average value for the corresponding feature.
#     * Step 2: Train-test splitting of the data, and then normalizing (standard) using the test dataframe. 
#     * Step 3: Trying different regression algorithms (Linear Regression, Elastic Net, Decision Tree, Random Forest, XGBoost, SVM, Neural Network) and picking the best one.
# * **Extra:**  Since NASA's own estimator uses $H$ and $albedo$ to calculate the diameter, I decided to drop those to add difficulty and have less linearity in problem to solve.
# 
# 
# **Dependencies: numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, keras.**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import seed 
seed(42)


# ## Part I: Importing, exploration & cleaning of the data

# In[ ]:


#Importing only the first 30000 rows
df = pd.read_csv('/kaggle/input/prediction-of-asteroid-diameter/Asteroid.csv',nrows = 30000)


# In[ ]:


df.head()


# **There are lots of feature columns to check:**   
# First I wanted to know if there are NaN values (there are and will be dealt with later)

# In[ ]:


#Checking which columns(features) have nan values
for column in df.columns:
    print(column, df[column].isna().sum()/df.shape[0]) #returns the fraction of NAN values


# Next thing is to understand what type of data we're dealing with

# In[ ]:


#Printing the first ten unique values of each feature
for column in df.columns:
    print(column, df[column].unique()[:10])


# * **Cleaning and prepping the dataframe:**  
# **Steps:**
#     * 0/ 'diameter' is string type, I will convert to numeric. This gave errors for some diameters because they were corrupted, so I added the argument "errors='coerce'" to set corrupted diameters to nan, and later dropped those.
#     * 1/ Dropping irrelevent features and choosing my battles:
#         * 1a/ dropping names because I dont believe asteroids are named according to their diameter.
#         * 1b/ Dropping all features with more than half nan values
#         * 1c/ dropping condition_code and neo and pha because most seems to be 0 or nan.
#     * 2/ Replace nans entries with mean value of column

# In[ ]:


#Steps 0
df['diameter']=pd.to_numeric(df['diameter'],errors='coerce') #transforming to numeric, setting errors to NaN
dropindexes = df['diameter'][df['diameter'].isnull()].index #rows with nan diameters to drop
dropped_df = df.loc[dropindexes] #saving dropped rows for the future
df = df.drop(dropindexes, axis=0) 


# In[ ]:


#Steps 1
tooMuchNa = df.columns[df.isna().sum()/df.shape[0] > 0.5]
df = df.drop(tooMuchNa,axis=1)
df = df.drop(['condition_code','full_name'],axis=1)
df = df.drop(['neo','pha'],axis=1)


# In[ ]:


#Step 2
df = df.fillna(df.mean())


# In[ ]:


df.head()


# In[ ]:


#Last sanity check for nan values
df.isna().values.any()


# Nasa's own Asteroid Size Estimator website uses the $H$ value and $\log(albedo)$: https://cneos.jpl.nasa.gov/tools/ast_size_est.html so I decided to drop those features for curiosity and added difficulty (regression scores went down from 0.98 to 0.8)

# In[ ]:


df = df.drop(['albedo','H'],axis = 1)


# Since a lot of values in physics are more relevant when you consider their log, I'll add columns to the dataframe corresponding to the log of the original columns:

# In[ ]:


df['diameter']= df['diameter'].apply(np.log)
for column in df.columns.drop(['diameter']):
    df['log('+column+')']=df[column].apply(np.log)
df = df.dropna(axis=1)


# Correlation analysis:

# In[ ]:


df.corr()['diameter'].abs().sort_values(ascending=False)


# ## Part II: Splitting the dataframe into train and test dataframes and normalizing them for our regressions.

# Splitting:

# In[ ]:


from sklearn.model_selection import train_test_split
predictors = df.drop('diameter',axis=1) 
target = df['diameter']
X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


# In[ ]:


X_train.head()


# Normalization:

# In[ ]:


from sklearn import preprocessing

#Input standard normalization:
std_scaler = preprocessing.StandardScaler().fit(X_train)

def scaler(X):
    x_norm_arr= std_scaler.fit_transform(X)
    return pd.DataFrame(x_norm_arr, columns=X.columns, index = X.index)

X_train_norm = scaler(X_train)
X_test_norm = scaler(X_test)

def inverse_scaler(X):
    x_norm_arr= std_scaler.inverse_transform(X)
    return pd.DataFrame(x_norm_arr, columns=X.columns, index = X.index)


# ## Part III:  Trying different regressions and ranking them according to their $R^2$ score.
# **Algorithms used:** Linear Regression, Elastic Net, k-Nearest Neighbours, Decision Tree, Random Forest, SVM, Neural Network and XGBoost. 

# In[ ]:


from sklearn.metrics import r2_score
import seaborn as sns

def plot(prediction):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,7)) 
    sns.distplot(Y_test.values,label='test values', ax=ax1)
    sns.distplot(prediction ,label='prediction', ax=ax1)
    ax1.set_xlabel('Distribution plot')
    ax2.scatter(Y_test,prediction, c='orange',label='predictions')
    ax2.plot(Y_test,Y_test,c='blue',label='y=x')
    ax2.set_xlabel('test value')
    ax2.set_ylabel('estimated $\log(radius)$')
    ax1.legend()
    ax2.legend()
    ax2.axis('scaled') #same x y scale
def score(prediction):
    score = r2_score(prediction,Y_test)
    return score
def announce(score):
    print('The R^2 score achieved using this regression is:', round(score,3))
algorithms = []
scores = []


# Linear Regression:

# In[ ]:


#Defining the model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

###Training
lr.fit(X_train,Y_train)

###Predicting
Y_pred_lr = lr.predict(X_test)

###Scoring
score_lr = score(Y_pred_lr)
announce(score_lr)

algorithms.append('LR')
scores.append(score_lr)


# In[ ]:


plot(Y_pred_lr)


# Elastic Net regression:

# In[ ]:


### Defining the Model
from sklearn.linear_model import ElasticNetCV
enet = ElasticNetCV(cv=9,max_iter=10000)

### Training
enet.fit(X_train_norm,np.ravel(Y_train))

### Predicting
Y_pred_enet = enet.predict(X_test_norm)

###Scoring
score_enet = score(Y_pred_enet)
announce(score_enet)

algorithms.append('eNet')
scores.append(score_enet)


# In[ ]:


plot(Y_pred_enet)


# k-Nearest Neighbours regression:

# In[ ]:


### Defining the Model

from sklearn.neighbors import KNeighborsRegressor

##For weighted metric, more accurate but longer calculation
#weights = X_train_norm.corrwith(Y_train).abs()
#neigh = KNeighborsRegressor(n_neighbors=3, metric_params={'w' : weights.values}, metric='wminkowski')

neigh = KNeighborsRegressor(n_neighbors=3)

### Training
neigh.fit(X_train_norm,Y_train)

### Predicting 
Y_pred_neigh = neigh.predict(X_test_norm)

### Scoring
score_neigh=score(Y_pred_neigh)
announce(score_neigh)

algorithms.append('k-NN')
scores.append(score_neigh)


# In[ ]:


plot(Y_pred_neigh)


# Decision Tree regression:

# In[ ]:


### Defining the model
from sklearn import tree
decTree = tree.DecisionTreeRegressor()

### Training
decTree = decTree.fit(X_train_norm,Y_train)

### Predicting
Y_pred_tree = decTree.predict(X_test_norm)

### Scoring
score_tree = score(Y_pred_tree)
announce(score_tree)

algorithms.append('DTree')
scores.append(score_tree)


# In[ ]:


plot(Y_pred_tree)


# Random Forest regression:

# In[ ]:


### Defining the model
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(max_depth=32, n_estimators=50)

### Training 
forest.fit(X_train_norm,np.ravel(Y_train))

###Predicting
Y_pred_forest = forest.predict(X_test_norm)

### Scoring
score_forest = score(Y_pred_forest)
announce(score_forest)

algorithms.append('RForest')
scores.append(score_forest)


# In[ ]:


plot(Y_pred_forest)


# Support Vector Machine regression:
# (too slow when n_samples > 30000)

# In[ ]:


### Defining the model
from sklearn import svm
svmreg = svm.SVR()

### Training
svmreg.fit(X_train_norm,np.ravel(Y_train))

### Predicting
Y_pred_svm = svmreg.predict(X_test_norm)

### Scoring
score_svm = score(Y_pred_svm)
announce(score_svm)

algorithms.append('SVM')
scores.append(score_svm)


# In[ ]:


plot(Y_pred_svm)


# Neural Network regression:

# In[ ]:


### Defining the model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

Adam(learning_rate=0.005)
model = Sequential()
model.add(Dense(24,activation='tanh',input_dim=X_train_norm.shape[1]))
model.add(Dense(12,activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')

### Training

model.fit(X_train_norm,Y_train,epochs=100,batch_size=256,verbose=False)

### Predicting

Y_pred_nn = model.predict(X_test_norm)

### Scoring
score_nn = score(Y_pred_nn)
announce(score_nn)

algorithms.append('NNet')
scores.append(score_nn)


# In[ ]:


plot(Y_pred_nn)


# XGBoost regression:

# In[ ]:


### Defining the model
import xgboost as xgb 
xgReg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, 
                         learning_rate = 0.08 ,
                max_depth = 4, n_estimators = 500)

### Training
xgReg.fit(X_train_norm,Y_train)

### Predicting
Y_pred_xgb = xgReg.predict(X_test_norm)

### Scoring
score_xgb = score(Y_pred_xgb)
announce(score_xgb)

algorithms.append('XGB')
scores.append(score_xgb)


# In[ ]:


plot(Y_pred_xgb)


# In[ ]:


# One bonus of using xgboost is being able to 
# simply see how important the different features where when creating the learners.

fig, ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(xgReg, height=0.5, ax=ax, importance_type='weight')
plt.show()


# Comparing all regression algorithms

# In[ ]:


sns.barplot(algorithms,scores)


# XGBoosting wins today with an $R^2$ score of 0.845

# Thank you for reading !
