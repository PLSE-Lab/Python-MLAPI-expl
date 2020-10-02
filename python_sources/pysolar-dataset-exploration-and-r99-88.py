#!/usr/bin/env python
# coding: utf-8

# ## Solar Radiation Prediction
# 
# > meteorological data from the HI-SEAS weather station from four months (September through December 2016) between Mission IV and Mission V.
# 
# Units:
# 
# * Solar radiation: watts per meter^2
# * Temperature: degrees Fahrenheit
# * Humidity: percent
# * Barometric pressure: Hg
# * Wind direction: degrees
# * Wind speed: miles per hour
# * Sunrise/sunset: Hawaii time

# ### Useful imports and read the data

# In[ ]:


import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb


# In[ ]:


# Read the data
df = pd.read_csv('../input/SolarPrediction.csv', parse_dates=['Data'])
df.head()


# In[ ]:


df.describe()


# ### Feature Engineering

# In[ ]:


# Convert all dates and times to unix timestamp (timezone doesn't matter now)
df['Data'] = df['Data'].dt.date.astype(str)
df['TimeSunRise'] = df['Data'] + ' ' + df['TimeSunRise']
df['TimeSunSet'] = df['Data'] + ' ' + df['TimeSunSet']
df['Data'] = df['Data'] + ' ' + df['Time']

# Convert to Unix timestamp
fields = ['Data', 'TimeSunRise', 'TimeSunSet']
for x in fields:
    df[x + '_UnixTimeStamp'] = df[x].apply(
        lambda k: int(datetime.strptime(k, "%Y-%m-%d %H:%M:%S").timestamp())
    )

# New sun time field
df['SunTime'] = df['TimeSunSet_UnixTimeStamp'] - df['TimeSunRise_UnixTimeStamp']

# Drop old columns
df.drop('UNIXTime', axis=1, inplace=True)
df.drop('Data', axis=1, inplace=True)
df.drop('Time', axis=1, inplace=True)
df.drop('TimeSunRise', axis=1, inplace=True)
df.drop('TimeSunSet', axis=1, inplace=True)


# ### Visualization

# In[ ]:


def dddraw(X_reduced,name):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,cmap=plt.cm.Paired)
    titel="First three directions of "+name 
    ax.set_title(titel)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()


# In[ ]:


from sklearn.decomposition import PCA, FastICA,SparsePCA,NMF, LatentDirichletAllocation,FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.cluster import KMeans,Birch
import statsmodels.formula.api as sm
from scipy import linalg
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
import matplotlib.pyplot as plt

n_col=12
X = df.drop('Radiation',axis=1) # we only take the first two features.

def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
def procenterror(y_predicted, y_real):
     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)

    
Y=df['Radiation']
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
poly = PolynomialFeatures(2)
X=poly.fit_transform(X)



names = [
         'PCA',
         'FastICA',
         'Gauss',
         'KMeans',
         #'SparsePCA',
         #'SparseRP',
         #'Birch',
         'NMF',    
         #'LatentDietrich',    
        ]

classifiers = [
    
    PCA(n_components=n_col),
    FastICA(n_components=n_col),
    GaussianRandomProjection(n_components=3),
    KMeans(n_clusters=24),
    #SparsePCA(n_components=n_col),
    #SparseRandomProjection(n_components=n_col, dense_output=True),
    #Birch(branching_factor=10, n_clusters=12, threshold=0.5),
    NMF(n_components=n_col),    
    #LatentDirichletAllocation(n_topics=n_col),
    
]
correction= [1,1,0,0,0,0,0,0,0]

temp=zip(names,classifiers,correction)
print(temp)

for name, clf,correct in temp:
    Xr=clf.fit_transform(X,Y)
    dddraw(Xr,name)
    res = sm.OLS(Y,Xr).fit()
    #print(res.summary())  # show OLS regression
    #print(res.predict(Xr).round()+correct)  #show OLS prediction
    #print('Ypredict',res.predict(Xr).round()+correct)  #show OLS prediction

    #print('Ypredict *log_sec',res.predict(Xr).round()+correct*Y.mean())  #show OLS prediction
    print(name,'%error',procenterror(res.predict(Xr)+correct*Y.mean(),Y),'rmsle',rmsle(res.predict(Xr)+correct*Y.mean(),Y))


# In[ ]:


from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier
from sklearn.preprocessing import MinMaxScaler

# import some data to play with
       # those ? converted to NAN are bothering me abit...        

from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

X = df.drop('Radiation',axis=1) # we only take the first two features.
le = preprocessing.LabelEncoder()
def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
def procenterror(y_predicted, y_real):
     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)

    
Y=np.round(np.log(df['Radiation'])*10)
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
poly = PolynomialFeatures(2)
X=poly.fit_transform(X)


names = [
         #'ElasticNet',
         #'SVC',
         #'kSVC',
         'KNN',
         'DecisionTree',
         'RandomForestClassifier',
         #'GridSearchCV',
         #'HuberRegressor',
         #'Ridge',
         #'Lasso',
         #'LassoCV',
         #'Lars',
         #'BayesianRidge',
         #'SGDClassifier',
         #'RidgeClassifier',
         #'LogisticRegression',
         #'OrthogonalMatchingPursuit',
         #'RANSACRegressor',
         ]

classifiers = [
    #ElasticNetCV(cv=10, random_state=0),
    #SVC(),
    #SVC(kernel = 'rbf', random_state = 0),
    KNeighborsClassifier(n_neighbors = 1),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators = 200),
    #GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),
    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,epsilon=2.95),
    Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),
    Lasso(alpha=0.05),
    LassoCV(),
    Lars(n_nonzero_coefs=10),
    BayesianRidge(),
    SGDClassifier(),
    RidgeClassifier(),
    LogisticRegression(),
    OrthogonalMatchingPursuit(),
    #RANSACRegressor(),
]
correction= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

temp=zip(names,classifiers,correction)
print(temp)

for name, clf,correct in temp:
    regr=clf.fit(X,Y)
    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)
    print(name,'%error',procenterror(regr.predict(X),Y),'rmsle',rmsle(regr.predict(X),Y))
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score

    # Confusion Matrix
    print(name,'Confusion Matrix')
    print(confusion_matrix(Y, np.round(regr.predict(X) ) ) )
    print('--'*40)

    # Classification Report
    print('Classification Report')
    print(classification_report(Y,np.round( regr.predict(X) ) ))

    # Accuracy
    print('--'*40)
    logreg_accuracy = round(accuracy_score(Y, np.round( regr.predict(X) ) ) * 100,2)
    print('Accuracy', logreg_accuracy,'%')
    df[name]=regr.predict(X)


# ### Model train

# In[ ]:


# Create the K-folds
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle = True)

# Prepare dataset
X = df.drop(['Radiation','Data_UnixTimeStamp','TimeSunRise_UnixTimeStamp','TimeSunSet_UnixTimeStamp'] , axis=1).as_matrix()
y = df['Radiation'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# ### XGBoost

# In[ ]:


xgb_params = {
    'n_trees': 50, 
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test)


# In[ ]:


cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=50,
    verbose_eval=200, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
plt.show()


# In[ ]:


num_boost_rounds = len(cv_output)
print(num_boost_rounds)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)


# In[ ]:


from sklearn.metrics import r2_score
print("R^2 in training: %s"  % r2_score(dtrain.get_label(), model.predict(dtrain)))
print("R^2 in testing: %s"  % r2_score(y_test, model.predict(dtest)))

