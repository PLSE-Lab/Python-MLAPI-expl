#!/usr/bin/env python
# coding: utf-8

# # Kaggle Competition: House Price Regression
# For this competiton, we are given a data set of 1,460 homes, each with a few dozen features of types: float, integer, and categorical. We are tasked with building a regression model to estimate a home's sale price. Since this is my first kaggle competition, and I'm still quite new to machine learning techniques, I'm going to use this problem as a way to explore common classifiers, namely:
# 
# * random trees and
# * random forests.
# 
# I appreciate any feedback! 
# 
# Let's get started by importing some libraries and getting the training data...

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # some plotting!
import seaborn as sns # so pretty!
from scipy import stats # I might use this
from sklearn.ensemble import RandomForestClassifier # checking if this is available
# from sklearn import cross_validation
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# import the training data set and make sure it's in correctly...
train = pd.read_csv('../input/train.csv')
train_original = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()


# # Feature First Impressions
# It looks like we have integer, float, and object (categorical) features. Also, it looks like some of the features only pertain to a small portion of the 1,460 samples. For now, let's ignore those features where data is missing.
# 
# ## Pre-processing Categorical Features
# Let's declare a quick function to convert categorical features into integer features, with the most common category of the feature being converted to integer 0, the next most common to 1, and so on. This may be useful later.

# In[ ]:


# define a function to convert an object (categorical) feature into an int feature
# 0 = most common category, highest int = least common.
def getObjectFeature(df, col, datalength=1460):
    if df[col].dtype!='object': # if it's not categorical..
        print('feature',col,'is not an object feature.')
        return df
    elif len([i for i in df[col].T.notnull() if i == True])!=datalength: # if there's missing data..
        print('feature',col,'is missing data.')
        return df
    else:
        df1 = df
        counts = df1[col].value_counts() # get the counts for each label for the feature
        df1[col] = [counts.index.tolist().index(i) for i in df1[col]] # do the conversion
        return df1 # make the new (integer) column from the conversion
# and test the function...
fcntest = getObjectFeature(train,'LotShape')
fcntest.head(10)


# # Target Variable Analysis: Is it Normal?
# 
# *This section was added after my first submission*
# 
# After going through a few other kernels on this problem to learn from the masters, I realized that checking for target variable normality (and enforcing normality through a transform) is helpful in developing accurate regression models. Namely, I want to cite [Marcelino's](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python) and [Serigne's](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard) kernels, from which I learned a ton.
# 
# So let's take a look at the Sale Price data and check for normality, and try to correct it otherwise...

# In[ ]:


#histogram and normal probability plot
from scipy.stats import norm
sns.distplot(train['SalePrice'],fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


# So, certainly not normal: we have right-skewness and the data is a bit peak-y. Let's apply a log transform on the data and see what happens...

# In[ ]:


train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice'],fit=norm);
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


# ## A Better Fit!
# 
# That looks much more normal, which will hopefully improve the regressions. We just have to remember to transform the output data back using an exponentiation before we submit anything.

# # First things first: A Random Tree Regressor
# To start off, let's try to train a simple model using ONLY the features on the "benchmark" solution provided with the data for this competition. Those features are:
# * Year and month of sale,
# * Lot square footage, and
# *  Number of bedrooms.
# 
# We will go for a very simple decision tree regression first. We can test for performance and overfitting using k-fold validation; here we take $k=10$. First, we take the data and make it useful...
# 

# In[ ]:


from sklearn.tree import DecisionTreeRegressor as dtr
# define the training data X...
X = train[['MoSold','YrSold','LotArea','BedroomAbvGr']]
Y = train[['SalePrice']]
# and the data for the competition submission...
X_test = test[['MoSold','YrSold','LotArea','BedroomAbvGr']]
print(X.head())
print(Y.head())


# ... and now we can use cross validation to see how well a proposed regression model performs. 
# 
# ## Explained Variance as a Performance Metric
# For now, we use explained variance, $EV$, as a metric to evaluate the performance of a model:
# 
# $EV = 1 - \frac{Var(y-\bar{y})}{Var(y)}$
# 
# where $y$ is the true price, $\bar{y}$ is the estimated price from the model, and $Var(\cdot)$ is the variance. The $\bar{y}$ estimates come from predictions made on the data witheld from training in each round of cross-validation. See: 
# http://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score 
# 
# Let's apply this metric...

# In[ ]:


# let's set up some cross-validation analysis to evaluate our model and later models...
from sklearn.model_selection import cross_val_score
# try fitting a decision tree regression model...
DTR_1 = dtr(max_depth=None) # declare the regression model form. Let the depth be default.
# DTR_1.fit(X,Y) # fit the training data
scores_dtr = cross_val_score(DTR_1, X, Y, cv=10,scoring='explained_variance') # 10-fold cross validation
print('scores for k=10 fold validation:',scores_dtr)
print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_dtr.mean(), scores_dtr.std() * 2))


# ## The Random Tree Regressor: A Terrible Model
# Wow, that's.... super bad. For explained variance, the best possible result is 1, which would correspond to $Var(y-y_{est})=0$. Values below 1 indicate error in the regression. Negative values imply $Var(y-\bar{y}) > Var(y)$, which is frankly embarrasing.
# 
# # Seeing the Random Forest for the Trees
# 
# So, using one tree is a bad idea... but what if we consider an ensemble of trees? Let's use a random forest regressor instead. We will consider forests with varying numbers of trees (estimators), each of which provides a weak regression solution that can be averaged to get the overall regression output. See: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# In[ ]:


from sklearn.ensemble import RandomForestRegressor as rfr
estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
mean_rfrs = []
std_rfrs_upper = []
std_rfrs_lower = []
yt = [i for i in Y['SalePrice']] # quick pre-processing of the target
np.random.seed(11111)
for i in estimators:
    model = rfr(n_estimators=i,max_depth=None)
    scores_rfr = cross_val_score(model,X,yt,cv=10,scoring='explained_variance')
    print('estimators:',i)
#     print('explained variance scores for k=10 fold validation:',scores_rfr)
    print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
    print('')
    mean_rfrs.append(scores_rfr.mean())
    std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting
    std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting


# In[ ]:


# and plot...
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(estimators,mean_rfrs,marker='o',
       linewidth=4,markersize=12)
ax.fill_between(estimators,std_rfrs_lower,std_rfrs_upper,
                facecolor='green',alpha=0.3,interpolate=True)
ax.set_ylim([-.3,1])
ax.set_xlim([0,80])
plt.title('Expected Variance of Random Forest Regressor')
plt.ylabel('Expected Variance')
plt.xlabel('Trees in Forest')
plt.grid()
plt.show()


# ## Random Forests: A Slight Improvement
# 
# Yeah, the results are still absolutely awful. But, at least the estimated means for explained variance are positive, which is a small improvement. We probably need more features, considering how poor even heavily populated forests perform. Let's start by adding a few more features and seeing what happens...

# In[ ]:


# list all the features we want. This is still arbitrary...
included_features = ['MoSold','YrSold','LotArea','BedroomAbvGr', # original data
                    'FullBath','HalfBath','TotRmsAbvGrd', # bathrooms and total rooms
                    'YearBuilt','YearRemodAdd', # age of the house
                    'LotShape','Utilities'] # some categoricals 
# define the training data X...
X = train[included_features]
Y = train[['SalePrice']]
# and the data for the competition submission...
X_test = test[included_features]
# transform categorical data if included in X...
for col in list(X):
    if X[col].dtype=='object':
        X = getObjectFeature(X, col)
X.head()


# In[ ]:


# define the number of estimators to consider
estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
mean_rfrs = []
std_rfrs_upper = []
std_rfrs_lower = []
yt = [i for i in Y['SalePrice']]
np.random.seed(11111)
# for each number of estimators, fit the model and find the results for 8-fold cross validation
for i in estimators:
    model = rfr(n_estimators=i,max_depth=None)
    scores_rfr = cross_val_score(model,X,yt,cv=10,scoring='explained_variance')
    print('estimators:',i)
#     print('explained variance scores for k=10 fold validation:',scores_rfr)
    print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
    print("")
    mean_rfrs.append(scores_rfr.mean())
    std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting
    std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting


# In[ ]:


# and plot...
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(estimators,mean_rfrs,marker='o',
       linewidth=4,markersize=12)
ax.fill_between(estimators,std_rfrs_lower,std_rfrs_upper,
                facecolor='green',alpha=0.3,interpolate=True)
ax.set_ylim([-.2,1])
ax.set_xlim([0,80])
plt.title('Expected Variance of Random Forest Regressor')
plt.ylabel('Expected Variance')
plt.xlabel('Trees in Forest')
plt.grid()
plt.show()


# Clearly better, but still pretty bad. At least we are moving in the right direction (towards expected variance of 1). 
# 
# # Scientific-ish Feature Analysis to Improve Random Forest Regressors
# Let's stick with random forest regression for now, but let's try to be more scientific about the features we select for training the forests. Let's do some feature analysis.
# 
# First, let's collect all the available features and transform the categorical features where necessary....

# In[ ]:


import sklearn.feature_selection as fs # feature selection library in scikit-learn
train = pd.read_csv('../input/train.csv') # get the training data again just in case
train['SalePrice'] = np.log(train['SalePrice'])
# first, let's include every feature that has data for all 1460 houses in the data set...
included_features = [col for col in list(train)
                    if len([i for i in train[col].T.notnull() if i == True])==1460
                    and col!='SalePrice' and col!='id']
# define the training data X...
X = train[included_features] # the feature data
Y = train[['SalePrice']] # the target
yt = [i for i in Y['SalePrice']] # the target list 
# and the data for the competition submission...
X_test = test[included_features]
# transform categorical data if included in X...
for col in list(X):
    if X[col].dtype=='object':
        X = getObjectFeature(X, col)
X.head()
# Y.head()


# ## Mutual Information Regression Metric for Feature Ranking
# We will use mutual information regression for feature ranking and selection. This metric measures the dependence between two random variables, in this case each feature in the data set and the sales price regression target. Note that this doesn't consider combinations of feature values (for example, the dependence between *sales price* and *the year of sale* combined with *overall quality*), which may also be useful.
# 
# See: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html

# In[ ]:


mir_result = fs.mutual_info_regression(X, yt) # mutual information regression feature ordering
feature_scores = []
for i in np.arange(len(included_features)):
    feature_scores.append([included_features[i],mir_result[i]])
sorted_scores = sorted(np.array(feature_scores), key=lambda s: float(s[1]), reverse=True) 
print(np.array(sorted_scores))


# ## MIR Results: What do homebuyers care about?
# Well, it seems like the most important factors (with respect to sales price) are overall quality, amount of living area, garage car capacity, kitchen quality, and exterior material quality. These seem like fairly intuitve results, at least for somebody with a distant notion of what matters when selecting a house (me).
# 
# Let's plot the results next to each other for a better visualization...

# In[ ]:


# and plot...
fig = plt.figure(figsize=(13,6))
ax = fig.add_subplot(111)
ind = np.arange(len(included_features))
plt.bar(ind,[float(i) for i in np.array(sorted_scores)[:,1]])
ax.axes.set_xticks(ind)
plt.title('Feature Importances (Mutual Information Regression)')
plt.ylabel('Importance')
# plt.xlabel('Trees in Forest')
# plt.grid()
plt.show()


# ## Feature Pruning
# It seems like the top few dozen features are fairly important... let's take the top 15, 20, 30, 40, and 50 features to train the random forest regressor model we've been working with and compare the performances. We will wrap the necessary model building and plotting code in functions first.

# In[ ]:


# define a function to do the necessary model building....
def getModel(sorted_scores,train,numFeatures):
    included_features = np.array(sorted_scores)[:,0][:numFeatures] # ordered list of important features
    # define the training data X...
    X = train[included_features]
    Y = train[['SalePrice']]
    # transform categorical data if included in X...
    for col in list(X):
        if X[col].dtype=='object':
            X = getObjectFeature(X, col)
    # define the number of estimators to consider
    estimators = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    mean_rfrs = []
    std_rfrs_upper = []
    std_rfrs_lower = []
    yt = [i for i in Y['SalePrice']]
    np.random.seed(11111)
    # for each number of estimators, fit the model and find the results for 8-fold cross validation
    for i in estimators:
        model = rfr(n_estimators=i,max_depth=None)
        scores_rfr = cross_val_score(model,X,yt,cv=10,scoring='explained_variance')
        mean_rfrs.append(scores_rfr.mean())
        std_rfrs_upper.append(scores_rfr.mean()+scores_rfr.std()*2) # for error plotting
        std_rfrs_lower.append(scores_rfr.mean()-scores_rfr.std()*2) # for error plotting
    return mean_rfrs,std_rfrs_upper,std_rfrs_lower

# define a function to plot the model expected variance results...
def plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,numFeatures):
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.plot(estimators,mean_rfrs,marker='o',
           linewidth=4,markersize=12)
    ax.fill_between(estimators,std_rfrs_lower,std_rfrs_upper,
                    facecolor='green',alpha=0.3,interpolate=True)
    ax.set_ylim([-.2,1])
    ax.set_xlim([0,80])
    plt.title('Expected Variance of Random Forest Regressor: Top %d Features'%numFeatures)
    plt.ylabel('Expected Variance')
    plt.xlabel('Trees in Forest')
    plt.grid()
    plt.show()
    return


# ...and let's run the regression model fitting for each of the scenarios listed before...

# In[ ]:


# top 15...
mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,15)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,15)


# In[ ]:


# top 20...
mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,20)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,20)


# In[ ]:


# top 30...
mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,30)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,30)


# In[ ]:


# top 40...
mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,40)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,40)


# In[ ]:


# top 50...
mean_rfrs,std_rfrs_upper,std_rfrs_lower = getModel(sorted_scores,train,50)
plotResults(mean_rfrs,std_rfrs_upper,std_rfrs_lower,50)


# # Random Forest Regression Impressions
# It seems like the mean expected variance of the regressions stops improving at around 20 features and 20 to 30 trees in the forest. The deviation in the expected variance score decreases with increasing features, which is intuitive. The 40-feature and 50-feature results look almost identical, probably because the 40th- to 50th-most important features are barely significant. Let's only consider the top 40 features from here on out.

# # The Finale: Building the Output for Submission
# Now, let's take the best regression model we have and build the competition output. For this model, we have:
# * A random forest resgression model, incorporating
# * the 40 most prominent features according to an MIR analysis, and
# * 60 regressor trees per forest, and
# * the default sklearn settings for the rest of the model parameters.
# 
# So let's apply this model to the test data and generate the submission!

# In[ ]:


# build the model with the desired parameters...
numFeatures = 40 # the number of features to inlcude
trees = 60 # trees in the forest
included_features = np.array(sorted_scores)[:,0][:numFeatures]
# define the training data X...
X = train[included_features]
Y = train[['SalePrice']]
# transform categorical data if included in X...
for col in list(X):
    if X[col].dtype=='object':
        X = getObjectFeature(X, col)
yt = [i for i in Y['SalePrice']]
np.random.seed(11111)
model = rfr(n_estimators=trees,max_depth=None)
scores_rfr = cross_val_score(model,X,yt,cv=10,scoring='explained_variance')
print('explained variance scores for k=10 fold validation:',scores_rfr)
print("Est. explained variance: %0.2f (+/- %0.2f)" % (scores_rfr.mean(), scores_rfr.std() * 2))
# fit the model
model.fit(X,yt)


# In[ ]:


# let's read the test data to be sure...
test = pd.read_csv('../input/test.csv')


# We will tweak the pre-processing function from before to handle missing data better, too...

# In[ ]:


# re-define a function to convert an object (categorical) feature into an int feature
# 0 = most common category, highest int = least common.
def getObjectFeature(df, col, datalength=1460):
    if df[col].dtype!='object': # if it's not categorical..
        print('feature',col,'is not an object feature.')
        return df
    else:
        df1 = df
        counts = df1[col].value_counts() # get the counts for each label for the feature
#         print(col,'labels, common to rare:',counts.index.tolist()) # get an ordered list of the labels
        df1[col] = [counts.index.tolist().index(i) 
                    if i in counts.index.tolist() 
                    else 0 
                    for i in df1[col] ] # do the conversion
        return df1 # make the new (integer) column from the conversion


# In[ ]:


# apply the model to the test data and get the output...
X_test = test[included_features]
for col in list(X_test):
    if X_test[col].dtype=='object':
        X_test = getObjectFeature(X_test, col, datalength=1459)
# print(X_test.head(20))
y_output = model.predict(X_test.fillna(0)) # get the results and fill nan's with 0
print(y_output)


# In[ ]:


# transform the data to be sure
y_output = np.exp(y_output)
print(y_output)


# In[ ]:


# define the data frame for the results
saleprice = pd.DataFrame(y_output, columns=['SalePrice'])
# print(saleprice.head())
# saleprice.tail()
results = pd.concat([test['Id'],saleprice['SalePrice']],axis=1)
results.head()


# In[ ]:


# and write to output
results.to_csv('housepricing_submission.csv', index = False)


# So, there's my random forest regression for the house price data set. I appreciate any feedback, comments, corrections, improvements!

# In[ ]:




