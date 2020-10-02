#!/usr/bin/env python
# coding: utf-8

# Hi all,
# second warm up competition, this time a regression problem
# 
# The idea is to do some EDA and submit a first result as a benchmark and then try to focus on feature selection and regressor tuning
# 
# I will use a library called speedml that can effectively speed up this part
# 
# Let's start!

# In[ ]:


#this is a library very useful (speeddml.com) helping to make faster analysis
from speedml import Speedml

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pylab
import plotly.plotly as py


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


################################################## Custom functions ###################

def plot_histogram(df, row,col,n_bins):
    g = sns.FacetGrid(df, col=col)
    g.map(plt.hist, row, bins=n_bins)

    
def box_plot(data):
    sns.boxplot(data=data)
    plt.xlabel("Attribute Index")
    plt.ylabel(("Quartile Ranges - Normalized "))
    
    
def split_and_train(X,y, test_size, classifier):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, random_state = 0)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    return y_test_pred, y_test, X_test, X_train, y_train    

def roc_graph(y_true, y_pred):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.figure()

def evaluate_classifier(y_true, y_pred, target_names):
    from sklearn.metrics import confusion_matrix    
    # Making the Confusion Matrix 
    print('Confusion Matrix')
    cm = confusion_matrix(y_true, y_pred)
    import seaborn as sn
    sn.heatmap(cm, annot=True)
    plt.figure()
    print(cm)
    print('\n')
    # Report
    print('Report')
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names = target_names))
    roc_graph(y_true, y_pred)
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    from sklearn.model_selection import learning_curve
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt  

def corr_heatmap(df):
    f, ax = plt.subplots(figsize=(10, 7))
    plt.xticks(rotation='90')
    sns.heatmap(DataFrame(df[df.columns].corr()), annot=True)
    
def split_and_train(X,y, test_size, regressor):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size, random_state = 0)
    y_train = y_train
    y_test = y_test
    regressor.fit(X_train, y_train)
    y_test_pred = regressor.predict(X_test)
    return y_test_pred, y_test, X_test, X_train, y_train        
    


# Step 1 - Acquiring data and preliminary EDA
# ===========================================

# In[ ]:


# using speedml too
sml = Speedml('../input/train.csv', 
              '../input/test.csv', 
              target = 'SalePrice',
              uid = 'Id')

sml.eda()


# In[ ]:


sml.train.describe()

# which nulls
sml.train.isnull().sum()
sml.test.isnull().sum()


# In[ ]:


#sml.plot.correlate()

sml.train.columns[0:]

#let's check numerical vcorrelation values against SalePrice
sml.train[sml.train.columns[0:]].corr()['SalePrice'][:-1].sort_values()


# Let's explore features in groups of five and fix when necessary
# For now we keep them all, then we think what to drop

# In[ ]:


sml.train[['MSSubClass', 'SalePrice']].groupby(['MSSubClass'], 
             as_index=False).mean().sort_values(by='SalePrice', ascending=False)

# --> feature density

sml.feature.density('MSSubClass')
sml.feature.drop('MSSubClass')

sml.train.MSSubClass_density


# In[ ]:


sml.train[['MSZoning', 'SalePrice']].groupby(['MSZoning'], 
             as_index=False).mean().sort_values(by='SalePrice', ascending=False)



sml.train[['MSZoning', 'SalePrice']].groupby(['MSZoning'], 
             as_index=False).count().sort_values(by='SalePrice', ascending=False)

# --> categorical feature

# fix nan (on test)
sml.feature.fillna('MSZoning', 'RL')

sml.feature.mapping('MSZoning', {'FV': 0, 'RL': 1, 'RH' : 2, 'RM' : 3, 'C (all)' : 4})

sml.train[['MSZoning', 'SalePrice']].groupby(['MSZoning'], 
             as_index=False).mean().sort_values(by='SalePrice', ascending=False)


# In[ ]:


sml.train[['LotFrontage', 'SalePrice']].groupby(['LotFrontage'], 
             as_index=False).mean().sort_values(by='SalePrice', ascending=False)

sns.jointplot(x='LotFrontage', y="SalePrice", data=sml.train);

# --> feature density

sml.feature.density('LotFrontage')
sml.feature.drop('LotFrontage')



sns.jointplot(x='LotFrontage_density', y="SalePrice", data=sml.train);


# In[ ]:



sns.distplot(sml.train.LotArea, kde=False, fit=stats.gamma);


# In[ ]:


sml.train[['LotArea', 'SalePrice']].groupby(['LotArea'], 
             as_index=False).mean().sort_values(by='SalePrice', ascending=False)

sns.jointplot(x='LotArea', y="SalePrice", data=sml.train);

#sml.feature.outliers('LotArea',upper=95)

# --> feature density

sml.feature.density('LotArea')
sml.feature.drop('LotArea')

sns.jointplot(x='LotArea_density', y="SalePrice", data=sml.train);


# In[ ]:


sns.distplot(sml.train.LotArea_density, kde=False, fit=stats.gamma);


# In[ ]:


sml.train[['Street', 'SalePrice']].groupby(['Street'], 
             as_index=False).mean().sort_values(by='SalePrice', ascending=False)

sml.feature.labels(['Street'])

sns.jointplot(x='Street', y="SalePrice", data=sml.train);


# In[ ]:


sml.train[['Alley', 'SalePrice']].groupby(['Alley'], 
             as_index=False).count().sort_values(by='SalePrice', ascending=False)

# a lot of missing data here..le'ts drop the column

sml.feature.drop('Alley')


# In[ ]:


sml.train[['LotShape', 'SalePrice']].groupby(['LotShape'], 
             as_index=False).mean().sort_values(by='SalePrice', ascending=False)

# --> categorical feature
sml.feature.labels(['LotShape'])

sns.jointplot(x='LotShape', y="SalePrice", data=sml.train);


# In[ ]:


sml.train[['LandContour', 'SalePrice']].groupby(['LandContour'], 
             as_index=False).count().sort_values(by='SalePrice', ascending=False)

sml.feature.labels(['LandContour'])



sns.jointplot(x='LandContour', y="SalePrice", data=sml.train);


# In[ ]:


sml.train[['Utilities', 'SalePrice']].groupby(['Utilities'], 
             as_index=False).count().sort_values(by='SalePrice', ascending=False)

# let's drop
sml.feature.drop(['Utilities'])


# In[ ]:


#too long using this approach
#better arrive to numerical data and then see correlation and do some feature reduction

sml.feature.impute()


#let's speed up...converting all categorical

feat_labels = sml.eda().iloc[8,0]

feat_labels
sml.feature.labels(feat_labels)


# In[ ]:


sml.eda()


# In[ ]:


feat_labels = sml.eda().iloc[7,0]

feat_labels

sml.feature.labels(feat_labels)


# In[ ]:


feat_density = sml.eda().iloc[4,0]

feat_density

sml.feature.density(['1stFlrSF',
 '2ndFlrSF',
 '3SsnPorch',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'EnclosedPorch',
 'Exterior1st',
 'Exterior2nd',
 'GarageArea',
 'GarageYrBlt',
 'GrLivArea',
 'LotArea_density',
 'LotFrontage_density',
 'LowQualFinSF',
 'MSSubClass_density',
 'MasVnrArea',
 'MiscVal',
 'MoSold',
 'Neighborhood',
 'OpenPorchSF',
'ScreenPorch',
 'TotRmsAbvGrd',
 'TotalBsmtSF',
 'WoodDeckSF',
 'YearBuilt',
 'YearRemodAdd'])

sml.feature.drop(['1stFlrSF',
 '2ndFlrSF',
 '3SsnPorch',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'EnclosedPorch',
 'Exterior1st',
 'Exterior2nd',
 'GarageArea',
 'GarageYrBlt',
 'GrLivArea',
 'LotArea_density',
 'LotFrontage_density',
 'LowQualFinSF',
 'MSSubClass_density',
 'MasVnrArea',
 'MiscVal',
 'MoSold',
 'Neighborhood',
 'OpenPorchSF',
'ScreenPorch',
 'TotRmsAbvGrd',
 'TotalBsmtSF',
 'WoodDeckSF',
 'YearBuilt',
 'YearRemodAdd'])


# In[ ]:




sml.eda()

sml.train[sml.train.columns[0:]].corr()['SalePrice'][:-1].sort_values()


# Step 2 - First submission
# =======

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

sml.model.data()

y_test_pred, y_test, X_test, X_train, y_train = split_and_train(X = sml.train_X,y = sml.train_y, 
                                                                test_size = 0.25, regressor = regressor)

regressor.score(X_test, y_test)

y_test
y_test_pred


# In[ ]:


plot_learning_curve(classifier, "Training", sml.train_X, sml.train_y.astype(int), (0.7, 1.01), cv=10)


# In[ ]:



data_test = pd.read_csv('../input/test.csv')

ids = data_test.iloc[:,0]

predictions = classifier.predict(sml.test_X)

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })
output.to_csv('house-prices.csv', index = False)
output.head()


# In[ ]:


sns.distplot(y_test, kde=False, fit=stats.gamma);

sns.distplot(y_test_pred, kde=False, fit=stats.gamma);


# Step 3 - Feature selection
# ==========================
# 
# let's try all the different feature selections method, using LinearRegression as is... then I'll choose the best one and I'll try to tune it

# In[ ]:


# let's use pipelines
from sklearn.pipeline import Pipeline
from sklearn import feature_selection

regressor = LinearRegression()


# Variance threshold

feature_selection1 = feature_selection.VarianceThreshold()
regression_pipe = Pipeline([('feature selection', feature_selection1), 
                            ('regressor', regressor)])
regression_pipe.fit(X=sml.train_X, y=sml.train_y)
print("Variance Threshold score :" , regression_pipe.score(X_test, y_test))

# KBest
feature_selection2 = feature_selection.SelectKBest(feature_selection.f_regression)
regression_pipe = Pipeline([('feature selection', feature_selection2), 
                            ('regressor', regressor)])
regression_pipe.fit(X=sml.train_X, y=sml.train_y)
print("SelectKBest score :" , regression_pipe.score(X_test, y_test))


# In[ ]:



def evaluate_rfe(regressor, X, y, step, cv):
    from sklearn.feature_selection import RFECV
    rfecv = RFECV(regressor, step=step, cv=cv) 
    rfecv.fit(X, y)
    # summarize the selection of the attributes
    print(rfecv.support_)
    print(rfecv.ranking_)
    print("Optimal number of features : %d" % rfecv.n_features_)    
    # Plot number of features VS. cross-validation scores
    # print("Score :" , rfecv.score())
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    
evaluate_rfe(regressor=regressor, X=sml.train_X, y=sml.train_y, step=1, cv=10) 


# In[ ]:


feature_selection3 = feature_selection.RFECV(regressor)
regression_pipe = Pipeline([('feature selection', feature_selection3), 
                            ('regressor', regressor)])
regression_pipe.fit(X=sml.train_X, y=sml.train_y)
print("RFECV score :" , regression_pipe.score(X_test, y_test))


# Step 4 - Second submission
# =======
# 
# score improved with LinearRegressor and feature selection
# 
# let's submit 

# In[ ]:


feature_selection1 = feature_selection.VarianceThreshold()
regression_pipe = Pipeline([('feature selection', feature_selection1), 
                            ('regressor', regressor)])
regression_pipe.fit(X=sml.train_X, y=sml.train_y)


predictions = regression_pipe.predict(sml.test_X)

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })
output.to_csv('house-prices2.csv', index = False)
output.head()

