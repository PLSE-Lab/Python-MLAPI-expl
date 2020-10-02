#!/usr/bin/env python
# coding: utf-8

# This notebook aims to make the use of pipelines a bit more user-friendly even for more unexperienced ML enthusiasts. In particular, most of the efforts here will be directed to the use of pipelines for cross-validation and model selection (for example, via GridSearch), which can get tricky when the pipeline becomes more complex.
# 
# We will touch the following topics:
# 
# * What is a Pipeline and why it matters
# * What is tricky about a pipeline
# * How to make your own custom transformers
# * A pipeline step by step (or tube by tube?)
# 
# We will make use of this dataset as it gives enough variety in input and enough opportunities for data transformation. However, we won't try to make an high scoring model but rather a very understandable one.
# 
# The final goal is thus making the reader able to use the concepts here presented to make their own high scoring model.

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

import warnings

pd.set_option('max_columns', 500)


# In[ ]:


def make_test(train, test_size, random_state, strat_feat=None):
    if strat_feat:
        
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        for train_index, test_index in split.split(train, train[strat_feat]):
            train_set = train.loc[train_index]
            test_set = train.loc[test_index]
            
    return train_set, test_set


# # Pipelines: what and why?
# 
# In the context of a ML project, we can define a pipeline as a sequence of objects that act on a set of data. The actions can include:
# 
# * learning a relation
# * transform some feature
# * impute missing values
# * create new features
# * fit a model
# * predict on unseen data
# 
# The purpose is to apply sequentially its element to **validate a process**. On this point, we can take a moment to stress out how important is to validate the entire process rather than just the ML model itself. It is well known that different problems call for different models since it is not possible to say a priori that one model will perform better than another one on any given dataset. This is one (fairly sloppy) formulation of the so-called No free lunch theorem.
# 
# Therefore, it is easy to imagine that choices in the processing stage (being this the imputation of missing entries, the scaling of the features or the feature engineered) will influence the choice of the final model, both in terms of algorithm or of hyperparameters. It is thus important to be able to assess the ability of a process to be better than another one. 
# 
# A trustable evaluation of your process is necessary to justify the choices you made, otherwise you can be less sure about, for example, creating a new feature as you don't know if you are increasing your cross-validation score because of a flawed process or because of the creation of a genuinely powerful predictor (or, more commonly, you are overestimating the effect of adding that new feature or not).
# 
# While the use of pipelines is not the silver bullet to this issue, it certainly helps in excluding some common validation errors. The main one, at first, is to not apply the same transformations to the train and test sets, resulting most of the time in an error or a very weird performing model. A pipeline will take care of this issue by always applying the same transformations in sequence until the final result is achieved, no matter what dataset are you applying it on.
# 
# To see why it is not enough to carefully apply the same sequence by hand and instead relying on a Pipeline, let's have a look at a simple example. In this competition, the training set (the only one we should be looking at to simulate a real-life problem) looks like this
# 

# In[ ]:


df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_train.head()


# And it is easy to see that we have quite a few missing values.

# In[ ]:


df_train.info()


# Now, let's suppose we want to fill the missing values with the average of each column. It is not hard to do it and we are now happy about the result.
# 
# We then split our data into training and validation sets and train some models to be evaluated.
# 
# **This is wrong!**
# 
# By imputing the missing values using the mean of the entire dataset and *then* splitting the data for validation, we are leaking some information from the validation set into the training one. This will most likely boost our evaluation score but not necessarily the quality of our model.
# 
# One can then argue that it is sufficient to split the data first and impute after on training and validation sets separately. However, it is important to keep in mind that the mean values (the values you will use to impute the missing entries) have to be calculated only on the training set and applied to both training and validation. In this way, you will be correctly simulating the performance of your model in your evaluation process.
# 
# A pipeline will take care of that, regardless of your validation strategy (k-fold, train_test split, both...)
# 
# # Existing instruments and what is tricky about them
# 
# By going through the sklearn documentation, it is easy to find that most of the operations you will commonly apply have already a very efficient implementation. For example, the imputation mentioned above can be done easily by using the `SimpleImputer`. Or, if we want to rescale our features, we can make use of the `StandardScaler`.
# 
# Let's thus see how to use one of these instruments. First, we prepare an evaluation set

# In[ ]:


train_set, test_set = make_test(df_train, 
                                test_size=0.2, random_state=654, 
                                strat_feat='Neighborhood')


# The validation set is 20% of the full training set and it correctly reproduces the proportion of houses in each Neighborhood (see the implementation in one of the hidden cells at the beginning). Now, let's suppose we only have this dataset

# In[ ]:


tmp = train_set[['GrLivArea', 'TotRmsAbvGrd']].copy()
tmp.head()


# and we want the data to be scaled before going to our model. If we apply the provided scaler, we get the following result

# In[ ]:


scaler = StandardScaler()  # initialize a StandardScaler object (more on this later)

tmp = scaler.fit_transform(tmp)  # apply a fit and a transform method (more on this later)

tmp


# The structure of the data is very different now. The reason is that, for speed and memory usage, the sklearn instruments use the dataframe as numpy arrays. However, one may want to operate again on the data after one of these transformations and losing the DataFrame structure may lead to very cumbersome code or, probably worse, very limited pipelines. In the next sections, we will implement methods and pipelines that aim to be very flexible in order to adapt to a larger variety of problems, the downside is that we will be a bit less efficient in terms of speed and memory usage.
# 
# A note before moving on: a pipeline with only 2 steps (scaler and model) will do just fine by not doing any further coding but I find it more useful to break things down and learn what happens inside.
# 
# # Transformers and classes
# 
# In this section, we will see how to use the existing instruments to build something that better suits our needs (having a pipeline and maintain the DataFrame structure).
# 
# First of all, we are going to be working with sklearn Pipelines. These will sequentially apply the `fit` and `transform` (or `predict`) methods at each step. We thus need to create objects that can do `fit` and `transform`. In python, the way to do so is to create a **class**. You can read about classes pretty much everywhere and everyone will explain them better than I could possibly do, let's just jump into an example.
# 
# We want a transformer that works like the scaler above but also maintain the DataFrame structure. It will look like this

# In[ ]:


class df_scaler(TransformerMixin):
    def __init__(self, method='standard'):
        self.scl = None
        self.scale_ = None
        self.method = method
        if self.method == 'sdandard':
            self.mean_ = None
        elif method == 'robust':
            self.center_ = None

    def fit(self, X, y=None):
        if self.method == 'standard':
            self.scl = StandardScaler()
            self.scl.fit(X)
            self.mean_ = pd.Series(self.scl.mean_, index=X.columns)
        elif self.method == 'robust':
            self.scl = RobustScaler()
            self.scl.fit(X)
            self.center_ = pd.Series(self.scl.center_, index=X.columns)
        self.scale_ = pd.Series(self.scl.scale_, index=X.columns)
        return self

    def transform(self, X):
        # X has to be a dataframe
        Xscl = self.scl.transform(X)
        Xscaled = pd.DataFrame(Xscl, index=X.index, columns=X.columns)
        return Xscaled


# If you have never seen a class before, it looks intimidating (at least it was for me). Let me break it down a bit.
# 
# * We use `self` a lot, this refers to the object itself (sorry for the repetition). This means that when you create an object of this type, it will have its properties (scale, method, etc) and its methods (fit, transform). The use of `self` is simply to refer to those properties or methods.
# * There is an `__init__` method, which initializes some variables, some of them can be external inputs (like the method for scaling) or properties that the methods can modify (and you can access to when you use it)
# * There is a `fit` method. In this case, it just calls the fit method of the selected pre-made scaler. It also finds the mean and scale of each column (by using the premade scaler as well)
# * There is a `transform` method. This applies the transformation of the scaler fitted above **and** rebuilds the dataframe with its structure.
# 
# Let's see how to use this new scaler on the same data as before.

# In[ ]:


tmp = train_set[['GrLivArea', 'TotRmsAbvGrd']].copy()
tmp.head()


# In[ ]:


scaler = df_scaler()  # initialize the oject

tmp = scaler.fit_transform(tmp)  # apply a fit and a transform method we defined above

tmp.head()  # this time it is a dataframe, we can use `head`


# We thus get the same numbers as before, but we still have the nice DataFrame structure, we will see in the next section how important this is.
# 
# All those properties defined in the `__init__` method are accessible like in the normal Scaler

# In[ ]:


scaler.mean_


# In[ ]:


scaler.scale_


# More often than not, you will need to create transformers that do nothing while fitting the data and do a lot of things when they transform it. For example, by following the documentation about this dataset, we can implement a transformer that cleans up the data.

# In[ ]:


class general_cleaner(BaseEstimator, TransformerMixin):
    '''
    This class applies what we know from the documetation.
    It cleans some known missing values
    If flags the missing values

    This process is supposed to happen as first step of any pipeline
    '''
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        #LotFrontage
        X.loc[X.LotFrontage.isnull(), 'LotFrontage'] = 0
        #Alley
        X.loc[X.Alley.isnull(), 'Alley'] = "NoAlley"
        #MSSubClass
        X['MSSubClass'] = X['MSSubClass'].astype(str)
        #MissingBasement
        fil = ((X.BsmtQual.isnull()) & (X.BsmtCond.isnull()) & (X.BsmtExposure.isnull()) &
              (X.BsmtFinType1.isnull()) & (X.BsmtFinType2.isnull()))
        fil1 = ((X.BsmtQual.notnull()) | (X.BsmtCond.notnull()) | (X.BsmtExposure.notnull()) |
              (X.BsmtFinType1.notnull()) | (X.BsmtFinType2.notnull()))
        X.loc[fil1, 'MisBsm'] = 0
        X.loc[fil, 'MisBsm'] = 1 # made explicit for safety
        #BsmtQual
        X.loc[fil, 'BsmtQual'] = "NoBsmt" #missing basement
        #BsmtCond
        X.loc[fil, 'BsmtCond'] = "NoBsmt" #missing basement
        #BsmtExposure
        X.loc[fil, 'BsmtExposure'] = "NoBsmt" #missing basement
        #BsmtFinType1
        X.loc[fil, 'BsmtFinType1'] = "NoBsmt" #missing basement
        #BsmtFinType2
        X.loc[fil, 'BsmtFinType2'] = "NoBsmt" #missing basement
        #BsmtFinSF1
        X.loc[fil, 'BsmtFinSF1'] = 0 # No bsmt
        #BsmtFinSF2
        X.loc[fil, 'BsmtFinSF2'] = 0 # No bsmt
        #BsmtUnfSF
        X.loc[fil, 'BsmtUnfSF'] = 0 # No bsmt
        #TotalBsmtSF
        X.loc[fil, 'TotalBsmtSF'] = 0 # No bsmt
        #BsmtFullBath
        X.loc[fil, 'BsmtFullBath'] = 0 # No bsmt
        #BsmtHalfBath
        X.loc[fil, 'BsmtHalfBath'] = 0 # No bsmt
        #FireplaceQu
        X.loc[(X.Fireplaces == 0) & (X.FireplaceQu.isnull()), 'FireplaceQu'] = "NoFire" #missing
        #MisGarage
        fil = ((X.GarageYrBlt.isnull()) & (X.GarageType.isnull()) & (X.GarageFinish.isnull()) &
              (X.GarageQual.isnull()) & (X.GarageCond.isnull()))
        fil1 = ((X.GarageYrBlt.notnull()) | (X.GarageType.notnull()) | (X.GarageFinish.notnull()) |
              (X.GarageQual.notnull()) | (X.GarageCond.notnull()))
        X.loc[fil1, 'MisGarage'] = 0
        X.loc[fil, 'MisGarage'] = 1
        #GarageYrBlt
        X.loc[X.GarageYrBlt > 2200, 'GarageYrBlt'] = 2007 #correct mistake
        X.loc[fil, 'GarageYrBlt'] = 0
        #GarageType
        X.loc[fil, 'GarageType'] = "NoGrg" #missing garage
        #GarageFinish
        X.loc[fil, 'GarageFinish'] = "NoGrg" #missing
        #GarageQual
        X.loc[fil, 'GarageQual'] = "NoGrg" #missing
        #GarageCond
        X.loc[fil, 'GarageCond'] = "NoGrg" #missing
        #Fence
        X.loc[X.Fence.isnull(), 'Fence'] = "NoFence" #missing fence
        #Pool
        fil = ((X.PoolArea == 0) & (X.PoolQC.isnull()))
        X.loc[fil, 'PoolQC'] = 'NoPool' 
        
        del X['Id']
        del X['MiscFeature']
        del X['MSSubClass']
        del X['Neighborhood']  # this should be useful
        del X['Condition1']
        del X['Condition2']
        del X['ExterCond']  # maybe ordinal
        del X['Exterior1st']
        del X['Exterior2nd']
        del X['Functional']
        del X['Heating']
        del X['PoolQC']
        del X['RoofMatl']
        del X['RoofStyle']
        del X['SaleCondition']
        del X['SaleType']
        del X['Utilities']
        del X['BsmtCond']
        del X['Electrical']
        del X['Foundation']
        del X['Street']
        del X['Fence']
        del X['LandSlope']
        
        return X


# As we can see, the fit method doesn't need to do anything, while the transform method fills in missing values, removes columns, and creates new columns. Moreover, we don't need to specify an `__init__` because there is nothing to be initialized in this case (it won't always be the case).
# 
# The usage is the same

# In[ ]:


tmp = train_set.copy()

gt = general_cleaner()

tmp = gt.fit_transform(tmp)

tmp.head()


# Note: we implemented a `fit` and a `transform` method but somehow we manage to use a `fit_transform` method. The reason is that we are making these objects inherit the properties of the `TransformerMixin`, which knows what to do when it finds a `fit_transform`. 
# 
# We are now ready to put all this knowledge together in the section we are all here for
# 
# # A Pipeline step-by-step
# 
# It is now time to build our complete pipeline for this dataset. We start with some transformers that are nothing more than simple wrappers around known sklearn functions.

# In[ ]:


class df_imputer(BaseEstimator, TransformerMixin):
    '''
    Just a wrapper for the SimpleImputer that keeps the dataframe structure
    '''
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imp = None
        self.statistics_ = None

    def fit(self, X, y=None):
        self.imp = SimpleImputer(strategy=self.strategy)
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    def transform(self, X):
        # X is supposed to be a DataFrame
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled
    
    
class df_scaler(BaseEstimator, TransformerMixin):
    '''
    Wrapper of StandardScaler or RobustScaler
    '''
    def __init__(self, method='standard'):
        self.scl = None
        self.scale_ = None
        self.method = method
        if self.method == 'sdandard':
            self.mean_ = None
        elif method == 'robust':
            self.center_ = None
        self.columns = None  # this is useful when it is the last step of a pipeline before the model

    def fit(self, X, y=None):
        if self.method == 'standard':
            self.scl = StandardScaler()
            self.scl.fit(X)
            self.mean_ = pd.Series(self.scl.mean_, index=X.columns)
        elif self.method == 'robust':
            self.scl = RobustScaler()
            self.scl.fit(X)
            self.center_ = pd.Series(self.scl.center_, index=X.columns)
        self.scale_ = pd.Series(self.scl.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xscl = self.scl.transform(X)
        Xscaled = pd.DataFrame(Xscl, index=X.index, columns=X.columns)
        self.columns = X.columns
        return Xscaled

    def get_feature_names(self):
        return list(self.columns)  # this is going to be useful when coupled with FeatureUnion
    

class dummify(BaseEstimator, TransformerMixin):
    '''
    Wrapper for get dummies
    '''
    def __init__(self, drop_first=False, match_cols=True):
        self.drop_first = drop_first
        self.columns = []  # useful to well behave with FeatureUnion
        self.match_cols = match_cols

    def fit(self, X, y=None):
        self.columns = []  # for safety, when we refit we want new columns
        return self
    
    def match_columns(self, X):
        miss_train = list(set(X.columns) - set(self.columns))
        miss_test = list(set(self.columns) - set(X.columns))
        
        err = 0
        
        if len(miss_test) > 0:
            for col in miss_test:
                X[col] = 0  # insert a column for the missing dummy
                err += 1
        if len(miss_train) > 0:
            for col in miss_train:
                del X[col]  # delete the column of the extra dummy
                err += 1
                
        if err > 0:
            warnings.warn('The dummies in this set do not match the ones in the train set, we corrected the issue.',
                         UserWarning)
            
        return X
        
    def transform(self, X):
        X = pd.get_dummies(X, drop_first=self.drop_first)
        if (len(self.columns) > 0): 
            if self.match_cols:
                X = self.match_columns(X)
            self.columns = X.columns
        else:
            self.columns = X.columns
        return X
    
    def get_features_name(self):
        return self.columns


# We have seen the scaler in action already, we can easily test the other transformers.

# In[ ]:


tmp = train_set[['HouseStyle']].copy()

dummifier = dummify()

tmp = dummifier.transform(tmp)  # no reason to call the fit method here

tmp.sample(5)


# We can use these transformers in many places but the imputer will probably come before the others as we don't want to deal with missing values again.
# 
# However, the imputation method changes a lot if we are dealing with categorical or numerical features, we need something that automatically selects the features to be fed to the right imputer. This is what the next class is for

# In[ ]:


class feat_sel(BaseEstimator, TransformerMixin):
    '''
    This transformer selects either numerical or categorical features.
    In this way we can build separate pipelines for separate data types.
    '''
    def __init__(self, dtype='numeric'):
        self.dtype = dtype

    def fit( self, X, y=None ):
        return self 

    def transform(self, X, y=None):
        if self.dtype == 'numeric':
            num_cols = X.columns[X.dtypes != object].tolist()
            return X[num_cols]
        elif self.dtype == 'category':
            cat_cols = X.columns[X.dtypes == object].tolist()
            return X[cat_cols]


# Which can be tested very simply

# In[ ]:


tmp = train_set.copy()

selector = feat_sel()  # it is numeric by default

tmp = selector.transform(tmp)  # no reason to fit again

tmp.head()


# ### Short note: the custom dummifier
# 
# One common issue one encounter when working with categorical features is that some categories may be very rare, resulting into a mismatch between the columns in the train and test (or validation set).
# 
# For this reason, the implementation here proposed takes care of the issue by creating or deleting any column that is not present in both sets. In other words, we assing the attribute columns when we first transform the training set and, when the transform method is called again, we check for missing or extra columns. If a column is missing, we add it with all 0's, otherwise we drop it.
# 
# A short demostration of this can be done if we take the following feature

# In[ ]:


tmp = train_set[['RoofMatl']].copy()

dummifier = dummify()

dummifier.fit_transform(tmp).sum()  # to get how many dummies are present


# However, in the test set we don't have all those values

# In[ ]:


test_set.RoofMatl.value_counts()


# The normal OneHotEncoding or the standard get_dummies would create a dataset with only 3 columns and, when a model is called, an error caused by the mismatch in shape. This custom dummifier takes care of it as follows

# In[ ]:


tmp = test_set[['RoofMatl']].copy()

dummifier.transform(tmp).sum()  # the same instance as before


# We thus see how the missing dummies were added with all 0's and the dummy for `Metal` got dropped. In this way, the pipeline will not break later on.
# 
# ## A pipeline for numeric features
# 
# We can finally explicitly build our first pipeline. Ideally, we want it as follows
# 
# * Clean the data following the documentation
# * Impute the missing values with the mean or the median (nothing stops us from using other types of imputations
# * Apply some transformations on some features
# * Create new features
# * Scale the data
# 
# We already have every element but one, let's make a custom transformer

# In[ ]:


class tr_numeric(BaseEstimator, TransformerMixin):
    def __init__(self, SF_room=True):
        self.columns = []  # useful to well behave with FeatureUnion
        self.SF_room = SF_room
        

    def fit(self, X, y=None):
        return self
    

    def remove_skew(self, X, column):
        X[column] = np.log1p(X[column])
        return X


    def SF_per_room(self, X):
        if self.SF_room:
            X['sf_per_room'] = X['GrLivArea'] / X['TotRmsAbvGrd']
        return X
    

    def transform(self, X, y=None):
        for col in ['GrLivArea', '1stFlrSF', 'LotArea']: # they can also be inputs
            X = self.remove_skew(X, col)

        X = self.SF_per_room(X)
        
        self.columns = X.columns 
        return X
    

    def get_features_name(self):  # again, it will be useful later
        return self.columns


# Please note that this transformer takes a parameter that determines whether or not to create a new feature. It is this kind of parameter that can be tuned with a GridSearch (more on this later).
# 
# Creating a new feature in that way would be impossible if the previous steps were not returning a DataFrame. There is naturally an alternative that includes specifying the index of the columns you want to use, but I find this approach way more user-friendly and robust.
# 
# A pipeline for numeric features would then look like this

# In[ ]:


numeric_pipe = Pipeline([('fs', feat_sel(dtype='numeric')),  # select only the numeri features
                         ('imputer', df_imputer(strategy='median')),  # impute the missing values with the median of each column
                         ('transf', tr_numeric(SF_room=True)),  # remove skew and create a new feature
                         ('scl', df_scaler(method='standard'))])  # scale the data

full_pipe = Pipeline([('gen_cl', general_cleaner()), ('num_pipe', numeric_pipe)])  # put the cleaner on top because we like it clean


# In other words, with the use of the sklearn `Pipeline`, we want to sequentially apply the transformations in the given list. The list is made of tuples, the first element is a label for that step, and the second element is the transformation (or the model, or another pipeline). The name is useful to identify every parameter of the Pipeline, as we will see later.
# 
# This pipeline, given the training data, acts as follows

# In[ ]:


tmp = train_set.copy()

tmp = full_pipe.fit_transform(tmp)

tmp.head()


# In[ ]:


tmp.info()


# As we wanted, the data flew through the pipeline, getting cleaned, transformed, and rescaled. Moreover, we still have a nice DataFrame structure.
# 
# The powerfulness of this pipeline is visible when we want to do the same thing to the validation set and it is evident when we implement it

# In[ ]:


tmp = test_set.copy()  # not ready to work on those sets yet

tmp = full_pipe.transform(tmp)  # the fit already happened with the training set, we don't want to fit again

tmp.head()


# We have to worry about nothing, if some column had missing values in the validation set and not in the train one, the pipeline is still able to take care of it. In other words, we are sure that our models will get the same data format both during training and validation, making the validation phase more trustable.
# 
# We mentioned in the comments the usefulness of exposing the parameters, this is how our pipeline looks like

# In[ ]:


full_pipe.get_params()


# ## A pipeline for categorical features
# 
# In the same way, we can create a similar pipeline for all the categorical features. The difference will be only that we impute differently (there is no mean or median for categories), and we transform differently.
# 
# Let's then create a simple custom transformer

# In[ ]:


class make_ordinal(BaseEstimator, TransformerMixin):
    '''
    Transforms ordinal features in order to have them as numeric (preserving the order)
    If unsure about converting or not a feature (maybe making dummies is better), make use of
    extra_cols and include_extra
    '''
    def __init__(self, cols, extra_cols=None, include_extra=True):
        self.cols = cols
        self.extra_cols = extra_cols
        self.mapping = {'Po':1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        self.include_extra = include_extra
    

    def fit(self, X, y=None):
        return self
    

    def transform(self, X, y=None):
        if self.extra_cols:
            if self.include_extra:
                self.cols += self.extra_cols
            else:
                for col in self.extra_cols:
                    del X[col]
        
        for col in self.cols:
            X.loc[:, col] = X[col].map(self.mapping).fillna(0)
        return X

    
class recode_cat(BaseEstimator, TransformerMixin):        
    '''
    Recodes some categorical variables according to the insights gained from the
    data exploration phase. Not presented in this notebook
    '''
    def fit(self, X, y=None):
        return self
    
    
    def tr_GrgType(self, data):
        data['GarageType'] = data['GarageType'].map({'Basment': 'Attchd',
                                                  'CarPort': 'Detchd', 
                                                  '2Types': 'Attchd' }).fillna(data['GarageType'])
        return data
    
    
    def tr_LotShape(self, data):
        fil = (data.LotShape != 'Reg')
        data['LotShape'] = 1
        data.loc[fil, 'LotShape'] = 0
        return data
    
    
    def tr_LandCont(self, data):
        fil = (data.LandContour == 'HLS') | (data.LandContour == 'Low')
        data['LandContour'] = 0
        data.loc[fil, 'LandContour'] = 1
        return data
    
    
    def tr_LandSlope(self, data):
        fil = (data.LandSlope != 'Gtl')
        data['LandSlope'] = 0
        data.loc[fil, 'LandSlope'] = 1
        return data
    
    
    def tr_MSZoning(self, data):
        data['MSZoning'] = data['MSZoning'].map({'RH': 'RM', # medium and high density
                                                 'C (all)': 'RM', # commercial and medium density
                                                 'FV': 'RM'}).fillna(data['MSZoning'])
        return data
    
    
    def tr_Alley(self, data):
        fil = (data.Alley != 'NoAlley')
        data['Alley'] = 0
        data.loc[fil, 'Alley'] = 1
        return data
    
    
    def tr_LotConfig(self, data):
        data['LotConfig'] = data['LotConfig'].map({'FR3': 'Corner', # corners have 2 or 3 free sides
                                                   'FR2': 'Corner'}).fillna(data['LotConfig'])
        return data
    
    
    def tr_BldgType(self, data):
        data['BldgType'] = data['BldgType'].map({'Twnhs' : 'TwnhsE',
                                                 '2fmCon': 'Duplex'}).fillna(data['BldgType'])
        return data
    
    
    def tr_MasVnrType(self, data):
        data['MasVnrType'] = data['MasVnrType'].map({'BrkCmn': 'BrkFace'}).fillna(data['MasVnrType'])
        return data


    def tr_HouseStyle(self, data):
        data['HouseStyle'] = data['HouseStyle'].map({'1.5Fin': '1.5Unf', 
                                                         '2.5Fin': '2Story', 
                                                         '2.5Unf': '2Story', 
                                                         'SLvl': 'SFoyer'}).fillna(data['HouseStyle'])
        return data
    
    
    def transform(self, X, y=None):
        X = self.tr_GrgType(X)
        X = self.tr_LotShape(X)
        X = self.tr_LotConfig(X)
        X = self.tr_MSZoning(X)
        X = self.tr_Alley(X)
        X = self.tr_LandCont(X)
        X = self.tr_BldgType(X)
        X = self.tr_MasVnrType(X)
        X = self.tr_HouseStyle(X)
        return X


# The pipeline for categorical features will then be

# In[ ]:


cat_pipe = Pipeline([('fs', feat_sel(dtype='category')),
                     ('imputer', df_imputer(strategy='most_frequent')), 
                     ('ord', make_ordinal(['BsmtQual', 'KitchenQual','GarageQual',
                                           'GarageCond', 'ExterQual', 'HeatingQC'])), 
                     ('recode', recode_cat()), 
                     ('dummies', dummify())])

full_pipe = Pipeline([('gen_cl', general_cleaner()), ('cat_pipe', cat_pipe)])


tmp = train_set.copy()

tmp = full_pipe.fit_transform(tmp)

tmp.head()


# And there we have it, some categories converted into numeric features, other first recoded and the dummified. This dataset is ready for a model and, as before, this pipeline is ready for the validation set as well

# In[ ]:


tmp = test_set.copy()

tmp = full_pipe.transform(tmp)

tmp.head()


# In[ ]:


full_pipe.get_params()


# ## Putting everything together
# 
# We have a pipeline for numeric features, one for categorical one, now we want a complete pipeline for the entire dataset.
# 
# Sklearn again helps us with `FeatureUnion` that, sadly, again compromises the DataFrame structure we are very much fun of. By now, we are confident enough to create our own version of it.

# In[ ]:


class FeatureUnion_df(TransformerMixin, BaseEstimator):
    '''
    Wrapper of FeatureUnion but returning a Dataframe, 
    the column order follows the concatenation done by FeatureUnion

    transformer_list: list of Pipelines

    '''
    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose  # these are necessary to work inside of GridSearch or similar
        self.feat_un = FeatureUnion(self.transformer_list, 
                                    self.n_jobs, 
                                    self.transformer_weights, 
                                    self.verbose)
        
    def fit(self, X, y=None):
        self.feat_un.fit(X)
        return self

    def transform(self, X, y=None):
        X_tr = self.feat_un.transform(X)
        columns = []
        
        for trsnf in self.transformer_list:
            cols = trsnf[1].steps[-1][1].get_features_name()  # getting the features name from the last step of each pipeline
            columns += list(cols)

        X_tr = pd.DataFrame(X_tr, index=X.index, columns=columns)
        
        return X_tr

    def get_params(self, deep=True):  # necessary to well behave in GridSearch
        return self.feat_un.get_params(deep=deep)


# I hope it is now evident why I kept implementing a `get_features_name` method in the previous classes. It was all for this moment.
# 
# The complete pipeline will then be

# In[ ]:


numeric_pipe = Pipeline([('fs', feat_sel('numeric')),
                         ('imputer', df_imputer(strategy='median')),
                         ('transf', tr_numeric())])


cat_pipe = Pipeline([('fs', feat_sel('category')),
                     ('imputer', df_imputer(strategy='most_frequent')), 
                     ('ord', make_ordinal(['BsmtQual', 'KitchenQual','GarageQual',
                                           'GarageCond', 'ExterQual', 'HeatingQC'])), 
                     ('recode', recode_cat()), 
                     ('dummies', dummify())])


processing_pipe = FeatureUnion_df(transformer_list=[('cat_pipe', cat_pipe),
                                                 ('num_pipe', numeric_pipe)])


full_pipe = Pipeline([('gen_cl', general_cleaner()), 
                      ('processing', processing_pipe), 
                      ('scaler', df_scaler())])  # the scaler is here to have also the ordinal features scaled

tmp = df_train.copy()

tmp = full_pipe.fit_transform(tmp)

tmp.head()


# The more perceptive of you will notice that this output has a different order of columns of the input. This is because `FeatureUnion` is essentially concatenating the results of each transformer: first the categorical features, then the numeric ones. This is also why the column list in our version of this transformer was build in that way.
# 
# Again, we can now apply the pipeline to the test set

# In[ ]:


tmp = test_set.copy()

tmp = full_pipe.transform(tmp)

tmp.head()


# This time, the parameters are a bit more complex

# In[ ]:


full_pipe.get_params()


# ## Using the pipeline in GridSearch
# 
# Having set up everything as we did, it is not difficult to tune our pipeline with GridSearch. We will put a simple model at the end of the pipeline just for the fun of it and tune both the hyperparameters of this model and the parameters of the pipeline. 
# 
# We thus make use of `GridSearch` to pick the best model configuration by varying several parameters, namely
# 
# * Whether or not we create the new feature describing the square feet per room
# * If we impute the numerical missing values with the mean or the median
# * If we drop one dummy or not
# * If we change the regularization parameter of the Lasso regression
# 
# Thanks to the fact that we have a pipeline, we are able to easily explore all these configurations without worrying too much about information leakage or by repeating the same steps over and over

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, GridSearchCV

folds = KFold(5, shuffle=True, random_state=541)

df_train['Target'] = np.log1p(df_train.SalePrice)

del df_train['SalePrice']

train_set, test_set = make_test(df_train, 
                                test_size=0.2, random_state=654, 
                                strat_feat='Neighborhood')

y = train_set['Target'].copy()
del train_set['Target']

y_test = test_set['Target']
del test_set['Target']


def grid_search(data, target, estimator, param_grid, scoring, cv):
    
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, 
                        cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    
    pd.options.mode.chained_assignment = None  # this is because the gridsearch throws a lot of pointless warnings
    tmp = data.copy()
    grid = grid.fit(tmp, target)
    pd.options.mode.chained_assignment = 'warn'
    
    result = pd.DataFrame(grid.cv_results_).sort_values(by='mean_test_score', 
                                                        ascending=False).reset_index()
    
    del result['params']
    times = [col for col in result.columns if col.endswith('_time')]
    params = [col for col in result.columns if col.startswith('param_')]
    
    result = result[params + ['mean_test_score', 'std_test_score'] + times]
    
    return result, grid.best_params_


# The grid search (here in an utility function just to have better looking results) looks like this

# In[ ]:


lasso_pipe = Pipeline([('gen_cl', general_cleaner()),
                       ('processing', processing_pipe),
                       ('scl', df_scaler()), 
                       ('lasso', Lasso(alpha=0.01))])

res, bp = grid_search(train_set, y, lasso_pipe, 
            param_grid={'processing__num_pipe__transf__SF_room': [True, False], 
                        'processing__num_pipe__imputer__strategy': ['mean', 'median'],
                        'processing__cat_pipe__dummies__drop_first': [True, False],
                        'lasso__alpha': [0.1, 0.01, 0.001]},
            cv=folds, scoring='neg_mean_squared_error')

res


# And the best parameters are

# In[ ]:


bp


# *I am still working on making easy to set these parameters into the full pipeline*
# 
# We thus see very easily how some parameters matter more than others and consider if it is worth it to keep spending time in tuning them (because the more parameters you want to tune, the more running time will take).
# 
# Please note how we refer to a specific parameter by calling every step of the pipeline by its name and concatenating those names by the double underscore. I think this is why you can't use parameters that start with and underscore, they mess up this reference system. (To be checked)
# 
# ## Other evaluation approaches
# 
# We see from the GridSearch that, if we ignore the fact that these models are vey simple and not high-performing, the best configurations of parameters are scoring very similar results. One may want to be sure that the model is really the best possible one and/or it is predicting reasonable prices.
# 
# Thanks to all that effort in preserving the feature names and making sure that everything happens inside of the pipeline, this model will fit pretty much in any validation approach you might want to adopt.
# 
# For example, I might be interested in seeing how the model performs in a 5-fold cross-validation setting, I might want to see how much the predictions are off, if I am missing something in the data, what are the most important features. With a few helper functions, we are going to do all of it.

# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

def cv_score(df_train, y_train, kfolds, pipeline):
    oof = np.zeros(len(df_train))
    train = df_train.copy()
    
    for train_index, test_index in kfolds.split(train.values):
            
        trn_data = train.iloc[train_index][:]
        val_data = train.iloc[test_index][:]
        
        trn_target = y_train.iloc[train_index].values.ravel()
        val_target = y_train.iloc[test_index].values.ravel()
        
        pipeline.fit(trn_data, trn_target)

        oof[test_index] = pipeline.predict(val_data).ravel()
            
    return oof


def get_coef(pipe):
    imp = pipe.steps[-1][1].coef_.tolist()
    feats = pipe.steps[-2][1].get_feature_names()  # again, this is why we implemented that method
    result = pd.DataFrame({'feat':feats,'score':imp})
    result = result.sort_values(by=['score'],ascending=False)
    return result

def _plot_diagonal(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    low = min(xmin, xmax)
    high = max(xmin, xmax)
    scl = (high - low) / 100
    
    line = pd.DataFrame({'x': np.arange(low, high ,scl), # small hack for a diagonal line
                         'y': np.arange(low, high ,scl)})
    ax.plot(line.x, line.y, color='black', linestyle='--')
    
    return ax


def plot_predictions(data, true_label, pred_label, feature=None, hue=None, legend=False):
    
    tmp = data.copy()
    tmp['Prediction'] = pred_label
    tmp['True Label'] = true_label
    tmp['Residual'] = tmp['True Label'] - tmp['Prediction']
    
    diag = False
    alpha = 0.7
    label = ''
    
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    
    if feature is None:
        feature = 'True Label'
        diag = True
    else:
        legend = 'full'
        sns.scatterplot(x=feature, y='True Label', data=tmp, ax=ax[0], label='True',
                         hue=hue, legend=legend, alpha=alpha)
        label = 'Predicted'
        alpha = 0.4

    sns.scatterplot(x=feature, y='Prediction', data=tmp, ax=ax[0], label=label,
                         hue=hue, legend=legend, alpha=alpha)
    if diag:
        ax[0] = _plot_diagonal(ax[0])
    
    sns.scatterplot(x=feature, y='Residual', data=tmp, ax=ax[1], 
                    hue=hue, legend=legend, alpha=0.7)
    ax[1].axhline(y=0, color='r', linestyle='--')
    
    ax[0].set_title(f'{feature} vs Predictions')
    ax[1].set_title(f'{feature} vs Residuals')


# In[ ]:


lasso_oof = cv_score(train_set, y, folds, lasso_pipe)

lasso_oof[:10]


# We have our predictions and we can see the coefficients of our regression

# In[ ]:


get_coef(lasso_pipe)  # it has been fitted in the cv_score function
# to be fair, these coefficients refer only to the last of the 5 folds


# Now we want to see if the predictions are too far off or if there is something odd in the residual plot (I suggest to read about them as they are very useful tools for diagnosing something wrong in your model)

# In[ ]:


plot_predictions(train_set, y, lasso_oof)


# There is a big outlier in our prediction and a visible pattern in the residual plot, both things that would require further investigation.
# 
# We can also plot the residuals against the most important features, for example

# In[ ]:


plot_predictions(train_set, y, lasso_oof, feature='GrLivArea')


# And there we find that our prediction so much off with respect to the real value was indeed a house too cheap for its size (to be fair, this is one of the outliers that everybody know about and they are documented in the official documentation).
# 
# So far, we have used the *default* parameters of our pipeline but we know that there is a better configuration thanks to our GridSearch. Let's see if something changes.

# In[ ]:


numeric_pipe = Pipeline([('fs', feat_sel('numeric')),
                         ('imputer', df_imputer(strategy='mean')),  # tuned above
                         ('transf', tr_numeric(SF_room=True))])  # tuned above


cat_pipe = Pipeline([('fs', feat_sel('category')),
                     ('imputer', df_imputer(strategy='most_frequent')), 
                     ('ord', make_ordinal(['BsmtQual', 'KitchenQual','GarageQual',
                                           'GarageCond', 'ExterQual', 'HeatingQC'])), 
                     ('recode', recode_cat()), 
                     ('dummies', dummify(drop_first=True))])  # tuned above


processing_pipe = FeatureUnion_df(transformer_list=[('cat_pipe', cat_pipe),
                                                    ('num_pipe', numeric_pipe)])

lasso_pipe = Pipeline([('gen_cl', general_cleaner()), 
                 ('processing', processing_pipe),
                  ('scl', df_scaler()), ('lasso', Lasso(alpha=0.01))])  # tuned above

lasso_oof = cv_score(train_set, y, folds, lasso_pipe)

get_coef(lasso_pipe)


# In[ ]:


plot_predictions(train_set, y, lasso_oof)


# In[ ]:


plot_predictions(train_set, y, lasso_oof, feature='GrLivArea')


# The coefficients are a bit different, but we did not solved much. This was expected since we were not changing too much from the default.
# 
# We can make further use of the fact that we are working with a pipeline and directly apply it to the test set and see if the behavior changes.

# In[ ]:


lasso_pred = lasso_pipe.predict(test_set)

plot_predictions(test_set, y_test, lasso_pred)


# In[ ]:


plot_predictions(test_set, y_test, lasso_pred, feature='GrLivArea')


# For the fans of the numeric metrics

# In[ ]:


print('Score in 5-fold cv')
print(f'\tRMSE: {round(np.sqrt(mean_squared_error(y, lasso_oof)), 5)}')
print(f'\tMAE: {round(mean_absolute_error(np.expm1(y), np.expm1(lasso_oof)), 2)} dollars')
print('Score on holdout test')
print(f'\tRMSE: {round(np.sqrt(mean_squared_error(y_test, lasso_pred)), 5)}')
print(f'\tMAE: {round(mean_absolute_error(np.expm1(y_test), np.expm1(lasso_pred)), 2)} dollars')


# Naturally, even though so far we didn't even loaded the test set from Kaggle, we can apply our pipeline to truly unseen data 

# In[ ]:


df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sub = df_test[['Id']].copy()

predictions = lasso_pipe.predict(df_test)


# And we again have to put no effort to make our pipeline work on new data. This is important for 2 reasons:
# 
# * we can put all our effort in making the model better rather than fighting with messy code
# * we are virtually ready to send our model to our client and it is ready to use
# 
# 
# ## Using this code in your Notebooks
# 
# After this notebook was first created, Kaggle implemented a very nice feature: import utility script. Therefore I created a utility script with the general parts of this pipeline (so nothing that refers to this specific competition). You find it here https://www.kaggle.com/lucabasa/df-pipeline
# 
# A simple example of how to use it is the following:
# 
# * add the utility script via the menu of your Notebook (File-> Add utility script)
# * import the script

# In[ ]:


import df_pipeline as dfp


# In[ ]:


dummifier = dfp.dummify()

tmp = train_set[['HouseStyle']].copy()

tmp = dummifier.transform(tmp)

tmp.sample(5)


# I hope this script can help you and that this notebook was useful for you to better understand the advantages and the functioning of a Pipeline. Please let me know if something is not clear or incorrect.
# 
# Cheers

# In[ ]:





# In[ ]:




