#!/usr/bin/env python
# coding: utf-8

# # Titanic exercise - trying to apply some knowledge from micro-courses

# # Setup

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn import preprocessing, metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import FactorAnalysis
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import os


# # I start from defining some functions and classes which will serve to train, evaluate and select the best model while experimenting with different compositions of features in the dataset

# # Split data helper function

# In[ ]:


def get_data_splits(dataframe, valid_fraction=0.1):
    """ Splits a dataframe into train, validation, and test sets. First, orders by 
        the column 'click_time'. Set the size of the validation and test sets with
        the valid_fraction keyword argument.
    """
    valid_rows = int(len(dataframe) * valid_fraction)
    train = dataframe[:-valid_rows * 2]
    valid = dataframe[-valid_rows * 2:]
    
    return train, valid


# # Several classes for data preprocessing
# * Column dropper
# To drop some columns
# * Missing values indicator features creator
# This will do the following add A_missing such as:

# In[ ]:


#    A     B              A     B   A_missing
# 0  4.5   ...        0  4.5   ...  FALSE
# 1  NaN   ...   =>   1  NaN   ...  TRUE
# 2  6.7   ...        2  6.7   ...  FALSE


# * Categorical features imputer
# > For missing values in the columns which correspond categorical features, this will impute some value

# In[ ]:


#     A         B            A         B 
#  0  female   ...        0  female   ...
#  1  NaN      ...   =>   1  female   ...
#  2  male     ...        2  male     ...
# 'Female' is most frequent value


# * Numerical features imputer
# > For missing values in the columns with numerical data, this will impute some value

# In[ ]:


#     A     B              A     B
#  0  4.5   ...        0  4.5   ...
#  1  NaN   ...   =>   1  5.6   ...
#  2  6.7   ...        2  6.7   ...
# 5.6 is a mean of column A


# * Count feature creator
# > This will create an additional column indicating a number of values in corresponding column:

# In[ ]:


#     A          B              A        B   A_count
# 0  female    ...        0  female     ...    2
# 1  male      ...   =>   1  male       ...    1
# 2  female    ...        2  female     ...    2


# * Discretizer
# > This will create categorical data from continuous. For example, an age range 0-80 may be binned into 4 bins: 0-20, 21-40, 41-60, 61-80
# * Custom Selector
# > This will apply feature selection using provided selector such as SelectKBest or SelectFromModel

# In[ ]:


from sklearn.pipeline import Pipeline

class CustomTransformBase:
    def __init__(self, **kwargs):
        if 'is_verbose' in kwargs.keys():
            self.is_verbose = kwargs['is_verbose']
        else:
            self.is_verbose = False
        if self.is_verbose:
            print(self.__class__.__name__)
            print(kwargs['columns'])
        self.cols = kwargs['columns']
    def get_cols(self):
        if self.is_verbose:
            print(self.__class__.__name__, ' get_cols')
        return self.cols
    def get_params(self, deep):
        return self.cols
    def fit_transform(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__, ' fit_transform')
        self.fit(X, y)
        return self.transform(X, y)

class ColumnsDropper(CustomTransformBase):
    def __init__(self, **kwargs):
        CustomTransformBase.__init__(self, **kwargs)
    def fit(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' fit')
        return self
    def transform(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' transform')
        for col in self.get_cols():
            X.drop(col, axis=1, inplace=True)
        if self.is_verbose:
            print(X)
        return X

class MissingValuesColumnCreator(CustomTransformBase):
    def __init__(self, **kwargs):
        CustomTransformBase.__init__(self, **kwargs)
    def fit(self, X, y=None):
        self.cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
        if self.is_verbose:
            print(self.__class__.__name__,' columns with missing')
            print(self.cols_with_missing)
        return self
    def transform(self, X, y=None):
        if self.cols_with_missing is None:
            self.fit(X, y)
        add_for_missing = pd.DataFrame()
        for col in self.cols_with_missing:
            add_for_missing[col + '_was_missing'] = X[col].isnull()
        add_for_missing.index = pd.RangeIndex(X.index.start, X.index.stop, X.index.step)
        return pd.concat([X, add_for_missing], axis=1)

class CatLabeler(CustomTransformBase):
    def __init__(self, **kwargs):
        CustomTransformBase.__init__(self, **kwargs)
        self.cat_labeler = LabelEncoder()
    def fit(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' fit')
        return self
    def transform(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' transform')
        for col in self.get_cols():
            X[col] = self.cat_labeler.fit_transform(X[col])
        if self.is_verbose:
            print(X)
        return X
    
class CustomImputer(CustomTransformBase):
    def __init__(self, **kwargs):
        CustomTransformBase.__init__(self, **kwargs)
    def fit(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' fit')
        self.imputer.fit(X[self.get_cols()])
        return self
    def transform(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' transform')
        X[self.get_cols()] = pd.DataFrame(self.imputer.transform(X[self.get_cols()]))
        X[self.get_cols()].columns = self.get_cols()
        if self.is_verbose:
            print(X)
        return X
    
class CatImputer(CustomImputer):
    def __init__(self, **kwargs):
        CustomImputer.__init__(self, **kwargs)
        self.imputer = SimpleImputer(strategy='most_frequent', copy=False)
    def fit(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' fit')
        self.imputer.fit(X[self.get_cols()])
        X[self.get_cols()].columns = self.get_cols()
        return self
    def transform(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' transform')
        X[self.get_cols()] = pd.DataFrame(self.imputer.transform(X[self.get_cols()]))
        X[self.get_cols()].columns = self.get_cols()
        return X
        
class NumImputer(CustomImputer):
    def __init__(self, **kwargs):
        CustomImputer.__init__(self, **kwargs)
        self.imputer = SimpleImputer(strategy='median', copy=False)
    def fit(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' fit')
        self.imputer.fit(X[self.get_cols()])
        X[self.get_cols()].columns = self.get_cols()
        return self
    def transform(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' transform')
        X[self.get_cols()] = self.imputer.transform(X[self.get_cols()])
        X[self.get_cols()].columns = self.get_cols()
        return X
    
class CountColumnsCreator(CustomTransformBase):
    def __init__(self, **kwargs):
        CustomTransformBase.__init__(self, **kwargs)
        self.countEncoder = ce.CountEncoder(cols=kwargs['columns'])
    def fit(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' fit')
        self.countEncoder.fit(X[self.get_cols()])
        return self
    def transform(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' transform')
        X_ = self.countEncoder.transform(X[self.get_cols()])
        #in case unseen categories encountered
        X_.fillna(value=0,inplace=True)
        return X.join(X_.add_suffix("_count"))

class Discretizer(CustomTransformBase):
    def __init__(self, **kwargs):
        CustomTransformBase.__init__(self, **kwargs)
        self.nbins = kwargs['bins']
        self.discretizer = KBinsDiscretizer(n_bins=self.nbins)
    def fit(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' fit')
        for col in self.get_cols():
            self.discretizer.fit(X[col].values.reshape(-1,1))
        return self
    def transform(self, X, y=None):
        if self.is_verbose:
            print(self.__class__.__name__,' transform')
        for col in self.get_cols():
            encoded = self.discretizer.transform(X[col].values.reshape(-1,1))
            binned_cols = []
            imputed_df = pd.DataFrame(encoded.toarray())
            for i in range(0,len(imputed_df.columns)):
                binned_cols.append(col+str(i))
            imputed_df.columns = binned_cols
            imputed_df.index = pd.RangeIndex(X.index.start, X.index.stop, X.index.step)
            X = pd.concat([X, imputed_df], axis=1)
            X.drop(col, axis=1, inplace=True)
        return X
    
class CustomSelector(CustomTransformBase):
    def __init__(self, **kwargs):
        CustomTransformBase.__init__(self, **kwargs)
        if 'selector' in kwargs.keys():
            self.selector = kwargs['selector']
        else:
            self.selector = None
        self.selected_columns = None
    def fit (self, X, y=None):
        if self.selector is None:
            return self
        X_new = self.selector.fit_transform(X,y)
        #print(X_new)
        selected_features = pd.DataFrame(self.selector.inverse_transform(X_new),
                                         index=X.index,
                                         columns=X.columns)

        self.selected_columns = selected_features.columns[selected_features.var() != 0]
    def transform (self, X, y=None):
        if self.selected_columns is None:
            return X
        return X[self.selected_columns]


# # Data preprocessor
# This uses a Pipeline to define a sequence of steps to pre-process the data
# 
# If create_features is True, additional features created:
# * Missing values columns
# * Count columns

# In[ ]:


class DataPreprocessor:
    def __init__(self,cols_to_drop,cat_features,numerical_cols,create_features=False, selector=None):
        dropper = ColumnsDropper(columns=cols_to_drop,is_verbose=False)
        catImputer = CatImputer(columns=cat_features,is_verbose=False)
        catLabeler = CatLabeler(columns=cat_features,is_verbose=False)
        num_imputer = NumImputer(columns=numerical_cols,is_verbose=False)
        discretizer = Discretizer(columns=numerical_cols, bins=4,is_verbose=False)
        customSelector = CustomSelector(columns=[],is_verbose=False,selector=selector)
        steps = []
        steps.append(('dropper', dropper))
        if create_features:
            missing = MissingValuesColumnCreator(columns=[],is_verbose=False)
            steps.append(('missing', missing))
        steps.append(('cat_imputer', catImputer))
        steps.append(('cat_labeler',catLabeler))
        steps.append(('num_imputer', num_imputer))
        if create_features:
            countColumnsCreator = CountColumnsCreator(columns=cat_features,is_verbose=False)
            steps.append(('count_col_creation', countColumnsCreator))
        steps.append(('discretizer', discretizer))
        steps.append(('selector', customSelector))
        self.preprocessor = Pipeline(steps=steps,verbose=True)

    def preprocess_data(self, df, y=None, is_train=False):
        if is_train:
            return pd.DataFrame(self.preprocessor.fit_transform(df, y))
        return pd.DataFrame(self.preprocessor.transform(df))

    def separate_data_to_features_and_label(train, label_name, validation=None, separate=False):
        train = train.copy()
        if validation is None and separate:
            train, validation = get_data_splits(train)
        train_y = train[label_name]
        if validation is not None:
            val_y = validation['Survived']
        else:
            val_y = None
        train.drop('Survived', axis=1, inplace=True)
        if validation is not None:
            validation.drop('Survived', axis=1, inplace=True)
        return train, train_y, validation, val_y
    def addCustomColumn(miniproc, dataset, **kwargs):
        col = miniproc(dataset, **kwargs)
        return pd.concat([dataset,col],axis=1)


# Since we use categorical features, I try FactorAnalysis in attempt to reduce dimensions rather than PCA

# In[ ]:


def get_number_of_components(fa_train_x, fa_train_y, fa_val_x, fa_val_y):
    best_fa_score = None
    best_fa_ncomp = None
    best_fa = None
    for i in range(2, len(fa_train_x.columns), 2):
        fa = FactorAnalysis(n_components=i)
        fa = fa.fit(fa_train_x)
        score = fa.score(fa_val_x)
        if best_fa_score is None or score > best_fa_score:
            best_fa_score = score
            best_fa_ncomp = i
            best_fa = fa
    print('FA best_score ', best_fa_score, ' with ', best_fa_ncomp, ' components')
    best_fa = FactorAnalysis(n_components=best_fa_ncomp)
    best_fa = best_fa.fit(fa_train_x)
    loading_matrix = pd.DataFrame(best_fa.components_,columns=fa_train_x.columns)
    loading_matrix = np.transpose(loading_matrix)
    print(loading_matrix)
    return best_fa


# There are two sets from which the parameters for RandomForest model are selected. Each member of a set contains a number of possible values. One set is used for RandomizedSerachCV and another is for GridSearchCV

# For RandomizedSearchCV:

# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# For GridSearchCV:

# In[ ]:


n_estimators = [2000, 1600, 1400, 1000, 400]
max_features = ['auto', 'sqrt']
max_depth = [10, 30, 60, 90]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2]
bootstrap = [True, False]
tuned_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# A function to train and evaluate RandomForest with SelectFromModel

# In[ ]:


def evaluate_rf_with_select(X, y, val_X, val_y, class_weights, preprocessor):
    rfModel = RandomForestClassifier(class_weight=class_weights)
        
    if type(X) == np.ndarray:
        X_cv = np.concatenate((X, val_X))
    else:
        X_cv = pd.concat([X, val_X], axis=0)
    if type(y) == np.ndarray:
        y_cv = np.concatenate((y, val_y))
    else:
        y_cv = pd.concat([y, val_y], axis=0)

    #choose one of the following
    #RandomizedSearchCV
    rf_random = RandomizedSearchCV(estimator = rfModel, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    #GridSearch
    #rf_random = GridSearchCV(estimator = rfModel, param_grid = tuned_grid, 
    #                      cv = 3, n_jobs = -1, verbose = 2)
    # Fit the random search model
    rf_random.fit(X_cv, y_cv)
    print(rf_random.best_params_)
    model = SelectFromModel(rf_random.best_estimator_, prefit=True)
    X_new = model.transform(X_cv)
    selected_features = pd.DataFrame(model.inverse_transform(X_new),
                                      index=X_cv.index,
                                      columns=X_cv.columns)

    selected_columns = selected_features.columns[selected_features.var() != 0]
    rf_random.best_estimator_.fit(selected_features[selected_columns], y_cv)
    preds_val = rf_random.best_estimator_.predict(val_X[selected_columns])
    f1score = f1_score(val_y, preds_val)
    print("RF RandomCV F1 score: ", f1score)
    print("RF RandomCV Training AUC score: ", metrics.roc_auc_score(val_y, preds_val))
    return f1score, rf_random.best_estimator_, selected_columns, None


# A function to train and evaluate RandomForest

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
def evaluate_rf_with_search_cv(X, y, val_X, val_y, class_weights, preprocessor):
    if type(X) == np.ndarray:
        X_cv = np.concatenate((X, val_X))
    else:
        X_cv = pd.concat([X, val_X], axis=0)
    if type(y) == np.ndarray:
        y_cv = np.concatenate((y, val_y))
    else:
        y_cv = pd.concat([y, val_y], axis=0)

    rfModel = RandomForestClassifier(class_weight=class_weights)
    #choose one of the following
    #RandomizedSearchCV
    rf_random = RandomizedSearchCV(estimator = rfModel, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    #GridSearch
    #rf_random = GridSearchCV(estimator = rfModel, param_grid = tuned_grid, 
    #                      cv = 3, n_jobs = -1, verbose = 2)
    # Fit the random search model
    rf_random.fit(X_cv, y_cv)
    print(rf_random.best_params_)
    preds_val = rf_random.best_estimator_.predict(val_X)
    f1score = f1_score(val_y, preds_val)
    print("RF RandomCV F1 score: ", f1score)
    print("RF RandomCV Training AUC score: ", metrics.roc_auc_score(val_y, preds_val))
    return f1score, rf_random.best_estimator_, None, None


# This function will apply FactorAnalysis before evaluating RandomForest

# In[ ]:


def evaluate_rf_with_fa(X, y, val_X, val_y, class_weights, preprocessor):
    fa = get_number_of_components(X, y, val_X, val_y)
    X = fa.transform(X)
    val_X = fa.transform(val_X)
    #f1score,clf, selected_columns, _ = evaluate_rf(X, y, val_X, val_y, class_weights, preprocessor)
    f1score,clf, selected_columns, _ = evaluate_rf_with_search_cv(X, y, val_X, val_y, class_weights, preprocessor)
    return f1score,clf, selected_columns,fa 


# This function is used to train and evaluate LightGMB classifier

# In[ ]:


def evaluate_lgbm(X, y, val_X, val_y, class_weights, preprocessor):
    lgbmClf = lgb.LGBMClassifier()
    lgbmClf.fit(X, y)
    preds_val = lgbmClf.predict(val_X)
    f1score = f1_score(val_y, preds_val)
    print("LGBM F1 score: ", f1score)
    print("LGBM Training AUC score: ", metrics.roc_auc_score(val_y, preds_val))
    return f1score, lgbmClf, None, None


# GradientBoost classifier train and evaluate

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
def evaluate_GradientBoost(X, y, val_X, val_y, class_weights, preprocessor):
    gbClf = GradientBoostingClassifier()
    gbClf.fit(X, y)
    preds_val = gbClf.predict(val_X)
    f1score = f1_score(val_y, preds_val)
    print("GradientBoost F1 score: ", f1score)
    print("GradientBoost Training AUC score: ", metrics.roc_auc_score(val_y, preds_val))
    return f1score, gbClf, None, None


# CatBoost classifier train and evaluate

# In[ ]:


import catboost
from catboost import CatBoostClassifier
def evaluate_catboost(X, y, val_X, val_y, class_weights, preprocessor):
    catboost = CatBoostClassifier(verbose=0, n_estimators=100, class_weights=[class_weights[1],class_weights[0]])
    catboost.fit(X, y)
    preds_val = catboost.predict(val_X)
    f1score = f1_score(val_y, preds_val)
    print("CatBoost F1 score: ", f1score)
    print("CatBoost Training AUC score: ", metrics.roc_auc_score(val_y, preds_val))
    return f1score, catboost, None, None


# XGB classifier train and evaluate

# In[ ]:


import xgboost
from xgboost import XGBClassifier
def evaluate_xgb(X, y, val_X, val_y, class_weights, preprocessor):
    xgbClf = XGBClassifier()
    xgbClf.fit(X, y)
    preds_val = xgbClf.predict(val_X)
    f1score = f1_score(val_y, preds_val)
    print("XGB F1 score: ", f1score)
    print("XGB Training AUC score: ", metrics.roc_auc_score(val_y, preds_val))
    return f1score, xgbClf, None, None

Here is a list of classifiers to evaluate
# In[ ]:


classifiers = [(evaluate_rf_with_select, 'RandomForest_with_SelectFromModel'),
               (evaluate_rf_with_fa, 'RandomForest_with_FactorAnalysis'),
            #commented this to shorten train time. Feel free to uncomment and try
              # (evaluate_lgbm,'LGBM'),
              # (evaluate_GradientBoost, 'GradientBoost'),
              # (evaluate_catboost, 'CatBoost'),
              # (evaluate_xgb, 'XGB'),
              # (evaluate_rf_with_search_cv, 'RandomForestRandomCV')
              ]


# And finally a function called for each dataset we want to train and evaluate the models, select the best model, make predictions and save the results

# In[ ]:


def train_predict_and_save(X, y, valX, val_y, test, preprocessor, extension, class_weights=None):
    best_f1_score = None
    selected_clf = None
    best_selected_columns = None
    best_fa = None
    best_clf_name = None
    print('Train set')
    print(X)
    print('Val set')
    print(valX)
    for get_clf in classifiers:     
        f1score,clf, selected_columns, fa = get_clf[0](X, y, valX, val_y, class_weights, preprocessor)
        print('f1 score ', f1score, ' ', best_f1_score)
        if best_f1_score is None or best_f1_score < f1score:
            best_f1_score = f1score
            selected_clf = clf
            best_clf_name = get_clf[1]
            best_selected_columns = selected_columns
            best_fa = fa
    passenger_id = test['PassengerId']
    X_test = preprocessor.preprocess_data(test,is_train=False)
    if best_selected_columns is not None:
        X_test = X_test[best_selected_columns]
    if best_fa is not None:
        X_test = best_fa.transform(X_test)
    print('Test set')
    print(X_test)
    print('Selected model ', best_clf_name, ' ', selected_clf)
    print('Selected mode F1 scores ', best_f1_score)
    predictions = selected_clf.predict(X_test)
    data_to_output = pd.DataFrame({'PassengerId': passenger_id, 'Survived':predictions})
    data_to_output.to_csv('predict_rf_'+extension+best_clf_name+'.csv', index=False)


# **Load Titanic dataset**

# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')


# First look at the dataset

# In[ ]:


explore_set = train.copy()
explore_set.head(10)


# The targer values are in 'Survived' column. We have 11 more columns.
# It looks like there are few columns which are unlikely to help but create noise:
# - PassengerId
# - Name
# - Ticket

# Now we check how imbalanced our classes are:

# In[ ]:


print('Survived ',explore_set.loc[explore_set['Survived'] == 1]['Survived'].count())
print('Died ',explore_set.loc[explore_set['Survived'] == 0]['Survived'].count())


# In[ ]:


sns.countplot(explore_set['Survived'])


# We have 342 survived passenger agains 549 who died. We can later try to:
# - Apply class weights
# - Resample the minor (survived positives) class
# We continue to explore the dataset. 
# We look at distributions of various features:

# In[ ]:


sns.countplot(explore_set['Sex'])


# In[ ]:


sns.countplot(explore_set['Pclass'])


# Now we are going to explore some dependencies

# In[ ]:


g = sns.catplot(x="Pclass", hue="Sex", col="Survived",
                data=explore_set, kind="count",
                height=4, aspect=.7);


# Now we check how many non-NA entries are in each column:

# In[ ]:


explore_set.count()


# We can see while some columns don't have NA entries at all (those which count 891) some columns have few NAs (Age, Embarked), one column has only 204 non-NA of 891. We are going to ignore the column with the very high number of NAs and impute the columns where we have few NAs.

# Now this is the time to prepare our data. We've already said we want to drop some columns and impute missing values. Some columns contain numerical data, however the numbers actually represent different categories. For example, Pclass columns has values 1, 2 and 3. These numbers correspond different cabin classes. We're going to label such columns to make it usable by the RandomForest algorithm.
# However, one columns with continuous numerical data - this is the Age and Fare columns. To make it usable for RandomForest, we're going to discretize the values in the Age and Fare columns by binning

# Until now, we've identified the following steps to preprocess our data:
# - drop PassengerId, Name, Ticket and Cabin columns
# - Impute Age and Embarked columns
# - Discretize the Age column

# Since we have imbalanced classes, calculate class weights:

# In[ ]:


class_weighted_train = train.copy()
y = class_weighted_train['Survived']
weight_for_0 = (1 / (class_weighted_train['Survived'].map(lambda is_survived: is_survived == 0)).sum())*(y.count())/2.0 
weight_for_1 = (1 / (class_weighted_train['Survived'].map(lambda is_survived: is_survived == 1)).sum())*(y.count())/2.0

class_weights = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


# Create labels, drop labels from train and validation sets:

# In[ ]:


class_weighted_train, train_y, class_weighted_valid, val_y = DataPreprocessor.separate_data_to_features_and_label(train,  'Survived', separate=True)


# Preprocess the data, choose the best number of estimators and max_depth (of the trees). 
# Then train on whole train set, predict and save results:

# In[ ]:


cols_to_drop = ['Cabin', 'Ticket', 'Name', 'PassengerId']
cat_features = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']
numerical_cols = ['Age','Fare']


# In[ ]:


preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols, create_features=False)
train_X = preprocessor.preprocess_data(class_weighted_train,is_train=True)
val_X = preprocessor.preprocess_data(class_weighted_valid,is_train=False)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train_predict_and_save(train_X, train_y, val_X, val_y, test, preprocessor, 'class_weighted', class_weights=class_weights)


# Now we're going to create some features (missing values and count columns, we set create_features=True for that) and repeat the steps:

# In[ ]:


class_weighted_train, train_y, class_weighted_valid, val_y =     DataPreprocessor.separate_data_to_features_and_label(train,  
                                                        'Survived', separate=True)
preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols,create_features=True)
train_X = preprocessor.preprocess_data(class_weighted_train,is_train=True)
val_X = preprocessor.preprocess_data(class_weighted_valid,is_train=False)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train_predict_and_save(train_X, train_y, val_X,val_y, test, preprocessor, 'class_weighted_more_features', class_weights=class_weights)


# Now we're going to remove SibSp and Parch features and repeat:

# In[ ]:


cols_to_drop = ['Cabin', 'Ticket', 'Name', 'PassengerId', 'SibSp', 'Parch']
cat_features = ['Sex', 'Pclass', 'Embarked']
numerical_cols = ['Age','Fare']
class_weighted_train, train_y, class_weighted_valid, val_y = DataPreprocessor.separate_data_to_features_and_label(train,  'Survived', separate=True)
preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols,create_features=True)
train_X = preprocessor.preprocess_data(class_weighted_train,is_train=True)
val_X = preprocessor.preprocess_data(class_weighted_valid,is_train=False)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train_predict_and_save(train_X, train_y, val_X, val_y, test, preprocessor, 'class_weighted_more_features2', class_weights=class_weights)


# Redefine category features to include SibSp and Parch

# In[ ]:


cols_to_drop = ['Cabin', 'Ticket', 'Name', 'PassengerId']
cat_features = ['Sex', 'Pclass', 'Embarked', 'SibSp', 'Parch']
numerical_cols = ['Age','Fare']


# Now we try to apply Factor Analysis to try to select the most important features

# In[ ]:


from sklearn.decomposition import FactorAnalysis

train_X, train_y, val_X, val_y = DataPreprocessor.separate_data_to_features_and_label(train,  'Survived', separate=True)
preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols,create_features=False)
train_X = preprocessor.preprocess_data(train_X,is_train=True)
val_X = preprocessor.preprocess_data(val_X,is_train=False)
fa = get_number_of_components(train_X, train_y, val_X, val_y)


# we can see that FactorAnalysis proposes the dimension reduction to 8 features

# This time we're going to do chi-squared feature selection. We tell SlectKBest to use all the features, then we print the scores

# In[ ]:


selector = SelectKBest(score_func=chi2, k='all')
select_k_best_train, train_y, _, _ = DataPreprocessor.separate_data_to_features_and_label(train,  'Survived')

preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols)
train_X = preprocessor.preprocess_data(select_k_best_train,is_train=True)
fs = selector.fit(train_X, train_y)
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()


# It looks we have up to 5 features to select. This time we're going to tell SelectKBest to select 4 features, then train and predict:

# In[ ]:


selector = SelectKBest(chi2, k=4)
select_k_best_train, train_y, val_X, val_y = DataPreprocessor.separate_data_to_features_and_label(train,  'Survived',separate=True)

preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols,selector=selector)
train_X = preprocessor.preprocess_data(select_k_best_train,y=train_y,is_train=True)
val_X = preprocessor.preprocess_data(val_X,y=val_y,is_train=False)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train_predict_and_save(train_X, train_y, val_X, val_y, test, preprocessor, 'selectk_chi2', class_weights=class_weights)


# We repeat the feature selection this time using mutual information classification

# In[ ]:


selector = SelectKBest(score_func=mutual_info_classif, k='all')
select_k_best_train, train_y, _, _ = DataPreprocessor.separate_data_to_features_and_label(train,  'Survived')
preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols)
train_X = preprocessor.preprocess_data(select_k_best_train,y=train_y, is_train=True)
fs = selector.fit(train_X, train_y)
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()


# Looks not more than 4 features is worth to try to select

# In[ ]:


selector = SelectKBest(mutual_info_classif, k=4)
select_k_best_train, train_y, val_X, val_y  = DataPreprocessor.separate_data_to_features_and_label(train,  'Survived', separate=True)

preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols,selector=selector)
train_X = preprocessor.preprocess_data(select_k_best_train, y=train_y, is_train=True)
val_X = preprocessor.preprocess_data(val_X, y=train_y, is_train=False)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train_predict_and_save(train_X, train_y, val_X, val_y, test, preprocessor, 'selectk_mutualinfo', class_weights=class_weights)


# Now it is time to try oversampling (the dataset is imbalanced with higher precense of negative cases so we're going to make increase a number of positive ones):

# In[ ]:


def oversample_dataset(train, column, minor_criteria, major_criteria):
    oversampled_train = train.copy()
    minor_features = oversampled_train.loc[oversampled_train[column] == minor_criteria]
    major_features = oversampled_train.loc[oversampled_train[column] == major_criteria]
    minor_labels = oversampled_train.loc[oversampled_train[column] == minor_criteria]
    major_labels = oversampled_train.loc[oversampled_train[column] == major_criteria]
    minor_labels = minor_labels[column]
    major_labels = major_labels[column]
    minor_features.drop(column, axis=1, inplace=True)
    major_features.drop(column, axis=1, inplace=True)
    #generate row indexes for minor class
    ids = np.arange(len(minor_features))
    #choose row indexes, amount equals major class size
    choices = np.random.choice(ids, len(major_features))
    #create resampled minor those size equals major size
    res_minor_features = minor_features.iloc[choices]
    res_minor_labels = minor_labels.iloc[choices]
    #concatenate minor and major
    resampled_features = pd.concat([res_minor_features, major_features])
    resampled_labels = pd.concat([res_minor_labels, major_labels])
    #create random order
    order = np.arange(len(resampled_labels))
    np.random.shuffle(order)
    #apply random order
    resampled_features = resampled_features.iloc[order]
    resampled_labels = resampled_labels.iloc[order]
    resampled_features.reset_index(inplace=True)
    resampled_features.drop('index', axis=1,inplace=True)
    resampled_labels = resampled_labels.reset_index()
    return resampled_features, resampled_labels


# In[ ]:


resampled_features, resampled_labels = oversample_dataset(train, 'Survived', 1.0, 0.0)


# Try to select a model with resampled set:

# In[ ]:


print(resampled_features)
print(resampled_labels)
oversampled_set = pd.concat([resampled_features, resampled_labels],axis=1)
oversampled_set.drop('index', axis=1,inplace=True)
train_X, train_y, val_X, val_y  = DataPreprocessor.separate_data_to_features_and_label(oversampled_set,  'Survived', separate=True)
preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols)
train_X = preprocessor.preprocess_data(train_X,is_train=True)
val_X = preprocessor.preprocess_data(val_X,is_train=False)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train_predict_and_save(train_X, train_y, val_X, val_y, test, preprocessor, 'oversampled', {0:1, 1:1})


# And another attempt with chi2 squared selection with the oversampled dataset:

# In[ ]:


resampled_features, resampled_labels = oversample_dataset(train, 'Survived', 1.0, 0.0)
oversampled_set = pd.concat([resampled_features, resampled_labels],axis=1)
oversampled_set.drop('index', axis=1,inplace=True)
train_X, train_y, val_X, val_y  = DataPreprocessor.separate_data_to_features_and_label(oversampled_set,  'Survived', separate=True)
selector = SelectKBest(chi2, k=4)
preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols,selector=selector)
train_X = preprocessor.preprocess_data(train_X,y=train_y,is_train=True)
val_X = preprocessor.preprocess_data(val_X,y=val_y,is_train=False)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train_predict_and_save(train_X, train_y, val_X, val_y, test, preprocessor, 'resampled_selectk_chi2', class_weights={0:1, 1:1})


# Now we're going to create another column, FamilySize. Below is the lambda function for this:

# In[ ]:


familySizeColCreator = lambda dataset, **kwargs: pd.DataFrame(dataset[kwargs['col1']] + dataset[kwargs['col2']], columns=[kwargs['col3']])


# Create a dataset with FamilySize feature and select a model:

# In[ ]:


extended_train = train.copy()
extended_train = DataPreprocessor.addCustomColumn(familySizeColCreator,extended_train, col1='SibSp', col2='Parch',col3='FamilySize')
print(extended_train)
resampled_features, resampled_labels = oversample_dataset(extended_train, 'Survived', 1.0, 0.0)
oversampled_set = pd.concat([resampled_features, resampled_labels],axis=1)
oversampled_set.drop('index', axis=1,inplace=True)
train_X, train_y, val_X, val_y  = DataPreprocessor.separate_data_to_features_and_label(oversampled_set,  'Survived', separate=True)
cols_to_drop = ['Cabin', 'Ticket', 'Name', 'PassengerId', 'SibSp', 'Parch']
cat_features = ['Sex', 'Pclass', 'Embarked', 'FamilySize']
numerical_cols = ['Age','Fare']
preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols)
train_X = preprocessor.preprocess_data(train_X,is_train=True)
val_X = preprocessor.preprocess_data(val_X,is_train=False)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
family_size_column = pd.DataFrame(test['SibSp'] + test['Parch'], columns=['FamilySize'])
test = pd.concat([test,family_size_column],axis=1)
train_predict_and_save(train_X, train_y, val_X, val_y, test, preprocessor, 'resampled_selectk_chi2', class_weights={0:1, 1:1})


# Now we're going to create another column, TravelAlone. Below is the lambda function for this:

# In[ ]:


travelAloneColCreator = lambda dataset, **kwargs: pd.DataFrame((dataset[kwargs['col1']] + dataset[kwargs['col2']])>0, columns=[kwargs['col3']])


# Create a dataset with TravelAlone feature and select a model:

# In[ ]:


travel_alone_train = train.copy()
travel_alone_train = DataPreprocessor.addCustomColumn(travelAloneColCreator,travel_alone_train, col1='SibSp', col2='Parch',col3='TravelAlone')
print(travel_alone_train)
resampled_features, resampled_labels = oversample_dataset(travel_alone_train, 'Survived', 1.0, 0.0)
oversampled_set = pd.concat([resampled_features, resampled_labels],axis=1)
oversampled_set.drop('index', axis=1,inplace=True)
train_X, train_y, val_X, val_y  = DataPreprocessor.separate_data_to_features_and_label(oversampled_set,  'Survived', separate=True)
cols_to_drop = ['Cabin', 'Ticket', 'Name', 'PassengerId', 'SibSp', 'Parch']
cat_features = ['Sex', 'Pclass', 'Embarked', 'TravelAlone']
numerical_cols = ['Age','Fare']
preprocessor = DataPreprocessor(cols_to_drop, cat_features, numerical_cols)
train_X = preprocessor.preprocess_data(train_X,is_train=True)
val_X = preprocessor.preprocess_data(val_X,is_train=False)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
travel_alone_column = pd.DataFrame((test['SibSp'] + test['Parch'])>0, columns=['TravelAlone'])
test = pd.concat([test,travel_alone_column],axis=1)
train_predict_and_save(train_X, train_y, val_X, val_y, test, preprocessor, 'travel_alone', class_weights={0:1, 1:1})

