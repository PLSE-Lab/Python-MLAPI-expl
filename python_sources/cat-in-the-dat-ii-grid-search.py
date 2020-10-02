#!/usr/bin/env python
# coding: utf-8

# # Cat in the Dat II - Grid Search and Pipeline
# This kernel is an exploration of using grid search and pipeline to optimize a binary classifier on the Cat in the Dat II data set.  Only the ordinal features are considered for this example, but the code easily supports the remaining feature classes, as shown in my other kernels.
# 
# The structure follows my pattern for building a pipeline and examining the transformed features at the end of the pipe.  While this adds complexity to an otherwise simple flow, the traceability of transformed features is priceless to understand the interactions in more sophisticated problems.
# 
# ### &#x1F534; CTL
# The CTL class is a singleton with static attributes used to control the behavior throughout the kernel.  Most notably, each of the input features can be turned on/off independently without digging through the code before invoking a long running commit.
# 
# ### &#x1F534; No data visualization
# This kernel does not do any visualization of the data, as we are only doing gridsearch/pipeline development here. My other, real, kernels contain input and label exploration.
# 
# ### &#x1F534; Scikit-learn get_feature_names()
# There is an ongoing effort in Scikit-learn's GitHub repo to implement `get_feature_names()` throughout the pipeline infrastructure (transformers, imputers, etc.).  The version of sklearn available by default in Kaggle contains only a partial implementation, and is most notably missing from Pipeline which is what glues it all together.
# 
# This kernel contains a limited sub-set of the code from sklearn's PR, with minor tweaks to get it to run.  Since this kernel only uses Pandas DataFrame, the implementation is much simpler than what is required in the full Scikit-learn solution.  The classes in this kernel that extend sklearn append *WithNames* to the classname and are contained in a single code cell.
# 
# ### &#x1F534; Custom Transformers
# This kernel contains several custom transformers useful in the Cat in the Dat II competition.  The transformations that they impelemnt are typically simple one-liners.  The surrounding code is there to support `get_feature_names()`, and to support analysis of the behaviors and datatypes passed through the Pipeline stack.
# 
# ### &#x1F534; Grid Search
# `RandomizedSearchCV` is used to evaluate different hyperparameter settings for optimal performance.  Scoring of `roc_auc` is used for Cat in the Dat II, as specified in the competition documents.
# 
# Most examples you see in other kernels run a grid search on the model only, not on a whole pipeline that includes input feature preprocessing.  Doing it on the full pipeline is a little slower since it will run preprocessing for each cross validation, but you gain so much in reduced bias that can be introduced by mixing validation data into training, either by design or accident.
# 
# ### &#x1F534; Transformed Feature Traceability
# Data about the transformed features in the best model is visualized at the end.  This is where all the effort to maintain `get_feature_names()` pays off, with transparent mappings of raw inputs to transformed output.

# In[ ]:


get_ipython().run_cell_magic('javascript', '', '/* Disable autoscrolling in kernel notebook */\nIPython.OutputArea.auto_scroll_threshold = -1')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Import libraries

# In[ ]:


from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns", 40)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

import xgboost as xgb


# In[ ]:


# set seaborn to work better with jupyter dark theme
sns.set_style("whitegrid")


# # Setup control constants

# In[ ]:


class CTL:
    """
    Constants and control switches to let me turn on/off certain code blocks from one spot.
    """
    random_state = 17
    
    input_dir = "/kaggle/input/cat-in-the-dat-ii"
    # set to None to write to current directory.
    output_dir = None
    
    train_file_name = "train.csv"
    test_file_name = "test.csv"
    
    # kaggle settings:
    sub_filename_pattern = "submission.csv"
    sub_file_date_pattern = None
    # local PC settings:
    #sub_filename_pattern = "sub-{0:s}-{1:0.5f}.csv"
    #sub_file_date_pattern = "%Y%m%d%H%M%S"
    
    enable_bin_features = False
    
    # enable_nom_features is global control for all the nom's
    enable_nom_features = False
    enable_nom_oh = True
    # The nominal high cardinality features are the hex strings, let's try mean encoding
    enable_nom_mean = True
    
    # enable_ord_features is global control for all the ord's
    enable_ord_features = True
    enable_ord0 = True
    enable_ord1 = True
    enable_ord2 = True
    enable_ord345 = True

   # enable_cyclical_features is global control for all the cyc's
    enable_cyclical_features = False
    enable_cyclical_month = True
    enable_cyclical_day = True
    
    # set active_model to one of the configured models
    model_xgboost = "xgboost"
    model_randomforest = "randomforest"
    active_model = ""
    
    gridsearch_n_iter = 10
    gridsearch_cv = 5
    gridsearch_scoring = "roc_auc"
    
    # for nested params, prefix with the name of the proper step in the pipeline
    gridsearch_param_grid = {
        'model__max_depth': [3, 4, 5, 6, 7],
        'model__learning_rate': [0.01, 0.05, 0.10, 0.20],
        'model__subsample': [0.5, 0.6, 0.7, 0.8, 0.9],
        'model__colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        'model__reg_alpha': [0.5, 1.0, 2.0, 5.0, 10.0],
        'model__reg_lambda': [0.5, 1.0, 2.0, 5.0, 10.0],
        'model__num_leaves': [7, 15, 31, 63, 127],
        'model__n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450],
        'model__min_data_in_leaf': [1, 3, 5, 10, 15, 25],
    }

    

# specify the active model outside the class def
CTL.active_model = CTL.model_xgboost


# # Load the Train and Test data

# In[ ]:


train_df = pd.read_csv(os.path.join(CTL.input_dir, CTL.train_file_name), index_col='id')
test_df = pd.read_csv(os.path.join(CTL.input_dir, CTL.test_file_name), index_col='id')

binary_features = [x for x in train_df.columns if x.startswith("bin")]
nominal_features = [x for x in train_df.columns if x.startswith("nom")]
ordinal_features = [x for x in train_df.columns if x.startswith("ord")]
cyclical_features = ["day", "month"]
label_column = "target"

print("Shape of raw train data:", train_df.shape)
print("Shape of raw test data :", test_df.shape)


# # Implementing Feature Names
# There is an unfinished PR in sklearn's GitHub repository, "RFC Implement Pipeline get feature names #12627".
# 
# The following code block is a minimalist way to implement get_feature_names in the transformers that I'm using in this notebook.

# In[ ]:


# This is minor modification of GitHub PR "RFC Implement Pipeline get feature names #12627"
class PipelineWithNames(Pipeline):
    def __init__(self, steps, memory=None, verbose=0):
        super().__init__(steps, memory, verbose)
        
    def get_feature_names(self, input_features=None):
        """Get feature names for transformation.
        Transform input features using the pipeline.
        If the last step is a transformer, it's included
        in the transformation, otherwise it's not.
        Parameters
        ----------
        input_features : array-like of string
            Input feature names.
        Returns
        -------
        feature_names : array-like of string
            Transformed feature names
        """
        feature_names = input_features
        with_final = hasattr(self._final_estimator, "transform")
        
        for i, name, transform in self._iter(with_final=with_final):
            if not hasattr(transform, "get_feature_names"):
                raise TypeError("Transformer {} does provide"
                                " get_feature_names".format(name))
            try:
                feature_names = transform.get_feature_names(
                    input_features=feature_names)
            except TypeError:
                    feature_names = transform.get_feature_names()
        return feature_names


class SimpleImputerWithNames(SimpleImputer):
    def __init__(self, missing_values=np.NaN, strategy='mean', fill_value=None, verbose=0,
                 copy=True, add_indicator=False):
        super().__init__(missing_values, strategy, fill_value, verbose, copy, add_indicator)
        self.feature_names = None
        
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names = X.columns
            if self.verbose:
                print("SimpleImputerWithNames.fit(", type(X), ") saving feature names:",
                      self.feature_names)
        else:
            if self.verbose:
                print("SimpleImputerWithNames.fit() input X (type=", type(X),
                      ") has no 'columns' attr, cannot save feature names")

        return super().fit(X, y)

    def transform(self, X):
        X_transform = super().transform(X)
        if self.verbose:
            print("SimpleImputerWithNames.transform(type=", type(X),
                  ") super return type is ", type(X_transform))
        # this next line ties this impl to pandas...
        # ...but this is only used for one notebook that's all pandas, so i'm ok with it
        X_df = pd.DataFrame(data=X_transform, columns=self.feature_names)
        return X_df
    
    def get_feature_names(self, input_features=None):
        return self.feature_names

    
class MissingIndicatorWithNames(MissingIndicator):
    def __init__(self, verbose=0, feature_suffix="_missing", **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.feature_suffix = feature_suffix
        self.feature_names = None
        
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names = [str(col)+self.feature_suffix for col in X.columns]
            if self.verbose:
                print("MissingIndicatorWithNames.fit(", type(X), ") saving feature names:",
                      self.feature_names)
        else:
            if self.verbose:
                print("MissingIndicatorWithNames.fit() input X (type=" + type(X) +
                      ") has no 'columns' attr, cannot save feature names")

        return super().fit(X, y)

    def get_feature_names(self, input_features=None):
        return self.feature_names
    
    
class StandardScalerWithNames(StandardScaler):
    def __init__(self, verbose=0, feature_suffix="_scaler", **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.feature_suffix = feature_suffix
        self.feature_names = None
        
    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names = [str(col)+self.feature_suffix for col in X.columns]
            if self.verbose:
                print("StandardScalerWithNames.fit(", type(X), ") saving feature names:",
                      self.feature_names)
        else:
            if self.verbose:
                print("StandardScalerWithNames.fit() input X (type=", type(X),
                      ") has no 'columns' attr, cannot save feature names")

        return super().fit(X, y)

    def get_feature_names(self, input_features=None):
        return self.feature_names


class OneToOneMixin(object):
    """Provides get_feature_names for simple transformers
    Assumes there's a 1-to-1 correspondence between input features
    and output features.
    """

    def get_feature_names(self, input_features=None):
        """Get feature names for transformation.
        Returns input_features as this transformation
        doesn't add or drop features.
        Parameters
        ----------
        input_features : array-like of string
            Input feature names.
        Returns
        -------
        feature_names : array-like of string
            Transformed feature names
        """
        if input_features is not None:
            return input_features
        else:
            raise ValueError("Don't know how to get"
                             " input feature names for {}".format(self))


# # Custom Transformers
# This next section has custom transformers, partly to implement new function, partly to include get_feature_names in the pipeline.
# 
# This is a contrast to the section above, which is copy-paste of minimal changes to sklearn library classes to make get_feature_names working prior to installation of the pull request.

# In[ ]:


def int2str(x, width):
    return str(x).rjust(width, " ")


def print_stamp(text: str = "", earlier: datetime = None):
    """
    Prints a timestamp with an optional text message, and an optional duration from an earlier
    datetime
    
    :return: the current timestamp, so you can pass it back in later to show a duration
    """
    d = datetime.now()
    if earlier is not None:
        print(text, str(d), "elapsed:", str(d - earlier), flush=True)
    else:
        print(text, str(d), flush=True)
    return d


class SinCosTransformer(BaseEstimator, TransformerMixin, OneToOneMixin):
    def __init__(self, max_vals: int, feature_suffix_sin: str = "_sin",
                 feature_suffix_cos: str = "_cos", label_feature: str = "target",
                 verbose: int = 0):
        super().__init__()
        self.max_vals = max_vals
        self.feature_suffix_sin = feature_suffix_sin
        self.feature_suffix_cos = feature_suffix_cos
        self.label_feature = label_feature
        self.verbose = verbose
        self.feature_names = []
    
    def debug(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    
    def fit(self, X, y=None):
        """Does nothing at fit time
        
        GitHub PR "RFC Implement Pipeline get feature names #12627" will introduce
        proper argument passing in the get_feature_names chain.
        Until that is available, we'll store the feature names at fit time.
        
        :return: self, so we can daisy-chain .transform()
        """
        self.debug("SinCosTransformer.fit(X=", type(X), "y=", type(y), ")")
        if hasattr(X, "columns"):
            for col in X.columns:
                self.feature_names.append(col + self.feature_suffix_sin)
                self.feature_names.append(col + self.feature_suffix_cos)
            self.debug("SinCosTransformer.fit() saving feature names:", self.feature_names)       
        else:
            self.debug("SinCosTransformer.fit() input X has no 'columns' attr,",
                       "cannot save feature names")
        return self

    def get_feature_names(self, input_features=None):
        """Overriding method in OneToOneMixin, because ColumnTransformer does not pass any value
        for input_features until GitHub PR "RFC Implement Pipeline get feature names #12627" is
        available
        """
        return self.feature_names
    
    def transform(self, X):
        X_new = pd.DataFrame()
        self.debug("SinCosTransformer.transform(", type(X), ") running")
        
        if hasattr(X, "columns"):
            self.debug("SinCosTransformer.transform() generating sin/cos values, X cols:", X.columns)
            
            for col_name in X.columns:
                if col_name == self.label_feature:
                    self.debug("SinCosTransformer.transform() skipping label_feature '",
                               col_name, "', not sure why sklearn passes it",
                               "for this transformer and not the others")
                else:
                    X_new[col_name + self.feature_suffix_sin] = np.sin(
                        2 * np.pi * X[col_name] / self.max_vals)
                    X_new[col_name + self.feature_suffix_cos] = np.cos(
                        2 * np.pi * X[col_name] / self.max_vals)
        else:
            raise AttributeError("SinTransformer input X (", type(X),
                                 ") has no attribute 'columns'")
        return X_new

    
class MeanTransformer(BaseEstimator, TransformerMixin, OneToOneMixin):
    def __init__(self, feature_suffix: str = "_mean", label_feature: str = "target",
                 verbose: int = 0):
        super().__init__()
        self.feature_suffix = feature_suffix
        self.label_feature = label_feature
        # mean_maps will be populated in fit().  It is a dict, k=col_name, value is another dict:
        #    k=value seen in traning, v=mean target label seen in training
        self.mean_maps = {}
        self.feature_names = []
        self.verbose = verbose
    
    def debug(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    
    def fit(self, X, y=None):
        """Calculates the mean target label value for each input value in X and retains
        a map to apply at transform() time.
        
        GitHub PR "RFC Implement Pipeline get feature names #12627" will introduce
        proper argument passing in the get_feature_names chain.
        Until that is available, we'll store the feature names at fit time.
        
        :return: self, so we can daisy-chain .transform()
        """
        self.debug("MeanTransformer.fit(X=", type(X), "y=", type(y), ")")
        
        if hasattr(X, "columns"):
            df = pd.DataFrame(X)
            
            df[self.label_feature] = y
            self.debug("MeanTransformer.fit(), df cols after adding labels:", str(df.columns))
            
            for col in df:
                self.debug("MeanTransformer.fit(), iterating on col:", col,
                           "comparing it to:", self.label_feature)
                
                if str(col) == self.label_feature:
                    self.debug("MeanTransformer.fit(), skipping fit on label column")
                else:
                    self.debug("MeanTransformer.fit(), adding column '", col,
                               "' to feature_names")
                    self.feature_names.append(col + self.feature_suffix)
                    self.mean_maps[col] = np.round(
                        df.groupby(col)[self.label_feature].mean(), decimals=2).to_dict()
            # drop the labels, lest they be given back to us in transform()
            df.drop(self.label_feature, axis=1, inplace=True)
        else:
            self.debug("MeanTransformer.fit() input X (", type(X),
                       ") has no 'columns' attr, cannot save feature names")
        return self

    def get_feature_names(self, input_features=None):
        """Overriding method in OneToOneMixin, because ColumnTransformer does not pass any value
        for input_features until GitHub PR "RFC Implement Pipeline get feature names #12627" is
        available
        """
        return self.feature_names
    
    def transform(self, X):
        X_new = pd.DataFrame()
        self.debug("MeanTransformer.transform(", type(X), ") running")
        
        if hasattr(X, "columns"):
            self.debug("MeanTransformer.transform() mapping mean values, X cols:", X.columns)
            
            for col_name in X.columns:
                if col_name == self.label_feature:
                    self.debug("MeanTransformer.transform() skipping label_feature '",
                                col_name, "', not sure why sklearn passes it",
                               "for this transformer and not the others")
                else:
                    if hasattr(X[col_name], "map"):
                        X_new[col_name+self.feature_suffix] = X[col_name].map(
                            self.mean_maps[col_name])
                    else:
                        raise AttributeError("MeanTransformer column (name=", col_name,
                                             ", type=", type(X[col_name]), ") has no attribute 'map'")
        else:
            raise AttributeError("MeanTransformer input X (", type(X),
                                 ") has no attribute 'columns'")
        return X_new
   
    
class MapTransformer(BaseEstimator, TransformerMixin, OneToOneMixin):
    def __init__(self, value_map: dict = {}, verbose: int = 0):
        super().__init__()
        self.value_map = value_map
        self.verbose = verbose
        self.feature_names = None
    
    def fit(self, X, y=None):
        """This transformer does nothing at fit time.
        WHY does TransformerMixin have no fit() method for us to invoke/override?
        
        GitHub PR "RFC Implement Pipeline get feature names #12627" will introduce
        proper argument passing in the get_feature_names chain.
        Until that is available, we'll store the feature names at fit time.
        
        :return: self, so we can daisy-chain .transform()
        """
        self.debug("MapTransformer.fit(", type(X), ") running")
        if hasattr(X, "columns"):
            self.feature_names = X.columns
            self.debug("MapTransformer.fit() saving feature names:", self.feature_names)
                   
        else:
            self.debug("MapTransformer.fit() input X has no 'columns' attr,",
                       "cannot save feature names")
        return self

    def get_feature_names(self, input_features=None):
        """Overriding method in OneToOneMixin, because ColumnTransformer does not pass any value
        for input_features until GitHub PR "RFC Implement Pipeline get feature names #12627" is
        available
        """
        return self.feature_names
    
    def transform(self, X):
        self.debug("MapTransformer.transform(", type(X), ") running, cols:", X.columns)
                   
        if hasattr(X, "replace"):
            #for k, v in self.value_map.items():
            #    X.replace(k, v, inplace=True)
            X.replace(self.value_map, inplace=True)
            self.debug("MapTransformer.transform() replacing with value_map",
                       str(self.value_map))
        else:
            raise AttributeError("MapTransformer input X (", type(X),
                                 ") has no attribute 'replace'")
        return X

    def debug(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)


class AsciiTransformer(BaseEstimator, TransformerMixin, OneToOneMixin):
    def __init__(self, feature_suffix: str = '_ascii', verbose: int = 0):
        super().__init__()
        self.verbose = verbose
        self.feature_names = None
        self.feature_suffix = feature_suffix
    
    def fit(self, X, y=None):
        """This transformer does nothing at fit time.
        WHY does TransformerMixin have no fit() method for us to invoke/override?
        
        GitHub PR "RFC Implement Pipeline get feature names #12627" will introduce
        proper argument passing in the get_feature_names chain.
        Until that is available, we'll store the feature names at fit time.
        
        :return: self, so we can daisy-chain .transform()
        """
        if hasattr(X, "columns"):
            # pd.DataFrame
            self.feature_names = [col + self.feature_suffix for col in X.columns]
            self.debug("AsciiTransformer.fit(", type(X), ") saving feature names:",
                       self.feature_names)
            
        elif hasattr(X, "dtype") and hasattr(X.dtype, "names") and X.dtype.names is not None:
            # np.ndarray
            self.feature_names = [col + self.feature_suffix for col in X.dtype.names]
            self.debug("AsciiTransformer.fit(", type(X), ") saving feature names:",
                       self.feature_names)
            
        else:
            raise AttributeError("AsciiTransformer.fit(), input X (", type(X),
                                 ") has no 'columns' or 'dtype.names' attr,",
                                 "cannot save feature names")
        return self

    def get_feature_names(self, input_features=None):
        """Overriding method in OneToOneMixin, because ColumnTransformer does not pass any value
        for input_features until GitHub PR "RFC Implement Pipeline get feature names #12627" is
        available
        """
        return self.feature_names
    
    def transform(self, X):
        X_new = pd.DataFrame()
        self.debug("AsciiTransformer.transform(", type(X), ") running")
        
        if hasattr(X, "columns"):
            self.debug("AsciiTransformer.transform() replacing with ascii values")

            for col_name in X.columns:
                if hasattr(X[col_name], "map"):
                    X_new[col_name+self.feature_suffix] = X[col_name].map(
                        ascii_ord, na_action='ignore')
                else:
                    raise AttributeError("AsciiTransformer column (name=", col_name,
                                         ", type=", type(X[col_name]), ") has no attribute 'map'")
        else:
            raise AttributeError("AsciiTransformer input X (", type(X),
                                 ") has no attribute 'columns'")
        return X_new

    def debug(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

                   
def ascii_ord(s):
    """
    Walks through each char in s backwards, and accumulates the ascii ord values
    
    :param s: str - the string to convert
    :return: int, the accumulated value
    """
    acc = 0
    for index, c in enumerate(reversed(s)):
        acc += ord(c) * (128 ** index)
    return acc


# # Create a simple model
# This is an exploration of grid search, so just build a preprocessor and model pipeline that is simple and quick to train.

# In[ ]:


###############################################################################################
# Initialize Preprocessor Steps
###############################################################################################

preprocessor_steps = []


###############################################################################################
# Ordinal Transformation Pipeline
###############################################################################################

# ord_0 is already ordinal numbers, so only need to replace missing and scale
ord0_transformer = PipelineWithNames(
    steps=[
        ('feature', SimpleImputerWithNames(strategy="most_frequent")),
        ('scaler', StandardScalerWithNames()),
    ])

if CTL.enable_ord_features and CTL.enable_ord0:
    preprocessor_steps.append(("ord0", ord0_transformer, ["ord_0"]))
    print("Enabling ordinal feature [ord_0]")
else:
    print("Disabling ord_0 feature in pipeline")

# Novice is the most common, so map NaN to Novice:1
ord1_value_map = {
    "Novice": 1,
    "Contributor": 2,
    "Expert": 3,
    "Master": 4,
    "Grandmaster": 5,
    np.nan: 1
}

ord1_transformer = PipelineWithNames(
    steps=[
        ('value_map', MapTransformer(value_map=ord1_value_map, verbose=0)),
        ('scaler', StandardScalerWithNames()),
    ])

if CTL.enable_ord_features and CTL.enable_ord1:
    preprocessor_steps.append(('ord1', ord1_transformer, ['ord_1']))
    print("Enabling ordinal feature [ord_1]")
else:
    print("Disabling ord_1 feature in pipeline")

# Freezing is the most common, so map NaN to Freezing:1
ord2_value_map = {
    "Freezing": 1,
    "Cold": 2,
    "Warm": 3,
    "Hot": 4,
    "Boiling Hot": 5,
    "Lava Hot": 6,
    np.nan: 1
}

ord2_transformer = PipelineWithNames(
    steps=[
        ('value_map', MapTransformer(value_map=ord2_value_map, verbose=0)),
        ('scaler', StandardScalerWithNames()),
    ])

if CTL.enable_ord_features and CTL.enable_ord2:
    preprocessor_steps.append(('ord2', ord2_transformer, ['ord_2']))
    print("Enabling ordinal feature [ord_2]")
else:
    print("Disabling ord_2 feature in pipeline")

# ord 3 & 4 get the ascii treatment
ascii_transformer = PipelineWithNames(
    steps=[
        ('missing', SimpleImputerWithNames(strategy="most_frequent", verbose=0)),
        ('ascii', AsciiTransformer()),
        ('scaler', StandardScalerWithNames()),
    ]
)
if CTL.enable_ord_features and CTL.enable_ord345:
    ordinal_ascii_features = ['ord_3', 'ord_4', 'ord_5']
    preprocessor_steps.append(
        ('ord345', ascii_transformer, ordinal_ascii_features)
    )
    print("Enabling ordinal ascii features", ordinal_ascii_features)
else:
    print("Disabling ordinal ascii features in pipeline")

    
###############################################################################################
# Assemble the All-Feature Preprocessor
###############################################################################################
feature_preprocessor = ColumnTransformer(transformers=preprocessor_steps)


print("")
print("Feature Preprocessor:")
print(str(feature_preprocessor))


# # Split training and validation data

# In[ ]:


t = print_stamp("Splitting train and validation data...")
X_train, X_valid, y_train, y_valid = train_test_split(
    train_df.drop(label_column, axis=1),
    train_df[label_column],
    random_state=CTL.random_state,
    stratify=train_df[label_column]
)
print_stamp("Splitting complete", t)
print("")

print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)
print("y_train shape:", y_train.shape)
print("y_valid shape:", y_valid.shape)


# # Train and Evaluate the Model

# In[ ]:


t = print_stamp("Setting up model...")

model = None

if CTL.active_model == CTL.model_randomforest:
    print("Using randomforest model")
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=CTL.random_state
    )
    
elif CTL.active_model == CTL.model_xgboost:
    print("Using xgboost model")
    params = {
        'objective': 'binary:logistic',
    }
    model = xgb.XGBRegressor(random_state=CTL.random_state)
    model.set_params(**params)

print_stamp("Model setup complete.", t)
print("")
    
t = print_stamp("Assembling final pipeline...")

eval_pipeline = Pipeline(
    steps=[
        ("preprocessor", feature_preprocessor),
        ("model", model)
    ])
print_stamp("Assembly complete", t)


print("")
t = print_stamp("Setting up grid search...")

grid_search = RandomizedSearchCV(
    estimator=eval_pipeline, param_distributions=CTL.gridsearch_param_grid,
    scoring=CTL.gridsearch_scoring, n_iter=CTL.gridsearch_n_iter, cv=CTL.gridsearch_cv,
    random_state=CTL.random_state,
    )
print_stamp("Setup complete", t)

print("")
train_start = print_stamp("Training grid search...")
grid_search.fit(X_train, y_train)
train_end = print_stamp("Training complete", train_start)
round_count = CTL.gridsearch_n_iter * CTL.gridsearch_cv
print("Grid search / CV rounds:", round_count)
print("Duration per round:", str((train_end - train_start) / round_count))
print("")

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Transformed feature names:")
print(best_model.named_steps["preprocessor"].get_feature_names())
print("")

t = print_stamp("Predicting on validation set...")
y_preds = best_model.predict(X_valid)
print_stamp("Predicting complete", t)
print("")

auc = roc_auc_score(y_valid, y_preds)
print("AUC: {0:.05f}".format(auc))
print("")


# # Visualize the Best Model

# In[ ]:


print("Details of best model (validation AUC={0:.05f}):".format(auc))
print("")
if isinstance(best_model, Pipeline):
    print(best_model.named_steps["model"])
else:
    print(best_model)
print("")

print("Best Grid Search parameters:")
for k, v in best_params.items():
    print("\t", k, v, sep="\t")

print("")

feature_weights = pd.DataFrame(
    data={
        'importance': best_model.named_steps["model"].feature_importances_,
        'feature': best_model.named_steps["preprocessor"].get_feature_names()}
)
feature_weights.sort_values(by="importance", ascending=False, inplace=True)

st = sns.axes_style()
sns.set_style(
    "whitegrid",
    {
        "axes.labelcolor": ".99",
        "axes.axisbelow": False,
    }
)

def plot_feature_weights(fw: pd.DataFrame, title: str = "Feature Importance"):
    plt.figure(figsize=(20, 10))

    ax = sns.barplot(x="feature", y="importance", data=fw)
    ax.set(xlabel='Feature', ylabel='Importance', title="Feature Importance")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    plt.show()

plot_feature_weights(feature_weights)

if feature_weights.shape[0] > 15:
    plot_feature_weights(feature_weights[:15], "Top 15 Most Important Features")

print()


# # Run a final training cycle on full data set
# 
# ### &#x1F534; Data Splits for Model Evaluation
# Splitting the data sets gets slightly complex so let's lay it all out here:
# 
# 1. **TRAIN_DF** is the full file provided as training data for the competition
# 1. We split **TRAIN_DF** into **TRAIN** and **VALID**
# 1. **TRAIN** is fed into the grid search
# 1. Grid search uses k-fold cross validation, splitting **TRAIN** into **FOLD_TRAIN** and **FOLD_VALID**
#     * There is no contamination between **VALID** and **FOLD_VALID**
#     * **FOLD_VALID** is exclusively a subset of **TRAIN**.
# 1. Grid search also uses **VALID** as an evaluation set for early stopping rounds in XGBoost.
# 1. After grid search finds the best model, we run a prediction on **VALID** and report the auc.
# 
# ### &#x1F534; Data for Final Predictions
# 
# 1. **TEST_DF** is the full file provided as test data for the competition
# 1. After the best parameters are identified, run one final training cycle, using the full **TRAIN_DF** (i.e. the full data set the competition provides in train.csv) and use the resulting model to make predictions on **TEST_DF**.
# 

# In[ ]:


t = print_stamp("Training best model...")
best_model.fit(train_df.drop(label_column, axis=1), train_df[label_column])
print_stamp("Training complete", t)
print("")


# # Make test predictions and submission file

# In[ ]:


X_test = test_df

t = print_stamp("Predicting on test set...")
test_preds = best_model.predict(X_test)
print_stamp("Predicting complete", t)
print("")

submission = pd.DataFrame({"id": X_test.index, label_column: test_preds})

# sub_filename_pattern = "preds-{0:s}-{1:0.5f}.csv"
# sub_file_date_pattern = "%Y%m%d%H%M%S"
sub_filename = CTL.sub_filename_pattern
if CTL.sub_file_date_pattern is not None:
    sub_filename = CTL.sub_filename_pattern.format(datetime.now().strftime(CTL.sub_file_date_pattern), auc)
    
if CTL.output_dir is not None:
    sub_filename = os.path.join(CTL.output_dir, sub_filename)

print("Submission file:", sub_filename)

submission.to_csv(sub_filename, index=False)


# In[ ]:




