"""
Submission for the Kaggle Titanic competition - Random Forest Classifier with sklearn pipeline

This script is a kernel predicting which passengers on Titanic survived.
It generates submission dataset for the Kaggle competition upon its execution.

## GENERAL DESCRIPTION
This kernel does some standard preprocessing steps on the data.
Then machine learning classifier can be searched using TPOT library (see bellow).
This part however is not supported on Kaggle and is therefore commented out.
Finally, model is estimated and predictions and saved into ouput .csv file.

## DETAILED FUNCTIONALITY

### DATA PREPROCESSING
Preprocessing is performed using scikit-learn pipelines with both standard and custom-written transformers.
There are two separate pipelines for the data preprocessing, for categorical and for numerical variables.
Variables' category is defined by NUMERICS_ATTRIBS and CAT_ATTRIBS constansts.
Each pipelines performs several of these steps (depending on variables category):
 - data imputation
 - columns creation (e.g. Title column)
 - data scaling
 - label binarizing (generating 0/1 columns from categorical variable

### MACHINE LEARNING MODEL SELECTION WITH TPOT LIBRARY
The script uses Random Forest Classifier to obtain predictions.
This classifier was chosen upon using TPOT library, A Python Automated Machine Learning tool.
This tool optimizes machine learning pipelines using genetic programming, see http://epistasislab.github.io/tpot/
However, as the Kaggle environment does not support this library, respective part of the code is commented out.
The Random Forest Classifier selected by the TPOT is hard-coded in the kernel.

### CLASSIFIER ESTIMATION
Chosen classifier (RandomForestClassifier, see part MACHINE LEARNING MODEL SELECTION WITH TPOT LIBRARY) is estimated.
Hyperparameters of the estimator are tuned using GridSearchCV with possible values taken as vicinty of TPOT estimate.

## LIST OF CLASSES

 - DataFrameSelector
 - EmbarkedImputer
 - GeneralImputer
 - TitleCreator

"""
import pandas as pd
import numpy as np
# import IPython
# from IPython import display
import sklearn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn_pandas import DataFrameMapper


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Subsets dataframe by selecting collumns supplied by attribute_names upon initialization.

    attribute_names - list of column names to be selected
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values
    
    
class EmbarkedImputer(BaseEstimator, TransformerMixin):
    """
    Imputes values into Embarked variable
    """
    def __init__(self): # no *args or **kargs
        return None
    def fit(self, X):
        return self  # nothing else to do
    def transform(self, X):
        # deep copy the df
        df = X.copy()
        
        # Clean up fares.
        value_to_input = df.loc[(df['Fare'] < 85) & (df['Fare'] > 75)  & (df['Pclass'] == 1)]['Embarked'].mode()

        value_to_input = value_to_input[0]

        df.loc[(df['Embarked'].isnull()),['Embarked']] = value_to_input

        return(df)
    
    
class GeneralImputer(BaseEstimator, TransformerMixin):
    """
    This is a general purpose imputer where you can choose columns for grouping and function to apply

    col_impute - column into which values are imputed
    col_group - dataframe is grouped by these columns
    impute_method - choose 'median' or 'average'
    """
    def __init__(self, col_impute, col_group, impute_method = 'median'): # no *args or **kargs
        self.col_impute = col_impute
        self.col_group = col_group
        self.impute_method = impute_method
        return None
    def fit(self, X):
        return self  # nothing else to do
    def transform(self, X):
        # deep copy the df because of transform
        df = X.copy()

        # Create a groupby object: by_sex_class
        grouped = df.groupby(self.col_group)

        # function to impute median
        def imputer_median(series):
            return series.fillna(series.median())
        # function to impute average
        def imputer_average(series):
            return series.fillna(series.mean())

        if self.impute_method == 'median':
            # impute median
            df[self.col_impute] = grouped[self.col_impute].transform(imputer_median)
            return(df)
        elif self.impute_method == 'average':
            # impute average
            df[self.col_impute] = grouped[self.col_impute].transform(imputer_average)
            return(df)
        else:
            return np.nan

        
class TitleCreator(BaseEstimator, TransformerMixin):
    """
    Generates Title column
    """
    def __init__(self): # no *args or **kargs
        return None

    def fit(self, X):
        return self  # nothing else to do

    def transform(self, X):
        # deep copy the df because of transform
        df = X.copy()

        df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev','Sir','Jonkheer','Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        df['Title'] = df['Title'].fillna(np.nan) 

        return df


if __name__ == '__main__':

    CAT_ATTRIBS = ['Sex','Embarked','Title']
    NUMERICS_ATTRIBS = ['Pclass','Age','SibSp','Parch','Fare']

    # Read train and test data
    train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
    test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

    # map transformers on variables
    my_mapper = DataFrameMapper([
        ('Sex', sklearn.preprocessing.LabelBinarizer()),
        ('Embarked', sklearn.preprocessing.LabelBinarizer()),
        ('Title', sklearn.preprocessing.LabelBinarizer())
        ], input_df=True)

    # categorical data pipeline
    categorical_data_pipeline = Pipeline([
        ('ebarked_imputer', EmbarkedImputer()),
        ('title_creator', TitleCreator()),
        ('label_binarizer_df', my_mapper),
    ])

    # numerical data pipeline
    numerical_data_pipeline = Pipeline([
        ('fare_imputer', GeneralImputer(col_impute=['Fare'],
                                                            col_group=['Sex', 'Pclass'],
                                                            impute_method='median')),
        ('age_imputer', GeneralImputer(col_impute=['Age'],
                                                           col_group=['Sex', 'Pclass'],
                                                           impute_method='average')), # median perhaps?
        ('selector', DataFrameSelector(NUMERICS_ATTRIBS)),
        ('std_scaler', StandardScaler()),

    ])

    # merge the numerical and the categorical pipelines
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", numerical_data_pipeline),
        ("cat_pipeline", categorical_data_pipeline),
    ])

    # execute the data preparation pipeline
    train_prepared = full_pipeline.fit_transform(train)
    test_prepared = full_pipeline.fit_transform(test)

    # ##########################################
    # # THE FOLLOWING PART IS NOT SUPPORTED BY KAGGLE ENVIRONMENT
    # # TPOT FUNCTION TRIES TO FIND AN OPTIMAL CLASSIFIER
    # # IN THIS CASE THE RandomForestClassifier() WAS FOUND AND IS HARDCODED BELLOW
    # from tpot import TPOTClassifier
    # tpot = TPOTClassifier(verbosity=2, max_time_mins=10)
    # tpot.fit(train_prepared, train['Survived'])
    # ##########################################

    # hyerparameters grid to search within
    param_grid = [
        {'bootstrap': [False, True],
         'n_estimators': [80,90, 100, 110, 130],
         'max_features': [0.6, 0.65, 0.7, 0.73, 0.7500000000000001, 0.78, 0.8],
         'min_samples_leaf': [10, 12, 14],
         'min_samples_split': [3, 5, 7]
        },
    ]

    # declare the classifier
    random_forest_classifier = RandomForestClassifier()

    # create the GridSearchCV object
    grid_search = GridSearchCV(random_forest_classifier, param_grid, cv=5,scoring='neg_mean_squared_error', refit=True)

    # fine-tune the hyperparameters
    grid_search.fit(train_prepared, train['Survived'])

    # get the best model
    final_model = grid_search.best_estimator_

    # predict using the test dataset
    final_predictions = final_model.predict(test_prepared)

    # generate submission datasets
    my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': final_predictions})
    my_submission.to_csv('submission.csv', index=False)