import numpy as np 
import pandas as pd 

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import cross_val_score

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb
import lightgbm as lgb

import time

import warnings
warnings.filterwarnings('ignore')



def log_transform(x):
    return np.log1p(x)

def inverse_log_transform(x):
    return np.expm1(x)


def get_classifiers():

    """
    Provide lists of classifiers and their names.
    """
    n_jobs       = -1
    random_state =  1

    classifiers = [
                   DummyClassifier(), 
                   LogisticRegression(n_jobs=n_jobs), 
                   RidgeClassifier(random_state=random_state), 
                   MultinomialNB(),
                   KNeighborsClassifier(n_jobs=n_jobs),
                   SGDClassifier(random_state=random_state),
                   SVC(random_state=random_state),
                   DecisionTreeClassifier(random_state=random_state),
                   RandomForestClassifier(n_jobs=n_jobs, random_state=random_state),
                   GradientBoostingClassifier(random_state=random_state),
                   lgb.LGBMClassifier(n_jobs=n_jobs, random_state=random_state),
                   xgb.XGBClassifier(n_jobs=n_jobs, random_state=random_state),
    ]

    clf_names = [
                "DummyClassifier        ",
                "LogisticRegression     ", 
                "RidgeClassifier        ",
                "MultinomialNB          ",
                "KNeighborsClassifier   ",
                "SGDClassifier          ",
                "SVC                    ",
                "DecisionTreeClassifier ",
                "RandomForestClassifier ", 
                "GBMClassifier          ", 
                "LGBMClassifier         ", 
                "XGBoostClassifier      ",
    ]

    return clf_names, classifiers



def prepare_data(df, target_name):

    """
    Separate descriptive variables and target variable.
    Separate numerical and categorical columns.
    """

    if target_name is not None:
        X = df.drop(target_name, axis=1)
        y = df[target_name]
    else:
        X = df
        y = None

    # get list of numerical & categorical columns in order to process these separately in the pipeline 
    num_cols = X.select_dtypes("number").columns
    cat_cols = X.select_dtypes("object").columns
    
    return X, y, num_cols, cat_cols


def get_pipeline(classifier, num_cols, cat_cols, impute_strategy, log_x):

    """
    Create Pipeline with a separate pipe for categorical and numerical data.
    Automatically impute missing values, scale and then one hot encode.
    """

    # the numeric transformer gets the numerical data acording to num_cols
    # first step: the imputer imputes all missing values to the provided strategy argument
    # second step: all numerical data gets stanadard scaled 
    if log_x == False:
        numeric_transformer = Pipeline(steps=[
            ('imputer', make_pipeline(SimpleImputer(strategy=impute_strategy))),
            ('scaler', StandardScaler()),
            ('scaler_min_max', MinMaxScaler()),
            ])
    # if log_x is "True" than log transform feature values
    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', make_pipeline(SimpleImputer(strategy=impute_strategy))),
            ('log_transform', FunctionTransformer(np.log1p)),
            ('scaler_std', StandardScaler()),
            ('scaler_min_max', MinMaxScaler()),
            ])
    
    # the categorical transformer gets all categorical data according to cat_cols
    # first step: imputing missing values
    # second step: one hot encoding all categoricals
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # the column transformer creates one Pipeline for categorical and numerical data each
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)])
    
    # return the whole pipeline for the classifier provided in the function call
    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])



def score_models(df, target_name, sample_size=None, 
    impute_strategy="mean", scoring_metric="accuracy", log_x=False, verbose=True):

    """
    This function yields scores for a large variety of common classifiers on provided training data. 

    Function separates numerical and categorical data based on dtypes of the dataframe columns. Missing values are imputed. Categorical data is one hot encoded, numerical data standard scaled. All classifiers are used with default settings and crossvalidated.
    
    The function returns a dataframe with scores of all classifiers as well as the mean of all results in the last row of the dataframe.

    Parameters
    ----------
    df : Pandas dataframe 
        Pandas dataframe with your training data
    target_name : str
        Column name of target variable
    sample_size : int, default "None" (score on all available samples)
        Number of samples for scoring the model
    impute_strategy : str, default "mean" 
        Strategy for SimpleImputer, can be "mean" (default), "median", "most_frequent" or "constant"
    scoring_metric : str, default "accuracy"
        scoring metric for classifier: "accuracy" (default), any scikit-learn metric for classification
        e.g. "roc_auc", "precision", "recall", "f1" see sklearn.metrics.SCORERS.keys()
    log_x : bool, default "False" 
        Log transform features variable(s)
    verbose : bool, default "True" 
        Print results during crossvalidation
    
    Returns
    -------
    DataFrame
        1st column : Name of classifier
        2nd column : scoring result

    Example
    -------
        X, y = sklearn.datasets.make_classification()
        X, X_test, y, _ = train_test_split(X, y)

        df = pd.DataFrame(X)
        df["target_variable"] = y
        df_test = pd.DataFrame(X_test)

        scores = score_models(df, "target_variable")
        display(scores)
        
        # further use: train and predict
        pipelines = train_models(df, "target_variable")
        predictions = predict_from_models(df_test, pipelines)
        predictions.head()

    """

    
    if sample_size is not None:
        df = df.sample(sample_size)
  
    # retrieve X, y and separated columns names for numerical and categorical data
    X, y, num_cols, cat_cols = prepare_data(df, target_name)

    scores = []

    clf_names, classifiers = get_classifiers()
    if verbose == True:
        print(f"Classifier             Metric ({scoring_metric})")
        print("-"*30)
    for clf_name, classifier in zip(clf_names, classifiers):
        start_time = time.time()
        
        # create a pipeline for each classifier
        clf = get_pipeline(classifier, num_cols, cat_cols, impute_strategy, log_x)
                
        # crossvalidate classifiers on training data
        cv_score = cross_val_score(clf, X, y, cv=3, scoring=scoring_metric)
        
        if verbose == True:
            print(f"{clf_name} {cv_score.mean(): .4f}  |  {(time.time() - start_time):.2f} secs")
        
        scores.append([clf_name.strip(), cv_score.mean()])

    scores = pd.DataFrame(scores, columns=["Classifier", scoring_metric]).sort_values(scoring_metric, ascending=False)
    
    # just for good measure: add the mean of all scores to dataframe
    scores.loc[len(scores) + 1, :] = ["mean_all", scores[scoring_metric].mean()]

    return scores.reset_index(drop=True)
    


def train_models(df, target_name, 
    impute_strategy="mean", log_x=False, verbose=True): 

    """
    This function trains a large variety of common classifiers on provided training data. 

    It separates numerical and categorical data based on dtypes of the dataframe columns. Missing values are imputed. Categorical data is one hot encoded, numerical data standard scaled. Each classifier is then trained with default settings.
    
    The function returns a list of fitted scikit-learn Pipelines.

    Parameters
    ----------
    df : Pandas dataframe 
        Pandas dataframe with your training data
    target_name : str
        Column name of target variable
    sample_size : int, default "None" (score on all available samples)
        Number of samples for scoring the model
    impute_strategy : str, default "mean" 
        Strategy for SimpleImputer, can be "mean" (default), "median", "most_frequent" or "constant"
    log_x : bool, default "False" 
        Log transform features variable(s)
    verbose : bool, default "True" 
        Print results during crossvalidation
    
    Returns
    -------
    List of fitted scikit-learn Pipelines

    Example:
        X, y = sklearn.datasets.make_classification()
        X, X_test, y, _ = train_test_split(X, y)

        df = pd.DataFrame(X)
        df["target_variable"] = y
        df_test = pd.DataFrame(X_test)

        scores = score_models(df, "target_variable")
        display(scores)
        
        pipelines = train_models(df, "target_variable")

        # further use: predict from pipelines
        predictions = predict_from_models(df_test, pipelines)
        predictions.head()
    
    """

    X, y, num_cols, cat_cols = prepare_data(df, target_name)

    pipelines = []

    if verbose == True:
        print(f"Classifier            Training time")
        print("-"*35)
    
    clf_names, classifiers = get_classifiers()
    for clf_name, classifier in zip(clf_names, classifiers):
        start_time = time.time()
        clf = get_pipeline(classifier, num_cols, cat_cols, impute_strategy, log_x)
        clf.fit(X, y)
        if verbose == True:
            print(f"{clf_name}     {(time.time() - start_time):.2f} secs")
        pipelines.append(clf)
    
    return pipelines



def predict_from_models(df_test, pipelines):

    """
    This function makes predictions with a list of pipelines. Test data is treated in the same pipeline the classifiers were trained on. 
    
    The function returns a dataframe with all predictions ordered columnwise. Each column is named with the respective classifiers.

    Parameters
    ----------
    df_test : Pandas dataframe 
        Dataframe with test data
    pipelines: array
        List of scikit-learn pipelines (preferably from train_models())

    Returns
    -------
    Pandas dataframe with prediction from each classifier, ordered columnwise. 
    1 column = results of 1 classifier.
    
    Example:
        X, y = sklearn.datasets.classification()
        X, X_test, y, _ = train_test_split(X, y)

        df = pd.DataFrame(X)
        df["target_variable"] = y
        df_test = pd.DataFrame(X_test)

        scores = score_models(df, "target_variable")
        display(scores)
        
        pipelines = train_models(df, "target_variable")

        # further use: predict from pipelines
        predictions = predict_from_models(df_test, pipelines)
        predictions.head()
    
    """
    
    X_test, _ , _, _ = prepare_data(df_test, None)
    predictions = []
    
    for pipeline in pipelines:
        preds = pipeline.predict(X_test)
        predictions.append(preds)
        
    df_predictions = pd.DataFrame(predictions).T
    clf_names, _ = get_classifiers()
    df_predictions.columns = [clf_name.strip() for clf_name in clf_names]

    return df_predictions


