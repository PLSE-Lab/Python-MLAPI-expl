# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold

class MeanTargetEncoding:
    """ A class for mean target encoding.
    
    Encoding categorical variables of high cardinality with a mean target value.
    This encoding ensures correlation between the encoded feature and the target
    variable. 
    For each unique value of the categorical feature, encode the value based on 
    the ratio of occurrence of the positive class in the target variable. As a
    result, mean target encoding solves both the encoding task and creates a new
    feature both training and test set that is more representative of the target 
    variable.
        
    Parameters
    ----------
    target   : str
               Target feature.
    category : str
               Categorical feature.              
    """
        
    # Method to create a new instance of MeanTargetEncoding class
    def __init__(self, target, category):
        # Store the value parameter to the value attribute
        self.target = target
        self.category = category
        
    
    def test_feature(self, train, test, alpha=5):
        """ A method to return encoded test features
        
        Calculate the category statistics on the training set to
        be applied to the test set. Set the smoothing parameter,
        alpha to 5.
        The smoothing parameter is used to prevent overfitting.
        
        
        Parameters
        ----------
        train    : array-like, shape = [num_samples, num_features] 
                   Training set.
        test     : array-like, shape = [num_samples, num_features]
                   Test set.
        alpha    : int, default=5
                   Smoothing parameter.            
        Return
        ------
        test_feature : array
                       Encoded test set feature.        
        """
        
        # Calculate global mean on the train data
        global_mean = train[self.target].mean()

        # Group by the categorical feature and calculate its properties
        train_groups = train.groupby(self.category)
        category_sum = train_groups[self.target].sum()
        category_size = train_groups.size()

        # Calculate smoothed mean target statistics
        train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)

        # Apply statistics to the test data and fill new categories
        test_feature = test[self.category].map(train_statistics).fillna(global_mean)
        encoded_test_values = test_feature.values
        
        # Return encoded test feature
        return encoded_test_values
    
    
    # Method for mean target encoding of the training set
    def train_feature(self, train, test):
        """ A method to return encoded train features
        
        Split the training set into 5-folds, calculate the out
        of fold mean for each fold and apply to the particular
        fold. Feature to be added to the training set.        
        
        Parameters
        ----------
        train    : array-like, shape = [num_samples, num_features] 
                   Training set.
        test     : array-like, shape = [num_samples, num_features]
                   Test set.        
        Return
        ------
        train_feature : array
                        Encoded training set feature.        
        """
        
        # Create 5-fold cross-validation
        kf = KFold(n_splits=5, random_state=123, shuffle=True)
        train_feature = pd.Series(index=train.index)

        # For each folds split
        for train_index, test_index in kf.split(train):
            cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

            # Calculate out-of-fold statistics and apply to cv_test
            cv_test_feature = self.test_feature(cv_train, cv_test)

            # Save new feature for this particular fold
            train_feature.iloc[test_index] = cv_test_feature   
            encoded_train_values = train_feature.values
            
        # Return encoded feature for training set
        return encoded_train_values
    
    
    def encoded_features(self, train, test):
        """ A method to return mean target encoded features
        
        Utilise _train_feature and _test_feature methods to return
        mean target encoded features for both training set and 
        test set.
        
        Parameters
        ----------
        train    : array-like, shape = [num_samples, num_features] 
                   Training set.
        test     : array-like, shape = [num_samples, num_features]
                   Test set.
        
        Return
        ------
        train_feature : array
                        Encoded training set feature.
        
        test_feature  : array
                        Encoded test set feature.
        
        """
        
        # Get the train feature
        train_feature = self.train_feature(train, test)
        # Get the test feature
        test_feature = self.test_feature(train, test)
    
        # Return new features to add to the model
        return train_feature, test_feature
    

    

    
