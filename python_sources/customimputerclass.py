import numpy as np 
import pandas as pd 

from sklearn.base import BaseEstimator, TransformerMixin

class CustomImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, fill_vals=None):
        '''
        Imputer to fill in missing data with specific values. Imputation strategies include mean, median, and most frequent values. 
    
        fill_vals (dictionary): 
                - key is column name with missing data
                - value is one of three:  
                    1: str of imputation strategy ('mean', 'median', 'most_frequent') 
                        - this will impute missing values based on entire column e.g. fill missing values of feature x w/mean of feature x
                    2: tuple of column to groupby and str of imputation strategy
                        - this will impute missing values based off groupby column 
                        e.g. fill missing values of feature x with mean of x grouped by column y: ('y', 'mean')
                    3: custom value such as 0 or a string
                        
        Returns DataFrame with filled in values
        '''
        
        self.fill_vals = fill_vals
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        for col, val in self.fill_vals.items():
            if val == 'mean':
                X[col] = X[col].fillna(X[col].mean())
                
            elif val == 'median':
                X[col] = X[col].fillna(X[col].median())
                
            elif val == 'most_frequent':
                X[col] = X[col].fillna(X[col].mode()[0])
                    
            elif type(val) == tuple:
                grpby_col = val[0]
                strategy = val[1]
                
                if strategy == 'mean':
                    X[col] = X.groupby(grpby_col)[col].transform(lambda x: x.fillna(x.mean()))
                    
                elif strategy == 'median':
                    X[col] = X.groupby(grpby_col)[col].transform(lambda x: x.fillna(x.median()))
                    
                elif strategy == 'most_frequent':
                    X[col] = X.groupby(grpby_col)[col].transform(lambda x: x.fillna(x.mode()[0]))
            
            else:
                X[col] = X[col].fillna(value=val)    

                
        return X