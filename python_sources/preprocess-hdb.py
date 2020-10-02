# Second version of script containing class to preprocess data the OOP way
# Helps reduce the amount of space needed for notebooks for model training
# Based on starter code in https://www.kaggle.com/raibosome/starter-notebook
# and code in https://www.kaggle.com/yxlee245/feature-engineering

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


class DataPreprocessModule(object):
    """
    Class to handle preprocessing of data
    """
    def __init__(self, train_path, test_path, address_to_stn_path):
        self.X = pd.read_csv(train_path)
        self.X_test_full = pd.read_csv(test_path)
        self.df_address_to_stn = pd.read_csv(address_to_stn_path)
        self._preprocess_ordinal_variables()
        self._feature_engineering()
        # Identify target variable
        # Remove rows with missing target, separate target from predictors, remove ids
        self.X.dropna(axis=0, subset=['resale_price'], inplace=True)
        self.train_indices = self.X['id']
        self.y = np.log1p(self.X.resale_price)  # calculate log(1 + x)
        self.X.drop(['id', 'resale_price'], axis=1, inplace=True)
        self.test_indices = self.X_test_full['id']
        self.X_test_full.drop(['id'], axis=1, inplace=True)
        # Split data into training and validation set
        self.X_train_full , self.X_val_full, self.y_train, self.y_val =\
        train_test_split(self.X, self.y, train_size=0.8, test_size=0.2,
                         random_state=4)
        self._select_columns()
        self._build_preprocesser()
        
    def get_preprocessed_data(self):
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val
    
    def get_indices(self):
        return self.train_indices, self.test_indices
    
    def get_preprocessor(self):
        return self.preprocessor
    
    def build_pipeline(self, model):
        return Pipeline(steps=[('preprocesser', self.preprocessor),
                               ('model', model)])
        
    def _preprocess_ordinal_variables(self):
        dictionaries = [
            {
                'name': 'num_rooms',
                'features': ['flat_type'],
                'lookup': {
                    '1 ROOM': 1,
                    '2 ROOM': 2,
                    '3 ROOM': 3,
                    '4 ROOM' : 4,
                    '5 ROOM' : 5,
                    'EXECUTIVE': 6,
                    'MULTI-GENERATION': 7,
                }
            },
        ]
        
        for df in [self.X, self.X_test_full]:
            
            for dictionary in dictionaries:
                for feature in dictionary['features']:
                    df[feature] = df[feature].map(dictionary['lookup']).fillna(0)
                    
            split_function = lambda x: x.split(' ')[0]
            df['remaining_lease_years'] =\
            pd.to_numeric(df['remaining_lease'].apply(split_function))
            df['storey_range_numerical'] =\
            pd.to_numeric(df['storey_range'].apply(split_function))
            df.drop(['remaining_lease','storey_range'],
                    axis=1, inplace=True)
            
    def _feature_engineering(self):
        # Change "C'WEALTH" to "COMMONWEALTH"
        self.X['street_name'] = self.X['street_name'].replace('C\'WEALTH', 'COMMONWEALTH', regex=True)
        self.X_test_full['street_name'] = self.X_test_full['street_name'].replace('C\'WEALTH', 'COMMONWEALTH', regex=True)
        # Concatenate block and street name to form address
        self.X['address'] = self.X['block'] + ' ' + self.X['street_name']
        self.X_test_full['address'] = self.X_test_full['block'] + ' ' + self.X_test_full['street_name']
        # Join nearest distance to station to train and test set
        self.X = self.X.merge(self.df_address_to_stn, on='address')
        self.X_test_full = self.X_test_full.merge(self.df_address_to_stn, on='address')
    
    def _select_columns(self):
        # "Cardinality" means the number of unique values in a column
        # Select categorical columns with relatively low cardinality (convenient but arbitrary)
        self.categorical_cols = [cname for cname in self.X_train_full.columns if
                                 self.X_train_full[cname].nunique() < 10 and
                                 self.X_train_full[cname].dtype == "object"]
        # Select numerical columns
        self.numerical_cols = [cname for cname in self.X_train_full.columns if
                               self.X_train_full[cname].dtype in ['int64', 'float64']]
        # Keep selected columns only
        my_cols = self.categorical_cols + self.numerical_cols
        self.X_train = self.X_train_full[my_cols].copy()
        self.X_val   = self.X_val_full[my_cols].copy()
        self.X_test  = self.X_test_full[my_cols].copy()
        
    def _build_preprocesser(self):
        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scale',   StandardScaler(with_mean=False))
        ])
        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot',  OneHotEncoder(handle_unknown='ignore')),
            ('scale',   StandardScaler(with_mean=False))
        ])
        # Bundle both preprocessing
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer,   self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])