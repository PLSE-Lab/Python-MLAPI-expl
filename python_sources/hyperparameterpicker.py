import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.gaussian_process
import wandb

class HyperParameterPicker:
    WANDB_ACCOUNT_NAME = 'rgdl'
    
    N_TEST = int(1e6)

    def __init__(self, project_name):
        self.project_name = project_name
        self.get_data()
        
        self._gpr = None
        self._ivs = []
        self._dvs = []
        self._detransformer = {}
        self._split_columns = []

    def get_data(self):
        api = wandb.Api()
        runs = api.runs(f'{self.WANDB_ACCOUNT_NAME}/{self.project_name}')
        self.data = pd.DataFrame.from_records([run.config for run in runs])

    def filter_data(self, filters):
        self.data = self.data[filters]
        self.data.reset_index(inplace=True, drop=True)

    def split_array_columns(self, array_columns, drop_original=True):
        if isinstance(array_columns, str):
            array_columns = [array_columns]
        for col in array_columns:
            self.data[f'n_{col}'] = self.data[col].apply(len)
            max_n = self.data[f'n_{col}'].max()
            for i in range(max_n):
                self.data[f'{col}_{i}_val'] = 0
            for i, row in self.data.iterrows():
                for n, val in enumerate(row[col]):
                    self.data.loc[i, f'{col}_{n}_val'] = val
        
        if drop_original:
            self.data.drop(array_columns, inplace=True, axis=1)
        
        self._split_columns += array_columns
        
    def transform_variables(self, discrete_vars=[], log_vars=[], var_mins={}, var_maxes={}):
        """
        Transform all values into the range (0, 1), storing the necessary info to undo the transformation
        Transformation can be:
        - Single value to 0 (store single value)
        - Discrete values, to even-width bins within the desired range
        - exponential values, logged then transformed to desired range
        - just transformed to desired range (default)
        """
        for col in self.data:
            try:
                if self.data[col].nunique() == 1:
                    self._detransformer[col] = ('single', self.data[col].dtype, self.data[col].iloc[0])
                    self.data[col] = 0
                elif col in discrete_vars:
                    unique_vals = self.data[col].sort_values().unique()
                    self._detransformer[col] = ('discrete', self.data[col].dtype, unique_vals)
                    trans_vals = pd.Series([-1 for _ in self.data[col]])
                    for i, val in enumerate(unique_vals):
                        trans_vals.loc[self.data[col] == val] = i / len(unique_vals)
                    self.data[col] = trans_vals
                elif col in log_vars:
                    self.data[col] = np.log(self.data[col])
                    log_min = np.log(var_mins[col]) if var_mins.get(col) else self.data[col].min()
                    log_max = np.log(var_maxes[col]) if var_maxes.get(col) else self.data[col].max()
                    self._detransformer[col] = ('log', self.data[col].dtype, (log_min, log_max))
                    self.data[col] -= log_min
                    self.data[col] /= log_max - log_min
                else:
                    _min = var_mins[col] if vat_mins.get(col) else self.data[col].min()
                    _max = var_maxes[col] if vat_maxes.get(col) else self.data[col].max()
                    self._detransformer[col] = ('default', self.data[col].dtype, (_min, _max))
                    self.data[col] -= _min
                    self.data[col] /= _max - _min
            except Exception as e:
                print('Something wrong with column:', col)
                raise e
    
    def detransform(self, x, unsplit):
        for col in x:
            if self._detransformer.get(col):
                trans, dtype, vals = self._detransformer[col]
                if trans == 'single':
                    x[col] = vals
                elif trans == 'discrete':
                    new_vals = pd.Series([-1 for _ in x[col]])
                    x[col] *= len(vals)
                    x[col] = round(x[col]).astype(int)
                    for i, val in enumerate(vals):
                        new_vals.loc[x[col] == i] = val
                    x[col] = new_vals
                elif trans == 'log':
                    x[col] *= vals[1] - vals[0]
                    x[col] += vals[0]
                    x[col] = np.exp(x[col])
                elif trans == 'default':
                    x[col] *= vals[1] - vals[0]
                    x[col] += vals[0]
                x[col] = x[col].astype(dtype)
        return x
    
    def unsplit_data(self, x):
        """
        NB this is slow, don't do it on a large dataframe, just a subset of interest
        """
        for split_col in self._split_columns:
            try:

                n_col = [col for col in x.columns if col == f'n_{split_col}'][0]
                val_cols = [col for col in x.columns if re.match(f'{split_col}_[0-9]+_val', col)]
                
                n_col_position = list(x.columns).index(n_col)
                val_col_positions = {col: list(x.columns).index(col) for col in val_cols}
                
                unsplit_col = []

                for row in x.itertuples(index=False):
                    array_length = row[n_col_position]
                    unsplit_col.append([row[val_col_positions[val_cols[i]]] for i in range(array_length)])
                x[split_col] = unsplit_col
                x.drop([n_col] + val_cols, axis=1, inplace=True)
            except Exception as e:
                print("Couldn't unsplit", split_col)
                raise e
        return x
    
    def fit_gpr(self, ivs, dvs):
        if not self._detransformer:
            raise ValueError('Call `transform_variables` first')
        self._gpr = sklearn.gaussian_process.GaussianProcessRegressor()
        self._gpr.fit(self.data[ivs], self.data[dvs])
        self._ivs = ivs
        self._dvs = dvs
    
    def predict(self, x):
        if self._gpr is None:
            raise ValueError('Call `fit_gpr` first')
        return self._gpr.predict(x[self._ivs], return_std=True)
    
    def acquisition_function(self, mu, theta):
        return mu + theta
    
    def random_search(self, n_test=N_TEST, detransform=True, unsplit=True):
        t0 = time.time()
        test_points = pd.DataFrame({
            iv: np.random.uniform(self.data[iv].min(), self.data[iv].max(), n_test)
            for iv in self._ivs
        })
        all_mu, all_theta = self.predict(test_points)
        for i, dv in enumerate(self._dvs):
            test_points[f'{dv}_utility'] = self.acquisition_function(all_mu[:, i], all_theta)
        print('Tested', n_test, 'points in', time.time() - t0, 'seconds.')
        return self.detransform(test_points, unsplit) if detransform else test_points