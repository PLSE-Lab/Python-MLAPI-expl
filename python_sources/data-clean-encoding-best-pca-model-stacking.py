from IPython.display import display

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
import sklearn.linear_model as linear
import sklearn.svm as svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import tree
from sklearn import neural_network

# lower ram usage of pandas DataFrame
def pd_mem_reducer(df):
    """ iterate through all the columns of a dataframe and modify 'int' & 'float'
        to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        
        col_type = df[col].dtype
        
        if str(col_type)[:3] == 'int':
            
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64) 
                
        elif str(col_type)[:5] == 'float':
            
            c_min = df[col].min()
            c_max = df[col].max()
            
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    return df

# macro data cleaning function
def cleaning_and_downsizing_data(data, headings=None):
    
    data = pd_mem_reducer(data)
    print('Original data shape: {}'.format(data.shape))
    
    # plug headings if provided
    if headings is not None:
        data.columns = headings
        
    # drop columns consisted of all NAN values
    data = data.dropna(axis=1, how='all')
    
    # drop duplicated columns
    data = data.loc[:,~data.columns.duplicated()]
    
    # drop duplicated rows
    data = data.drop_duplicates()
    
    print('Cleaned data shape: {}'.format(data.shape))
    end_mem = data.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB\n'.format(end_mem))
    
    return data

# convert categorical into numeric encodings if needed
def category_encoding(data, cat_columns):
    
    for col_name in cat_columns:
        # print out the encoding's meaning
        print('Encodings: {}'.format(dict(enumerate(data[col_name].cat.categories))))
        # assign numeric encoding
        data[col_name] = data[col_name].cat.codes
        
    return data

# PCA graph for subset of dataset after normalization(1. minMax scale to between 0 and 1, 2.centered at mean=0)
def best_pca_graph(train_cleaned, test_cleaned):
    
    bins = 5
    nos_col = train_cleaned.shape[1]
    n_trials = np.arange(nos_col//bins, nos_col, nos_col//bins)
    n_explained_variance = []
    
    # pca is reducing dimensionality of both train and test set to find the 'natural' directions
    all_data = np.concatenate((train_cleaned, test_cleaned))
    for n_features in n_trials:
        
        pca = PCA(n_components=n_features)
        pca.fit(all_data)
        n_explained_variance.append(sum(pca.explained_variance_ratio_)*100)
    
    print(n_trials)
    print(n_explained_variance)
    plt.plot(n_trials, n_explained_variance)
    plt.show()

    
# 2-layer stacking meta-model(combining weak learners to form strong learner on big enough dataset)
def two_lvl_model_stacking(X_train_variant, X_test_variant, y_train_universal):
    # 1st level regression models: GBT_R, RF_R, SVR, LR. KNN_R 
    # 2nd level regression model: Linear Model
    models = [[ensemble.GradientBoostingRegressor(),
               ensemble.RandomForestRegressor(),
               svm.SVR(),
               linear.BayesianRidge(),
               GaussianProcessRegressor(),
               tree.DecisionTreeRegressor(),
               neural_network.MLPRegressor(),
               neighbors.KNeighborsRegressor()], 
                linear.BayesianRidge()]
    
    # creates dataframe to store first level result
    first_level_B = pd.DataFrame()
    first_level_C = pd.DataFrame()
    i = 0
    
    # train_variant for differently prepared X-dataset, e.g different features or scaling based on same X_test originally
    for (AB, C) in zip(X_train_variant, X_test_variant):
        A_x, B_x, A_y, B_y = train_test_split(AB, y_train_universal, test_size=0.5, random_state=42)
        # 1st lvl models
        for model in models[0]:
            model.fit(A_x, A_y)
            first_level_B[type(model).__name__+'_'+str(i)] = model.predict(B_x)
            first_level_C[type(model).__name__+'_'+str(i)] = model.predict(C)
        i += 1

    first_level_B['B_y'] = B_y.to_numpy()
    display(first_level_B.head())
    display(first_level_C.head())
    
    # perform second level stacking
    final_model = models[1]
    # last col in first_lvl_b is the target y values
    final_model.fit(first_level_B.iloc[:,:-1], first_level_B.iloc[:,-1])
    
    # series of y_values for X_test
    submission = final_model.predict(first_level_C.iloc[:,:])
    
    return submission
    
    
