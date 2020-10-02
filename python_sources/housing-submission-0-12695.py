import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

def preprocess():
    
    def count_na(data):
        
        na_count = pd.DataFrame(data.isnull().sum(), columns=['Count']).sort_values(by=['Count'], ascending=False)
        na_count = na_count[na_count['Count'] != 0]
        na_count['Pct'] = na_count['Count']/data.shape[0]
        return na_count    
    
    def str_to_cat(data):
        
        for c in data.select_dtypes(include='object'):

            uq = list(pd.unique(data[c]))
            uq = [ i if isinstance(i, str) else 'nan' for i in uq ]        
            uq.sort()
            data[c] = data[c].apply(lambda x: uq.index(str(x))).astype('category')

        return data
    
    def drop_cols(data, excl, threshold=0.15):
        
        for c in data.columns:
            if c not in excl:
                if (data[data[c].isna()].shape[0])/data.shape[0] > threshold:
                    data = data.drop([c], axis=1)
        return data
    
    def fill_median(data, excl, threshold=0.05):
        
        for c in  data.select_dtypes(include=['int64', 'float64']):
            if c not in excl:
                if (data[data[c].isna()].shape[0])/data.shape[0] < threshold:
                    data[c] = data[c].fillna(data[c].median())
        return data

    def impute(data, excl):
        
        cols = data.isna().sum().to_frame(name='Na')
        cols = cols[(cols['Na'] > 0) & (~cols.index.isin(excl))]
        cols = cols.reset_index().drop(['Na'], axis=1)['index'].tolist()
        #print(cols)
        
        for c in cols:
            
            train = data[~data[c].isna()].drop(excl, axis=1)
            missing = data[data[c].isna()].drop(excl, axis=1)
            
            #print(train.shape, missing.shape)
            
            est = RandomForestRegressor(random_state=1, n_jobs=6, n_estimators=100)            
            est.fit(train.drop([c], axis=1), train[c])
            missing[c] = est.predict(missing.drop([c],axis=1))            
            
            data.loc[data.index.isin(missing[c].index.tolist()), c] = missing[c]
             
        return data
    
    def remove_outliers(data, excl, sd=5):
        for c in data.select_dtypes(include=['int64', 'float64']):
            if c not in excl:
                data = data[ ~(abs(data[c] - data[c].mean())/data[c].std() > sd) ]
        return data
    
    def unskew(data, excl):
        for c in data.select_dtypes(include=['int64', 'float64']):
            if c not in excl and data[c].skew() > 3:                                        
                data[c] = np.log1p(data[c])
            
        data['SalePrice'] = np.log1p(data['SalePrice'])            
        return data
                
        
    def features(data):
        pass
    
    test = pd.read_csv('../input/test.csv', encoding='ISO-8859-1', engine='c')
    train = pd.read_csv('../input/train.csv', encoding='ISO-8859-1', engine='c')
    full = pd.concat([test.copy(), train.copy()]).reset_index(drop=True)
    excl = ['SalePrice', 'Id']
    
    # Count nans
    #nans = count_na(full)
    #print(nans)
    
    # Drop useless columns
    full = drop_cols(data=full, excl=excl)

    # Map categoricals to ints
    full = str_to_cat(data=full)   
    
    # Fill 0 < x < 5 with median
    full = fill_median(data=full, excl=excl)
    
    # Impute
    full = impute(data=full, excl=excl)
        
    # Unskew
    full = unskew(data=full, excl=excl)
                
    # Dummies
    full = pd.get_dummies(data=full, drop_first=True)    
                
    train = full[~full['SalePrice'].isna()]
    test = full[full['SalePrice'].isna()]    
        
    # Outliers
    train = remove_outliers(data=train, excl=excl)        
    return train, test

def scores(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred)**(1/2)    
    r2 = r2_score(y_true, y_pred)
    return [rmse,r2]

def best_params(gs):
    score_data = []
    for p in list(zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score'])):
        sc = list(p[0].values())
        sc.append(p[1])
        score_data.append(sc)
    scores = pd.DataFrame(data=score_data,columns=list(p[0].keys())+['mean_test_score'] )    
    scores = scores.sort_values(by=['mean_test_score'], ascending=False)
    print(scores[0:10])

if __name__ == "__main__":

    start = time.time()    


    # Read files, clean
    train,test = preprocess()

    # Test train split to check for best model
    X_train, X_test, y_train, y_test = train_test_split(
        train.drop(['SalePrice', 'Id'], axis=1), 
        train['SalePrice'],
        random_state=1234,
        test_size=0.2)    

    reg1 = RandomForestRegressor()
    reg2 = AdaBoostRegressor()
    reg3 = GradientBoostingRegressor()
    reg4 = BaggingRegressor()
    reg5 = ExtraTreesRegressor()

    reg = [reg1, reg2, reg3, reg4, reg5]


    metrics=[]
    for r in reg:
        r.fit(X_train, y_train)
        
        y_true_train = np.exp(y_train)-1
        y_pred_train = np.exp(r.predict(X_train))-1
            
        y_true_test = np.exp(y_test)-1
        y_pred_test = np.exp(r.predict(X_test))-1                
        
        all_scores = scores(y_true=y_true_train, y_pred=y_pred_train) + scores(y_true=y_true_test, y_pred=y_pred_test)
        all_scores.append(str(type(r)) )
        metrics.append(all_scores)

    metrics = pd.DataFrame(data=metrics, columns=['RMSE_train', 'R^2_train', 'RMSE_test', 'R^2_test', 'Model'])
    metrics.sort_values(by=['RMSE_test'])

    # GradientBoosting has highest score, perform grid search
    regr = GradientBoostingRegressor(random_state=1234)

    params = {"n_estimators": [300, 400],
              "max_depth": [2, 4, 6], 
              "max_features": [25, 50, 75]}

    gs_cv = GridSearchCV(estimator=regr, 
                         cv=10, 
                         param_grid=params, 
                         n_jobs=6, 
                         verbose=1)

    gs_cv.fit(train.drop(['SalePrice', 'Id'], axis=1), train['SalePrice'])
        
    best_params(gs = gs_cv)

    test['SalePrice'] = np.exp(gs_cv.predict(test.drop(['SalePrice', 'Id'], axis=1)))-1

    # Save predictions
    path = 'submission.csv' 
    test = test[['Id', 'SalePrice']]
    test.to_csv(path, sep=',', index=False)
    test.shape

    end = time.time()
    print('Time: %s' %(end-start))