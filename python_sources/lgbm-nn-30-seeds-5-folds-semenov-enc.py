#!/usr/bin/env python
# coding: utf-8

# In[ ]:


MAX_BOOST_ROUNDS = 10000
EARLY_STOPPING = 500
NUM_FOLDS = 5


# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('/kaggle/input/infopulsehackathon/train.csv', index_col='Id')
CATEGORICAL_FEATURES = df.columns[(df.dtypes != float) & (df.nunique() <= 12) & (df.nunique() >= 4)]

def create_cat_encode(X, X_val, random_state=42, reg=10, categorical_features=CATEGORICAL_FEATURES):
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=random_state)
    bins = np.linspace(0, X.shape[0], 10)
    y_binned = np.digitize(X['Energy_consumption'], bins)
    
    for f in categorical_features:
        X[f'target_encode_{f}'] = X['Energy_consumption'].mean()
        
    for train_idx, test_idx in skf.split(X, y_binned):
        abs_mean_value = X.iloc[train_idx]['Energy_consumption'].mean()
        for f in categorical_features:
            group_obf = X.iloc[train_idx].groupby(f)['Energy_consumption']
            group_mean = group_obf.mean().to_dict()
            group_count = group_obf.count().to_dict()
            X.loc[test_idx, f'target_encode_{f}'] = X.iloc[test_idx][f].apply(lambda x: 
                (group_mean.get(x, abs_mean_value)*group_count.get(x, 0) + abs_mean_value*reg)/(group_count.get(x, 0) + reg))

    
    abs_mean_value = X['Energy_consumption'].mean()
    for f in categorical_features:
        X_val[f'target_encode_{f}'] = abs_mean_value
        group_obf = X.groupby(f)['Energy_consumption']
        group_mean = group_obf.mean().to_dict()
        group_count = group_obf.count().to_dict()
        X_val[f'target_encode_{f}'] = X_val[f].apply(lambda x: 
                (group_mean.get(x, abs_mean_value)*group_count.get(x, 0) + abs_mean_value*reg)/(group_count.get(x, 0) + reg))       
    return X, X_val


# In[ ]:


import lightgbm as lgb
from tqdm import tqdm
from copy import deepcopy


import matplotlib.pyplot as plt
import seaborn as sns

def plotImp(model, col_names , num = 20):
    feature_imp = pd.DataFrame({'Value':model.feature_importance(),'Feature':col_names})
    plt.figure(figsize=(40, 20))
    sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    
    plt.title('LightGBM Features (avg over folds)')
    plt.show()

class MyRegressor(object):
    def __init__(self, ml_params, categoricals, cols_to_drop=[], tgt_variable='Energy_consumption'):
        self.ml = None
        self.ml_params = ml_params
        
        self.tgt_variable = tgt_variable
        self.categoricals = categoricals
        self.cols_to_drop = cols_to_drop
        
    def fit(self, X, X_val=None, plot_feature_imp=False):        
        y = X[self.tgt_variable]
        X = X.drop(columns=[self.tgt_variable] + self.cols_to_drop)
        col_names = X.columns
        
        X = lgb.Dataset(X, y)        
        if X_val is not None:
            X_val = lgb.Dataset(X_val.drop(columns=[self.tgt_variable] + self.cols_to_drop), X_val[self.tgt_variable])
            self.ml = lgb.train(self.ml_params,
                                X,
                                num_boost_round=MAX_BOOST_ROUNDS,
                                valid_sets=(X, X_val),
                                early_stopping_rounds=EARLY_STOPPING,
                                verbose_eval = 500)
        else:
            self.ml = lgb.train(self.ml_params,
                                X,
                                valid_sets=(X),
                                num_boost_round=MAX_BOOST_ROUNDS,
                                verbose_eval = 500)
        if plot_feature_imp:
            plotImp(self.ml, col_names)
            
        return self
    
    def predict(self, X):
        cols_to_drop = list(set(['row_id', self.tgt_variable] + self.cols_to_drop) & set(X.columns))
            
        return self.ml.predict(X.drop(columns=cols_to_drop))


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import gc
from copy import deepcopy

def time_val(data, model, metric=mean_squared_error, target_var_name='Energy_consumption', test_to_predict=None, random_state=42):
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=random_state)
    bins = np.linspace(0, data.shape[0], 10)
    y_binned = np.digitize(data[target_var_name], bins)
    y_val_preds = deepcopy(data[target_var_name])
    print('Starting Validation')
    results = []
    if test_to_predict is not None:
        test_prediction = []
    for train_idx, test_idx in kf.split(data, y_binned):
        train_df, val_df = create_cat_encode(data.iloc[train_idx].reset_index(drop=True), data.iloc[test_idx].reset_index(drop=True), random_state=random_state)
        print('New Itter')
        model.fit(train_df, val_df)
        pred = model.predict(val_df)
        y_val_preds.iloc[test_idx] = pred
        
        train_with_ce, test_with_ce = create_cat_encode(data, test_to_predict, random_state=random_state)
        if test_to_predict is not None:
            test_prediction.append(model.predict(test_with_ce))
            
        itter_metric = metric(data.iloc[test_idx][target_var_name], pred)
        print('Itter metric: '+str(itter_metric))
        results.append(itter_metric)
        
        gc.collect()
     
    if test_to_predict is not None:
        return results, sum(test_prediction)/NUM_FOLDS, y_val_preds
    else:
        return results


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


def lgb_df():
    train = pd.read_csv('/kaggle/input/infopulsehackathon/train.csv', index_col='Id')
    test = pd.read_csv('/kaggle/input/infopulsehackathon/test.csv', index_col='Id')
    def label_encode(train_df, test_df, feature_name):
        map_dict = {k:i for i, k in enumerate(train_df[feature_name].unique())}
        print(map_dict)
        train_df[feature_name] = train_df[feature_name].map(map_dict)
        test_df[feature_name] = test_df[feature_name].map(map_dict)
        return train_df, test_df
    
    for f in ['feature_3', 'feature_4', 'feature_257', 'feature_258']:
        train, test = label_encode(train, test, f)
        
    return train, test


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def nn_df():
    train = pd.read_csv('/kaggle/input/infopulsehackathon/train.csv', index_col='Id')
    test = pd.read_csv('/kaggle/input/infopulsehackathon/test.csv', index_col='Id')
    y_train = train.Energy_consumption
    X_train = train.drop(columns=['Energy_consumption'])
    X_test = test
    
    to_drop = (X_train.std() <= 0.01).index[(X_train.std() <= 0.01).values].values
    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)
    
    continuous_columns = list(X_train.columns[X_train.dtypes == float]) + list(X_train.columns[(X_train.apply(pd.Series.nunique) > 8) & (X_train.dtypes != object)])
    continuous_columns = sorted(set(continuous_columns))
    
    scaler = StandardScaler()

    X_train[continuous_columns] = pd.DataFrame(scaler.fit_transform(X_train[continuous_columns]), columns=continuous_columns)
    X_test[continuous_columns] = pd.DataFrame(scaler.transform(X_test[continuous_columns]), columns=continuous_columns)
    
    categorical_columns = list(filter(lambda x: x not in continuous_columns, X_train.columns))
    binary_columns = list(filter(lambda f: X_train[f].append(X_test[f]).nunique() <= 2, categorical_columns))
    categorical_columns = list(filter(lambda x: x not in binary_columns, categorical_columns))
    
    for f in binary_columns:
        le = LabelEncoder()
        le.fit(X_train[f].append(X_test[f]))
        X_train[f] = le.transform(X_train[f])
        X_test[f] = le.transform(X_test[f])
        
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe_cols = categorical_columns

    ohe_data = pd.DataFrame(ohe.fit_transform(X_train[ohe_cols]).toarray(), dtype=int)
    X_train = pd.concat([X_train.drop(columns = ohe_cols), ohe_data], axis=1)

    ohe_data = pd.DataFrame(ohe.transform(test[ohe_cols]).toarray(), dtype=int)
    X_test = pd.concat([X_test.drop(columns = ohe_cols), ohe_data], axis=1)
    
    return X_train, y_train, X_test


# In[ ]:


lgb_train, lgb_test = lgb_df()
nn_X_train, nn_y_train, nn_X_test = nn_df()


# In[ ]:


from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam, Nadam
from keras import callbacks

def create_model(inp_dim):
    inps = Input(shape=(inp_dim,))
    x = Dense(256, activation='relu')(inps)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(1)(x)
    model = Model(inputs=inps, outputs=x)
    model.compile(
        optimizer=Nadam(lr=1e-3),
        loss=['mse']
    )
    #model.summary()
    return model


# In[ ]:


def nn_cv(X_train, y_train, X_test, seed):
    test_predictions = []
    metric_results = []
    cross_val_predicts = deepcopy(y_train)
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)
    bins = np.linspace(0, y_train.shape[0], 10)
    y_binned = np.digitize(y_train, bins)

    for ind, (tr, val) in enumerate(skf.split(X_train, y_binned)):
        X_tr = X_train.iloc[tr]
        y_tr = y_train.iloc[tr]
        X_vl = X_train.iloc[val]
        y_vl = y_train.iloc[val]

        model = create_model(X_train.shape[1])
        es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=50, verbose=False, mode='auto', restore_best_weights=True)
        rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, mode='auto', verbose=False)
        model.fit(
            X_tr, y_tr, epochs=500, batch_size=256, validation_data=(X_vl, y_vl), verbose=False, callbacks=[es, rlr]
        )
        test_predictions.append(model.predict(X_test).flatten())
        cross_val_predicts.iloc[val] = model.predict(X_vl).flatten()
        metric_results.append(mean_squared_error(y_vl, cross_val_predicts.iloc[val].values))
        
    return metric_results, sum(test_predictions)/NUM_FOLDS, cross_val_predicts


# In[ ]:


from sklearn.metrics import accuracy_score, roc_auc_score

all_predicts = []
all_results = []
all_seeds = [0, 42, 2906, 1999, 2019, 1204, 1207, 15, 3, 7, 29975, 77383, 95657, 58553, 34851, 31093, 69823, 46869, 98800,
        5391, 2848, 34806, 96965, 9961, 62236, 1364, 88418, 14141, 14865, 530]

for seed in all_seeds:
    boost_model = MyRegressor(ml_params={
                "objective": "regression",
                "boosting": "gbdt",
                "num_leaves": 15,
                'max_depth': 8,
                "learning_rate": 0.01,
                "feature_fraction": 0.6,
                "reg_lambda": 2,
                "metric": "mse",
                'seed': seed,
                'subsample_freq': 3,
                'bagging_seed': seed,
                'subsample': 0.6
                }, categoricals=[])
    
    lgb_res, lgb_predicts, lgb_val_predicts = time_val(lgb_train, boost_model, test_to_predict=lgb_test, random_state=seed)
    print(f'LGBM  Seed: {seed} Result: {round(np.mean(lgb_res),5)} +/- {round(np.std(lgb_res),5)}')
    print(mean_squared_error(lgb_train.Energy_consumption, lgb_val_predicts))
    
    nn_res, nn_predicts, nn_val_predicts = nn_cv(nn_X_train, nn_y_train, nn_X_test, seed=seed)
    print(f'NN  Seed: {seed} Result: {round(np.mean(nn_res),5)} +/- {round(np.std(nn_res),5)}')
    print(mean_squared_error(lgb_train.Energy_consumption, nn_val_predicts))   
    
    weight_space = np.arange(0.5, 1.01, 0.01)
    blend_results = []
    for w in weight_space:
        blend_results.append(mean_squared_error(lgb_train.Energy_consumption, w*lgb_val_predicts + (1-w)*nn_val_predicts))
        
    best_weight = weight_space[np.argmin(blend_results)]
    print(min(blend_results))
    print(best_weight)
    best_weight = 0.7*best_weight + 0.3*0.8
    
    all_predicts.append(best_weight*lgb_predicts + (1 - best_weight)*nn_predicts)


# In[ ]:


np.mean(all_results), np.std(all_results)


# In[ ]:


test = pd.read_csv('/kaggle/input/infopulsehackathon/test.csv', index_col='Id')
test['Energy_consumption'] = sum(all_predicts)/len(all_predicts)
test.loc[test['Energy_consumption'] < lgb_train.Energy_consumption.min(), 'Energy_consumption'] = lgb_train.Energy_consumption.min()
test[['Energy_consumption']].to_csv('submission.csv', index=True)


# In[ ]:


import seaborn as sns

sns.distplot(lgb_train.Energy_consumption, bins=30)
sns.distplot(test.Energy_consumption, bins=30);

