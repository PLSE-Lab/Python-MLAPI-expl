# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingClassifier

print("Load the training, test and store data using pandas")
dataset_treino = pd.read_csv("../input/dataset_treino.csv")
dataset_teste = pd.read_csv("../input/dataset_teste.csv")
lojas = pd.read_csv("../input/lojas.csv")

# verifying files

print('Shape - Dataset Treino', dataset_treino.shape)
print('Shape - Dataset Teste', dataset_teste.shape)
print('Shape - Lojas', lojas.shape)

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

# Missing values in dataset teste
#column open has 11 missing values
missing_values_table(dataset_teste)

# Missing values in dataset treino
# dataset treino doesn't have missing value.
missing_values_table(dataset_treino)

# Missing values in lojas
# column lojas has missing values in six columns
missing_values_table(lojas)

# Let's use the mode in dataset_treino open feature to imputate the missing value in dataset_teste
# The mode of a set of values is the value that appears most often.
mode = dataset_treino['Open'].mode()[0]
dataset_teste['Open'] = dataset_teste['Open'].fillna(mode)
print(dataset_teste.isnull().sum())

# The columns storetype, assortment, and promo2 have full information in the 3 lines that CompetitionDistance is null, so we find mean value

competition_mean = lojas['CompetitionDistance'][(lojas['StoreType']=='d')&
                                            (lojas['Assortment']=='a')&
                                            (lojas['Promo2']==0)].mean()
    

lojas['CompetitionDistance'] = lojas['CompetitionDistance'].fillna(competition_mean)
missing_values_table(lojas)

#Replace NA by 0 in other feature
lojas = lojas.fillna(0)
missing_values_table(lojas)

# no columns with missing values
lojas.info()

#Let's get the data out of the stores that are open and have zero sales
dataset_treino = dataset_treino[dataset_treino['Sales']!=0]
print(dataset_treino.shape)

#The column Date is string... So we need convert string value to pandas date value
dataset_treino['Date'] = pd.to_datetime(dataset_treino['Date'])
dataset_teste['Date'] = pd.to_datetime(dataset_teste['Date'])
dataset_treino.info()

# Add more time feature
# Pandas The year, month, day and weekofyear of the datetime

def get_time_feature(df):
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.weekofyear
    return df

dataset_treino = get_time_feature(dataset_treino)
dataset_teste = get_time_feature(dataset_teste)


print(dataset_treino.shape)
print(dataset_teste.shape)

print(dataset_treino.info())
print(dataset_teste.info())

# We need to convert the column StateHoliday of categorical text data into model-understandable numerical data, we use the Label Encoder class. 

print(dataset_treino.groupby('StateHoliday').size())
print(dataset_teste.groupby('StateHoliday').size())

#the data type is not unique for value 0, so we need replace the categorical 0 and numeric 0 to a new categorical variable

dataset_treino['StateHoliday'] = dataset_treino['StateHoliday'].replace([0, '0'], 'no_Holiday')
dataset_teste['StateHoliday'] = dataset_teste['StateHoliday'].replace([0, '0'], 'no_Holiday')


print(dataset_treino.groupby('StateHoliday').size())
print(dataset_teste.groupby('StateHoliday').size())


# encode
dataset_treino['StateHoliday'] = dataset_treino['StateHoliday'].replace(['no_Holiday', 'a', 'b', 'c'], [1,2,3,4])
dataset_teste['StateHoliday'] = dataset_teste['StateHoliday'].replace(['no_Holiday', 'a', 'b', 'c'], [1,2,3,4])

print(dataset_treino.groupby('StateHoliday').size())
print(dataset_teste.groupby('StateHoliday').size())

# We need to convert the columns 'StoreType', 'Assortment', 'PromoInterval' of categorical text data into model-understandable numerical data, we use the Label Encoder class. 

# check the value categories in three feature
col_name = ['StoreType', 'Assortment', 'PromoInterval']
for col in col_name:
    print(lojas.groupby(col).size())


#change column promointerval to same datatype because there is 544 values 0
lojas['PromoInterval'] = lojas['PromoInterval'].replace(0, 'None')


# label encode by sklearn
le = LabelEncoder()
for col in col_name:
    le.fit(lojas[col])
    lojas[col] = le.transform(lojas[col])
    

lojas.info()

# merge the store dataset and train test by store id
treino = pd.merge(dataset_treino, lojas, on='Store')
teste = pd.merge(dataset_teste, lojas, on='Store')

print(treino.shape)
print(teste.shape)

print(treino.info())
print(teste.info())

# reset index by Date
treino = treino.set_index('Date')
teste = teste.set_index('Date')

print(treino.shape)
print(teste.shape)

# evaluation
def rmspe(y_pred, y_true):
    rmspe = np.sqrt(np.mean(((y_true-y_pred)/y_true)**2))
    return rmspe
    
# split the train/test by store_id
# Sales is our lable and Customers is the feature corresponding to Sales.
# Since Customers feature is not in the testset, we drop the Sales and Customers from train set.
def store_split(store_id, train_stores, test_stores):
    train = train_stores[store_id]
    label = train.loc[:, 'Sales']
    train_feature = train.drop(['Sales', 'Customers'], axis=1)

    # Create a new variable store_id to store Id
    test = test_stores[store_id]
    test_id = test.loc[:, 'Id'] 
    test_feature = test.drop(['Id'], axis=1)
        
    return train_feature, label, test_id, test_feature,
    

# split the train/test set by store
train_stores = dict(list(treino.groupby('Store')))
test_stores = dict(list(teste.groupby('Store')))


# Build model Xgboost
xgb_result = pd.Series()
for s_xgb in test_stores:
    xgb_train, xgb_label, xgb_id, xgb_test = store_split(s_xgb, train_stores, test_stores)
    
    xgb_class = XGBRegressor()
        
    X_train, X_valid, y_train, y_valid = train_test_split(xgb_train, xgb_label, test_size=0.2)
    
    # params 
    xgb_params = {'subsample': [0.7, 0.9],
                  'colsample_bytree': [0.7],
                  'max_depth': [5, 10]}
    
    # cv to find best params
    xgb_grid = GridSearchCV(xgb_class, xgb_params)
    xgb_grid.fit(X_train, y_train)
    
    # predict the validation
    pred = xgb_grid.best_estimator_.predict(X_valid)
    print('RMSPE', round(rmspe(pred, np.array(y_valid)), 3))
     
    # predict the test set 
    xgb_pred = xgb_grid.best_estimator_.predict(xgb_test)      
    xgb_result = xgb_result.append(pd.Series(xgb_pred, index=xgb_id))


# Build model LightGBM
lgb_result = pd.Series()
for s_lgb in test_stores:
    lgb_train, lgb_label, lgb_id, lgb_test = store_split(s_lgb, train_stores, test_stores)

    lgb_class = LGBMRegressor()
        
    X_train, X_valid, y_train, y_valid = train_test_split(lgb_train, lgb_label, test_size=0.2)
    
    # params
    lgb_params = {'max_depth': [5, 15], 
                 'learning_rate': [0.1, 1]}

    # cv to find best params
    lgb_grid = GridSearchCV(xgb_class, lgb_params)
    lgb_grid.fit(X_train, y_train)
    
    # predict the validation
    pred = lgb_grid.best_estimator_.predict(X_valid)
    print('RMSPE', round(rmspe(pred, np.array(y_valid)), 3))
     
    # predict the test set 
    lgb_pred = lgb_grid.best_estimator_.predict(lgb_test)      
    lgb_result = lgb_result.append(pd.Series(lgb_pred, index=lgb_id))

# the result of predict teste XGB
list(xgb_result.values )

# the result of predict teste LGB
list(lgb_result.values )

# concat the result of the two models XGB and LGB
results = pd.concat([pd.Series(xgb_result.sort_index()), pd.Series(lgb_result.sort_index())], axis=1)
weight = [0.7, 0.3]
results = results * weight
results['final'] = results.sum(axis=1)

# the result of predict teste LGB 0,3 x XGB 0,7
#list(results.final.values)

#resultado_xgb = pd.DataFrame({"Id": list(dataset_teste.loc[:, 'Id']), 'Sales': list(xgb_result.values)})
#resultado_xgb.to_csv('submission_competicao3_v4_xgb.csv', index=False)

#resultado_lgb = pd.DataFrame({"Id": list(dataset_teste.loc[:, 'Id']), 'Sales': list(lgb_result.values)})
#resultado_lgb.to_csv('submission_competicao3_v4_lgb.csv', index=False)

resultado_xgb_lgb = pd.DataFrame({"Id": list(dataset_teste.loc[:, 'Id']), 'Sales': list(results.final.values)})
resultado_xgb_lgb.to_csv('submission_competicao3_v4_lgb_xgb.csv', index=False)

#resultado = pd.DataFrame({"Id": list(dataset_teste.loc[:, 'Id']), 'Sales': list(xgb_result.values)})
#resultado.to_csv('resultado.csv', index=False)

#google colab
#resultado.to_csv('/content/drive/My Drive/Colab Notebooks/competicao3/submission_competicao3_v1.csv', index=False)