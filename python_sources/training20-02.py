import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import datetime
import time
from collections import Counter
from datetime import datetime
from xgboost import plot_importance
from matplotlib import pyplot




path = '../'

# True for validation or False to generate predictions
is_valid = False
temporal = True
useUseless = True
useUnified = True
useTime = True

def split_validation(df, numberOfRows):
    df.sort_values(by=["user_id", "ts_listen"], inplace=True)
    df.reset_index(inplace=True)
    val_indexes = df.groupby('user_id')['index'].tail(numberOfRows)
    df_train = df[~df['index'].isin(val_indexes)]
    df_valid = df[df['index'].isin(val_indexes)]
    del df_train['index'], df_valid['index']
    return df_train, df_valid
#----------------------------------------------------Model parameters-----------------------------------------------------------------------------------------

def train_lgb(seed, dtrain, val_sets, n_round):
    params = {   
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.15,
        'num_leaves': 50,
        'min_data_in_leaf': 10, 
        'max_depth': 10,
        'feature_fraction': 1,
        'bagging_freq': 1,
        'bagging_fraction': 0.7,
        #'lambda_l1': 0.1,
        'random_state': seed,
        'verbosity': -1}

    model = lgb.train(params,
                    dtrain,
                    num_boost_round = n_round,
                    valid_sets = val_sets,
                    verbose_eval=10,
                    early_stopping_rounds = 10)
    return model
    
def train_xgb(x_train, x_val, y_train, y_val, seed):
    model = xgb.XGBRegressor( tree_method = 'exact',
                              objective='binary:logistic',
                              base_score=0.76,
                              colsample_bylevel=1, 
                              colsample_bytree=0.5,
                              gamma=0.9, 
                              learning_rate=0.15, 
                              max_delta_step=0, 
                              max_depth=15,
                              min_child_weight=1, 
                              missing=9999999999, 
                              n_estimators=15,
                              n_jobs=-1, 
                              reg_alpha=0.03,
                              reg_lambda=1, 
                              scale_pos_weight=1, 
                              seed=seed, 
                              silent=False,
                              subsample=0.6).fit(x_train, x_val, eval_set=[(y_train, y_val)], early_stopping_rounds=5)
    
    return model
    
#---------------------------------------------------------------------------------------------------------------------------------------------
    
def splitByTimestamp(distancia ,lst):
    indices = [i+1 for (x, y, i) in zip(lst, lst[1:], range(len(lst))) if 1800 < abs(x - y)]
    result = [lst[start:end] for start, end in zip([0] + indices, indices + [len(lst)])]
    return result
    
multiply = lambda x : x*2
    
################################# Import #########################################################################

train = pd.read_csv(path+'input/train.csv')
test = pd.read_csv(path+'input/test.csv')

divide = len(train)

tudo = pd.concat([train, test], ignore_index=True, sort=False)
########################################## heat map function ###############################################################

def getHeatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(correlation_matrix)
    
def plotFeatureImportance(model):
    plot_importance(model).figure.set_size_inches(15, 15)
############################################### Temporal features  ###########################################################

if(temporal):
    auxiliar = tudo.groupby('user_id')['ts_listen'].unique()
    teste = [splitByTimestamp(1800, sorted(x)) for x in auxiliar]
    #teste = list(map(splitByTimestamp, auxiliar))
    #splitByTimestamp(7200, sorted(teste[0]))
    
    def numeroNaSublista2(vetor_usuarios):
        index_lista = [[[i]*len(lista) for (i, lista) in zip(range(len(vetor)), vetor)] for vetor in vetor_usuarios]
        index_lista = [[item for sublist in sublista_index_lista for item in sublist] for sublista_index_lista in index_lista]
        return index_lista
    
    auxiliar2 = numeroNaSublista2(teste)
    
    def createColumnWithTimestamps(listOfGroupTimestamps, listOfCrudeTimestamps):
        #lista = pd.DataFrame([], columns=['user_id', 'ts_listen', 'group'])
        listOfDataframes = []
        i=0
        for (GroupTimestamp, CrudeTimestamp) in zip(listOfGroupTimestamps, listOfCrudeTimestamps):
            aux1 = pd.DataFrame(GroupTimestamp, columns=['listen_session'])
            aux2 = pd.DataFrame(CrudeTimestamp, columns=['ts_listen'])
            usuario = pd.DataFrame([i] * len(GroupTimestamp), columns=['user_id'])
            i = i+1
            #aux3 = pd.DataFrame(list(range(0, len(GroupTimestamp))))
            dataAuxiliar = pd.DataFrame.merge(usuario, aux2, how='inner', left_index=True, right_index=True)
            dataAuxiliar2 = pd.DataFrame.merge(dataAuxiliar, aux1, how='inner', left_index=True, right_index=True)
            listOfDataframes.append(dataAuxiliar2)
            #lista = pd.concat([lista, dataAuxiliar2], ignore_index=True, sort=False)
            
            #lista.append(list(zip(CrudeTimestamp, GroupTimestamp)))
            
            #lista.extend(np.column_stack((usuario, CrudeTimestamp, GroupTimestamp)))
        
        lista = pd.concat(listOfDataframes, ignore_index=True, sort=False)
        return lista
        
    auxiliar3 = createColumnWithTimestamps(auxiliar2, auxiliar)

#-----------------------------------------------Usefull features----------------------------------------------------------------------------------------------

del tudo['listen_type']

#dia da semana q que esta ouvindo
aux = pd.DataFrame({'weekday': pd.DatetimeIndex(tudo['ts_listen']*(10**9)).weekday})
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_index=True, right_index=True)
train = pd.DataFrame.merge(train, aux, how='left', left_index=True, right_index=True)

#porcentagem skipada por dia da semana 
aux = pd.DataFrame({'weekdaySkipped': train.groupby(['user_id', 'weekday'])['is_listened'].mean()})
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id', 'weekday'], right_index=True)
tudo['weekdaySkipped'].fillna(0.5, inplace=True)

#Quantas musicas cada usuario ouviu
aux = pd.DataFrame(train.groupby('user_id').size(), columns=['musics_listened'])
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on='user_id', right_index=True)
tudo['musics_listened'].fillna(0, inplace=True)
tudo['musics_listened'] = [int(i) for i in tudo['musics_listened']]
#tudo['musics_listened'] = preprocessing.normalize([np.array(tudo['musics_listened'])])

#Porcentagem de músicas ouvidas contra skipadas
aux = pd.DataFrame(train.groupby('user_id')['is_listened'].mean())
aux.rename(columns={'is_listened':'percentage_listened'}, inplace=True)
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on='user_id', right_index=True)
tudo['percentage_listened'].fillna(0.5, inplace=True)

#-----------------------------------------------------------Plataform Unified----------------------------------------------------------------------------------
if(useUnified):  
    aux = pd.DataFrame(train.groupby(['platform_name', 'platform_family'])['is_listened'].mean())
    aux['is_listened'] = np.arange(0,5,1)
    aux.rename(columns={'is_listened':'platform_unified'}, inplace=True)
    tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['platform_name', 'platform_family'], right_index=True)
    del tudo['platform_name']
    del tudo['platform_family']
    
    aux = pd.DataFrame(tudo.groupby('platform_unified')['is_listened'].mean())
    aux.rename(columns={'is_listened':'platform_listened'}, inplace=True)
    tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on='platform_unified', right_index=True)
    
    aux = pd.DataFrame(tudo.groupby(['platform_unified', 'context_type'])['is_listened'].mean())
    aux.rename(columns={'is_listened':'platformContext'}, inplace=True)
    tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['platform_unified', 'context_type'], right_index=True)

#----------------------------------------------------------Useless features-----------------------------------------------------------------------------------
if(useUseless):    
    #pocentagem de pulos de um determinado artista por usuario
    aux = pd.DataFrame(train.groupby(['user_id', 'artist_id'])['is_listened'].mean())
    aux.rename(columns={'is_listened':'ArtistLiked'}, inplace=True)
    tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id', 'artist_id'], right_index=True)
    tudo['ArtistLiked'].fillna(0.5, inplace=True)
    
    
    aux = pd.DataFrame(train.groupby(['user_gender', 'context_type'])['is_listened'].mean())
    aux.rename(columns={'is_listened':'genderAndContext'}, inplace=True)
    tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_gender', 'context_type'], right_index=True)
    
    aux = pd.DataFrame(train.groupby(['user_age', 'context_type'])['is_listened'].mean())
    aux.rename(columns={'is_listened':'ageContext'}, inplace=True)
    tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_age', 'context_type'], right_index=True)
    
    aux = pd.DataFrame(train.groupby(['genre_id', 'user_gender'])['is_listened'].mean())
    aux.rename(columns={'is_listened':'genreGender'}, inplace=True)
    tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['genre_id','user_gender'], right_index=True)
    tudo['genreGender'].fillna(0.5, inplace=True)
    
    aux = pd.DataFrame(train.groupby(['user_id', 'context_type'])['is_listened'].mean())
    aux.rename(columns={'is_listened':'userContext'}, inplace=True)
    tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id', 'context_type'], right_index=True)
    tudo['userContext'].fillna(0.5, inplace=True)
    
    
    #Porcentagem de músicas ouvidas pelo gênero
    aux = pd.DataFrame(tudo.groupby(['user_id','genre_id'])['is_listened'].mean())
    aux.rename(columns={'is_listened':'genre_listened_user'}, inplace=True)
    tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id','genre_id'], right_index=True)
    tudo['genre_listened_user'].fillna(0.5, inplace=True)


#----------------------------------------------------Time features---------------------------------------------------------------------------------------------
if(useTime):  
    #Pega auxiliar3 que contem músicas ouvidas em determinada faixa de tempo (O vetor traz a faixas faixas de tempo)
    aux = auxiliar3
    tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id', 'ts_listen'], right_on=['user_id', 'ts_listen'])
    
#----------------------------------------------------testing-----------------------------------------------------------------------------------------
    

#----------------------------------------------------Model and training-----------------------------------------------------------------------------------------
if (is_valid == True):
    train = tudo.iloc[:divide]
    del train['sample_id']
    
    train, valid = split_validation(train)

    y_train = train['is_listened']
    y_valid = valid['is_listened']
    del train['is_listened'], valid['is_listened']

    d_train = lgb.Dataset(train, y_train)
    d_valid = lgb.Dataset(valid, y_valid)

    model = train_lgb(d_train, val_sets=[d_train, d_valid], n_round=10000)
else:
    train = tudo.iloc[:divide]
    del train['sample_id']
    test = tudo.iloc[divide:]
    del test['is_listened']
    
    X_train, y_train = split_validation(train, 50)
    X_val = X_train['is_listened']
    y_val = y_train['is_listened']
    
    del X_train['is_listened']
    del train['index']
    del y_train['is_listened']
    
    validation2 = train['is_listened']
    del train['is_listened']
    
    #X_train, y_train, X_val, y_val = train_test_split(train, validation2, test_size=0.2, random_state=123)
    
    
    features = train.columns
    d_train = lgb.Dataset(train, validation2)
    predictionsXGB = []
    predictionsLGB = []
    for i in range(0,10):
            model = train_xgb(X_train, X_val, y_train, y_val, i*189)
            preds = model.predict(test[features])
            predictionsXGB.append(preds)

            #model = train_lgb(69, d_train, val_sets=[d_train], n_round=10)
            #preds = model.predict(test[features])
            #predictionsLGB.append(preds)
            
            print("Iteração de numero: ",i)
      
    #sub1 = pd.DataFrame({'sample_id': test['sample_id'].astype(int), 'is_listened': np.mean(predictionsXGB, axis=0)})
    #sub2 = pd.DataFrame({'sample_id': test['sample_id'].astype(int), 'is_listened': np.mean(predictionsLGB, axis=0)})
    predictions = np.mean([np.mean(predictionsXGB, axis=0), np.mean(predictionsLGB, axis=0)], axis=0)

    sub = pd.DataFrame({'sample_id': test['sample_id'].astype(int), 'is_listened': predictions})
    #sub = pd.DataFrame({'sample_id': test['sample_id'].astype(int), 'is_listened': preds})
    sub
    sub.to_csv('sub_001.csv', index=False)
    