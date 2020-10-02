import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from datetime import datetime
from collections import Counter
from xgboost import plot_importance
from matplotlib import pyplot
import datetime
import timeit

path = '../'

# True for validation or False to generate predictions
is_valid = False
entraXgb = True
temporal = True

def split_validation(df):
    df.sort_values(by=["user_id", "ts_listen"], inplace=True)
    df.reset_index(inplace=True)
    val_indexes = df.groupby('user_id')['index'].max()
    df_train = df[~df['index'].isin(val_indexes)]
    df_valid = df[df['index'].isin(val_indexes)]
    del df_train['index'], df_valid['index']
    return df_train, df_valid

def train_lgb(seed, dtrain, val_sets, n_round):
    params = {   
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        'num_leaves': 50,
        'min_data_in_leaf': 10, 
        'max_depth': 10,
        'feature_fraction': 1,
        'bagging_freq': 1,
        'bagging_fraction': 1,
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
    
def train_xgb(x_train, x_val, y_train, y_val):
    model = xgb.XGBRegressor(objective='binary:logistic',
                              tree_method='exact',
                              base_score=0.5, 
                              colsample_bynode=0.6,
                              colsample_bylevel=0.5, 
                              colsample_bytree=0.4,
                              gamma=0.9, 
                              learning_rate=0.07, 
                              max_delta_step=0, 
                              max_depth=20,
                              min_child_weight=1, 
                              n_estimators=250,
                              n_jobs=-1, 
                              reg_alpha=0.03,
                              reg_lambda=1, 
                              scale_pos_weight=1, 
                              seed=123, 
                              silent=False,
                              subsample=0.7).fit(x_train, x_val, eval_set=[(y_train, y_val)], early_stopping_rounds=20)
    
    return model
    
##################################################################################
    
def splitByTimestamp(distancia ,lst):
    indices = [i+1 for (x, y, i) in zip(lst, lst[1:], range(len(lst))) if 1800 < abs(x - y)]
    result = [lst[start:end] for start, end in zip([0] + indices, indices + [len(lst)])]
    return result
    
multiply = lambda x : x*2
    
##########################################################################################################

train = pd.read_csv(path+'input/train.csv')
test = pd.read_csv(path+'input/test.csv')

divide = len(train)

tudo = pd.concat([train, test], ignore_index=True, sort=False)

aux = pd.DataFrame(train.groupby(['platform_name', 'platform_family'])['is_listened'].mean())
aux['is_listened'] = np.arange(0,5,1)
aux.rename(columns={'is_listened':'platform_unified'}, inplace=True)
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['platform_name', 'platform_family'], right_index=True)
del tudo['platform_name']
del tudo['platform_family']

##########################################################################################################

def getHeatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(14,12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(correlation_matrix)

def plotFeatureImportance(model):
    plot_importance(model).figure.set_size_inches(15, 15)
    
##########################################################################################################

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
    
##################################################################################################################################################################

def createFeature(indexes, oldnName, newName, dataframe, byIndex):
    aux = pd.DataFrame(dataframe)
    aux.rename(columns={oldName: newName}, inplace=True)
    if(byIndex):
        tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=indexes, right_index=True)
    else:
        tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=indexes, right_on=indexes)

##################################################################################################################################################################
df = tudo[['user_id', 'ts_listen']].sort_values(['user_id', 'ts_listen'])
tslistenToDays = pd.DatetimeIndex(df['ts_listen']*(10**9))
tslistenDelta = tslistenToDays.to_series().diff()
tslistenDiference = [1 if abs(i.days)*86400 + abs(i.seconds) > 1800 else 0 for i in tslistenDelta]
tslistenDiference[0] = 1
aux = pd.DataFrame({'listenSession': tslistenDiference})
aux = pd.merge(pd.DataFrame(df['ts_listen']), aux, how='inner', left_index=True, right_index=True)
aux = pd.merge(pd.DataFrame(df['user_id']), aux, how='inner', left_index=True, right_index=True)
tudo = pd.merge(tudo, aux, how='left', left_on=['user_id', 'ts_listen'], right_on=['user_id', 'ts_listen'])
train = tudo.iloc[:divide]
del train['sample_id']

aux = pd.DataFrame({'weekday': pd.DatetimeIndex(tudo['ts_listen']*(10**9)).weekday})
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_index=True, right_index=True)
train = pd.DataFrame.merge(train, aux, how='left', left_index=True, right_index=True)

aux = pd.DataFrame({'weekdaySkipped': train.groupby(['user_id', 'weekday'])['is_listened'].mean()})
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id', 'weekday'], right_index=True)
tudo['weekdaySkipped'].fillna(0.5, inplace=True)

#Quantas musicas cada usuario ouviu
aux = pd.DataFrame({'musics_listened': train.groupby('user_id').size()})
#aux = (aux - aux.min()) / (aux.max() - aux.min())
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on='user_id', right_index=True)
tudo['musics_listened'].fillna(0, inplace=True)
tudo['musics_listened'] = [int(i) for i in tudo['musics_listened']]
#tudo['musics_listened'] = preprocessing.normalize([np.array(tudo['musics_listened'])])

#Porcentagem de músicas ouvidas contra skipadas
aux = pd.DataFrame({'percentage_listened': train.groupby('user_id')['is_listened'].mean()})
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on='user_id', right_index=True)
tudo['percentage_listened'].fillna(0.5, inplace=True)

train = tudo.iloc[:divide]
del train['sample_id']

aux = pd.DataFrame({'platform_listened': train.groupby('platform_unified')['is_listened'].mean()})
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on='platform_unified', right_index=True)

aux = pd.DataFrame({'genderAndContext': train.groupby(['user_gender', 'context_type'])['is_listened'].mean()})
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_gender', 'context_type'], right_index=True)

aux = pd.DataFrame({'ageContext': train.groupby(['user_age', 'context_type'])['is_listened'].mean()})
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_age', 'context_type'], right_index=True)

aux = pd.DataFrame({'platformContext': train.groupby(['platform_unified', 'context_type'])['is_listened'].mean()})
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['platform_unified', 'context_type'], right_index=True)

#aux = pd.DataFrame(train.groupby(['genre_id', 'user_gender'])['is_listened'].mean())
#aux.rename(columns={'is_listened':'genreGender'}, inplace=True)
#tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['genre_id','user_gender'], right_index=True)
#tudo['genreGender'].fillna(0.5, inplace=True)

aux = pd.DataFrame({'userContext': train.groupby(['user_id', 'context_type'])['is_listened'].mean()})
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id', 'context_type'], right_index=True)
tudo['userContext'].fillna(0.5, inplace=True)

aux = pd.DataFrame({'durationOfMusics': train.groupby(['user_id'])['media_duration'].mean()})
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id'], right_index=True)
musicas = tudo[['media_id', 'media_duration']].sort_values(by=['media_id', 'media_duration']).drop_duplicates(subset='media_id', keep='last')

tudo['durationOfMusics'].fillna(float(musicas['media_duration'].median()), inplace=True)

#Porcentagem de músicas ouvidas pelo tamanho
#aux = pd.DataFrame(tudo.groupby('media_duration')['is_listened'].mean())
#aux.rename(columns={'is_listened':'duration_notSkipped'}, inplace=True)
#tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on='media_duration', right_index=True)

#Porcentagem de músicas ouvidas pelo artista
#aux = pd.DataFrame(tudo.groupby('artist_id')['is_listened'].mean())
#aux.rename(columns={'is_listened':'artist_listened'}, inplace=True)
#tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on='artist_id', right_index=True)

#Porcentagem de músicas ouvidas pelo gênero
#aux = pd.DataFrame(tudo.groupby(['user_id','genre_id'])['is_listened'].mean())
#aux.rename(columns={'is_listened':'genre_listened_user'}, inplace=True)
#tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id','genre_id'], right_index=True)
#tudo['genre_listened_user'].fillna(0.5, inplace=True)

#Porcentagem de músicas ouvidas pela plataforma
#aux = pd.DataFrame(tudo.groupby('context_type')['is_listened'].mean())
#aux.rename(columns={'is_listened':'context_listened'}, inplace=True)
#tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on='context_type', right_index=True)

#Hora do dia quando a música foi ouvida
#aux = [int(datetime.fromtimestamp(time).strftime('%H%M')) for time in tudo['ts_listen']]
#aux = pd.DataFrame(aux)
#aux.rename(columns={0:'timeInDay'}, inplace=True)
#tudo = pd.DataFrame.merge(tudo, aux, how='left', left_index=True, right_index=True)

#Outro meio de contar quantas músicas cada usuário ouviu. Deixe comentado
#aux = pd.DataFrame.from_dict(Counter(train['user_id']), orient='index').reset_index()
#aux.rename(columns={0:'entries', 'index':'user_id'}, inplace=True)
#train = pd.DataFrame.merge(train, aux, how='left', left_on='user_id', right_on='user_id')
#test = pd.DataFrame.merge(test, aux, how='left', left_on='user_id', right_on='user_id')

#Porcentagem por gênero que cada pessoa ouviu
#aux = pd.DataFrame(train.groupby(['user_id', 'genre_id'])['is_listened'].mean())
#aux.rename(columns={'is_listened':'preferenciaGenero'}, inplace=True)
#tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id', 'genre_id'], right_on=['user_id', 'genre_id'])
#tudo['preferenciaGenero'].fillna(0.5, inplace=True)

#Pega auxiliar3 que contem músicas ouvidas em determinada faixa de tempo (O vetor traz a faixas faixas de tempo)
aux = auxiliar3
tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id', 'ts_listen'], right_on=['user_id', 'ts_listen'])
train = tudo.iloc[:divide]
del train['sample_id']

#parteTudo = tudo.iloc[:divide]
#parteTudo.reset_index(inplace=True)
#aux = pd.DataFrame(parteTudo.groupby(['user_id'])['listen_session'].max())

#aux2 = parteTudo[['user_id', 'listen_session', 'is_listened']].sort_values(by=['user_id', 'listen_session'])
#aux2.reset_index(inplace=True)

#aux3 = pd.DataFrame.merge(aux, aux2, how='left', left_on=['user_id', 'listen_session'], right_on=['user_id', 'listen_session'])

#Genero skipado
#aux = pd.DataFrame({'GenreListened': train.groupby(['user_id', 'listen_session', 'genre_id'])['is_listened'].mean()})
#tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id', 'listen_session', 'genre_id'], right_on=['user_id', 'listen_session', 'genre_id'])
#tudo['GenreListened'].fillna(0.5, inplace=True)


#aux = pd.DataFrame(tudo.groupby(['user_id', 'listen_session'])['is_listened'].mean())
#aux.rename(columns={'is_listened':'QuantasOuvidasNaSessao'}, inplace=True)
#tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id', 'listen_session'], right_on=['user_id', 'listen_session'])
#tudo['QuantasOuvidasNaSessao'].fillna(0.5, inplace=True)

#aux = pd.DataFrame(tudo.groupby(['user_id', 'listen_session', 'genre_id'])['is_listened'].mean())
#aux.rename(columns={'is_listened':'MediaOuvidasNaSessao'}, inplace=True)
#tudo = pd.DataFrame.merge(tudo, aux, how='left', left_on=['user_id', 'listen_session', 'genre_id'], right_on=['user_id', 'listen_session', 'genre_id'])
#tudo['MediaOuvidasNaSessao'].fillna(0, inplace=True)

#tudo = pd.get_dummies(tudo, columns=['weekday'])

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
    
    validation2 = train['is_listened']
    del train['is_listened']
    
    X_train, y_train, X_val, y_val = train_test_split(train, validation2, test_size=0.2, random_state=123)
    
    features = train.columns
    
    samples = test['sample_id']
    del test['sample_id']

    if(entraXgb):
        model = train_xgb(X_train, X_val, y_train, y_val)
        preds = model.predict(test[features])
    else:
        d_train = lgb.Dataset(train, validation2)
        model = train_lgb(123, d_train, val_sets=[d_train], n_round=100)
        preds = model.predict(test[features])
        
    sub = pd.DataFrame({'sample_id': samples, 'is_listened': preds})
    sub
    sub.to_csv('sub_001.csv', index=False)
    

    #test = pd.read_csv(path+'input/test.csv')