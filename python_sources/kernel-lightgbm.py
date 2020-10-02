import numpy as np
import pandas as pd
import os

# Leemos datos
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Definimos dataframe de outputs solo con ID
outputs=pd.DataFrame(test['ID'])

# Unimos train y test para tratar variables
test['TARGET']=np.nan
full_data=pd.concat([train, test], ignore_index=True,sort=False)

# Eliminamos los warnings de "A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead"
pd.options.mode.chained_assignment = None


##########################################################################################################
########################################                     #############################################
######################################## FEATURE ENGINEERING #############################################
########################################                     #############################################
##########################################################################################################

# 1) Correlaciones y constantes
################################################################################
# Primero de todo vemos que ciertas columnas están altamente correlacionadas
# o que son la misma o un "shift" de la misma:

thresholdCorrelation = 0.99
def InspectCorrelated(df):
    corrMatrix = df.corr().abs()
    upperMatrix = corrMatrix.where(np.triu(
                                   np.ones(corrMatrix.shape),
                                   k=1).astype(np.bool))
    correlColumns=[]
    for col in upperMatrix.columns:
        correls=upperMatrix.loc[upperMatrix[col]>thresholdCorrelation,col].keys()
        if (len(correls)>=1):
            correlColumns.append(col)
            print("\n",col,'->', end=" ")
            for i in correls:
                print(i, end=" ")
    print('\nSelected columns to drop:\n',correlColumns)
    return(correlColumns)

correlColumns=InspectCorrelated(full_data.iloc[:,1:-1])

# Si vemos que todo ok las tiramos a la basura:
full_data=full_data.drop(correlColumns,axis=1)
train=train.drop(correlColumns,axis=1)
test=test.drop(correlColumns,axis=1)


# Vemos si hay alguna columna constante...
for col in list(full_data):
    print(len(full_data[col].unique()),'\t',full_data[col].dtypes,'\t',col)
# Bien, todas con muchos valores


# 2) Missings
################################################################################
# Función para sacar columnas con más de n_miss de missings
def miss(ds,n_miss):
    for col in list(ds):
        if ds[col].isna().sum()>=n_miss:
            print(col,ds[col].isna().sum())

# Vemos qué columnas contienen al menos 1 missing...
miss(full_data,1)


# 2.1) Missings -> Eliminamos algunas filas
################################################################################
# Se repiten muchas columnas con 3 missings, miramos los datos y...
# ...vemos que hay 4 observaciones con muchas columnas missings, que son:
# A1039
# A2983
# A3055
# A4665

# Si están todas en train, las quitamos:
len(train.loc[train['ID']=='A1039',])
len(train.loc[train['ID']=='A2983',])
len(train.loc[train['ID']=='A3055',])
len(train.loc[train['ID']=='A4665',])
# Perfect! FUERA:
train = train[train['ID']!='A1039']
train = train[train['ID']!='A2983']
train = train[train['ID']!='A3055']
train = train[train['ID']!='A4665']

# Volvemos a unir full_data
full_data=pd.concat([train, test], ignore_index=True,sort=False)


# 2.2) Missings -> Rellenamos columnas
################################################################################
# Volvemos a ver qué columnas contienen missings una vez eliminadas las 3 filas conflictivas:
miss(full_data,1)

# Pintamos comportamiento individual de features:
# Barras = Población en cada bucket (eje izquierda)
# Linea = TMO (eje derecha)
import matplotlib.pyplot as plt

def feat_graph(df,icol,binary_col,n_buckets):
    feat_data=df[[icol,binary_col]]
    feat_data['bucket']=pd.qcut(feat_data.iloc[:,0], q=n_buckets,labels=False,duplicates='drop')+1

    if len(feat_data.loc[feat_data[icol].isna(),'bucket'])>0:
        feat_data.loc[feat_data[icol].isna(),'bucket']=0

    hist_data_p=pd.DataFrame(feat_data[['bucket',binary_col]].groupby(['bucket'])[binary_col].mean()).reset_index()
    hist_data_N=pd.DataFrame(feat_data[['bucket',binary_col]].groupby(['bucket'])[binary_col].count()).reset_index()
    hist_data=pd.merge(hist_data_N,hist_data_p,how='left',on='bucket')
    hist_data.columns=['bucket','N','p']

    width = .70 # width of a bar
    hist_data['N'].plot(kind='bar', width = width, color='darkgray')
    hist_data['p'].plot(secondary_y=True,marker='o')
    ax = plt.gca()
    plt.xlim([-width, len(hist_data)-width/2])
    if len(feat_data.loc[feat_data[icol].isna(),'bucket'])>0:
        ax.set_xticklabels(('Missing', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6' ,'G7', 'G8', 'G9', 'G10'))
    else:
        ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5', 'G6' ,'G7', 'G8', 'G9', 'G10'))

    plt.title(icol)
    plt.show()

for icol in list(train.iloc[:,1:-1]):
    feat_graph(train,icol,'TARGET',10) # Poner como máximo 10 grupos

# Imputación de missings:
# Vamos variable a variable que tenga suficientes missings y en función de su TMO observada
# Le assignamos el valor correspondiente a:
# Percentil 5 si se comporta como la TMO de G1
# Percentil 95 si se comporta como la TMO de G10
# Percentil 50 si el missing no aporta información (se comporta como el centro)

# Listado de variables y # de missings totales en full_data que tengan al menos 30 missings:
miss(full_data,30)

# X21 102
# X24 132
# X27 390
# X28 105
# X32 46
# X37 2544
# X41 84
# X45 267
# X47 35
# X52 36
# X53 105
# X54 105
# X64 105

# Definimos qué hacemos mirando los dibujos
Imputa={'X21':'P5',
        'X24':'P50', # missing tiene TMO = 0
        'X27':'P5',
        'X28':'P5',
        'X32':'P5',  # missing tiene TMO = 0
        'X37':'P5',
        'X41':'P50', # missing tiene TMO = 0
        'X45':'P5',
        'X47':'P50', # missing tiene TMO = 0
        'X52':'P5',  # missing tiene TMO = 0
        'X53':'P5',
        'X54':'P5',
        'X64':'P5'}

# Imputamos percentil correspondiente
for col, percentil in Imputa.items():
    if percentil=='P5':
        pctl=0.05
    elif percentil=='P50':
        pctl=0.5
    elif percentil=='P95':
        pctl=0.95
    imput_val=full_data[col].quantile(q=pctl)
    full_data.loc[full_data[col].isna(),col]=imput_val

# El resto de missings los imputamos por K-Nearest Neighbours
# Primero estandarizamos los datos:
from sklearn import preprocessing

X_full_data=full_data.drop(['ID','TARGET'],axis=1)
X=X_full_data.values
X_scaled = preprocessing.StandardScaler().fit_transform(X)

# Imputamos
from fancyimpute import KNN
# X_filled_knn será la matriz completada
# X_scaled contiene missings pero ya está estandarizada
# Usamos k=10 filas cercanas por distancia Euclídea para rellenar los missings:
X_filled_knn = KNN(k=10).fit_transform(X_scaled)
X_full_data_std_knn = pd.DataFrame(X_filled_knn, columns=list(X_full_data))

# Comprobamos que ya no quedan missings:
miss(X_full_data_std_knn,1)

# Volvemos a redefinir full_data, train y test
full_data=pd.concat([full_data['ID'],X_full_data_std_knn,full_data['TARGET']],axis=1)
train=full_data.loc[~full_data['TARGET'].isna(),].reset_index(drop=True)
test=full_data.loc[full_data['TARGET'].isna(),].reset_index(drop=True)
test.drop(['TARGET'],axis=1, inplace=True)


# 3) Creamos Alertas (Automatizado)
################################################################################
# Para cada feature original vamos a obtener el corte óptimo que nos 
# separe la morosidad a izquierda o derecha de la mejor manera (cross-entropy).
# Ordenaremos estas "alertas" de más discriminantes a menos a través de la
# Tasa de Morosidad Relativa (TMR) =
# = TM de las obs. para las que se activa la alerta / TM de la muestra entera 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

alertas=pd.DataFrame()
GINIS=list()
ACTIVACIONES=list()
TMRS=list()

for FACTOR in list(train)[1:-1]:
    # Construimos árbol de profundidad 1 (un solo split óptimo)
    X=train[[FACTOR]].reset_index(drop=True)
    Y=train['TARGET'].reset_index(drop=True)
    dtree=DecisionTreeClassifier(max_depth=1)
    dtree.fit(X,Y)
    # Split óptimo
    threshold = dtree.tree_.threshold[0]
    # Creación alerta
    alertas[FACTOR]=full_data[FACTOR]
    alertas[FACTOR+'_b']=np.zeros(len(full_data))
    alertas.loc[alertas[FACTOR]<=threshold,FACTOR+'_b']=1

    # subconjunto de train
    alerta_train=alertas.loc[0:len(train)-1,[FACTOR+'_b']]

    # Calculamos GINI de la alerta
    gini=roc_auc_score(Y,alerta_train)*2-1

    # Si el GINI sale negativo, giramos 1 y 0
    if gini<0:
        alertas[FACTOR+'_b']=np.logical_not(alertas[FACTOR+'_b']).astype(int)

    # Volvemos a calcular GINI para asegurarnos que todos son +
    alerta_train=alertas.loc[0:len(train)-1,[FACTOR+'_b']]
    gini=roc_auc_score(Y,alerta_train)*2-1

    # ACTIVACIONES
    activ=int(alerta_train[FACTOR+'_b'].sum())

    # TMR
    TMO=pd.DataFrame(pd.concat([alerta_train,Y],axis=1).groupby([FACTOR+'_b'])['TARGET'].mean()).reset_index()
    TMR=float(TMO.loc[TMO[FACTOR+'_b']==1,'TARGET'])/Y.mean()

    # Tiramos factor original de la tabla
    alertas.drop([FACTOR],axis=1,inplace=True)

    # Añadimos GINI, ACTIVACIONES y TMR a la secuencia
    GINIS.append(gini*100)
    ACTIVACIONES.append(activ)
    TMRS.append(TMR*100)

# Tabla de severidades
severidad=pd.DataFrame({'Alerta':list(alertas),
                        'Gini (%)':GINIS,
                        'Activaciones (N)': ACTIVACIONES,
                        'TMR (%)': TMRS,
                        'TMR/Act': [a/b for a,b in zip(TMRS,ACTIVACIONES)]})

severidad=severidad.sort_values(by="TMR (%)", ascending=False).reset_index(drop=True)

# Vemos si algunas alertas están altamente correlacionadas
# Primero reordenamos alertas por su importancia
alertas=alertas[severidad['Alerta']]

thresholdCorrelation = 0.95
correlColumns=InspectCorrelated(alertas)

# Si vemos que todo ok las tiramos a la basura:
alertas=alertas.drop(correlColumns,axis=1)
for col in correlColumns:
    severidad=severidad[severidad['Alerta']!=col].reset_index(drop=True)

# Vamos a separar en alertas leves, medias y graves en función de su efectividad:
# Hay 48 Alertas
# Cortamos Graves si TMR>=500 --> Nos dejamos 4 que están por encima de 490. Hacemos si TMR>=490
# Cortamos Medias si TMR>=350 (y hasta TMR<490)
# Leves si TMR<350
graves=severidad.loc[severidad['TMR (%)']>=490,'Alerta'].tolist()
medias=severidad.loc[(severidad['TMR (%)']<490) & (severidad['TMR (%)']>=350),'Alerta'].tolist()
leves=severidad.loc[severidad['TMR (%)']<350,'Alerta'].tolist()

# Creamos contadores de Alertas
# graves
alertas_graves=alertas[graves]
alertas_graves['CONT_GRAVES']=alertas_graves.sum(axis=1)
# medias
alertas_medias=alertas[medias]
alertas_medias['CONT_MEDIAS']=alertas_medias.sum(axis=1)
# leves
alertas_leves=alertas[leves]
alertas_leves['CONT_LEVES']=alertas_leves.sum(axis=1)

# Añadimos contadores a los datos
full_data['CONT_GRAVES']=alertas_graves['CONT_GRAVES']
full_data['CONT_MEDIAS']=alertas_medias['CONT_MEDIAS']
full_data['CONT_LEVES']=alertas_leves['CONT_LEVES']

# Reordenamos columnas (TARGET al final)
cols=list(full_data)
cols.insert(len(cols), cols.pop(cols.index('TARGET')))
full_data = full_data.reindex(columns= cols)

# Separamos
train=full_data.loc[~full_data['TARGET'].isna(),].reset_index(drop=True)
test=full_data.loc[full_data['TARGET'].isna(),].reset_index(drop=True)
test.drop(['TARGET'],axis=1, inplace=True)


##########################################################################################################
############################################                  ############################################
############################################ MODELO Light GBM ############################################
############################################                  ############################################
##########################################################################################################

################################################################################
# Light GBM
# Toda la información aquí: "https://lightgbm.readthedocs.io/en/latest/index.html"
################################################################################
import lightgbm as lgb

# 1) Definimos conjuntos de entrenamiento y test:
################################################################################
predictoras=list(train)[1:-1]
X_train=train[predictoras].reset_index(drop=True)
Y_train=train['TARGET'].reset_index(drop=True)
X_test=test[predictoras].reset_index(drop=True)

# 2) Parámetros
################################################################################
RE=123 # Seed que utilizaremos para los folds y la parte random del modelo
esr=100 # early_stopping_rounds
n_folds=5 # Número de folds para la cross-validación

params = {'objective': 'binary',
          'learning_rate': 0.01,
          'num_leaves': 40,
          'colsample_bytree': 0.5,
          'bagging_fraction': 0.5,
          'random_seed': RE}

# 3) Cross-Validación
################################################################################
# Datos en formato lgbm:
train_lgbm=lgb.Dataset(data = X_train, label = Y_train, feature_name = list(X_train))

print('\nLightGBM CV...\n')
cv_lightgbm = lgb.cv(params,
                train_set=train_lgbm,
                metrics='auc', # Es la métrica para mirar efectividad en los folds e implementar el early_stopping_rounds
                num_boost_round=5000, # Número de rondas máximo (árboles). Poner valor alto y que pare por "esr"
                nfold=n_folds,
                early_stopping_rounds=esr,
                verbose_eval=50, # Nos muestra la métrica CV cada tantos árboles
                seed=RE)

# Obtenemos número óptimo de rondas
n_opt=len(cv_lightgbm.get('auc-mean'))
# Cuando entrenemos todo el train, utilizaremos más rondas (proporcional al numero de folds empleado)
nrounds=int(n_opt/(1-1/n_folds))

# 4) Modelo sobre todo el train con los parámetros óptimos y número de rondas óptimas del CV
################################################################################
print('\nLightGBM Fit...\n')
model_lightgbm = lgb.LGBMClassifier()
model_lightgbm.set_params(**params)
model_lightgbm.set_params(n_estimators=nrounds)
model_lightgbm.fit(X=X_train,y=Y_train)

##########################################################################################################
###############################################            ###############################################
############################################### RESULTADOS ###############################################
###############################################            ###############################################
##########################################################################################################

# Importancia de los factores
################################################################################
import shap
explainer = shap.TreeExplainer(model_lightgbm)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, max_display=25)

# Predicción final (submission)
################################################################################
test['Pred']=model_lightgbm.predict_proba(X_test)[:,1]
outputs_lightgbm=pd.merge(outputs, test[['ID','Pred']], on='ID', how='left')

# Outputs a .csv
################################################################################
outputs_lightgbm.to_csv('outputs_lightgbm.csv', index = False)
print('\nEND')

##########################################################################################################
#############################################                #############################################
############################################# FIN DE LA CITA #############################################
#############################################                #############################################
##########################################################################################################
