##########################################################################################################
############################################            ##################################################
############################################ INITIALIZE ##################################################
############################################            ##################################################
##########################################################################################################
import pandas as pd
import numpy as np

train = pd.read_csv("../input//train.csv")
test = pd.read_csv("../input//test.csv")

# Renombramos alertas que al estar estandarizadas se pierde el efecto WOE
# Les quitamos la cola "_W".
train = train.rename(columns={'aler_leves_W': 'aler_leves', 'aler_medias_W': 'aler_medias', 'aler_graves_W': 'aler_graves'})
test = test.rename(columns={'aler_leves_W': 'aler_leves', 'aler_medias_W': 'aler_medias', 'aler_graves_W': 'aler_graves'})

# Nos aseguramos que no hay missings
train.isnull().values.any()
test.isnull().values.any()
outputs=pd.DataFrame(test.ID)

########################################################
###################### Features ########################
########################################################

# Cálculo del WOE
###################################################################################
# Malos y Buenos totales
BAD=sum(train.TARGET)
GOOD=len(train)-BAD

# Juntamos train y test para hacer bucketizaciones globales
test['TARGET']=np.nan
full_data_woe=pd.concat([train, test], ignore_index=True,sort=False)
test = test.drop('TARGET', 1)

# Función de cálculo de WOE sobre un dataset (ds) y columna NUMÉRICA
# El dataset debe tener parte train (con TARGET informado) y parte test con TARGET Null
# Devuelve el mismo dataset pero con la columna transformada (y su nombre acabado en _W)
def WoE_num(ds,i_col,n_buckets):
    ds['bucket']=pd.qcut(ds[i_col], q=n_buckets,labels=False,duplicates='drop')
    tabla_woe=ds[['bucket','TARGET']].groupby(['bucket']).sum(skipna=True).reset_index()

    if 0 in tabla_woe['TARGET'].values:
        ds['bucket']=pd.qcut(ds[i_col], q=n_buckets-1,labels=False,duplicates='drop')
        tabla_woe=ds[['bucket','TARGET']].groupby(['bucket']).sum(skipna=True).reset_index()
# La bucketización en Python está bien hecha (he visto que el "ntile" de R parte los empates
# como le sale de los huevos.
# Nos aseguramos que al tirar los cortes repetidos con "duplicates='drop'" (esto pasa cuando
# una variable tiene acumulación de valores repetidos) almenos queden 5 buckets restantes.
# Si no, no woeizamos la variable.
    if len(ds['bucket'].unique())>=5:
        tabla_woe = tabla_woe.rename(columns={'TARGET': 'BAD'})
        tabla_woe['TOTAL']=ds[['bucket','TARGET']].groupby(['bucket']).count().reset_index()['TARGET']
        tabla_woe['GOOD']=(tabla_woe['TOTAL']-tabla_woe['BAD']).astype(int)
        # Cálculo WOE por bucket
        tabla_woe['WOE']=np.log((tabla_woe['GOOD']/GOOD)/(tabla_woe['BAD']/BAD))
        # Nueva variable "WOEizada"
        ds=pd.merge(ds, tabla_woe[['bucket','WOE']], on='bucket', how='left')
        # Gestión de nombres
        ds = ds.drop('bucket', axis=1)
        ds = ds.rename(columns={'WOE': i_col+"_W"})
        # Eliminamos variable original
        ds = ds.drop(i_col, axis=1)
    else:
        ds = ds.drop(i_col, axis=1)
        ds = ds.drop('bucket', axis=1)
    return(ds)

# Lista con las variables a las que vamos a "WOEizar"
lista = list(set(list(train))-set(["ID","TARGET","aler_leves","aler_medias","aler_graves"]))
# Bucle sobre las variables
for nombre_columna in lista:
  full_data_woe=WoE_num(full_data_woe,nombre_columna,10)

# Función de cálculo de WOE sobre un dataset (ds) y columna NUMÉRICA pero que representa categorías
# El dataset debe tener parte train (con TARGET informado) y parte test con TARGET Null
# Devuelve el mismo dataset pero con la columna transformada (y su nombre acabado en _W)
def WoE_aler(ds,i_col):
    tabla_woe=ds[[i_col,'TARGET']].groupby([i_col]).sum(skipna=True).reset_index()
    tabla_woe = tabla_woe.rename(columns={'TARGET': 'BAD'})
    tabla_woe['TOTAL']=ds[[i_col,'TARGET']].groupby([i_col]).count().reset_index()['TARGET']
    tabla_woe['GOOD']=(tabla_woe['TOTAL']-tabla_woe['BAD']).astype(int)
    # Cálculo WOE por bucket
    tabla_woe['WOE']=np.log((tabla_woe['GOOD']/GOOD)/(tabla_woe['BAD']/BAD))
    # Nueva variable "WOEizada"
    ds=pd.merge(ds, tabla_woe[[i_col,'WOE']], on=i_col, how='left')
    # Gestión de nombres
    ds = ds.drop(i_col, axis=1)
    ds = ds.rename(columns={'WOE': i_col+"_W"})
    return(ds)

# Lista con las alertas a las que vamos a "WOEizar"
lista_aler=["aler_leves","aler_medias","aler_graves"]
# Bucle sobre las alertas
for nombre_columna in lista_aler:
  full_data_woe=WoE_aler(full_data_woe,nombre_columna)

# Separamos train y test
train_woe=full_data_woe[~(np.isnan(full_data_woe.TARGET))]
test_woe=full_data_woe[np.isnan(full_data_woe.TARGET)]
test_woe = test_woe.drop('TARGET', 1)

# Juntamos a las variables originales las WoEizadas (y tiramos alertas originales)
train=pd.merge(train[list(set(list(train))-set(["aler_leves","aler_medias","aler_graves"]))], train_woe[list(set(list(train_woe))-set(["TARGET"]))], on="ID", how='left')
test=pd.merge(test[list(set(list(test))-set(["aler_leves","aler_medias","aler_graves"]))], test_woe, on="ID", how='left')
###################################################################################


########################################################
####################### MODELO #########################
########################################################

from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, auc

# Definimos métrica de evaluación por Cross-Validación
# Ponemos NÚMERO DE FOLDS = 8
def auc_cv(model, X, y):
	return (cross_val_score(model, X, y, scoring='roc_auc', cv=8)).mean()

#########
# Lasso #
#########
from sklearn.linear_model import LogisticRegression

# Definimos dataset de entreno
trainLasso=train.sort_values(by=['TARGET'])
predictoras=list(set(list(trainLasso))-set(["ID","TARGET"]))
X_train=trainLasso[predictoras]
Y_train=trainLasso.TARGET

# Grid Search del parámetro "C", Cross-Validado
# Aquí, C no equivale a la lambda de glmnet de R
# Debería ser C=1/(N*lambda) (siendo N el número de observaciones)
# Ver: https://stats.stackexchange.com/questions/203816/logistic-regression-scikit-learn-vs-glmnet
# Un Lasso en R da lambda=0.00019, que equivale a C=0.09

# Hacemos algunos tests
# En caso de no tener ni idea de por dónde va la C, inicialmente probaríamos
# diversos órdenes de magnitud, p.e. Cs=[0.001,0.01,0.1,1,10,100,1000] etc... e iríamos acotando
Cs=[0.08,0.09,0.1]
cv_lasso = [auc_cv(LogisticRegression(C = c, penalty='l1',solver='liblinear',random_state=12345), X_train, Y_train) for c in Cs]
print([item*2 - 1 for item in cv_lasso])

# Entrenamos todo el train con el parámetro óptimo:
model_lasso = LogisticRegression(C=0.09, penalty='l1',solver='liblinear')
model_lasso.fit(X_train, Y_train)
weights = pd.DataFrame({'feature':np.array(X_train.columns),'coefs':np.transpose(np.array(model_lasso.coef_))[:,0]})

########################################################
##################### RESULTADOS #######################
########################################################
# Predicción
Y_test=test[predictoras]
# Al predecir, la primera columna (0) corresponde a la probabilidad de ser 0 y la seggunda (1) a la
# probabilidad de ser 1, que es la que nos interesa:
test['Pred']=model_lasso.predict_proba(Y_test)[:,1]
outputs_lasso=pd.merge(outputs, test[['ID','Pred']], on='ID', how='left')

# Outputs a .csv
outputs_lasso.to_csv("outputs.csv", index = False)
weights.to_csv("lasso_features.csv", index = False)

##########################################################################################################
##########################################################################################################
########################################### FIN DE LA CITA ###############################################
##########################################################################################################
##########################################################################################################