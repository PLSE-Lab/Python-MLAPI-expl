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

# Definimos dataframe de outputs solo con ID
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
# una variable tiene acumulación de valores repetidos) almenos queden 2 buckets restantes.
# Si no, no woeizamos la variable.
    if len(ds['bucket'].unique())>=2:
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
################## STACKING NIVEL 1 ####################
########################################################

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# Definimos conjuntos de entrenamiento y test:
predictoras=list(set(list(train))-set(["ID","TARGET"]))
X_train=train[predictoras].reset_index(drop=True)
Y_train=train.TARGET.reset_index(drop=True)
X_test=test[predictoras].reset_index(drop=True)

# Función de entrenamiento de cada fold a través de los otros para un modelo dado
# Genera predicciones a test
# Genera predicciones (concatenadas y libres de overfitting) a train
def Model_cv(MODEL, k, X_train, X_test, y, RE):
	# Creamos los k folds
	kf=StratifiedKFold(n_splits=k, shuffle=True, random_state=RE)

	# Creamos los conjuntos train y test de primer nivel
	Nivel_1_train = pd.DataFrame(np.zeros((X_train.shape[0],1)), columns=['train_yhat'])
	Nivel_1_test = pd.DataFrame()

	# Bucle principal para cada fold. Iniciamos contador
	count=0
	for train_index, test_index in kf.split(X_train, Y_train):
		count+=1
		# Definimos train y test en función del fold que estamos
		fold_train= X_train.loc[train_index.tolist(), :]
		fold_test=X_train.loc[test_index.tolist(), :]
		fold_ytrain=y[train_index.tolist()]
		fold_ytest=y[test_index.tolist()]

		# Ajustamos modelo con los k-1 folds
		model_fit=MODEL.fit(fold_train, fold_ytrain)

		# Predecimos sobre el fold libre para calcular el error del CV y muy importante:
		# Para hacer una prediccion a train libre de overfitting para el siguiente nivel
		p_fold=model_fit.predict_proba(fold_test)[:,1]

		# Sacamos el score de la prediccion en el fold libre
		score=roc_auc_score(fold_ytest,p_fold)
		print(k, "- cv, Fold", count, "AUC:", score)
		# Gardamos a Nivel_1_train  las predicciones "libres" concatenadas
		Nivel_1_train.loc[test_index.tolist(),'train_yhat'] = p_fold

		# Tenemos que predecir al conjunto test para hacer la media de los k modelos
		# Definimos nombre de la predicción (p_"número de iteración")
		name = 'p_' + str(count)
		# Predicción al test real
		real_pred = model_fit.predict_proba(X_test)[:,1]
		# Ponemos nombre
		real_pred = pd.DataFrame({name:real_pred}, columns=[name])
		# Añadimos a Nivel_1_test
		Nivel_1_test=pd.concat((Nivel_1_test,real_pred),axis=1)

	# Caluclamos la métrica de la predicción total concatenada (y libre de overfitting) a train
	print("")
	print(k, "- cv, TOTAL AUC:", roc_auc_score(y,Nivel_1_train['train_yhat']))

	# Hacemos la media de las k predicciones de test
	Nivel_1_test['model']=Nivel_1_test.mean(axis=1)

    # Devolvemos los conjuntos de train y test con la predicción
	return Nivel_1_train, pd.DataFrame({'test_yhat':Nivel_1_test['model']},columns=['test_yhat'])


# Entrenamos los diferentes modelos
# Se supone que ya conocemos los parámetros óptimos (en caso que no, se tienen que testear Cross-Validados)
###################################################################################
# Lasso
from sklearn.linear_model import LogisticRegression
print("\nCalculando Lasso...")
Lasso_train, Lasso_test = Model_cv(LogisticRegression(C=0.9, penalty='l1',solver='liblinear'),
                                                     8,X_train,X_test,Y_train,12345)
# Ridge
print("\nCalculando Ridge...")
Ridge_train, Ridge_test = Model_cv(LogisticRegression(C=0.2, penalty='l2',solver='liblinear'),
                                                     8,X_train,X_test,Y_train,54321)

# Xgboost
from xgboost import XGBClassifier
print("\nCalculando XGBoost...")
XGB_train, XGB_test = Model_cv(XGBClassifier(objective='binary:logistic',
                                            n_estimators=10000,
                                            learning_rate=0.005,
                                            max_depth=5,
                                            min_child_weight=0,
                                            gamma=5,
                                            alpha=0.1,
                                            colsample_bytree=0.7,
                                            subsample=0.7,
                                            early_stopping_rounds=200,
                                            silent=True),
											5,X_train,X_test,Y_train,41235)

# Neural Network
from sklearn.neural_network import MLPClassifier
print("\nCalculando Neural Network...")
NN_train, NN_test = Model_cv(MLPClassifier(hidden_layer_sizes=(2),
                                          alpha=1e-5,
                                          solver='lbfgs'),
                                          8,X_train,X_test,Y_train,15423)
###################################################################################

# Nuevo train con las predicciones (concatenadas de los modelos cross-validados):
X1_train=pd.DataFrame({
					   "Lasso":Lasso_train['train_yhat'],
					   "Ridge":Ridge_train['train_yhat'],
					   "XGB":XGB_train['train_yhat'],
                       "NN":NN_train['train_yhat']
					  },columns=['Lasso','Ridge','XGB','NN'])

# Nuevo test con las predicciones de cada modelo (para cada modelo, es la media de los k submodelos surgidos de los k folds que hayamos establecido):
X1_test=pd.DataFrame({
					   "Lasso":Lasso_test['test_yhat'],
					   "Ridge":Ridge_test['test_yhat'],
					   "XGB":XGB_test['test_yhat'],
                       "NN":NN_test['test_yhat']
					  },columns=['Lasso','Ridge','XGB','NN'])

########################################################
################## STACKING NIVEL 2 ####################
########################################################
# Entrenamos un modelo de segundo nivel a través de las predicciones de los modelos
# del primer nivel. Este XGBoost ya tiene unos parámetros (muy distintos al de primer nivel)
# que ya han sido (poco) optimizados:
model_Nivel_2 = XGBClassifier(objective='binary:logistic',
                             n_estimators=500,
                             learning_rate=0.01,
					         max_depth=3,
					         gamma=1,
					         subsample=0.8,
                             early_stopping_rounds=50,
					         silent=True)
print("\nCalculando XGBoost Nivel 2...")
model_fit=model_Nivel_2.fit(X1_train, Y_train)

########################################################
##################### RESULTADOS #######################
########################################################
# Predicción final (submission)
test['Pred']=model_fit.predict_proba(X1_test)[:,1]
outputs_stacking=pd.merge(outputs, test[['ID','Pred']], on='ID', how='left')

# Outputs a .csv
outputs_stacking.to_csv("outputs_stacking.csv", index = False)
##########################################################################################################
##########################################################################################################
########################################### FIN DE LA CITA ###############################################
##########################################################################################################
##########################################################################################################
