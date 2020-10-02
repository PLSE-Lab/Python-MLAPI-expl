##########################################################################################################
#############################################                #############################################
############################################# INICIALIZACIÓN #############################################
#############################################                #############################################
##########################################################################################################

import pandas as pd
import numpy as np

# Lectura de datos
train = pd.read_csv('../input/bc-real-estate/train.csv',encoding='iso-8859-1')
test = pd.read_csv('../input/bc-real-estate/test.csv',encoding='iso-8859-1')

# Definimos dataframe de outputs solo con ID
outputs=pd.DataFrame(test.ID)

# Unimos train y test para tratar variables
test['PRECIO_VENTA']=np.nan
full_data=pd.concat([train, test], ignore_index=True,sort=False)

# Eliminamos los warnings de "A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead"
pd.options.mode.chained_assignment = None

##########################################################################################################
##########################################                     ###########################################
########################################## FEATURE ENGINEERING ###########################################
##########################################                     ###########################################
##########################################################################################################

print('\nInicio del Feature Engineering...\n')

# Cambiamos algunas fechas por su año
################################################################################
def anyo(x):
    if not pd.isnull(x):
        return str(x)[0:4]
    else:
        return 'NS'

full_data['FecAdjudicacionBanco']=full_data['FecAdjudicacionBanco'].apply(anyo)
full_data['FecAltaSistema']=full_data['FecAltaSistema'].apply(anyo)
full_data['FecCaducidadCedula']=full_data['FecCaducidadCedula'].apply(anyo)
full_data['FecSolicitudCedula']=full_data['FecSolicitudCedula'].apply(anyo)
full_data['FecPrescripcionVPO']=full_data['FecPrescripcionVPO'].apply(anyo)

full_data.loc[(full_data.FecCaducidadCedula>='2099') & (full_data.FecCaducidadCedula<'NS'),'FecCaducidadCedula']='NS'
full_data.loc[(full_data.FecSolicitudCedula>'2021') & (full_data.FecSolicitudCedula<'NS'),'FecSolicitudCedula']='NS'

# Año de Construcción
################################################################################
for p in [0,20,40,60,80,100]: print(p,np.nanpercentile(full_data['AnyoConstruccion'], p))

def antig(x):
    if x<=1966:
        return 'MuyAntiguo'
    elif x<=1975:
        return 'Antiguo'
    elif x<=1991:
        return 'Medio'
    elif x<=2004:
        return 'Nuevo'
    elif x<=2015:
        return 'MuyNuevo'
    else:
        return 'NS'

full_data['AntigCasa']=full_data['AnyoConstruccion'].apply(antig)
full_data=full_data.drop('AnyoConstruccion',axis=1)

# Indicadores de demanda
################################################################################
full_data.loc[full_data.DiasPublicados == 'No esta publicado', 'DiasPublicados'] = '10 dias'
full_data['DiasPublicados']=full_data['DiasPublicados'].apply(lambda x: int(x.replace(' dias','')))
full_data['p_Ofertas']=full_data['Ofertas']/full_data['DiasPublicados']
full_data['p_Visitas']=full_data['Visitas']/full_data['DiasPublicados']
full_data['p_Favoritos']=full_data['Favoritos']/full_data['DiasPublicados']

# Categorización de superficie
################################################################################
# Cambiamos un claro error de superfície
full_data.loc[full_data.ID == 'A77470', 'Superficie'] = 276 # Ponemos su superficie entre 100
for p in [0,20,40,60,80,100]: print(p,np.nanpercentile(full_data['Superficie'], p))

def super(x):
    if x<=64:
        return 'MuyPeq'
    elif x<=78:
        return 'Peq'
    elif x<=93:
        return 'Medio'
    elif x<=117:
        return 'Gran'
    elif x<=2000:
        return 'MuyGran'

full_data['TamCasa']=full_data['Superficie'].apply(super)

# Missings habitaciones (ponemos las que tocarían por tamaño). Da igual que no sea un entero
################################################################################
tabla_dormi=full_data[['TamCasa','TotalDormitorios']].groupby(['TamCasa']).mean().reset_index()
print(tabla_dormi)
full_data.loc[(full_data.TotalDormitorios.isnull()) & (full_data.TamCasa=='MuyPeq'),'TotalDormitorios']=2.06
full_data.loc[(full_data.TotalDormitorios.isnull()) & (full_data.TamCasa=='Peq'),'TotalDormitorios']=2.58
full_data.loc[(full_data.TotalDormitorios.isnull()) & (full_data.TamCasa=='Medio'),'TotalDormitorios']=2.85
full_data.loc[(full_data.TotalDormitorios.isnull()) & (full_data.TamCasa=='Gran'),'TotalDormitorios']=3.08
full_data.loc[(full_data.TotalDormitorios.isnull()) & (full_data.TamCasa=='MuyGran'),'TotalDormitorios']=3.48

# Missings baños (ponemos los que tocarían por tamaño). Da igual que no sea un entero
################################################################################
tabla_banyos=full_data[['TamCasa','TotalBanyos']].groupby(['TamCasa']).mean().reset_index()
print(tabla_banyos)
full_data.loc[(full_data.TotalBanyos.isnull()) & (full_data.TamCasa=='MuyPeq'),'TotalBanyos']=1.06
full_data.loc[(full_data.TotalBanyos.isnull()) & (full_data.TamCasa=='Peq'),'TotalBanyos']=1.20
full_data.loc[(full_data.TotalBanyos.isnull()) & (full_data.TamCasa=='Medio'),'TotalBanyos']=1.42
full_data.loc[(full_data.TotalBanyos.isnull()) & (full_data.TamCasa=='Gran'),'TotalBanyos']=1.68
full_data.loc[(full_data.TotalBanyos.isnull()) & (full_data.TamCasa=='MuyGran'),'TotalBanyos']=2.22

# Homogenización categóricas
################################################################################
lista_SI_NO=['Calefaccion','AireAcondicionado','Parquet','Rejas','PreInstAC','Chimenea','Barbacoa',
'SistemaAlarma','Piscina','VideoPortero','Electrodomesticos']
for col in lista_SI_NO:
    full_data.loc[full_data[col]=="SI",col]='S'
    full_data.loc[full_data[col]=="NO",col]='N'

# Relleno missings categóricas
################################################################################
lista_mis=['MotivoAdjudicacion','CambioBombin','Reformar','CondicionHabitabilidad','TipoCocina',
'CalificacionEnergtica','Residencia','NotaSimple','VPO','VPOSusceptible','Poblacion',
'Provincia','Area','Comunidad','Delegacion']+lista_SI_NO
for col in lista_mis:
    full_data.loc[full_data[col].isnull(),col]='NS'

# Contador de características
################################################################################
lista_contador=['GarajeIncluido','TrasteroIncluido','Calefaccion','AireAcondicionado','Parquet','Rejas',
'PreInstAC','Chimenea','Barbacoa','SistemaAlarma','Piscina','VideoPortero','Electrodomesticos']
tabla_cont=full_data[lista_contador]
for col in lista_contador:
    tabla_cont.loc[tabla_cont[col]=="S",col+'_num']=1
    tabla_cont.loc[tabla_cont[col]=="N",col+"_num"]=-1
    tabla_cont.loc[tabla_cont[col]=="NS",col+"_num"]=0

tabla_cont=tabla_cont[tabla_cont.columns[-int(tabla_cont.shape[1]/2):]]
full_data['Vec_caract']=tabla_cont.sum(axis=1)

# Información de precio regional al mínimo nivel disponible
################################################################################
# 1) Arreglamos Código Postal. Algunos vienen con "0" y algunos son missing
# Assignamos "0" también a los missing y más adelante tiraremos los "0"
full_data.loc[full_data.CodigoPostal.isnull(),'CodigoPostal']=0
full_data['CodigoPostal']=full_data['CodigoPostal'].apply(int)

# 2) Precio/m^2 de cada piso vendido (luego tiraremos esta variable ya que en test no la tendremos)
full_data['Precio_m2']=full_data['PRECIO_VENTA']/full_data['Superficie']

# 3) Función de agregación de precio/m^2 por segmentación geográfica. Exigimos un mínimo de "min_obs"
# para "creernos" la media de precios m^2
def p_segm(ds,col,min_obs):
    tabla_P=ds[[col,'Precio_m2']].groupby([col]).mean().reset_index()
    tabla_P=tabla_P.rename(columns={'Precio_m2': 'P_'+col})
    tabla_N=ds[[col,'Precio_m2']].groupby([col]).count().reset_index()
    tabla_N=tabla_N.rename(columns={'Precio_m2': 'N'})
    tabla_T=pd.merge(tabla_P, tabla_N, how='left', on=[col])
    tabla_T=tabla_T.loc[tabla_T.N>=min_obs,]
    return tabla_T

# 4) Aplicamos a cada segmentación geográfica. Ponemos min_obs=5 (éste parámetro se debería optimizar)
CP=p_segm(full_data,'CodigoPostal',5)
Pob=p_segm(full_data,'Poblacion',5)
Prov=p_segm(full_data,'Provincia',5)
Area=p_segm(full_data,'Area',5)
CA=p_segm(full_data,'Comunidad',5)
Dele=p_segm(full_data,'Delegacion',5)

# Quitamos el CP "0" de la tabla CP ya que correspondía a los 0 (errores y missings)
CP = CP[CP.CodigoPostal != 0]

# 5) Agregamos a full_data
precio_geogr=pd.merge(full_data[['ID','CodigoPostal','Poblacion','Provincia','Area','Comunidad','Delegacion']],
CP[['CodigoPostal','P_CodigoPostal']], how='left', on=['CodigoPostal'])
precio_geogr=pd.merge(precio_geogr,Pob[['Poblacion','P_Poblacion']],how='left', on=['Poblacion'])
precio_geogr=pd.merge(precio_geogr,Prov[['Provincia','P_Provincia']],how='left', on=['Provincia'])
precio_geogr=pd.merge(precio_geogr,Area[['Area','P_Area']],how='left', on=['Area'])
precio_geogr=pd.merge(precio_geogr,CA[['Comunidad','P_Comunidad']],how='left', on=['Comunidad'])
precio_geogr=pd.merge(precio_geogr,Dele[['Delegacion','P_Delegacion']],how='left', on=['Delegacion'])

# 6) Conjunto CP y Población. A veces un CP puede estar en dos poblaciones y, evidentemente,
# una población puede contener varios CP. Por eso nos quedamos con la media de manera que si
# uno de los dos es Null, nos quedamos con el otro (si los dos están informados, con la media) y
# solo no tendremos la info a este mínimo nivel geográfico si faltan los dos
precio_geogr['P_CP_Pob']=precio_geogr[['P_CodigoPostal','P_Poblacion']].mean(axis=1)

# 7) Nos quedamos con la info al nivel más "local" que tengamos
precio_geogr['Precio_m2_zona']=np.where(~(np.isnan(precio_geogr['P_CP_Pob'])), precio_geogr['P_CP_Pob'],
np.where(~(np.isnan(precio_geogr['P_Provincia'])), precio_geogr['P_Provincia'],
np.where(~(np.isnan(precio_geogr['P_Area'])), precio_geogr['P_Area'],
np.where(~(np.isnan(precio_geogr['P_Comunidad'])), precio_geogr['P_Comunidad'],
np.where(~(np.isnan(precio_geogr['P_Delegacion'])), precio_geogr['P_Delegacion'],0)))))

# 8) Finalmente, para cada vivienda añadimos el precio/m^2 de su zona
# y su "precio de mercado" multiplicando por su superfiície
full_data=pd.merge(full_data,precio_geogr[['ID','Precio_m2_zona']],how='left',on=['ID'])
full_data['Precio_zona']=full_data.Precio_m2_zona*full_data.Superficie

# Tiramos variable precio/m^2
full_data=full_data.drop('Precio_m2',axis=1)

# Tasación actualizada. El objetivo es llevar todas las tasaciones al mismo punto temporal
# para que sean comparables entre ellas
################################################################################
# 1) Llevamos la fecha de tasación a su trimestre. Llevamos 2 tasaciones de 2002 y 2003
# a 2004T1 pq la base de datos de precios de vivienda que utilizaremos empieza en 2004
def trim(x):
    if str(x)[0:4] in ('2002','2003'): # Hay 2 tasaciones anteriores a 2004
        return '2004T1'
    elif str(x)[5:7] in ('01','02','03'):
        return str(x)[0:4]+'T1'
    elif str(x)[5:7] in ('04','05','06'):
        return str(x)[0:4]+'T2'
    elif str(x)[5:7] in ('07','08','09'):
        return str(x)[0:4]+'T3'
    elif str(x)[5:7] in ('10','11','12'):
        return str(x)[0:4]+'T4'
    else:
        return 'NS'

full_data['TrimTasacion']=full_data['FecTasacion'].apply(trim)

# Ya podemos cambiar la fecha tasación a año como hemos hecho con las otras fechas
full_data['FecTasacion']=full_data['FecTasacion'].apply(anyo)

# 2) Cargamos BBDD con la evolución de precios de vivienda del Ministerio de Fomento.
# Contiene datos desde 2004T1. Fuente "https://apps.fomento.gob.es/BoletinOnline2/?nivel=2&orden=34000000",
# apartado "3.1 Valor de las transacciones inmobiliarias de vivienda libre".
# El fichero que cargamos es una modificación del original con los datos bien puestos, con los nombres adaptados
# a nuestra BBDD y de manera que para cada trimestre y provincia hay el multiplicador de precio respecto 20014T1
EPV = pd.read_csv('../input/precios-vivienda-ministerio-fomento/EvolucionPrecioVivienda.csv',encoding='iso-8859-1',sep=';')
EPV = EPV.rename(columns={'Territorio': 'Provincia'})

# 3) Rellenamos missings de Provincia por 'NS'
full_data.Provincia = full_data.Provincia.fillna('NS')

# 4) Obtención del multiplicador en la fecha de tasación (origen)
# Dataframes auxiliares para traer multiplicador tasación en fecha tasación
s1=pd.merge(full_data[['Provincia','TrimTasacion']], EPV, how='left', on=['Provincia'])
s2=pd.DataFrame()

# Obtenemos multiplicador en origen
for v in s1.columns.drop(['Provincia', 'TrimTasacion']):
    s2['Tasacion_' + v] = (s1['TrimTasacion'] == v).astype(int) * s1[v]

full_data['Mult_tas_ori'] = s2.sum(axis=1)

# 5) Multiplicador a "presente" (Todas las tasaciones son como mucho a 2015T2).
# Las ventas son todas en 2014 / 2015 -> Traemos los precios a 2015T2 para igualar
full_data = pd.merge(full_data,EPV[['Provincia','2015T2']],how='left',on=['Provincia'])
full_data = full_data.rename(columns={'2015T2': 'Mult_tas_fin'})
full_data['ValorTasacionAct']=full_data.ValorTasacion/full_data.Mult_tas_ori*full_data.Mult_tas_fin

# 6) Había valores de tasación missing y 0. Lo arreglamos poniendo el precio que le correspondería por su zona.
# Es un arreglo cutre corrigiendo por un multiplicador en función del tamaño ya que las tasaciones están
# en general por encima del precio de mercado (estamos hablando de una época en que las tasaciones se hinchaban
# bastante para meter en la hipoteca las reformas de la casa y el mercedes).
tabla_Tam1=full_data[['TamCasa','ValorTasacionAct']].groupby(['TamCasa']).mean().reset_index()
tabla_Tam2=full_data[['TamCasa','Precio_zona']].groupby(['TamCasa']).mean().reset_index()
tabla_Tam=pd.merge(tabla_Tam1,tabla_Tam2,how='left',on=['TamCasa'])
tabla_Tam['mult']=tabla_Tam.ValorTasacionAct/tabla_Tam.Precio_zona

full_data=pd.merge(full_data,tabla_Tam[['TamCasa',"mult"]],how='left',on=['TamCasa'])
full_data['Tas_implicita']=full_data.mult*full_data.Precio_zona
full_data['ValorTasacionAct']=np.where((np.isnan(full_data['ValorTasacionAct'])) | (full_data['ValorTasacionAct']==0),
full_data['Tas_implicita'], full_data['ValorTasacionAct'])

# 7) Tiramos columnas auxiliares y tasación original
full_data=full_data.drop('TrimTasacion',axis=1)
full_data=full_data.drop('Mult_tas_ori',axis=1)
full_data=full_data.drop('Mult_tas_fin',axis=1)
full_data=full_data.drop('ValorTasacion',axis=1)
full_data=full_data.drop('Tas_implicita',axis=1)
full_data=full_data.drop('mult',axis=1)

# Rehabilitación
################################################################################
# Porcentaje de coste en función de la tasación
full_data['p_CosteRehabilitacion']=full_data.CosteRehabilitacion/full_data.ValorTasacionAct
# Indicador de necesidad de rehabilitación
full_data['IndRehabilitacion']='N'
full_data.loc[full_data.CosteRehabilitacion>0,'IndRehabilitacion']='S'

# Número de plantas
################################################################################
# En los casos missing ponemos una planta
full_data.loc[full_data.NumPlantas.isnull(),'NumPlantas']=1
# Pasamos dato a string pq no sabemos si el número de plantas correlaciona con el precio directamente
full_data['NumPlantas']=full_data['NumPlantas'].astype(str)

# Logaritmos y drops
################################################################################
# Logaritmo de precios y superfície
full_data['ValorTasacionAct']=np.log(full_data['ValorTasacionAct'])
full_data['ValorContable']=np.log1p(full_data['ValorContable'])
full_data['Superficie']=np.log(full_data['Superficie'])
full_data['Precio_zona']=np.log(full_data['Precio_zona'])
# y logaritmo de la variable target
full_data['PRECIO_VENTA']=np.log(full_data['PRECIO_VENTA'])

# Drop de variables que no queremos utilizar
full_data=full_data.drop('Longitud',axis=1)
full_data=full_data.drop('Latitud',axis=1)
full_data=full_data.drop('FecCaducidadCedula',axis=1)
full_data=full_data.drop('FecSolicitudCedula',axis=1)
full_data=full_data.drop('FecPrescripcionVPO',axis=1)

# Detectamos aquellas columnas con más de 50 valores distintos
for col in list(full_data):
    if (len(set(full_data[col]))>50):
        print(col, len(set(full_data[col])))

# Todas son "contínuas" excepto Código Postal y Población.
# Como ya tenemos el precio por m^2 al menor nivel de agregación las tiramos
full_data=full_data.drop('CodigoPostal',axis=1)
full_data=full_data.drop('Poblacion',axis=1)

# Rename de la columna TARGET:
################################################################################
full_data = full_data.rename(columns={'PRECIO_VENTA': 'TARGET'})

# Nos aseguramos que no hay missings
print(full_data.drop('TARGET',axis=1).isnull().values.any())
################################################################################

# Separamos train y test
################################################################################
train=full_data[~(np.isnan(full_data.TARGET))]
test=full_data[np.isnan(full_data.TARGET)]
test = test.drop('TARGET',axis=1)

##########################################################################################################
################################################         #################################################
################################################ MODELOS #################################################
################################################         #################################################
##########################################################################################################

################################################################################
# Toda la información aquí: "https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/"
################################################################################
import catboost as cat

# 1) Definimos conjuntos de entrenamiento y test:
################################################################################
predictoras=list(set(list(train))-set(['ID','TARGET']))
X_train=train[predictoras].reset_index(drop=True)
Y_train=train.TARGET.reset_index(drop=True)
X_test=test[predictoras].reset_index(drop=True)

 # 2) Preprocesmos datos para CatBoost (Pasamos datos a Pool que es el formato que requiere CatBoost)
################################################################################
# Lista de categóricas
lista_str=['FecAdjudicacionBanco','MotivoAdjudicacion','FecAltaSistema','TipoVivienda','FecTasacion',
'GarajeIncluido','TrasteroIncluido','CambioBombin','Reformar','CondicionHabitabilidad',
'TipoCocina','CalificacionEnergtica','Calefaccion','AireAcondicionado','Parquet','Rejas','NumPlantas',
'PreInstAC','Chimenea','Barbacoa','SistemaAlarma','Piscina','VideoPortero','Electrodomesticos',
'Uso','Residencia','NotaSimple','VPO','VPOSusceptible','TipoFoto','AntigCasa','TamCasa','IndRehabilitacion',
'Provincia','Area','Comunidad','Delegacion']

# Calculamos índice (posición) de las categóricas para pasar a CatBoost
Pos=list()
for col in lista_str:
    Pos.append((X_train.columns.get_loc(col)))

# Pasamos a clase Pool
pool=cat.Pool(X_train, Y_train, cat_features=Pos)


# 3) Modelo CV de CatBoost. Aquí deberemos tunear los parámetros "a mano". Hacer un grid-search
# es demasiado costoso computacionalmente. Vamos haciendo pruebas con valores razonables hasta
# encontrar unos que funcionen bien
################################################################################
# Parámetros básicos
RE=12345 # Seed que utilizaremos para partición de folds y la parte random del modelo
esr=50 # Early Stopping Rounds (si en "esr" árboles no mejora test -> se detiene).
# El objetivo es ajustar el par iterations / leraning_rate de manera que antes de acabar se detenga por "esr"
# Si llegamos al número final de rondas -> subir número de árboles o bajar learning_rate
n_folds=5 # Folds para CV

# Parámetros del booster. Ver todoas las opciones en:
# Python: "https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/"
# R: "https://tech.yandex.com/catboost/doc/dg/concepts/r-training-parameters-docpage/"
params = {'loss_function': 'RMSE',
          'learning_rate': 0.01, # En sintonía al número de árboles (iterations). Si no acaba por "esr", bajarlo
          'depth': 6, # Profundidad de los árboles (poner valores entre 5 y 10, más alto -> más overfitting)
          'l2_leaf_reg': 10, # Regularización L2 (poner entre 3 y 20, más alto -> menos overfitting).
          'rsm': 0.5, # % de features para hacer cada split (más bajo: acelera la ejecución y reduce overfitting)
          'random_seed': RE}

# Cross-Validation. Puede tardar horas... (cuando ponemos muchas iteraciones y learning rate bajo)
print('\nCatBoost CV...\n')
cv_catboost = cat.cv(pool, params,
                iterations=50000, # Número de rondas máximo (árboles). Poner valor alto y que pare por "esr"
                nfold=n_folds,
                early_stopping_rounds=esr,
                verbose=50, # Nos muestra la métrica train/test cada tantos árboles
                partition_random_seed=RE)

# Pintamos comportamiento train/test
import matplotlib.pyplot as plt
plt.plot(cv_catboost.loc[esr:,['test-RMSE-mean','train-RMSE-mean']])

# Obtenemos número óptimo de rondas
best_nrounds=int((cv_catboost.shape[0]-esr)/(1-1/n_folds))

# 4) Modelo sobre todo el train con los parámetros óptimos y número de rondas óptimas del CV
################################################################################
model_catboost = cat.CatBoostRegressor(
          iterations=best_nrounds,
          learning_rate=0.01,
          depth=6,
          l2_leaf_reg=10,
          rsm=0.5,
          random_seed=RE,
          verbose=200)

print('\nCatBoost Fit...\n')
model_catboost.fit(X=pool)

# 5) Feature Importance
################################################################################
# Sacamos importancia de los factores ya calculada automáticamente por la estimación del modelo
# según "https://tech.yandex.com/catboost/doc/dg/concepts/fstr-docpage/#fstr__regular-feature-importance"
FI=model_catboost.feature_importances_
FI=pd.DataFrame({'Feature':list(X_train),'Importance':FI}).sort_values(by=['Importance'],ascending=False)
print(FI)

# Sacamos importancia de los factores con la metodología Shap:
# "https://canopylabs.com/resources/interpreting-complex-models-with-shap/"
# CatBoost ya lo tiene integrado
ShapImportance=model_catboost.get_feature_importance(data=pool,fstr_type='ShapValues',prettified=True,verbose=500)

# Para visualizar bien los datos necesitamos del módulo shap
# (Se instala como "pip install Shap", con "S" mayúscula y se llama con "s")
from shap import summary_plot
summary_plot(ShapImportance[:,:-1], X_train)

##########################################################################################################
###############################################            ###############################################
############################################### RESULTADOS ###############################################
###############################################            ###############################################
##########################################################################################################

# Predicción final (submission). Hacemos exp pq la submission es con el precio
################################################################################
test['Pred']=np.exp(model_catboost.predict(X_test))
outputs_catboost=pd.merge(outputs, test[['ID','Pred']], on='ID', how='left')

# Outputs a .csv
################################################################################
outputs_catboost.to_csv('outputs_catboost.csv',index=False)
print('\nEND')

# Códigos por si queremos salvar/recuperar un modelo largo de ejecutar
################################################################################
# Ejemplo de como salvar un modelo:
# from sklearn.externals import joblib
# joblib.dump(model_catboost,'BBDD Output/model_catboost.sav')

# Como cargarlo y hacer predicciones
# loaded_model=joblib.load('BBDD Output/model_catboost.sav')
# loaded_model.predict(X1_test)
################################################################################

##########################################################################################################
#############################################                #############################################
############################################# FIN DE LA CITA #############################################
#############################################                #############################################
##########################################################################################################