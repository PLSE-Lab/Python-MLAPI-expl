##########################################################################################################
############################################            ##################################################
############################################ INITIALIZE ##################################################
############################################            ##################################################
##########################################################################################################
import numpy as np
import pandas as pd
import os

# Leemos datos
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

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
		lab=['Missing']
		for i in range(1,n_buckets+1):
			lab.append('G'+str(i))
		ax.set_xticklabels(lab)
	else:
		lab=[]
		for i in range(1,n_buckets+1):
			lab.append('G'+str(i))
		ax.set_xticklabels(lab)

	plt.title(icol)
	plt.show()

for icol in list(train.iloc[:,1:-1]):
	feat_graph(train,icol,'TARGET',10)

# Imputación de missings:
# Vamos variable a variable que tenga suficientes missings y le assignamos un valor
# que esté por debajo del mínimo y creamos una dummy indicando que esas
# observaciones tienen missing en la variable

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

Imputa=['X21', 'X24', 'X27', 'X28', 'X32', 'X37', 'X41', 'X45', 'X47', 'X52', 'X53', 'X54', 'X64']

miss_dummy=pd.DataFrame()

# Creamos dummies e imputamos mínimo-1
for col in Imputa:
	# Creamos columna para dummy con el valor original:
	miss_dummy[col+'_m']=full_data[col]
	miss_dummy.loc[~miss_dummy[col+'_m'].isna(),col+'_m']=0 # No missing -> 0
	miss_dummy.loc[miss_dummy[col+'_m'].isna(),col+'_m']=1 # Missing -> 1
	# Rellenamos en full_data con min-1
	full_data.loc[full_data[col].isna(),col]=full_data[col].min()-1

# El resto de missings los imputamos por K-Nearest Neighbours
# Comprobamos que quedan pocos:
miss(full_data,1)

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

severidad=severidad.sort_values(by="TMR (%)",ascending=False).reset_index(drop=True)

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

# Añadimos dummies de missings que habían quedado pendientes
full_data=pd.concat([full_data,miss_dummy],axis=1)

# Reordenamos columnas (TARGET al final)
cols=list(full_data)
cols.insert(len(cols), cols.pop(cols.index('TARGET')))
full_data = full_data.reindex(columns= cols)


# 4) WoEizado con suavizado de Tasche (para modelos lineales)
################################################################################
# Metodología de D.Tasche "The art of Probability of default curve calibration"
# que aplicaremos aquí, no para calibrar, sino para "suavizar" la curva de TMOs
# antes de calcular el WoE para evitar overfitting

# Los códigos vienen de "https://github.com/Densur/LDPD" pero contienen algunos errores,
# sobretodo con "portfolio_cnd_no_dft: conditional on no default portfolio distribution",
# que se han corregido en aquí.

from statsmodels import distributions
from scipy.optimize import fsolve
from scipy import stats
from operator import sub
import math

# Definimos clase Portfolio para calcular Accuracy Ratio implícito y Tendencia Central
class LDPortfolio:
	'''
	Basic functionality for all LDP calibration facilities (LDP=Low Default Portfolio).
	:attribute self.ar: Estimated Accuracy ratio given portfolio distribution and PD values.
	:attribute self.ct: Central Tendency (mean PD) ratio given portfolio distribution and PD values.
	'''
	def __init__(self, portfolio, pd_cnd=None):
		'''
		:param portfolio: Unconditional portfolio distribution from the worst to the best credit quality.
						Each 'portfolio' item contains total number of observations in a given rating class.
		:param pd_cnd: Current conditional PD distribution from the worst to the best credit quality.
						Used for current AR estimation.
		'''
		self.portfolio = np.array(portfolio)
		self.pd_cnd = np.array(pd_cnd)
		self.portfolio_size = self.portfolio.sum()
		# self.portfolio_dist = self.portfolio.cumsum() / self.portfolio_size
		# self.portfolio_dist = (np.hstack((0, self.portfolio_dist[:-1])) + self.portfolio_dist) / 2
		self.rating_prob = self.portfolio / self.portfolio_size

		self.ct = None
		self.ar = None
		if not pd_cnd is None:
			self.ct, self.ar = self._ar_estimate(self.pd_cnd)

	def _ar_estimate(self, pd_cnd):
		ct = self.rating_prob.T.dot(pd_cnd)
		ar_1int_1 = self.rating_prob * pd_cnd
		ar_1int_1 = np.hstack((0, ar_1int_1[:-1]))
		ar_1int_1 =  (1 - pd_cnd) * self.rating_prob * ar_1int_1.cumsum()
		ar_1 = 2 * ar_1int_1.sum()
		ar_2 =  (1 - pd_cnd) * pd_cnd * self.rating_prob * self.rating_prob
		ar = (ar_1 + ar_2.sum()) * (1.0 / (ct * (1 - ct))) - 1
		return ct, ar.sum()

# Clase que implementa el Quasi-Moment Matching
class QMM(LDPortfolio):
	"""
	Calibrates conditional probabilities of default according to Quasi Moment Matching algorithm
	:attribute self.pd_cnd: calibrated conditional PD
	:attribute self.alpha: intercept calibration parameter
	:attribute self.beta: slope calibration parameter
	"""
	def __init__(self, portfolio, portfolio_cnd_no_dft = None):
		"""
		:param portfolio: Unconditional portfolio distribution from the worst to the best credit quality.
						Each 'portfolio' item contains total number of observations in a given rating class.
		:param portfolio_cnd_no_dft: Conditional on no default portfolio distribution (in case None, unconditional
						portfolio distribution is used as a proxy)
		:return: initialized QMM class object
		"""
		super().__init__(portfolio, pd_cnd=None)
		if portfolio_cnd_no_dft is None:
			self.portfolio_cnd_no_dft = self.portfolio
			self.cum_cond_ND = (np.hstack((0, self.portfolio_cnd_no_dft.cumsum()[:-1])) + self.portfolio_cnd_no_dft.cumsum()) / 2
		else:
			self.cum_cond_ND = (np.hstack((0, portfolio_cnd_no_dft.cumsum()[:-1])) + portfolio_cnd_no_dft.cumsum()) / 2

		self.alpha = None
		self.beta = None

	def fit(self, ct_target, ar_target):
		"""
		:param ct_target: target Central Tendency
		:param ar_target: target Accuracy Ratio
		:return: calibrated QMM class
		"""
		a = self.__get_pd((0, 0))
		tf = lambda x: tuple(map(sub, self._ar_estimate(self.__get_pd(x)), (ct_target, ar_target)))
		params = fsolve(tf, (0, 0))
		self.alpha, self.beta = params
		self.pd_cnd = self.__get_pd(params)
		self.ct, self.ar = self._ar_estimate(self.pd_cnd)
		return self

	def __get_pd(self, params):
		return self._robust_logit(self.cum_cond_ND, params)

	@staticmethod
	def _robust_logit(x, params):
		alpha, beta = params
		return 1 / (1 + np.exp(- alpha - beta * stats.norm.ppf(x)))


# Función de WoE con suavizado de Tasche
def WoE_Tasche(icol,binary_col,ori,df,n_buckets):
	# Bucketizamos sobre todo el dataset
	df['bucket'], bins = pd.qcut(df[icol],q=n_buckets,labels=False,duplicates='drop',retbins=True)
	real_bins=len(bins)-1

	# Girado de etiquetas si la orientación en TMO es ascendente
	if ori=='asc':
		df['bucket']=(real_bins-1-df['bucket'])

	# Filtramos todos los datos a la parte train (donde 'binary_col' está definido)
	df_tr=df.loc[~df[binary_col].isna(),].reset_index(drop=True)

	# Creamos tabla de Ratings (bucketización) para aplicar Tasche
	rating_data_N=pd.DataFrame(df_tr[['bucket',binary_col]].groupby(['bucket'])[binary_col].count()).reset_index() # Totales
	rating_data_D=pd.DataFrame(df_tr[['bucket',binary_col]].groupby(['bucket'])[binary_col].sum()).reset_index() # Defaults
	rating_data=pd.merge(rating_data_N,rating_data_D,how='left',on='bucket')
	rating_data.columns=['bucket','N','D']
	rating_data['ND']=rating_data['N']-rating_data['D'] # Buenos
	rating_data['COND_PD']=rating_data['D']/rating_data['N'] # TMO observada (Conditional PD)
	total_ND=rating_data['ND'].sum() # Buenos totales
	rating_data['COND_ND']=rating_data['ND']/total_ND # Distribución Buenos

	# Aplicamos modelo Tasche
	p1=LDPortfolio(portfolio=rating_data['N'],pd_cnd=rating_data['COND_PD'])
	q1=QMM(portfolio=rating_data['N'], portfolio_cnd_no_dft=rating_data['COND_ND'])
	q1.fit(ct_target=p1.ct, ar_target=p1.ar)
	rating_data['CALIB']=q1.pd_cnd # Nueva curva suavizada

	# Malos y Buenos según curva suavizada
	rating_data['Model_D']=rating_data['CALIB']*rating_data['N']
	rating_data['Model_ND']=(1-rating_data['CALIB'])*rating_data['N']

	# Malos y buenos totales según curva suaviada.
	# Como hemos calibrado con la tendencia central original, tendrían que ser
	# los mismos que originalmente
	Model_BAD=rating_data['Model_D'].sum()
	Model_GOOD=rating_data['Model_ND'].sum()

	# Cálculo del WoE "suavizado"
	rating_data['WOE']=np.log((rating_data['Model_ND']/Model_GOOD)/(rating_data['Model_D']/Model_BAD))

	# Creamos etiquetas para pintar
	lab=[]
	for i in range(1,real_bins+1):
		lab.append('G'+str(i))

	rating_data['labels']=lab

	if ori=='asc':
		rating_data.sort_values(by=['bucket'],ascending=False,inplace=True)
		rating_data=rating_data.reset_index(drop=True)
		rating_data['labels']=lab

	# Dibujo
	# Da error en esta versión de matplotlib, comento código:
	
	# plt.plot(rating_data['labels'],rating_data['COND_PD'],color='red')
	# plt.plot(rating_data['labels'],rating_data['CALIB'],color='darkblue',marker='o')
	# plt.title(icol)
	# plt.show()

	# Unimos nuevo factor woeizado a la tabla y eliminamos factor original
	df=pd.merge(df, rating_data[['bucket','WOE']], on='bucket', how='left')
	df=df.rename(columns={'WOE': icol+"_W"})
	df=df.drop(icol, axis=1)
	df=df.drop('bucket', axis=1)

	return df

# Función de WoE clásica
def WoE_num(icol,binary_col,df,n_buckets):
	# Bucketizamos sobre todo el dataset
	df['bucket'], bins = pd.qcut(df[icol],q=n_buckets,labels=False,duplicates='drop',retbins=True)
	real_bins=len(bins)-1

	# Creamos tabla WoE
	tabla_woe=df[['bucket',binary_col]].groupby(['bucket']).sum(skipna=True).reset_index()

	# Si algun bucket no tiene morosos, bucketizamos con uno menos (se supone que arreglamos el problema):
	# Podría ser que no... se tendría que hacer con un while
	if 0 in tabla_woe[binary_col].values:
		df['bucket'], bins = pd.qcut(df[icol], q=real_bins-1,labels=False,duplicates='drop',retbins=True)
		real_bins=len(bins)-1
		# Creamos tabla WoE
		tabla_woe=df[['bucket',binary_col]].groupby(['bucket']).sum(skipna=True).reset_index()

	# Buenos y malos totales
	BAD=df[binary_col].sum(skipna=True)
	GOOD=df.loc[~df[binary_col].isna(),binary_col].count()-BAD

	# Nos aseguramos que al tirar los cortes repetidos con "duplicates='drop'" (esto pasa cuando
	# una variable tiene acumulación de valores repetidos) almenos queden 5 buckets restantes.
	# Si no, no woeizamos la variable.
	if real_bins>=5:
		tabla_woe = tabla_woe.rename(columns={binary_col: 'BAD'}) # Defaults
		tabla_woe['TOTAL']=df[['bucket',binary_col]].groupby(['bucket']).count().reset_index()[binary_col] # Totales
		tabla_woe['GOOD']=(tabla_woe['TOTAL']-tabla_woe['BAD']).astype(int) # Buenos

		# Cálculo WOE por bucket
		tabla_woe['WOE']=np.log((tabla_woe['GOOD']/GOOD)/(tabla_woe['BAD']/BAD))

		# Unimos nuevo factor woeizado a la tabla y eliminamos factor original
		df=pd.merge(df, tabla_woe[['bucket','WOE']], on='bucket', how='left')
		df = df.rename(columns={'WOE': icol+"_W"})
		df = df.drop(icol, axis=1)
		df = df.drop('bucket', axis=1)
	else:
		df = df.drop(icol, axis=1)
		df = df.drop('bucket', axis=1)

	return(df)


# Miramos otra vez las variables que quedan:
for icol in list(full_data.loc[~full_data['TARGET'].isna(),].iloc[:,1:-1]):
	feat_graph(full_data.loc[~full_data['TARGET'].isna(),],icol,'TARGET',20)

# Después de mirar los dibujos, hacemos:
full_data_woe=full_data.copy()

# Descendentes
full_data_woe=WoE_Tasche('X1','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X3','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X4','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X5','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X6','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X7','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X8','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X10','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X11','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X12','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X13','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X16','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X19','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X21','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X24','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X25','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X28','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X29','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X31','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X33','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X37','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X38','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X39','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X45','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X49','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X53','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X54','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X55','TARGET','desc',full_data_woe,20)
full_data_woe=WoE_Tasche('X56','TARGET','desc',full_data_woe,20)

# Ascendentes
full_data_woe=WoE_Tasche('X2','TARGET','asc',full_data_woe,20)
full_data_woe=WoE_Tasche('X30','TARGET','asc',full_data_woe,20)
full_data_woe=WoE_Tasche('X32','TARGET','asc',full_data_woe,20)
full_data_woe=WoE_Tasche('X51','TARGET','asc',full_data_woe,20)
full_data_woe=WoE_Tasche('X52','TARGET','asc',full_data_woe,20)
full_data_woe=WoE_Tasche('X58','TARGET','asc',full_data_woe,20)
full_data_woe=WoE_Tasche('X62','TARGET','asc',full_data_woe,20)
full_data_woe=WoE_Tasche('CONT_GRAVES','TARGET','asc',full_data_woe,20)
full_data_woe=WoE_Tasche('CONT_MEDIAS','TARGET','asc',full_data_woe,20)
full_data_woe=WoE_Tasche('CONT_LEVES','TARGET','asc',full_data_woe,20)

# Sin sentido global
full_data_woe=WoE_num('X9','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X15','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X20','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X27','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X34','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X36','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X41','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X42','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X43','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X44','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X47','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X57','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X59','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X61','TARGET',full_data_woe,10)
full_data_woe=WoE_num('X64','TARGET',full_data_woe,10)


# Reordenamos columnas (TARGET al final)
cols=list(full_data_woe)
cols.insert(len(cols), cols.pop(cols.index('TARGET')))
full_data_woe = full_data_woe.reindex(columns= cols)

# Separamos parte WoE
train_woe=full_data_woe.loc[~full_data_woe['TARGET'].isna(),].reset_index(drop=True)
test_woe=full_data_woe.loc[full_data_woe['TARGET'].isna(),].reset_index(drop=True)
test_woe.drop(['TARGET'],axis=1, inplace=True)

# Separamos parte normal
train=full_data.loc[~full_data['TARGET'].isna(),].reset_index(drop=True)
test=full_data.loc[full_data['TARGET'].isna(),].reset_index(drop=True)
test.drop(['TARGET'],axis=1, inplace=True)

##########################################################################################################
#############################################                 ############################################
############################################# MODELOS NIVEL 0 ############################################
#############################################                 ############################################
##########################################################################################################

# 1) Definimos conjuntos de entrenamiento y test:
################################################################################

# Para modelos no lineales
predictoras=list(train)[1:-1]
X_train=train[predictoras].reset_index(drop=True)
Y_train=train['TARGET'].reset_index(drop=True)
X_test=test[predictoras].reset_index(drop=True)

# Para modelos lineales
predictoras_woe=list(train_woe)[1:-1]
X_train_woe=train_woe[predictoras_woe].reset_index(drop=True)
X_test_woe=test_woe[predictoras_woe].reset_index(drop=True)

# 2) Función de Cross-Validación
################################################################################
# Función de entrenamiento de cada fold a través de los otros para un modelo dado.
# Genera predicciones (concatenadas y libres de overfitting) a train.
# Genera predicciones a test (como media de los k modelos del CV).
# Está adaptada para aceptar una NN definida en Keras donde algunos parámetros de
# entrenamiento se pasan en el propio fit.

from sklearn.model_selection import StratifiedKFold

def Model_cv(MODEL, k, X_train, X_test, y, RE, makepred=True, NN_params=None):
	# Creamos los k folds
	kf=StratifiedKFold(n_splits=k, shuffle=True, random_state=RE)

	# Creamos los conjuntos train y test de primer nivel
	Nivel_1_train = pd.DataFrame(np.zeros((X_train.shape[0],1)), columns=['train_yhat'])
	if makepred==True:
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
		if NN_params:
			MODEL.fit(fold_train, fold_ytrain,
					  batch_size=NN_params['batch_size'],
					  epochs=NN_params['epochs'],
					  shuffle=True,
					  verbose=False)
		else:
			MODEL.fit(fold_train, fold_ytrain)

		# Predecimos sobre el fold libre para calcular el error del CV y muy importante:
		# Para hacer una prediccion a train libre de overfitting para el siguiente nivel
		if NN_params:
			p_fold=MODEL.predict(fold_test)[:,0]
			p_fold_train=MODEL.predict(fold_train)[:,0]
		else:
			p_fold=MODEL.predict_proba(fold_test)[:,1]
			p_fold_train=MODEL.predict_proba(fold_train)[:,1]

		# Sacamos el score de la prediccion en el fold libre
		score=roc_auc_score(fold_ytest,p_fold)
		score_train=roc_auc_score(fold_ytrain,p_fold_train)
		print(k, '-cv, Fold ', count, '\t --test AUC: ', round(score,4), '\t--train AUC: ', round(score_train,4),sep='')
		# Gardamos a Nivel_1_train  las predicciones "libres" concatenadas
		Nivel_1_train.loc[test_index.tolist(),'train_yhat'] = p_fold

		# Tenemos que predecir al conjunto test para hacer la media de los k modelos
		# Definimos nombre de la predicción (p_"número de iteración")
		if makepred==True:
			name = 'p_' + str(count)
			# Predicción al test real
			if NN_params:
				real_pred = MODEL.predict(X_test)[:,0]
			else:
				real_pred = MODEL.predict_proba(X_test)[:,1]
			# Ponemos nombre
			real_pred = pd.DataFrame({name:real_pred}, columns=[name])
			# Añadimos a Nivel_1_test
			Nivel_1_test=pd.concat((Nivel_1_test,real_pred),axis=1)

	# Caluclamos la métrica de la predicción total concatenada (y libre de overfitting) a train
	score_total=roc_auc_score(y,Nivel_1_train['train_yhat'])
	print('\n',k, '- cv, TOTAL AUC:', round(score_total*100,4),'%')

	# Hacemos la media de las k predicciones de test
	if makepred==True:
		Nivel_1_test['model']=Nivel_1_test.mean(axis=1)

	# Devolvemos los conjuntos de train y test con la predicción y el rendimiento
	if makepred==True:
		return Nivel_1_train, pd.DataFrame({'test_yhat':Nivel_1_test['model']}), score_total
	else:
		return score_total


# 3) Cross-Validación Nivel 0
################################################################################

# 3.1) Light GBM
################################################################################
# Toda la información aquí: "https://lightgbm.readthedocs.io/en/latest/index.html"
################################################################################
from lightgbm import LGBMClassifier

# Parámetros de la CV:
RS=2305 # Seed que utilizaremos para los folds y la parte random del modelo
n_folds=5 # Número de folds para la cross-validación

# Parámetros del modelo:
params = {'objective': 'binary',
		  'learning_rate': 0.005,
		  'num_leaves': 40,
		  'min_data_in_leaf': 5,
		  'colsample_bytree': 0.7,
		  'max_bin': 20,
		  'random_seed': RS}

# Definimos modelo para diferentes iteraciones y hacemos un search del número óptimo:
model_lightgbm=LGBMClassifier()
model_lightgbm.set_params(**params)
iter=[2000,2500,3000]

print('\nLightGBM CV...')
print('########################################################')
scores=[]
for nrounds in iter:
	model_lightgbm.set_params(n_estimators=nrounds)
	print('\nn rounds: ',nrounds)
	Pred_train, Pred_test, s = Model_cv(model_lightgbm,n_folds,X_train,X_test,Y_train,RS,makepred=True)

	# Miramos si estamos en la primera prueba
	if len(scores)==0:
		max_score=float('-inf')
	else:
		max_score=max(scores)

	# Si el score es el mayor, nos quedamos con esta predicción
	if s>=max_score:
		print('BEST')
		LGBM_train=Pred_train.copy()
		LGBM_test=Pred_test.copy()
	# Añadimos score
	scores.append(s)

# El mejor score cross-validado se ha obtenido en:
print('\n###########################################')
print('LightGBM optimal rounds: ',iter[scores.index(max(scores))])
print('LightGBM optimal AUC: ',round(max(scores)*100,4),'%')
print('###########################################')


# 3.2) Neural Network (Keras)
################################################################################
# Toda la información aquí: "https://keras.io/"
################################################################################
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.constraints import max_norm
from keras.optimizers import adadelta
from keras.regularizers import l2
from tensorflow import set_random_seed

# Parámetros de la CV:
RS=2305 # Seed que utilizaremos para los folds y la parte random del modelo
n_folds=5 # Número de folds para la cross-validación
np.random.seed(RS)
set_random_seed(RS)
# Aquí no obtendremos ejecuciones reproducibles a pesar de las seeds definidas.
# Hay alguna parte random que se me escapa...

# Parámetros del modelo:
# Regla heurísitca: El número de parámetros del modelo (weights) tiene que
# ser como mucho el número de instancias del train / 30. Es decir, si tenemos aprox 4.100
# en train, deberíamos tener como mucho unos 140 parámetros. Con 30 neuronas (hidden) y
# un imput size de 67, tendremos 67*30 weights de la primera capa + 30 weights de la final,
# en total unos 2000 parámetros!!! Sin controles de overfitting sería una locura...

params = {
		  'hidden_dim': 30, # Neuronas en la hidden layer (hacemos una sola capa)
		  'l2': 0.001, # Control overfitting 1: Normalización l2 en la capa
		  'max_norm': 2, # Control overfitting 2: Norma máxima de los weights de la capa
		  'dropout': 0.2, # Control overfitting 3: Al entrenar, el 20% de las neuronas de la capa se desactivan
		  'batch_size': 128, # Número de instancias del train para calcular gradiente. Mayor número no imlica mejor, ya que podríamos no escapar de un mínimo local
		  'opt': 'adadelta' # Adapta el paso del stochastic gradient
		  }

# Definimos modelo:
model_NN=Sequential()
model_NN.add(Dense(units=params['hidden_dim'], input_dim=len(list(X_train)), activation='relu', # REctified Linear Unit
				   kernel_regularizer=l2(params['l2']),
				   kernel_constraint=max_norm(params['max_norm'])))
model_NN.add(Dropout(params['dropout']))
model_NN.add(Dense(units=1, activation='sigmoid')) # Sacamos logística para que nos dé probabilidad
model_NN.compile(loss='binary_crossentropy',optimizer=params['opt'],metrics=['accuracy'])

# Entrenamos modelo para diferentes epochs (número total de pasadas sobre el
# conjunto de entrenamiento entero) y hacemos un search del número óptimo:
epochs=[100,200,300]

print('\nNeural Network CV...')
print('########################################################')
scores=[]
for e in epochs:
	fit_params = {'batch_size': params['batch_size'],'epochs': e}
	print('\nEpochs: ',e)
	Pred_train, Pred_test, s = Model_cv(model_NN,n_folds,X_train,X_test,Y_train,RS,makepred=True,NN_params=fit_params)

	# Miramos si estamos en la primera prueba
	if len(scores)==0:
		max_score=float('-inf')
	else:
		max_score=max(scores)

	# Si el score es el mayor, nos quedamos con esta predicción
	if s>=max_score:
		print('BEST')
		NN_train=Pred_train.copy()
		NN_test=Pred_test.copy()
	# Añadimos score
	scores.append(s)

# El mejor score cross-validado se ha obtenido en:
print('\n###########################################')
print('NN optimal epochs: ',epochs[scores.index(max(scores))])
print('NN optimal AUC: ',round(max(scores)*100,4),'%')
print('###########################################')


# 3.3) Lasso
################################################################################
from sklearn.linear_model import LogisticRegression

# Parámetros de la CV:
RS=2305 # Seed que utilizaremos para los folds y la parte random del modelo
n_folds=5 # Número de folds para la cross-validación

# Parámetros del modelo:
params = {'penalty': 'l1',
		  'solver': 'liblinear',
		  'random_state': RS}

# Definimos modelo para diferentes C's y hacemos un search de la óptima:
model_lasso = LogisticRegression()
model_lasso.set_params(**params)
Cs=[1e-2,1e-1,0.5,1,5,10]

print('\nLasso CV...')
print('########################################################')
scores=[]
for c in Cs:
	model_lasso.set_params(C=c)
	print('\nRegularization C: ',c)
	Pred_train, Pred_test, s = Model_cv(model_lasso,n_folds,X_train_woe,X_test_woe,Y_train,RS,makepred=True)

	# Miramos si estamos en la primera prueba
	if len(scores)==0:
		max_score=float('-inf')
	else:
		max_score=max(scores)

	# Si el score es el mayor, nos quedamos con esta predicción
	if s>=max_score:
		print('BEST')
		Lasso_train=Pred_train.copy()
		Lasso_test=Pred_test.copy()
	# Añadimos score
	scores.append(s)

# El mejor score cross-validado se ha obtenido en:
print('\n###########################################')
print('Lasso optimal C: ',Cs[scores.index(max(scores))])
print('Lasso optimal AUC: ',round(max(scores)*100,4),'%')
print('###########################################')


# 3.4) Ridge
################################################################################
from sklearn.linear_model import LogisticRegression

# Parámetros de la CV:
RS=2305 # Seed que utilizaremos para los folds y la parte random del modelo
n_folds=5 # Número de folds para la cross-validación

# Parámetros del modelo:
params = {'penalty': 'l2',
		  'solver': 'liblinear',
		  'random_state': RS}

# Definimos modelo para diferentes C's y hacemos un search de la óptima:
model_ridge = LogisticRegression()
model_ridge.set_params(**params)
Cs=[1e-2,1e-1,0.5,1,5,10]

print('\nRidge CV...')
print('########################################################')
scores=[]
for c in Cs:
	model_ridge.set_params(C=c)
	print('\nRegularization C: ',c)
	Pred_train, Pred_test, s = Model_cv(model_ridge,n_folds,X_train_woe,X_test_woe,Y_train,RS,makepred=True)

	# Miramos si estamos en la primera prueba
	if len(scores)==0:
		max_score=float('-inf')
	else:
		max_score=max(scores)

	# Si el score es el mayor, nos quedamos con esta predicción
	if s>=max_score:
		print('BEST')
		Ridge_train=Pred_train.copy()
		Ridge_test=Pred_test.copy()
	# Añadimos score
	scores.append(s)

# El mejor score cross-validado se ha obtrnido en:
print('\n###########################################')
print('Ridge optimal C: ',Cs[scores.index(max(scores))])
print('Ridge optimal AUC: ',round(max(scores)*100,4),'%')
print('###########################################')

##########################################################################################################
#############################################                 ############################################
############################################# MODELOS NIVEL 1 ############################################
#############################################                 ############################################
##########################################################################################################

# 1) Nuevos train y test con las predicciones (concatenadas de los modelos cross-validados)
################################################################################
X1_train=pd.DataFrame({
					   "LGBM":LGBM_train['train_yhat'],
					   "NN":NN_train['train_yhat'],
					   "Lasso":Lasso_train['train_yhat'],
					   "Ridge":Ridge_train['train_yhat']
					  })

X1_test=pd.DataFrame({
					   "LGBM":LGBM_test['test_yhat'],
					   "NN":NN_test['test_yhat'],
					   "Lasso":Lasso_test['test_yhat'],
					   "Ridge":Ridge_test['test_yhat']
					  })

# Añadimos predicciones sobre los conjuntos de train/test iniciales
X1_train_complete=pd.concat([X_train,X1_train],axis=1)
X1_test_complete=pd.concat([X_test,X1_test],axis=1)

# 2) Cross-Validación Nivel 1 (Light GBM)
################################################################################
# Parámetros de la CV:
RS=2305 # Seed que utilizaremos para los folds y la parte random del modelo
n_folds=5 # Número de folds para la cross-validación

# Parámetros del modelo: (ligeramente diferentes que en el nivel anterior)
params = {'objective': 'binary',
		  'learning_rate': 0.005,
		  'num_leaves': 40,
		  'min_data_in_leaf': 1,
		  'colsample_bytree': 0.7,
		  'max_bin': 10,
		  'random_seed': RS}

# Definimos modelo para diferentes iteraciones y hacemos un search de la óptima:
model_lightgbm_L1=LGBMClassifier()
model_lightgbm_L1.set_params(**params)
iter=[1300,1500,1700]

print('\nLightGBM Level 1 CV...')
print('########################################################')
scores=[]
for nrounds in iter:
	model_lightgbm_L1.set_params(n_estimators=nrounds)
	print('\nn rounds: ',nrounds)
	s = Model_cv(model_lightgbm_L1,n_folds,X1_train_complete,X1_test_complete,Y_train,RS,makepred=False)

	# Miramos si estamos en la primera prueba
	if len(scores)==0:
		max_score=float('-inf')
	else:
		max_score=max(scores)

	if s>=max_score:
		print('BEST')

	# Añadimos score
	scores.append(s)

# El mejor score cross-validado se ha obtenido en:
print('\n###########################################')
print('LightGBM Level 1 optimal rounds: ',iter[scores.index(max(scores))])
print('LightGBM Level 1 optimal AUC: ',round(max(scores)*100,4),'%')
print('###########################################')

# 3) Modelo de Nivel 1 sobre todo el train con los parámetros óptimos y número de rondas óptimas del CV
################################################################################
# Cuando entrenemos todo el train, utilizaremos más rondas (proporcional al numero de folds)
nrounds=int(iter[scores.index(max(scores))]/(1-1/n_folds))

print('\nLightGBM Level 1 Fit with %d rounds...\n' % nrounds)
model_lightgbm_L1_TOTAL=LGBMClassifier()
model_lightgbm_L1_TOTAL.set_params(**params)
model_lightgbm_L1_TOTAL.set_params(n_estimators=nrounds)
model_lightgbm_L1_TOTAL.fit(X=X1_train_complete,y=Y_train)

# 4) Importancia de los factores del modelo final de Nivel 1:
import shap
print('\nCalculating Feature Importance...\n')
explainer = shap.TreeExplainer(model_lightgbm_L1_TOTAL)
shap_values = explainer.shap_values(X1_train_complete)
shap.summary_plot(shap_values, X1_train_complete,max_display=len(X1_train_complete))

##########################################################################################################
###############################################            ###############################################
############################################### RESULTADOS ###############################################
###############################################            ###############################################
##########################################################################################################

# Predicción final (submission)
################################################################################
test['Pred']=model_lightgbm_L1_TOTAL.predict_proba(X1_test_complete)[:,1]
outputs=pd.DataFrame(test[['ID','Pred']])

# Outputs a .csv
################################################################################
outputs.to_csv('outputs_stacking.csv', index = False)
print('END')

# Código por si queremos salvar/recuperar un modelo largo de ejecutar
################################################################################
# Ejemplo de como salvar un modelo:
# from sklearn.externals import joblib
# joblib.dump(model_xxx,'./BBDD Output/model_xxx.sav')

# Como cargarlo y hacer predicciones
# loaded_model=joblib.load('./BBDD Output/model_xxx.sav')
# loaded_model.predict_proba(X1_test)[:,1]
################################################################################

##########################################################################################################
#############################################                #############################################
############################################# FIN DE LA CITA #############################################
#############################################                #############################################
##########################################################################################################