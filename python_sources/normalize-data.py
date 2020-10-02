# %% [code]
import pandas as pd


def is_outlier(group):
	''' Para cada grupo devuelve por cada valor si es o no outlier. '''
	Q1 = group.quantile(0.25)
	Q3 = group.quantile(0.75)
	IQR = Q3 - Q1
	precio_min = Q1 - 1.5 * IQR
	precio_max = Q3 + 1.5 * IQR
	return ~group.between(precio_min, precio_max)


def getNormalizedDataset():
	''' Devuelve el dataset de propiedades, completando valores nulos, arreglando errores y removiendo outliers. '''

	df = pd.read_csv('../input/mexican-zonaprop-datasets/train.csv',
		index_col='id',
        dtype={'gimnasio': bool,
            	'usosmultiples': bool,
                'escuelascercanas': bool,
                'piscina': bool,
                'centroscomercialescercanos': bool,
                'tipodepropiedad': 'category',
                'provincia': 'category',
                'ciudad': 'category'
            },
		parse_dates=['fecha'])
	pd.set_option('display.float_format', '{:.2f}'.format)
    
	# Elimino columnas innecesarias.
	df.drop(['direccion','idzona','lat','lng'], axis=1, inplace=True)

	# Elimino cualquier propiedad que no sea terreno y no tenga metros cubiertos.
	indices_invalidos = df[(df.metroscubiertos.isnull()) & (~df.tipodepropiedad.isin(['Terreno','Terreno comercial','Lote']))].index
	df.drop(index=indices_invalidos, inplace=True)
    
	print(df.shape)
    
	# Elimino propiedades que no tengan ni ciudad o provincia.
	df.dropna(subset=['ciudad','provincia'], inplace=True)

	print(df.shape)
    
    
	# Completo los nans de metros totales, con lo que tengan en la columna de metros cubiertos.
	df['metrostotales'].fillna(df['metroscubiertos'], inplace=True)

	# El resto de las filas con metros cubiertos nulo, son terrenos. Por ende los completo con 0.
	df['metroscubiertos'].fillna(0, inplace=True)

	# Elimino aquellas propiedades sin tipo. (Son pocas).
	df.dropna(subset=['tipodepropiedad'],inplace=True)
    
	print(df.shape)

	# Asigno como baños y habitaciones nulas el promedio para el tipo de propiedad por ciudad en el que esta. 
	# No se puede asignarle "0" baños o "0" habitaciones porque no tendria sentido.
	df['banos'] = df.groupby(['tipodepropiedad','ciudad'])['banos'].transform(lambda x: x.fillna(x.mode()))
	df['habitaciones'] = df.groupby(['tipodepropiedad','ciudad'])['habitaciones'].transform(lambda x: x.fillna(x.mode()))
	df['banos'] = df.groupby(['tipodepropiedad'])['banos'].transform(lambda x: x.fillna(x.mode()))
	df['habitaciones'] = df.groupby(['tipodepropiedad'])['habitaciones'].transform(lambda x: x.fillna(x.mode()))

	# Aquellos que no tenian un valor de baño o habitacion en los grupos anteriores, los relleno con 0.
	df['banos'].fillna(0, inplace = True)
	df['habitaciones'].fillna(0, inplace = True)

	# Completo con la moda, despues si no se relleno ahi, completo con 0
	df['garages'] = df.groupby(['tipodepropiedad','ciudad'])['garages'].transform(lambda x: x.fillna(x.mode()))
	df['garages'].fillna(0, inplace=True)

	df['precio_m2'] = df['precio']/df['metrostotales']
       
	# Limpio los outliers
	df = df[~df.groupby('tipodepropiedad')['precio_m2'].apply(is_outlier)]
    
	print(df.shape)
    
	# Hay 45000 registros que tienen este problema. La antiguedad no se puede seleccionar en base a un grupo porque estaría, probablemente, dando informacion falsa.
	# Consideramos que todos los que vienen sin información de antiguedad, es porque son nuevos.
	# df['antiguedad'].fillna(0, inplace=True)

	VALOR_CAMBIO_A_DOLAR = 19.54

	# Hay 70000 filas donde los metros totales son menores a los cubiertos. Esto es invalido, pero son muchos datos para descartar
	# Se asigna para estos casos, los metros cubiertos como totales.
	df.loc[df['metrostotales']<df['metroscubiertos'], 'metrostotales'] = df['metroscubiertos']
    
	# Nuevas columnas
	df['precio_dolar'] = df['precio']/VALOR_CAMBIO_A_DOLAR
	df['extras'] = df['garages']+df['piscina']+df['usosmultiples']+df['gimnasio']

	return df