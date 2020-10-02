#!/usr/bin/env python
# coding: utf-8

# # 1. Importing Packages and Collecting Data, Defining Evaluation 

# ## 1.1 Importing Packages

# In[ ]:


'''Importing Data Manipulattion Modules'''
import re
import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from scipy.stats import norm, skew
from scipy.special import boxcox1p
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
pd.set_option("display.max_columns", 81)
pd.set_option("display.max_rows", 101)
pd.set_option("display.max_colwidth", 100)

'''Seaborn and Matplotlib Visualization'''
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')                    
sns.set_style({'axes.grid':False}) 
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

'''Validation'''
from sklearn.model_selection import KFold, cross_val_score

'''Ignore deprecation and future, and user warnings.'''
import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning) 
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = UserWarning) 


# ## 1.2 Collecting Data

# In[ ]:


'''Check the files'''
from subprocess import check_output
print(check_output(["ls", "../input/grupo-bimbo-inventory-demand/"]).decode("utf8"))


# ### town_state

# In[ ]:


dtype = {'Agencia_ID': 'int16'}
town_state_df = pd.read_csv("../input/grupo-bimbo-inventory-demand/town_state.csv", dtype=dtype)

town_state_df['Town_ID'] = town_state_df.Town.apply(lambda x: x.split(' ')[0]).astype('int16')
town_state_df['Town_name'] = town_state_df.Town.apply(lambda x: ' '.join(x.split(' ')[1:]))

print(f'town_state\'s shape: {town_state_df.shape}')
town_state_df.head(5)


# In[ ]:


town_state_df.info()


# ### product

# In[ ]:


dtype = {'Producto_ID': 'int32'}
product_df = pd.read_csv("../input/grupo-bimbo-inventory-demand/producto_tabla.csv", dtype=dtype)

product_df['popular_name'] = product_df.NombreProducto.str.extract(r'^(.*?)(\d*\s\d+(kg|Kg|g|G|ml| ml|p|Reb)\s)', expand=False)[0]
product_df['property'] = product_df.NombreProducto.str.extract(r'^.*\d+(kg|Kg|g|G|ml| ml|p|Reb)\s(.*?)\s\d+$', expand=False)[1]
product_df['unit'] = product_df.NombreProducto.str.extract(r'(\d*\s\d+(kg|Kg|g|G|ml| ml))', expand=False)[0]
product_df['pieces'] =  product_df.NombreProducto.str.extract('(\d+(p|Reb)) ', expand=False)[0]

print(f'product\'s shape: {product_df.shape}')
product_df.head(5)


# In[ ]:


product_df.info()


# ### client

# In[ ]:


dtype = {'Cliente_ID': 'int32'}
client_df = pd.read_csv("../input/grupo-bimbo-inventory-demand/cliente_tabla.csv", dtype=dtype)

dup_sr = client_df.groupby('Cliente_ID')['Cliente_ID'].count().astype('int8')
dup_sr.name = 'dup_num'
client_df = pd.merge(client_df, pd.DataFrame(dup_sr).reset_index())
del dup_sr

print(f'client\'s shape: {client_df.shape}')
client_df.head(5)


# In[ ]:


client_df.info()


# ### test

# In[ ]:


dtype = {
    'id': 'int32',
    'Semana': 'int8',
    'Agencia_ID': 'int16',
    'Canal_ID': 'int8',
    'Ruta_SAK': 'int16',
    'Cliente_ID': 'int32',
    'Producto_ID': 'int32',
}
test_df = pd.read_csv("../input/grupo-bimbo-inventory-demand/test.csv", dtype=dtype)

print(f'test\'s shape: {test_df.shape}')
test_df.head()


# In[ ]:


test_df.info()


# ### train

# In[ ]:


dtype = {
    'Semana': 'int8',
    'Agencia_ID': 'int16',
    'Canal_ID': 'int8',
    'Ruta_SAK': 'int16',
    'Cliente_ID': 'int32',
    'Producto_ID': 'int32',
    'Venta_uni_hoy': 'int16',
    'Venta_hoy': 'float32',
    'Dev_uni_proxima': 'int32',
    'Dev_proxima': 'float32',
    'Demanda_uni_equil': 'int16',
}
train_df = pd.read_csv("../input/grupo-bimbo-inventory-demand/train.csv", dtype=dtype)

print(f'train\'s shape: {train_df.shape}')
train_df.head()


# In[ ]:


train_df.info()


# ### product_price

# In[ ]:


sale_price_sr = (train_df.Venta_hoy / train_df.Venta_uni_hoy)
return_price_sr = (train_df.Dev_proxima / train_df.Dev_uni_proxima)
product_price_df = pd.DataFrame({'Producto_ID': train_df.Producto_ID, 'sale_price': sale_price_sr, 'return_price': return_price_sr})

del sale_price_sr
del return_price_sr

print(f'product price\'s shape: {product_price_df.shape}')
product_price_df.head(5)


# In[ ]:


product_price_df.info()


# ## 1.3 Defining Evaluation

# In[ ]:


'''KFold for cross validation'''
kf = KFold(n_splits=3, shuffle=True, random_state=2)

'''Define the validation function'''
def rmsle_cv(model, X, y, cv=kf):
    rmsle = np.sqrt(
        -cross_val_score(
            model,
            X, y,
            scoring="neg_mean_squared_log_error",
            cv=cv,
        )
    )
    return(rmsle)


# # 2. Adjusting Data

# ## 2.1 Cleansing

# ### town_state

# In[ ]:


town_state_df['Town'] = town_state_df['Town'].str.upper()
town_state_df['Town_name'] = town_state_df['Town_name'].str.upper()
town_state_df['State'] = town_state_df['State'].str.upper()


# In[ ]:


town_state_df.groupby(['Town_name'])['Town_ID'].nunique().sort_values(ascending=False)[:6]


# In[ ]:


town_state_df[(town_state_df['Town_name'].isin(['LOS MOCHIS', 'PINOTEPA']))].sort_values(by='Town_name')


# In[ ]:


town_state_df.loc[498, 'Town_ID'] = 2561


# In[ ]:


town_state_df.head(5)


# In[ ]:


town_state_df.groupby('Town_ID')['Town'].nunique().sort_values(ascending=False)


# In[ ]:


town_state_df[town_state_df.Town_ID.isin([2561, 2169, 2152])].sort_values(by='Town_ID')


# In[ ]:


town_state_df['Town_ID'].max()


# In[ ]:


town_state_df.loc[199, 'Town_ID'] = 3217
town_state_df.loc[311, 'Town_ID'] = 3218


# In[ ]:


town_state_df['Town_ID'].nunique()


# ### product_price

# In[ ]:


sale_prices_df = product_price_df.drop('return_price', axis=1).dropna().rename(columns={'sale_price': 'price'})
return_prices_df = product_price_df.drop('sale_price', axis=1).dropna().rename(columns={'return_price': 'price'})
prices_df = pd.concat([sale_prices_df, return_prices_df])
prices_df = prices_df.groupby('Producto_ID')['price'].median().reset_index()
prices_df.head(5)


# In[ ]:


prices_df.shape


# ### product

# In[ ]:


product_df['in_train'] = 0
product_df['in_test'] = 0
product_df.loc[product_df['Producto_ID'].isin(test_df['Producto_ID'].unique()), 'in_test'] = 1
product_df.loc[product_df['Producto_ID'].isin(train_df['Producto_ID'].unique()), 'in_train'] = 1
product_df = product_df[(product_df['in_test'] == 1) | (product_df['in_train'] == 1)]


# In[ ]:


product_df[product_df['property'].isnull()]


# In[ ]:


product_df.loc[117, 'popular_name'] = 'Donas'
product_df.loc[117, 'property'] = 'Prom BIM'
product_df.loc[117, 'unit'] = None
product_df.loc[117, 'pieces'] = '6p'

product_df.loc[190, 'popular_name'] = 'Paletina para Cafe'
product_df.loc[190, 'property'] = 'NES'
product_df.loc[190, 'unit'] = None
product_df.loc[190, 'pieces'] = None

product_df.loc[381, 'popular_name'] = 'Camioncitos Bimbo'
product_df.loc[381, 'property'] = 'BIM'
product_df.loc[381, 'unit'] = None
product_df.loc[381, 'pieces'] = None

product_df.loc[1152, 'popular_name'] = 'Burrito Vaporero FrijolChorizo'
product_df.loc[1152, 'property'] = 'CU LON'
product_df.loc[1152, 'unit'] = '90g'
product_df.loc[1152, 'pieces'] = None

product_df.loc[1677, 'popular_name'] = 'Tarima Twin Pack Thins Multig'
product_df.loc[1677, 'property'] = 'CU ORO'
product_df.loc[1677, 'unit'] = None
product_df.loc[1677, 'pieces'] = None

product_df.loc[1888, 'popular_name'] = 'Deliciosas Chochochispas'
product_df.loc[1888, 'property'] = 'Prom MTA LAR'
product_df.loc[1888, 'unit'] = '204g'
product_df.loc[1888, 'pieces'] = None

product_df.loc[1889, 'popular_name'] = 'Deliciosas Chochochispas'
product_df.loc[1889, 'property'] = 'Prom LAR'
product_df.loc[1889, 'unit'] = '204g'
product_df.loc[1889, 'pieces'] = None

product_df.loc[2449, 'popular_name'] = 'Galleta Granel Classics Chocolate'
product_df.loc[2449, 'property'] = 'GBI'
product_df.loc[2449, 'unit'] = None
product_df.loc[2449, 'pieces'] = None


# In[ ]:


product_df[product_df['popular_name'].isnull()]


# In[ ]:


product_df.loc[877, 'popular_name'] = 'Tortilla Hna Chihuahua'
product_df.loc[877, 'unit'] = '535g'
product_df.loc[877, 'pieces'] = '10p'

product_df.loc[1585, 'popular_name'] = 'Principe Cho Bco MG'
product_df.loc[1585, 'unit'] = '110g'
product_df.loc[1585, 'pieces'] = '10p'

product_df.loc[1748, 'popular_name'] = 'Combo Salma mas Levite'
product_df.loc[1748, 'unit'] = '1360g'
product_df.loc[1748, 'pieces'] = None


# In[ ]:


product_df['pieces'] = product_df['pieces'].str.extract(r'(\d+)(p|Reb)')[0]


# In[ ]:


product_df['weight'] = product_df['unit'].str.strip()
product_df['weight'] = product_df['weight'].str.replace(' ', '.')
product_df['weight'] = product_df['weight'].str.upper()
w = product_df['weight'].str.extract('(.+?)(KG|G|ML)', expand=True)
product_df['weight'] = w[0].astype('float') * w[1].map({'KG':1000, 'G':1, 'ML':1})


# In[ ]:


product_df = pd.merge(product_df, prices_df, how='left')


# ## 2.2 Imputing Missing Data

# ### product

# In[ ]:


product_df['pieces'] = product_df['pieces'].fillna(1)


# In[ ]:


product_df[product_df['weight'].isnull()]['price'].max()


# In[ ]:


product_df[product_df['price'].isnull()]['weight'].max()


# In[ ]:


df = product_df.dropna()
df = df[(df['price'] <= 311) & (df['weight'] <= 1880)]
plt.figure(figsize=(16,8))
sns.scatterplot(x='weight', y='price', data=df)
del df


# In[ ]:


from sklearn.linear_model import LinearRegression

df = product_df.dropna()
df = df[(df['price'] <= 100) & (df['weight'] <= 1880)]

# predict missing prices
lf = LinearRegression()
lf.fit(df['weight'].values.reshape(-1, 1), df['price'])

prices = lf.predict(product_df[product_df['price'].isnull()]['weight'].values.reshape(-1, 1))

product_df.loc[product_df['price'].isnull(), 'price'] = prices

# predict missing weights
lf = LinearRegression()
lf.fit(df['price'].values.reshape(-1, 1), df['weight'])

weights = lf.predict(product_df[product_df['weight'].isnull()]['price'].values.reshape(-1, 1))

product_df.loc[product_df['weight'].isnull(), 'weight'] = weights

del df
del prices
del weights


# In[ ]:


(product_df.drop(['unit'], axis=1).isnull().sum() == 0).all()


# ## 2.3 Transforming Data Type

# In[ ]:


product_df['pieces'] = product_df['pieces'].astype('int16')
product_df['in_train'] = product_df['in_train'].astype('bool')
product_df['in_test'] = product_df['in_test'].astype('bool')
product_df['weight'] = product_df['weight'].astype('float32')
product_df['price'] = product_df['price'].astype('float32')


# In[ ]:


town_state_df['State'] = town_state_df['State'].astype('category')


# In[ ]:


train_df['Canal_ID'] = train_df['Canal_ID'].astype('category')
test_df['Canal_ID'] = test_df['Canal_ID'].astype('category')


# ## 2.4 Dropping Features

# In[ ]:


train_df.drop(['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima'], axis=1, inplace=True)


# ## 2.5 Merging Data

# ### town_state

# In[ ]:


train_df = pd.merge(train_df, town_state_df[['Agencia_ID', 'Town_ID']], how='left')
test_df = pd.merge(test_df, town_state_df[['Agencia_ID', 'Town_ID']], how='left')
train_df.drop('Agencia_ID', axis=1, inplace=True)
test_df.drop('Agencia_ID', axis=1, inplace=True)


# ### product

# In[ ]:


train_df = pd.merge(
    train_df,
    product_df[[
        'Producto_ID', 'popular_name', 'property',
        'pieces', 'weight', 'price'
    ]], how='left')
test_df = pd.merge(
    test_df,
    product_df[[
        'Producto_ID', 'popular_name', 'property',
        'pieces', 'weight', 'price'
    ]], how='left')


# In[ ]:


train_df.head(5)


# ## 2.6 Bin-Counting

# ### Semana

# In[ ]:


semana_med_s = train_df.groupby('Semana')['Demanda_uni_equil'].median()


# In[ ]:


semana_med_s


# In[ ]:


train_df.drop('Semana', axis=1, inplace=True)
test_df.drop('Semana', axis=1, inplace=True)
del semana_med_s


# ### Cliente_ID

# In[ ]:


client_med_s = train_df.groupby('Cliente_ID')['Demanda_uni_equil'].median().astype('int16')
client_med_s.name = 'client_med'


# ### popular_name

# In[ ]:


popular_name_med_s = train_df.groupby('popular_name')['Demanda_uni_equil'].median().astype('int16')
popular_name_med_s.name = 'popular_name_med'


# ### Town_ID

# In[ ]:


town_id_med_s = train_df.groupby('Town_ID')['Demanda_uni_equil'].median().astype('int16')
town_id_med_s.name = 'town_id_med'


# ### Ruta_SAK

# In[ ]:


ruta_id_med_s = train_df.groupby('Ruta_SAK')['Demanda_uni_equil'].median().astype('int16')
ruta_id_med_s.name = 'ruta_id_med'


# ## 2.7 Merging Bin-Counting Data

# In[ ]:


test_df = pd.merge(test_df, client_med_s.reset_index(), how='left')
test_df = pd.merge(test_df, popular_name_med_s.reset_index(), how='left')
test_df = pd.merge(test_df, town_id_med_s.reset_index(), how='left')
test_df = pd.merge(test_df, ruta_id_med_s.reset_index(), how='left')


# In[ ]:


train_df = pd.merge(train_df, client_med_s.reset_index(), how='left')
train_df = pd.merge(train_df, popular_name_med_s.reset_index(), how='left')
train_df = pd.merge(train_df, town_id_med_s.reset_index(), how='left')
train_df = pd.merge(train_df, ruta_id_med_s.reset_index(), how='left')


# ## 2.8 Imputing Test Missing Data

# In[ ]:


test_df['popular_name_med'] = test_df['popular_name_med'].fillna(test_df['popular_name_med'].mean())
test_df['client_med'] = test_df['client_med'].fillna(test_df['client_med'].mean())
test_df['ruta_id_med'] = test_df['ruta_id_med'].fillna(test_df['ruta_id_med'].mean())


# ## 2.9 Transforming Data Type

# In[ ]:


train_df['client_med'] = train_df['client_med'].astype('int16')
train_df['popular_name_med'] = train_df['popular_name_med'].astype('int16')
train_df['town_id_med'] = train_df['town_id_med'].astype('int16')
train_df['ruta_id_med'] = train_df['ruta_id_med'].astype('int16')


# In[ ]:


test_df['client_med'] = test_df['client_med'].astype('int16')
test_df['popular_name_med'] = test_df['popular_name_med'].astype('int16')
test_df['town_id_med'] = test_df['town_id_med'].astype('int16')
test_df['ruta_id_med'] = test_df['ruta_id_med'].astype('int16')


# ## 2.10 Dropping Features

# In[ ]:


train_df.drop(
    ['Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Town_ID', 'popular_name', 'property', 'pieces'],
    axis=1, inplace=True)
test_df.drop(
    ['Ruta_SAK', 'Cliente_ID', 'Producto_ID', 'Town_ID', 'popular_name', 'property', 'pieces'],
    axis=1, inplace=True)


# In[ ]:


train_df.head(3)


# In[ ]:


test_df.head(3)


# # 3. Data Preprocessing

# ## 3.1 Take a glance at all variables

# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# In[ ]:


'''Plot histogram of numerical variables to validate pandas intuition.'''
def draw_histograms(df, variables, n_rows, n_cols, size):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows, n_cols, i+1)
        df[var_name].hist(bins=40, ax=ax, color='skyblue', alpha=0.8, figsize=size)
        ax.set_title(var_name, fontsize=43)
        ax.tick_params(axis='both', which='major', labelsize=35)
        ax.tick_params(axis='both', which='minor', labelsize=35)
        ax.set_xlabel('')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# In[ ]:


skewness = train_df.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float32', 'float64']).apply(lambda x: skew(x))
skew_index = skewness[abs(skewness) >= 0.75].index
skewness[skew_index].sort_values(ascending=False)


# In[ ]:


skewness = test_df.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 'float32', 'float64']).apply(lambda x: skew(x))
skew_index = skewness[abs(skewness) >= 0.75].index
skewness[skew_index].sort_values(ascending=False)


# ## 3.2 BoxCox Transform

# In[ ]:


'''BoxCox Transform'''
lam = 0.01
for column in skew_index:
    train_df[column] = boxcox1p(train_df[column], lam)
    test_df[column] = boxcox1p(test_df[column], lam)


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# ## 3.3 Transforming Data Type

# In[ ]:


train_df['client_med'] = train_df['client_med'].astype('float32')
train_df['popular_name_med'] = train_df['popular_name_med'].astype('float32')
train_df['town_id_med'] = train_df['town_id_med'].astype('float32')
train_df['ruta_id_med'] = train_df['ruta_id_med'].astype('float32')


# In[ ]:


test_df['client_med'] = test_df['client_med'].astype('float32')
test_df['popular_name_med'] = test_df['popular_name_med'].astype('float32')
test_df['town_id_med'] = test_df['town_id_med'].astype('float32')
test_df['ruta_id_med'] = test_df['ruta_id_med'].astype('float32')


# # 4. Exploratory Data Analysis 

# ## 4.1 Analyzing Target 

# In[ ]:


sample_train_df = train_df.sample(n=10000)
sample_train_df['log_target'] = np.log1p(sample_train_df['Demanda_uni_equil'])


# In[ ]:


sample_train_df.head(5)


# In[ ]:


'''correlation matrix'''
plt.subplots(figsize=(20, 16))
k = 20 #number of variables for heatmap
corrmat = sample_train_df.corr()
cols = corrmat.nlargest(k, 'log_target')['log_target'].index

cm = np.corrcoef(sample_train_df[cols].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size': 10}, cmap='Blues',
                 yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[ ]:


'''Check feature inportance by applying LightGBM'''
import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(num_leaves=1000,
                              max_depth=5,
                              learning_rate=0.1,
                              random_state=2)
model_lgb.fit(sample_train_df.drop(['Demanda_uni_equil', 'log_target'], axis=1), sample_train_df['log_target'])


# In[ ]:


df = pd.DataFrame(model_lgb.feature_importances_,
             index=sample_train_df.drop(['Demanda_uni_equil', 'log_target'], axis=1).columns,
             columns=['importance']).sort_values('importance', ascending=False)
df[df.importance > 10]


# In[ ]:


# weight
plt.figure(figsize=(16, 8))
sns.scatterplot(x='weight', y='log_target', data=sample_train_df, palette='Blues_d')


# In[ ]:


# price
plt.figure(figsize=(16, 8))
sns.scatterplot(x='price', y='log_target', data=sample_train_df, palette='Blues_d')


# In[ ]:


# client_med
plt.figure(figsize=(16, 8))
sns.scatterplot(x='client_med', y='log_target', data=sample_train_df, palette='Blues_d')


# In[ ]:


# popular_name_med
plt.figure(figsize=(16, 8))
sns.scatterplot(x='popular_name_med', y='log_target', data=sample_train_df, palette='Blues_d')


# In[ ]:


# town_id_med
plt.figure(figsize=(16, 8))
sns.scatterplot(x='town_id_med', y='log_target', data=sample_train_df, palette='Blues_d')


# In[ ]:


# ruta_id_med
plt.figure(figsize=(16, 8))
sns.scatterplot(x='ruta_id_med', y='log_target', data=sample_train_df, palette='Blues_d')


# # 5. Model Building and Evaluation

# ## 5.1 Importing Packages

# In[ ]:


'''Importing Modeling Interested Modules'''
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from lightgbm import LGBMRegressor


# ## 5.2 Preparation before Building Models 

# In[ ]:


'''Adjust dataframe for modeling'''
train_y = train_df['Demanda_uni_equil']
train_df.drop(['Demanda_uni_equil'], axis=1, inplace=True)
train_X = train_df
test_X = test_df.drop('id', axis=1)

'''Transform categorical features to dummy variables'''
train_X = pd.get_dummies(train_X)
test_X = pd.get_dummies(test_X)


# In[ ]:


sample_train_df['Demanda_uni_equil'] = np.expm1(sample_train_df['log_target']).astype('int32')


# In[ ]:


sample_train_df.head(5)


# In[ ]:


'''Prepare sample train for the fast training'''
sample_train_y = sample_train_df['Demanda_uni_equil']
sample_train_df.drop(['Demanda_uni_equil', 'log_target'], axis=1, inplace=True)
sample_train_X = sample_train_df

sample_train_X = pd.get_dummies(sample_train_X)


# In[ ]:


'''We should use the log transform of the target value'''
class MyEstimator(BaseEstimator):
    def __init__(self, model):
        self.model = model
        
    def fit(self, X, y):
        self.model.fit(X, np.log1p(y))
        return self 

    def predict(self, X):
        predicts = np.expm1(self.model.predict(X))
        mask = (predicts <= 0)
        predicts[mask] = 0
        return predicts


# In[ ]:


'''Define evaluation function for Convienience'''
def evaluation_model(model, train_X, train_y, test_X):
    cv = rmsle_cv(model, train_X, train_y)
    cv_mean = np.round(cv.mean(), 5)
    cv_std = np.round(cv.std(), 5)
    sample_prediction = model.predict(test_X.loc[:3, :])
    return {'cv_mean': cv_mean, 'cv_std': cv_std, 'sample_prediction': sample_prediction}


# In[ ]:


'''Define Hyperparameters Tuning Function'''
def tune_hyperparameters(model, param_grid, train_X, train_y):
    grid = GridSearchCV(
        model, param_grid, 
        scoring='neg_mean_squared_log_error',
        cv=3, n_jobs=-1,
    )
    grid.fit(train_X, train_y)
    best_params = grid.best_params_ 
    best_score = np.round(np.sqrt(-1 * grid.best_score_), 5)
    return best_params, best_score


# ## 5.3 Building Models

# ### LinearRegression

# In[ ]:


model = make_pipeline(
    RobustScaler(),
    LinearRegression(),
)
lr_model = MyEstimator(model)
lr_model.fit(sample_train_X, sample_train_y)
lr_eval = evaluation_model(lr_model, sample_train_X, sample_train_y, test_X)
print(lr_eval)


# ### LassoCV

# In[ ]:


model = make_pipeline(
    RobustScaler(),
    LassoCV(
        alphas=(0.00001, 0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10),
    ),
)
lasso_cv_model = MyEstimator(model)
lasso_cv_model.fit(sample_train_X, sample_train_y)
lasso_cv_eval = evaluation_model(lasso_cv_model, sample_train_X, sample_train_y, test_X)
print(lasso_cv_eval)

opt_alpha = lasso_cv_model.model.steps[1][1].alpha_
print(f'\nopt_alpha: {opt_alpha}')


# ### RidgeCV

# In[ ]:


model = make_pipeline(
    RobustScaler(),
    RidgeCV(
        alphas=(0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1, 0.3, 1, 3, 5, 10),
    ),
)
ridge_cv_model = MyEstimator(model)
ridge_cv_model.fit(sample_train_X, sample_train_y)
ridge_cv_eval = evaluation_model(ridge_cv_model, sample_train_X, sample_train_y, test_X)
print(ridge_cv_eval)

opt_alpha = ridge_cv_model.model.steps[1][1].alpha_
print(f'\nopt_alpha: {opt_alpha}')


# ### ElasticNetCV

# In[ ]:


model = make_pipeline(
    RobustScaler(),
    ElasticNetCV(
        alphas=(0.00001, 0.0001, 0.0002, 0.0003), 
        l1_ratio=(0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5),
    ),
)
elsnt_cv_model = MyEstimator(model)
elsnt_cv_model.fit(sample_train_X, sample_train_y)
elsnt_cv_eval = evaluation_model(elsnt_cv_model, sample_train_X, sample_train_y, test_X)
print(elsnt_cv_eval)

opt_alpha = elsnt_cv_model.model.steps[1][1].alpha_
opt_l1_ratio = elsnt_cv_model.model.steps[1][1].l1_ratio_
print(f'\nopt_alpha: {opt_alpha} opt_l1_ratio: {opt_l1_ratio}')


# ### SVR

# In[ ]:


# {'cv_mean': 0.57025, 'cv_std': 0.00775, 'sample_prediction': array([3.85650165, 1.24517075, 3.45866592])}

# grid best_params: {'model__svr__C': 10, 'model__svr__epsilon': 0.1, 'model__svr__gamma': 0.01}

# ### build basemodel
# model = make_pipeline(
#     RobustScaler(),
#     SVR(),
# )
# svr_model = MyEstimator(model)
# 
# ### optimize hyperparameters
# param_grid = {'model__svr__C': [1, 10, 20],
#               'model__svr__epsilon': [0.001, 0.01, 0.1],
#               'model__svr__gamma': [0.0001, 0.001, 0.01]}
# best_params, best_score = \
#     tune_hyperparameters(svr_model, param_grid, sample_train_X, sample_train_y)
# 
# ### fit using best_params
# svr_model.set_params(**best_params)
# svr_model.fit(sample_train_X, sample_train_y)
# svr_eval = evaluation_model(svr_model, sample_train_X, sample_train_y, test_X)
# print(svr_eval)
# 
# print(f'\ngrid best_params: {best_params}')


# ### KernelRidge

# In[ ]:


# {'cv_mean': 0.56925, 'cv_std': 0.00546, 'sample_prediction': array([4.15061605, 1.21544493, 3.65678685, 1.81862787])}

# grid best_params: {'model__kernelridge__alpha': 0.5, 'model__kernelridge__coef0': 3, 'model__kernelridge__degree': 2, 'model__kernelridge__kernel': 'polynomial'}

# ### build basemodel
# model = make_pipeline(
#     RobustScaler(),
#     KernelRidge(),
# )
# kr_model = MyEstimator(model)
# 
# ### optimize hyperparameters
# param_grid = {'model__kernelridge__alpha': [0.01, 0.1, 0.5, 1],
#               'model__kernelridge__kernel': ['linear', 'polynomial'],
#               'model__kernelridge__degree': [1, 1.5, 2, 3],
#               'model__kernelridge__coef0': [3, 4, 5]}
# best_params, best_score = \
#     tune_hyperparameters(kr_model, param_grid, sample_train_X, sample_train_y)
# 
# ### fit using best_params
# kr_model.set_params(**best_params)
# kr_model.fit(sample_train_X, sample_train_y)
# kr_eval = evaluation_model(kr_model, sample_train_X, sample_train_y, test_X)
# print(kr_eval)
# 
# print(f'\ngrid best_params: {best_params}')


# ### LightGBM

# In[ ]:


model = LGBMRegressor(learning_rate=0.01, n_estimators=3000,
                      num_leaves=5,
                      objective='regression',
                      max_bin=55, bagging_fraction=0.8,
                      bagging_freq=5, feature_fraction=0.2319,
                      feature_fraction_seed=9, bagging_seed=9,
                      min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
lgb_model = MyEstimator(model)
lgb_model.fit(sample_train_X, sample_train_y)
lgb_eval = evaluation_model(lgb_model, sample_train_X, sample_train_y, test_X)
print(lgb_eval)


# ## 5.4 Submission

# In[ ]:


def output_submission_file(model, test_X, filename='submission.csv'):
    prediction = model.predict(test_X)
    df = pd.DataFrame({'id': test_df['id'], 'Demanda_uni_equil': prediction})
    print(f'{df.shape}')
    print(f'{df.head(5)}')
    df.to_csv(filename, index=False)
    df.to_csv(filename + '.gz', index=False, compression='gzip')


# In[ ]:


'''Submission'''
output_submission_file(lgb_model, test_X)

