#!/usr/bin/env python
# coding: utf-8

# # Covid-19 in Italy from 2020-02-24 to 2020-06-28

# 1. Loading the Python's libraries needed

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.metrics import r2_score
#
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#
from sklearn.metrics import r2_score
#
from sklearn.metrics import r2_score
#
import warnings
warnings.filterwarnings('ignore')


# 2.  Loading the Data and previewing the first 10 rows of the dataframe

# In[ ]:


df_regions = pd.read_csv('/kaggle/input/covid19-italy-per-regions/dpc-covid19-ita-regioni.csv.txt')
df_regions.head(10)


# 3. Looking for missing values

# In[ ]:


missing_data2 = df_regions.isnull() 
for column in missing_data2.columns.values.tolist():
    print(column)
    print(missing_data2[column].value_counts())
    print('')


# There are 1491 missing data for Tested patients because the testing started later tha the 24th of Februrary
# "note_en" and "note_it" are annotation about the data but I won't be using it.

# 4. Dropping columns I won't use

# In[ ]:


df_regions = df_regions.drop(columns=['note_it', 'note_en'])


# 5. converting the "data" column to Datetime64

# In[ ]:


df_regions['data'] = df_regions['data'].astype('datetime64')


# 6. Graphic visualization of correlation between features

# In[ ]:


corr = df_regions.corr()
corr.style.background_gradient()


# 7. Statistical analysis on the data

# In[ ]:


df_regions.describe()


# 8. graphic visualization of the distribution of patients in intensive care from 2020-02-24 to 2020-06-28

# In[ ]:


df_regions.plot(kind='line', x='data', y='terapia_intensiva', figsize=(12,8))
plt.title('Number of Cases in Intensive Care from 2020-02-24 to 2020-06-28')
plt.xlabel('Date')
plt.ylabel('Number of Cases in Intensive Care')


# 9. Number of patients in intensive care the 28th of June per region 

# In[ ]:


longitude = 12.87194
latitude = 42.56738
IT_map = folium.Map(location=[latitude, longitude], zoom_start=5.5)
regions = folium.map.FeatureGroup()
for lat, lng in zip(df_regions.lat, df_regions.long):
    regions.add_child(
        folium.CircleMarker([lat, lng],
                            radius=5,
                            color='yellow',
                            fill=True,
                            fill_color='blue',
                            fill_opacity=0.6
                            )
    )

    
latitudes = list(df_regions.lat)
longitudes = list(df_regions.long)
labels = list(df_regions.terapia_intensiva)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(IT_map)    
    
IT_map.add_child(regions)


# 10. Dividing the date column into Day (giorno) an Month (mese) columns

# In[ ]:


df_regions['data'] = df_regions['data'].astype('str')
new = df_regions['data'].str.split("-", n=2, expand=True)
df_regions['mese'] = new[1]
df_regions['Day']= new[2]
df_regions.drop(columns=['data'], inplace=True)


# In[ ]:


new = df_regions['Day'].str.split(" ", n=2, expand=True)
df_regions['giorno'] = new[0]


# In[ ]:


df_regions.drop(columns=['Day'], inplace=True)


# In[ ]:


df_regions.head()


# 11. replacing nAn values with 0's

# In[ ]:


df_regions['casi_testati'] = df_regions['casi_testati'].fillna(0)


# 12. Changing some column's data types

# In[ ]:


df_regions['codice_regione'] = df_regions['codice_regione'].astype('str')
df_regions['giorno'] = df_regions['giorno'].astype('int')
df_regions['mese'] = df_regions['mese'].astype('int')


# 13. Dropping the regional code's column

# In[ ]:


df_regions.drop(columns=['codice_regione'], inplace=True)


# 14. Getting dummyes for the cathegorical variable "region name"

# In[ ]:


data = pd.get_dummies(df_regions, drop_first=True)


# 15. Dividing the data into predictors and target

# In[ ]:


X = data[['ricoverati_con_sintomi','totale_ospedalizzati', 'isolamento_domiciliare', 'totale_positivi',
       'variazione_totale_positivi', 'nuovi_positivi', 'dimessi_guariti',
       'deceduti', 'totale_casi', 'tamponi', 'casi_testati', 'mese', 'giorno',
       'denominazione_regione_Basilicata', 'denominazione_regione_Calabria',
       'denominazione_regione_Campania',
       'denominazione_regione_Emilia-Romagna',
       'denominazione_regione_Friuli Venezia Giulia',
       'denominazione_regione_Lazio', 'denominazione_regione_Liguria',
       'denominazione_regione_Lombardia', 'denominazione_regione_Marche',
       'denominazione_regione_Molise', 'denominazione_regione_P.A. Bolzano',
       'denominazione_regione_P.A. Trento', 'denominazione_regione_Piemonte',
       'denominazione_regione_Puglia', 'denominazione_regione_Sardegna',
       'denominazione_regione_Sicilia', 'denominazione_regione_Toscana',
       'denominazione_regione_Umbria', "denominazione_regione_Valle d'Aosta",
       'denominazione_regione_Veneto']]
Y = data['terapia_intensiva']


# 16. Creating train and test sets

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)


# 17. Loading and comparing regression's models

# In[ ]:


models = []
models.append(('Linear Regression', LinearRegression()))
models.append(('Decision Trees', DecisionTreeRegressor()))
models.append(('K Nearest Neighbor',KNeighborsRegressor()))
models.append(('Support Vector', SVR()))
results = []
names = []
for i, j in models: 
    k = KFold(n_splits=10 , random_state=26)
    result = cross_val_score(j, x_train,y_train, cv=k, scoring='r2')
    results.append(result)
    names.append(i)
    print('Model: ', i,'Score: %.2f' % result.mean(), "Model's Standard Deviation: %.2f" % result.std())


# 18. Decision Tree Regressor's accuracy

# In[ ]:


DT =DecisionTreeRegressor()
DT.fit(x_train,y_train)
yhat = DT.predict(x_test)
print('Accuracy of the model: %.2f' % r2_score(y_test, yhat))

