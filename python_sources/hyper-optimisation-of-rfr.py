#!/usr/bin/env python
# coding: utf-8

# # EDA of US cars for auction

# 1. Brief inspection of data
# 2. Data visualisations and exploratory analysis
# 3. Categoric transformations
# 4. Model evaluations
# 5. Hyper optimisation of Random Forest
# 6. Feature evaluation

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3 as sq3
import matplotlib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


us_car_file_path = pd.read_csv('../input/usa-cers-dataset/USA_cars_datasets.csv')
us = us_car_file_path


# **SQL Connection**

# In[ ]:


conn = sq3.connect('USA_cars_datasets.db')
us.to_sql('cars', conn, if_exists='replace', index=False)
e = pd.read_sql_query


# In[ ]:


sns.set(font_scale=2, style='white')


# ### 1. Brief inspection of data

# In[ ]:


print (us.shape)


# In[ ]:


print (us.dtypes)


# In[ ]:


print (us.head())


# In[ ]:


print (us.isna().sum())


# ### 2. Data visualisation and exploratory analysis

# **Origin of cars by state**

# In[ ]:


get_ipython().run_cell_magic('HTML', '', "<div class='tableauPlaceholder' id='viz1588413042357' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ca&#47;Car_state&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Car_state&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ca&#47;Car_state&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1588413042357');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


model_quantity = e('''
                    select count(brand) as total, brand
                    from cars
                    group by brand
                    order by total desc
                    ''', conn)

chart = sns.catplot(x='brand', y='total', data=model_quantity, kind='bar', height=10, aspect=3, palette='plasma')
chart.set_xticklabels(rotation=90)
chart.set_axis_labels('Vehicle Brand','Number of Vehicles')
plt.show()


# In[ ]:


price_brand = e('''
                select avg(price) as average, brand
                from cars
                group by brand
                order by average desc
                
                ''', conn)

plt.figure(figsize=(26,12))
chart = sns.barplot(x='brand',y='average', data=price_brand)
plt.xticks(rotation=90)
plt.xlabel('Vehicle Brand')
plt.ylabel('Average Price [$]')
plt.title('Average Price of Vehicle By Brand')
plt.show()


# In[ ]:


cars_comp = e('''
                select price as price_$, brand, mileage as mileage_km, year
                from cars
                where brand in('ford', 'dodge', 'nissan', 'chevrolet')
                
                ''', conn)



chart = sns.pairplot(cars_comp, hue='brand', palette='husl', height=9, aspect=1)
plt.show()


# We can see from the figures above we have some vehicles set at zero for price.

# In[ ]:


zero_price = us.loc[us['price'] == 0]
print ('Number of cars set at zero for price =', len(zero_price))


# In[ ]:


car_median = (us['price'].median())
us['price'] = us['price'].replace(0, car_median)


# Fill the cells with the mean value from the price column

# In[ ]:


zero_price = us.loc[us['price'] == 0]
print ('Number of cars set at zero for price =',len(zero_price))


# Investigate the price variation between the classifications 'clean vehicle' and 'salvage insurance'.

# In[ ]:


price_status = e('''
                    select title_status, price, brand
                    from cars
                    where brand in ('ford', 'dodge', 'nissan', 'chevrolet')
                    
                    ''', conn)

chart = sns.catplot(x='brand', y='price', hue='title_status', data=price_status, kind='bar', height=11, aspect=2, palette='BuGn_r', saturation=1)
chart.set_axis_labels('Vehicle Brand', 'Price [$]')
plt.show()


# ### 3. Categoric Transformations
# 
# **Categorical conversions and bin distributions**

# Earlier we observed the different datatypes in the dataset, now we make some more subtle observations. 
# First, the year the car was made is better off as a categorical variable rather than numeric, if we carry on with the model as is when we evaluate feaures at the end of the model we would see that the year the car was made gets an insignificant weighting, as a categorical variable it's importance is scaled better. 

# In[ ]:


us['year'] = [str(i) for i in us['year']]


# Secondly, the mileage can be distributed to unequal sized bins. The difference in mileage on cars with lower mileages scales much greater than large differences of mileage for cars with a higher amount of mileage recorded.

# In[ ]:


bin_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
cut_bins = [0,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,250000,1000000]
us['mileage_class'] = pd.cut(us['mileage'], bins=cut_bins, labels=bin_labels)


# **Feature selection**

# The vin and the lot number can be dropped from the dataframe as they will cause of overfitting of the model, naturally a reg and a lot number are not contributing factors when determining the price of a car, also, we renamed the mileage according to a class number so we can drop the mileage column. As we have only 8 cars from Canada we will drop country from the dataframe as well.

# In[ ]:


numeric_features = us.select_dtypes(include=['int64','float64'])
numeric_features = numeric_features.drop(['Unnamed: 0', 'lot','price','mileage'], axis=1)
numeric_features = list(numeric_features.columns)


# In[ ]:


categoric_features = us.select_dtypes(include=['object','category'])
categoric_features = categoric_features.drop(['vin','country'], axis=1)
categoric_features = list(categoric_features.columns)


# In[ ]:


all_features =  categoric_features


# In[ ]:


X = us.drop(['price'],axis=1)[all_features]
y = us['price']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=1)


cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))])

process_features = ColumnTransformer(transformers=[('cat', cat_transformer, categoric_features)])


# ### 4. Model Evaluations

# In[ ]:


models = []
models.append(('SVM', svm.SVR()))
models.append(('DTR', tree.DecisionTreeRegressor()))
models.append(('RFR', RandomForestRegressor()))

results = []
names = []
results_mean = []
for name, model in models:
    pl = Pipeline([('process', process_features), ('models', model)])
    kf = KFold(n_splits=10)
    cv_results = cross_val_score(pl, X_train, y_train, cv=kf, scoring='r2')
    results.append(cv_results)
    names.append(name)
    results_mean.append(cv_results.mean())
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[ ]:


model_comp = {'Model':names,'Results':results_mean}
model_comp = pd.DataFrame(data=model_comp)
plt.figure(figsize=(14,7))
chart = sns.barplot(x='Model',y='Results',data=model_comp)
plt.show()


# ### 5. Hyper optimisation of Random Forest

# We will use RandomizedSearchCV to evaluate the best parameters for the RFR

# In[ ]:


n_estimators = [10,20,50,75,100,200]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 150, num = 10)]
max_depth.append(None)
min_samples_split = [2, 4, 6, 8, 10]
min_samples_leaf = [1, 2, 3, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Then, instantiate the RSCV by incorporating it into our pipeline

# In[ ]:


rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(),param_distributions = random_grid, n_iter = 100, 
                               cv = 3, verbose=1, random_state=42)

pl_rand = Pipeline([('process', process_features), ('models', rf_random)])
pl_rand.fit(X_train, y_train)

print (pl_rand['models'].best_params_)


# Now we will compare the base model from earlier to the tuned model using the parameters printed above to see if the tuned parameters make a significant difference from the base model.

# In[ ]:


rf_random_tuned = RandomForestRegressor(n_estimators=200, min_samples_split=2,min_samples_leaf=1,
                                        max_features='sqrt',max_depth=87,bootstrap=False)

results_tuned = []

pl_rf = Pipeline([('process', process_features), ('model', rf_random_tuned)])
kf = KFold(n_splits=10)
cv_results = cross_val_score(pl_rf, X_train, y_train, cv=kf, scoring='r2')
results_tuned.append(cv_results.mean())
print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[ ]:


comp = results_mean + results_tuned
names_ = ['SVM','DTR','RFR','RFR_Tuned']


result_params = {'Model':names_,'Score':comp}
result_params = pd.DataFrame(data=result_params)

plt.figure(figsize=(14,7))
chart = sns.barplot(x='Model',y='Score',data=result_params)
plt.show()


# Using the tuned parameters with RSCV we have managed a ~3% increase in R2 score, which isn't neglible.

# ### 6. Feature evaluation

# 1. Get encoded column names from columntransformer
# 2. Store names in a list and zip together with results from RandomForestRegressor feature_importances_
# 3. Evaluate the top results i.e. features with the most weighting

# In[ ]:


pl_rf.fit(X_train, y_train)
onehot_cols = list(pl_rf.named_steps['process'].
                      named_transformers_['cat'].
                      named_steps['onehot'].
                      get_feature_names(input_features=categoric_features))


feature_list = onehot_cols


imp = pl_rf['model'].feature_importances_


feature_imp = zip(feature_list,imp)

top_features = []

for i in feature_imp:
    top_features.append(i)
    
top_features = pd.DataFrame(data=top_features,columns=['Feature','Weight'])
top_features = top_features.sort_values(by='Weight',ascending=False).head(n=30)

plt.figure(figsize=(26,12))
chart = sns.barplot(x='Feature',y='Weight', data=top_features, saturation=1, palette='plasma')
plt.xticks(rotation=90)
plt.show()


# We can make some estimations about the importance of some features from the figure above.
# 1. Model, mileage and condition carry the most weighting in general
# 2. The status of a vehicle i.e salvage or clean carry the same weighting, which we would expect, so it's reassuring to see
# 3. While some states appear in the top results of the feature evaluation, this could detract from the generalistion of a model if it were included in further optimising the model
#  
