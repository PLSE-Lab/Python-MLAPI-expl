#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# plotly
# import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# Disabling warnings
import warnings
warnings.simplefilter("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
df = data.copy()


# # General Information and Data Cleaning

# ### This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981).
# ### We would like to find out which features are the most indicative of a poisonous or edible mushroom.
# 
# ### For this purpose;
# 1. Have a general idea about the data set. 
# 1. Use necessary methods to clean and prepare the data for analysis.
# 1. Conduct Explanatory Data Analysis (EDA) and visualization. 
# 1. Pre-process the data. 
# 1. Conduct Decision Tree and RF analyses with all of the categories and check the results.
# 1. Look into the top most influential 5 features on the to classify the mushroom class whether the mushroom is poisonous or not. 
# 1. Conduct Decision Tree and RF analyses with these top 5 features one more time and check the results.

# ### Attribute Information: (classes: edible=e, poisonous=p)
# 
# * cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 
# * cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
#  
# * cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
#  
# * bruises: bruises=t,no=f
#  
# * odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
#  
# * gill-attachment: attached=a,descending=d,free=f,notched=n
#  
# * gill-spacing: close=c,crowded=w,distant=d
#  
# * gill-size: broad=b,narrow=n
#  
# * gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
#  
# * stalk-shape: enlarging=e,tapering=t
#  
# * stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
#  
# * stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
#  
# * stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
#  
# * stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
#  
# * stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
#  
# * veil-type: partial=p,universal=u
#  
# * veil-color: brown=n,orange=o,white=w,yellow=y
#  
# * ring-number: none=n,one=o,two=t
#  
# * ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
#  
# * spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
#  
# * population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
#  
# * habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

# In[ ]:


display(df.head())
display(df.tail())


# ### We have 23 categorical variables and 8124 observations.

# In[ ]:


df.info()


# ### We do not have any nan values. 

# In[ ]:


df.isnull().sum()


# ### We have keyword such as class and '-'between the words of variable names. So, it would be good to rename the columns. 

# In[ ]:


df.columns


# In[ ]:


len(df.columns)


# In[ ]:


df.columns = ['Mushroom_class', 'Cap_shape', 'Cap_surface', 'Cap_color', 'Bruises', 'Odor', 'Gill_attachment', 'Gill_spacing', 'Gill_size', 'Gill_color',
            'Stalk_shape', 'Stalk_root', 'Stalk_surface_above_ring', 'Stalk_surface_below_ring', 'Stalk_color_above_ring', 'Stalk_color_below_ring', 
            'Veil_type', 'Veil_color', 'Ring_number', 'Ring_type', 'Spore_print_color', 'Population', 'Habitat']
df.head()


# In[ ]:


df.columns


# ### We convert object type variables into categorical variables since we will use all of these categorical variables in our analysis. 

# In[ ]:


cols = df.columns
df[cols] = df[cols].astype('category')


# In[ ]:


# Alternative way
# df.columns.apply(lambda x: x.astype('category'))


# In[ ]:


df.info()


# ### We check the names and number of categories in each variable. 

# In[ ]:


for col in df.columns:
    print(df[col].unique())


# # EDA

# ### We can only see the properties such as count, unique categories, the most frequent category and its frequency of each categorical variable. We do not have any other statistical information since they are not quantitative. 

# In[ ]:


df.describe()


# ### We check the value counts of categories in each variable. 

# In[ ]:


for col in df.columns:
    print(df[col].value_counts())


# ## Countplot of Categorical Variables

# ### We use Seaborn Countplot graph to visualize the frequency of categories in each variable. 

# In[ ]:


for i, col in enumerate(df.columns):
    plt.figure(i)
    plt.title(col, color = 'blue',fontsize=15)
    sns.countplot(x=col, data=df ,order=df[col].value_counts().index)


# # Data Preprocessing

# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


# ### We would like to know whether our result will be edible or poisonous (0/1 in numerical terms) through our independent variables. For this purpose, we need to label our variables and convert the categories into numerical values. I have decided to use one-hot encoding. 

# In[ ]:


df1 = df.copy()
df1.drop('Mushroom_class',axis=1,inplace=True)


# In[ ]:


def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df
df1 = one_hot(df1,df1.columns)
df1.head()


# In[ ]:


df1.info()


# In[ ]:


from sklearn.preprocessing import LabelEncoder

lbe = LabelEncoder()
df['Mushroom_class_bin'] = lbe.fit_transform(df['Mushroom_class'])
df.head()


# In[ ]:


y = df["Mushroom_class_bin"]
X = df1.select_dtypes(exclude='category')
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)


# # Decision Tree

# ### We have accuracy score of 1.0 which means that the features can successfully predict whether a mushroom is poisonous or not with a percentage of 100%. Since the accuracy score is already at the maximum value, the model tuning does not change the accuracy score.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train, y_train)


# In[ ]:


cart_model


# In[ ]:


get_ipython().system('pip install skompiler')


# In[ ]:


y_pred = cart_model.predict(X_test)
accuracy_score(y_test, y_pred)


# ## Model Tuning

# In[ ]:


cart_grid = {"max_depth": range(1,10),
            "min_samples_split" : list(range(2,50)) }


# In[ ]:


cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)


# In[ ]:


print("Best Parameters: " + str(cart_cv_model.best_params_))


# In[ ]:


cart = tree.DecisionTreeClassifier(max_depth = 8, min_samples_split = 2)
cart_tuned = cart.fit(X_train, y_train)


# In[ ]:


y_pred = cart_tuned.predict(X_test)
accuracy_score(y_test, y_pred)


# ### When we look at the importance levels of different categories, we see that Odor_n, Stalk_root_c, Stalk_surface_below_ring_y, Spore_print_color_r and Odor_a can be considered the top 5 important sub-categories.
# ### Note: The order of the important sub-categories with relatively lower importance scores may change in each trial.

# In[ ]:


Importance = pd.DataFrame({"Importance": cart_tuned.feature_importances_*100}, index = X_train.columns)


# In[ ]:


Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[:10]


# In[ ]:


Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[:10].plot(kind ="barh", color = "r")

plt.xlabel("Importance Levels of Categories");


# In[ ]:


# df2 = df.copy()
# df2.drop('Mushroom_class',axis=1,inplace=True)
# df2 = df2[['Odor', 'Stalk_root', 'Stalk_surface_below_ring', 'Spore_print_color', 'Ring_type']]
# df2.head()


# In[ ]:


# def one_hot(df, cols):
#     for each in cols:
#         dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
#         df = pd.concat([df, dummies], axis=1)
#     return df
# df2 = one_hot(df2,df2.columns)
# df2 = df2.select_dtypes(exclude=['category'])
# df2.head()


# ### We execute decision tree model again with the top 5 features. It can be seen that accuracy score of the model with these top 5 features is 0.9888 compared to 1.0 with all of the features that comprise of all sub-categories. In this framework, top 5 features have a very high accuracy score to predict whether a mushroom is poisonous or not. 

# In[ ]:


y = df["Mushroom_class_bin"]
X = df1[['Odor_n', 'Stalk_root_c', 'Stalk_surface_below_ring_y', 'Spore_print_color_r', 'Odor_a']]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)


# In[ ]:


cart1 = DecisionTreeClassifier()
cart_model1 = cart.fit(X_train, y_train)
cart_model1


# In[ ]:


y_pred = cart_model1.predict(X_test)
accuracy_score(y_test, y_pred)


# ## Model Tuning

# In[ ]:


cart_grid = {"max_depth": range(1,10),
            "min_samples_split" : list(range(2,50)) }


# In[ ]:


cart1 = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart1, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)


# In[ ]:


print("Best Parameters: " + str(cart_cv_model.best_params_))


# In[ ]:


cart1 = tree.DecisionTreeClassifier(max_depth = 4, min_samples_split = 2)
cart_tuned1 = cart1.fit(X_train, y_train)


# In[ ]:


y_pred = cart_tuned1.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


Importance1 = pd.DataFrame({"Importance": cart_tuned1.feature_importances_*100}, index = X_train.columns)


# In[ ]:


Importance1.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[:10]


# In[ ]:


Importance1.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[:10].plot(kind ="barh", color = "r")

plt.xlabel("Importance Levels of Categories");


# # RF

# ### Similar to Decision Tree approach, we have accuracy score of 1.0 which means that the features can successfully predict whether a mushroom is poisonous or not with a percentage of 100%. Since the accuracy score is already at the maximum value, the model tuning does not change the accuracy score.

# In[ ]:


y = df["Mushroom_class_bin"]
X = df1.select_dtypes(exclude='category')
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf_model = RandomForestClassifier().fit(X_train, y_train)


# In[ ]:


rf_model


# In[ ]:


y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)


# ## Model Tuning

# In[ ]:


rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,5,8],
            "n_estimators": [10,500,1000],
            "min_samples_split": [2,5,10]}


# In[ ]:


rf_model = RandomForestClassifier()

rf_cv_model = GridSearchCV(rf_model, 
                           rf_params, 
                           cv = 10, 
                           n_jobs = -1, 
                           verbose = 2) 


# In[ ]:


rf_cv_model.fit(X_train, y_train)


# In[ ]:


print("Best Parameters: " + str(rf_cv_model.best_params_))


# In[ ]:


rf_tuned = RandomForestClassifier(max_depth = 8, 
                                  max_features = 5, 
                                  min_samples_split = 2,
                                  n_estimators = 500)

rf_tuned.fit(X_train, y_train)


# In[ ]:


y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred)


# ### We execute RF model again with the top 5 features. It can be seen that accuracy score of the model with these top 5 features is 0.9664 compared to 1.0 with all of the features that comprise of all sub-categories. In this framework, top 5 features have a very high accuracy score to predict whether a mushroom is poisonous or not.

# In[ ]:


Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)


# In[ ]:


Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[0:10]


# In[ ]:


Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[0:10].plot(kind ="barh", color = "r")

plt.xlabel("Importance Levels of Variables")


# In[ ]:


y = df["Mushroom_class_bin"]
X = df1[['Odor_n', 'Odor_f', 'Gill_size_b', 'Gill_size_n', 'Gill_color_b']]
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)


# In[ ]:


rf_model1 = RandomForestClassifier().fit(X_train, y_train)


# In[ ]:


y_pred = rf_model1.predict(X_test)
accuracy_score(y_test, y_pred)


# ## Model Tuning

# In[ ]:


rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,5,8],
            "n_estimators": [10,500,1000],
            "min_samples_split": [2,5,10]}


# In[ ]:


rf_model1 = RandomForestClassifier()

rf_cv_model1 = GridSearchCV(rf_model1, 
                           rf_params, 
                           cv = 10, 
                           n_jobs = -1, 
                           verbose = 2) 


# In[ ]:


rf_cv_model1.fit(X_train, y_train)


# In[ ]:


print("Best Parameters: " + str(rf_cv_model1.best_params_))


# In[ ]:


rf_tuned1 = RandomForestClassifier(max_depth = 2, 
                                  max_features = 2, 
                                  min_samples_split = 2,
                                  n_estimators = 500)

rf_tuned1.fit(X_train, y_train)


# In[ ]:


y_pred = rf_tuned1.predict(X_test)
accuracy_score(y_test, y_pred)


# In[ ]:


Importance = pd.DataFrame({"Importance": rf_tuned1.feature_importances_*100},
                         index = X_train.columns)


# In[ ]:


Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[0:10]


# In[ ]:


Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[0:10].plot(kind ="barh", color = "r")

plt.xlabel("Importance Levels of Variables")


# In[ ]:




