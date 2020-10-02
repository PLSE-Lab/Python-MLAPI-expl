#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import pandas_profiling as pp
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# data load

"../input/"
X_full = pd.read_csv("../input/learn-together/train.csv", index_col=0)
test = pd.read_csv("../input/learn-together/test.csv", index_col=0)


# In[ ]:


X_full.dtypes


# Excellent!! all features are numeric
# `Soil_TypeX` and `Wilderness_AreaX` are OHE features of categorical variables. Hence they need to be transformed to binary columns.

# In[ ]:


X_full.iloc[:,10:-1] = X_full.iloc[:,10:-1].astype("category")
test.iloc[:,10:] = test.iloc[:,10:].astype("category")

TARGET = 'Cover_Type'
X = X_full.copy()
y = X_full[TARGET]


# In[ ]:


print(X.shape)
X.head()


# In[ ]:


print(y.value_counts(), '\n', 'sum of y count is ', y.value_counts().sum())


# The dataset is **balanced**, there are only 7 values for the label (Cover_Type), and each type have same sample size (2160 rows)
# It is a multiclassification problem.

# In[ ]:


X.describe()


# In[ ]:


X.isna().sum().sum()


# There is no NANs!!! Wonderfull databases :) but not hard work :/  
# 
# Let's make a descriptive stats before training our first model
# 

# In[ ]:


f,ax = plt.subplots(figsize=(9,5))
sns.heatmap(X.corr(),annot=True, linewidths=.2, fmt='.1f', ax=ax)
plt.show()


# 
# Most important correlations are:  
# `Horizontal Distance To Hydrology` and `Vertical Distance To Hydrology` with `70%`.  
# `Aspect` and `Hillshade 3pm` with `60%`;  
# `Hillshade Noon` and `Hillshade 3pm` with `60%`;  
# `Elevation` and `Horizontal Distance To Roadways` with `60%`.

# #### Viz of corr columns

# In[ ]:


X.plot(kind='scatter', x='Vertical_Distance_To_Hydrology',
       y='Horizontal_Distance_To_Hydrology', alpha=0.5,color='darkblue', figsize = (12,9))

plt.title('Vertical And Horizontal Distance To Hydrology')
plt.xlabel("Vertical Distance")
plt.ylabel("Horizontal Distance")

plt.show()


# In[ ]:


X.plot(kind='scatter', x='Aspect', 
              y='Hillshade_3pm', alpha=0.5, 
              color='grey', figsize = (12,9))

plt.title('Vertical And Horizontal Distance To Hydrology')
plt.xlabel("Vertical Distance")
plt.ylabel("Horizontal Distance")

plt.show()


# Let's use boxplot to see outliers and some patterns.

# In[ ]:


box1 = go.Box(y=X["Vertical_Distance_To_Hydrology"],
                name = 'Vertical Distance',marker = dict(color = 'rgb(100,125,219)'))

box2 = go.Box(y=X["Horizontal_Distance_To_Hydrology"],
                name = 'Horizontal Distance', marker = dict(color = 'rgb(59, 19, 224)'))

data = [box1, box2]
layout = dict(autosize=False, width=800,height=500, title='Distance To Hydrology', paper_bgcolor='rgb(243, 243, 243)', 
              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)
iplot(fig)


# We notice u a big difference in median 32 for V and 180 for H. With outliers in uper box

# In[ ]:


box3 = go.Box(y=X["Hillshade_Noon"],name = 'Hillshade Noon',
                marker = dict(color = 'rgb(255,111,145)'))

box4 = go.Box(y=X["Hillshade_3pm"],name = 'Hillshade 3pm',
                marker = dict(color = 'rgb(132,94,194)'))

data = [box3, box4]
layout = dict(autosize=False, width=700,height=500, title='Hillshade 3pm and Noon', paper_bgcolor='rgb(243, 243, 243)', 
              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)
iplot(fig)


# Compare H and V distance separately

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(15,7))
X.Vertical_Distance_To_Hydrology.plot.hist(ax=ax[0],bins=30,
                                                  edgecolor='black',color='crimson') 
                                       
ax[0].set_title('Vertical Distance To Hydrology')
x1=list(range(-150,350,50))
ax[0].set_xticks(x1)

X.Horizontal_Distance_To_Hydrology.plot.hist(ax=ax[1],bins=30,
                                                    edgecolor='black',color='darkmagenta') 
                                                                                                        
ax[1].set_title('Horizontal Distance To Hydrology')
x2=list(range(0,1000,100))
ax[1].set_xticks(x2)

plt.show


# Normal negative values for deep distance, we get tailled representation both.

# Let's see soil_types variables

# In[ ]:


soil_types = X.iloc[:,14:-1].sum(axis=0)

plt.figure(figsize=(18,9))
sns.barplot(x=soil_types.index, y=soil_types.values, palette="rocket")

plt.xticks(rotation= 75)
plt.ylabel('Total')
plt.title('Count of Soil Types With Value 1', color = 'darkred',fontsize=12)

plt.show()


# We se that type _8, type_7, type_15 , type_25 are likely empty. We'l see them  before applying ML.

# ## Distances analysis
# 
# Distances features are: 
# - Vertical_Distance_To_Hydrology
# - Horizontal_Distance_To_Hydrology
# - Horizontal_Distance_To_Roadways
# - Horizontal_Distance_To_Fire_Points

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

X['Euclidean_distance_to_hydro'] = (X.Vertical_Distance_To_Hydrology**2 + X.Horizontal_Distance_To_Hydrology**2)**.5

f, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

sns.distplot(X.Horizontal_Distance_To_Hydrology, color="b", ax=axes[0])
sns.distplot(X.Vertical_Distance_To_Hydrology, color="b", ax=axes[1])
sns.distplot(X['Euclidean_distance_to_hydro'], color="g", ax=axes[2])


# ### Interpretation
# The first plot (horizontal distance to hydrology)
# As expected, vegetation seems to be more abundant near hydrology.
# We see that our third graph looks like the first, this is because the horizontal distance has a wider distribution compared to the vertical one.  
# However, this Euclidean distance is also better suited to the line, which will improve our model.

# Let's see what give us `Horizontal_Distance_To_Hydrology` vs TARGET

# In[ ]:


sns.violinplot(x=TARGET, y='Horizontal_Distance_To_Hydrology', data=X)


# In[ ]:


def euclidean(df):
    df['Euclidean_distance_to_hydro'] = (df.Vertical_Distance_To_Hydrology**2 
                                         + df.Horizontal_Distance_To_Hydrology**2)**.5

    return df
# Calculate euclidian dist in two cols
X = euclidean(X)
test = euclidean(test)


# In[ ]:


from itertools import combinations

def distances(df):
    cols = [
        'Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Fire_Points',
        'Horizontal_Distance_To_Hydrology',
    ]
    
    df['distance_mean'] = df[cols].mean(axis=1)
    df['distance_sum'] = df[cols].sum(axis=1)
    df['distance_road_fire'] = df[cols[:2]].mean(axis=1)
    df['distance_hydro_fire'] = df[cols[1:]].mean(axis=1)
    df['distance_road_hydro'] = df[[cols[0], cols[2]]].mean(axis=1)
    
    df['distance_sum_road_fire'] = df[cols[:2]].sum(axis=1)
    df['distance_sum_hydro_fire'] = df[cols[1:]].sum(axis=1)
    df['distance_sum_road_hydro'] = df[[cols[0], cols[2]]].sum(axis=1)
    
    df['distance_dif_road_fire'] = df[cols[0]] - df[cols[1]]
    df['distance_dif_hydro_road'] = df[cols[2]] - df[cols[0]]
    df['distance_dif_hydro_fire'] = df[cols[2]] - df[cols[1]]
    
    # Vertical distances measures
    colv = ['Elevation', 'Vertical_Distance_To_Hydrology']
    
    df['Vertical_dif'] = df[colv[0]] - df[colv[1]]
    df['Vertical_sum'] = df[colv].sum(axis=1)
    
    return df

X = distances(X)
test = distances(test)


# In[ ]:


X.head(200)


# ## Shade analysis

# In[ ]:


f, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

sns.distplot(X['Hillshade_9am'], color="y", ax=axes[0])
sns.distplot(X['Hillshade_Noon'], color="b", ax=axes[1])
sns.distplot(X['Hillshade_3pm'], color="g", ax=axes[2])


# Hillshade noon have more light which is expected, shade_3pm is normal also as expected, because of less sun light part.

# In[ ]:


X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].kurt()


# ### Interpretation
# 
# Here we can see the variation in the amount of sunlight among three different day hours.
# Between 9 am and noon, we see how the sunlight is increasing with a high positive kurtosis (>1) a huge peak in approx 225, that is almost the max value measurable (254). 
# 

# In[ ]:


def shade(df):
    SHADES = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    
    df['shade_noon_diff'] = df['Hillshade_9am'] - df['Hillshade_Noon']
    df['shade_3pm_diff'] = df['Hillshade_Noon'] - df['Hillshade_3pm']
    df['shade_all_diff'] = df['Hillshade_9am'] - df['Hillshade_3pm']
    df['shade_sum'] = df[SHADES].sum(axis=1)
    df['shade_mean'] = df[SHADES].mean(axis=1)
    return df

X = shade(X)
test = shade(test)


# ## Elevation analysis
# 
# Elevation is the most important feature (see feature importance section). Hence we would compare this with other features (bivariate analysis), and make some transformations creating new features that help our tree algos to make better splits.

# In[ ]:


sns.violinplot(x=TARGET, y='Elevation', data=X)


# #### Plot shade bassed vs elevation

# In[ ]:


f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Hillshade_9am', y='Elevation', 
                hue=TARGET, data=X, y_jitter=True)


# In[ ]:


f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Hillshade_Noon', y='Elevation', 
                hue=TARGET, data=X, y_jitter=True)


# In[ ]:


f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Hillshade_3pm', y='Elevation', 
                hue=TARGET, data=X, y_jitter=True)


# In[ ]:



f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Euclidean_distance_to_hydro', y='Elevation', 
                hue=TARGET, data=X)


# In[ ]:


def elevation(df):
    df['ElevationHydro'] = df['Elevation'] - 0.25 * df['Euclidean_distance_to_hydro']
    return df

X = elevation(X)
test = elevation(test)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Euclidean_distance_to_hydro', y='ElevationHydro', 
                hue=TARGET, data=X)


# In[ ]:


f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Vertical_Distance_To_Hydrology', y='Elevation', 
                hue=TARGET, data=X)


# In[ ]:


def elevationV(df):
    df['ElevationV'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    return df

X = elevationV(X)
test = elevationV(test)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Vertical_Distance_To_Hydrology', y='ElevationV', 
                hue=TARGET, data=X)


# In[ ]:


f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Horizontal_Distance_To_Hydrology', y='Elevation', 
                hue=TARGET, data=X)


# In[ ]:


def elevationH(df):
    df['ElevationH'] = df['Elevation'] - 0.19 * df['Horizontal_Distance_To_Hydrology']
    return df

X = elevationH(X)
test = elevationH(test)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Horizontal_Distance_To_Hydrology', y='ElevationH', 
                hue=TARGET, data=X)


# In[ ]:


f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Horizontal_Distance_To_Roadways', y='Elevation', 
                hue=TARGET, data=X)


# In[ ]:


f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Horizontal_Distance_To_Fire_Points', y='Elevation', 
                hue=TARGET, data=X)


# In[ ]:


f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='distance_road_fire', y='Elevation', 
                hue=TARGET, data=X)


# In[ ]:


def kernel_features(df):
    df['Elevation2'] = df['Elevation']**2
    df['ElevationLog'] = np.log1p(df['Elevation'])
    return df

X = kernel_features(X)
test = kernel_features(test)


# In[ ]:


f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Aspect', y='Elevation', 
                hue=TARGET, data=X)


# In[ ]:


f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Slope', y='Elevation', 
                hue=TARGET, data=X)


# In[ ]:


# drop label 
if TARGET in X.columns:
    X.drop(TARGET, axis=1, inplace=True)


# ## Profiling the dataset

# In[ ]:


report = pp.ProfileReport(X)

report.to_file("report.html")

report


# Some feature need to be ride of before model.

# ### Make a model

# In[ ]:


from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingCVClassifier
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

SEED = 1234

models = {
    'LGBM': LGBMClassifier(n_estimators=400, metric='multi_logloss', num_leaves=100,
                           verbosity=0, random_state=SEED, n_jobs=-1), 
    
    'Random Forest': RandomForestClassifier(n_estimators=700,
                                            n_jobs=-1, random_state=SEED),
    
    'Extra Tree': ExtraTreesClassifier(max_depth=500, n_estimators=450,
                                       n_jobs=-1, oob_score=False, random_state=SEED,
                                       warm_start=True)
    }


# ## Feautures importances

# In[ ]:


clf = models['Random Forest']

def feature_importances(clf, X, y, figsize=(18, 6)):
    clf = clf.fit(X, y)
    
    importances = pd.DataFrame({'Features': X.columns, 
                                'Importances': clf.feature_importances_})
    
    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)

    fig = plt.figure(figsize=figsize)
    sns.barplot(x='Features', y='Importances', data=importances)
    plt.xticks(rotation='vertical')
    plt.show()
    return importances
    
importances = feature_importances(clf, X, y)    


# In[ ]:


def select(importances, edge):
    c = importances.Importances >= edge
    cols = importances[c].Features.values
    return cols

col = select(importances, 0.004)
X = X[col]
test = test[col]    


# Reduce feature less than 0.004

# Let's validate our model(s)

# In[ ]:


# cross validation
from sklearn.model_selection import KFold, cross_val_score

# model selection functions

cv = KFold(n_splits=10, shuffle=True, random_state=SEED)

def cross_val(models, X=X, y=y):
    r = dict()
    for name, model in models.items():
        cv_results = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        r[name] = cv_results
        
        print(name, 'Accuracy Mean {0:.4f}, Std {1:.4f}'.format(
              cv_results.mean(), cv_results.std()))
    return r
    
def choose_best(results):
    errors = dict()

    for name, arr in results.items():
        errors[name] = arr.mean()

    best_model =  [m for m, e in errors.items() 
                   if e == max(errors.values())][0]
    return best_model


# In[ ]:


results = cross_val(models)


# In[ ]:


best_model_name = choose_best(results)

model = models[best_model_name]


# In[ ]:


def predict(model, filename, X=X, y=y, test=test):
    model.fit(X, y)
    predicts = model.predict(test)

    output = pd.DataFrame({'ID': test.index,
                       TARGET: predicts})
    output.to_csv(filename+'.csv', index=False)
    return predicts


# ### Stacked model

# In[ ]:


estimators = [m for m in models.values()]

stack = StackingCVClassifier(classifiers=estimators,
                             meta_classifier=model,
                             cv=cv,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=SEED,
                             n_jobs=-1)

predict_stack = predict(stack, 'stacked')
print('Ready!')


# In[ ]:


print('Predited are', predict_stack)


# ## I'll try out without using correlated feature.

# ### Forked from [Evimar](https://www.kaggle.com/evimarp/top-6-roosevelt-national-forest-competition) and [Fatih Belgin](https://www.kaggle.com/fatihbilgin/quick-eda-and-data-visualization-for-beginners)
