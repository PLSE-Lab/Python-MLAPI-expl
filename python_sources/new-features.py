#!/usr/bin/env python
# coding: utf-8

# **Importing Libraries**

# In[ ]:



import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import gc
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import missingno


# In[ ]:


train=pd.read_csv('../input/learn-together/train.csv')
test=pd.read_csv('../input/learn-together/test.csv')
sample_submission=pd.read_csv('../input/learn-together/sample_submission.csv')


# In[ ]:


X_train=train.copy()
x_test=test.copy()


# feature:
# 
# Euclidean_distance_to_hydrology: square root of the sum of square of vertical and horizontal distance

# In[ ]:


X_train['Euclidean_distance_to_hydrology']=(X_train.Horizontal_Distance_To_Hydrology**2+X_train.Vertical_Distance_To_Hydrology**2)**(1/2)
x_test['Euclidean_distance_to_hydrology']=(x_test.Horizontal_Distance_To_Hydrology**2 + x_test.Vertical_Distance_To_Hydrology**2)**(1/2)


# interaction_9amnoon: Product of hillshades at 9AM and Noon
# 

# In[ ]:


X_train['interaction_9amnoon']=X_train.Hillshade_9am*X_train.Hillshade_Noon
x_test['interaction_9amnoon']=x_test.Hillshade_9am*test.Hillshade_Noon


# cosine of slope :  used to partially model the relationships between hillshade

# In[ ]:


X_train['Cosine_slope']=pd.DataFrame(np.cos(X_train.Slope))
x_test['Cosine_slope']= pd.DataFrame(np.cos(x_test.Slope))


# **soils families**

# family                  : :                 soil_Type
# 
# Cantheral  : :  1                            
# 
# Ratake : : 2,4
# 
# vanet : :5
# 
# wetmore :: 6
# 
# limber : :8
# 
# troutville : :9
# 
# catamount :: 10,11,13,26,28,31,32,33
# 
# legault : :12,29,30
# 
# gateview:: 17
# 
# rogert :: 18
# 
# leighcan : : 21,22,23,24,25,27
# 
# bross : : 36
# 
# moran : : 38,39,40

# In[ ]:


X_train['Canteral_family']=X_train['Soil_Type1']
x_test['Canteral_family']=x_test['Soil_Type1']

X_train['Ratake_family'] = np.where((X_train['Soil_Type2'] ==1) | (X_train['Soil_Type4'] == 1), 1, 0)
x_test['Ratake_family'] = np.where((x_test['Soil_Type2'] ==1) | (x_test['Soil_Type4'] == 1), 1, 0)

X_train['Vanet_family'] = np.where((X_train['Soil_Type5'] ==1) , 1, 0)
x_test['Vanet_family'] = np.where((x_test['Soil_Type5'] ==1) , 1, 0)

X_train['Wetmore_family'] = np.where((X_train['Soil_Type6'] ==1) , 1, 0)
x_test['Wetmore_family'] = np.where((x_test['Soil_Type6'] ==1) , 1, 0)

X_train['Limber_family'] = np.where((X_train['Soil_Type8'] ==1) , 1, 0)
x_test['Limber_family'] = np.where((x_test['Soil_Type8'] ==1) , 1, 0)

X_train['Troutwill_family'] = np.where((X_train['Soil_Type9'] ==1) , 1, 0)
x_test['Troutwill_family'] = np.where((x_test['Soil_Type9'] ==1) , 1, 0)

X_train['Catamount_family'] = np.where((X_train['Soil_Type10'] ==1) | (X_train['Soil_Type11'] == 1) | (X_train['Soil_Type13'] == 1) | (X_train['Soil_Type26'] == 1) |
                                       (X_train['Soil_Type28'] == 1) | (X_train['Soil_Type31'] == 1) | (X_train['Soil_Type32'] == 1) | (X_train['Soil_Type33'] == 1), 1, 0)
x_test['Catamount_family'] = np.where((x_test['Soil_Type10'] ==1) | (x_test['Soil_Type11'] == 1) | (x_test['Soil_Type13'] == 1) | (x_test['Soil_Type26'] == 1) |
                                       (x_test['Soil_Type28'] == 1) | (x_test['Soil_Type31'] == 1) | (x_test['Soil_Type32'] == 1) | (x_test['Soil_Type33'] == 1), 1, 0)

X_train['Legualt_family'] = np.where((X_train['Soil_Type12'] ==1) | (X_train['Soil_Type29'] == 1) | (X_train['Soil_Type30']), 1, 0)
x_test['Legualt_family'] = np.where((x_test['Soil_Type12'] ==1) | (x_test['Soil_Type29'] == 1) | (x_test['Soil_Type30']), 1, 0)

X_train['Gateview_family'] = np.where((X_train['Soil_Type17'] ==1) , 1, 0)
x_test['Gateview_family'] = np.where((x_test['Soil_Type17'] ==1) , 1, 0)

X_train['Rogert_family'] = np.where((X_train['Soil_Type18'] ==1) , 1, 0)
x_test['Rogert_family'] = np.where((x_test['Soil_Type18'] ==1) , 1, 0)

X_train['Leighcan_family'] = np.where((X_train['Soil_Type21'] ==1) | (X_train['Soil_Type22'] == 1) | (X_train['Soil_Type23'] == 1) | (X_train['Soil_Type24'] == 1) |
                                       (X_train['Soil_Type25'] == 1) | (X_train['Soil_Type26'] == 1), 1, 0)
x_test['Leighcan_family'] = np.where((x_test['Soil_Type21'] ==1) | (x_test['Soil_Type22'] == 1) | (x_test['Soil_Type23'] == 1) | (x_test['Soil_Type24'] == 1) |
                                       (x_test['Soil_Type25'] == 1) | (x_test['Soil_Type27'] == 1), 1, 0)

X_train['Bross_family'] = np.where((X_train['Soil_Type36'] ==1) , 1, 0)
x_test['Bross_family'] = np.where((x_test['Soil_Type36'] ==1) , 1, 0)

X_train['Ratake_family'] = np.where((X_train['Soil_Type38'] ==1) | (X_train['Soil_Type39'] == 1) | (X_train['Soil_Type40'] ), 1, 0)
x_test['Ratake_family'] = np.where((x_test['Soil_Type38'] ==1) | (x_test['Soil_Type39'] == 1) | (X_train['Soil_Type40'] ), 1, 0)


# **Rock Type**

# 
# Rock_type :: soil_type
# 
# rubbly  : : 3,4,5,10,11,13
# 
# stony : : 6,2,9,12
# 
# extremely stony : : 1,24,25,27,28,29,30,31,32,33,34,36,37,38,39,40

# In[ ]:


X_train['Rubbly'] = np.where((X_train['Soil_Type3'] ==1) | (X_train['Soil_Type4'] == 1) | (X_train['Soil_Type5'] == 1) | (X_train['Soil_Type10'] == 1) |
                                       (X_train['Soil_Type11'] == 1) | (X_train['Soil_Type13'] == 1), 1, 0)
x_test['Rubbly'] = np.where((x_test['Soil_Type3'] ==1) | (x_test['Soil_Type4'] == 1) | (x_test['Soil_Type5'] == 1) | (x_test['Soil_Type10'] == 1) |
                                       (x_test['Soil_Type13'] == 1) | (x_test['Soil_Type11'] == 1), 1, 0)


X_train['stony'] = np.where((X_train['Soil_Type2'] ==1) | (X_train['Soil_Type6'] == 1) | (X_train['Soil_Type9'] == 1) | (X_train['Soil_Type12'] == 1), 1, 0)
x_test['stony'] = np.where((x_test['Soil_Type2'] ==1) | (x_test['Soil_Type6'] == 1) | (x_test['Soil_Type9'] == 1) | (x_test['Soil_Type12'] == 1), 1, 0)


X_train['extremely_stony'] = np.where((X_train['Soil_Type1'] ==1) | (X_train['Soil_Type24'] == 1) | (X_train['Soil_Type25'] == 1) | (X_train['Soil_Type27'] == 1) |
                                       (X_train['Soil_Type28'] == 1) | (X_train['Soil_Type29'] == 1) | (X_train['Soil_Type30'] == 1) | (X_train['Soil_Type31'] == 1) | (X_train['Soil_Type32'] == 1) | (X_train['Soil_Type33'] == 1) |
                                      (X_train['Soil_Type34'] == 1) | (X_train['Soil_Type36'] == 1) | (X_train['Soil_Type37'] == 1) | (X_train['Soil_Type38'] == 1) | (X_train['Soil_Type39'] == 1) | (X_train['Soil_Type40'] == 1), 1, 0)
x_test['extremely_stony'] = np.where((x_test['Soil_Type1'] ==1) | (x_test['Soil_Type24'] == 1) | (x_test['Soil_Type25'] == 1) | (x_test['Soil_Type27'] == 1) |
                                       (x_test['Soil_Type28'] == 1) | (x_test['Soil_Type29'] == 1) | (x_test['Soil_Type30'] == 1) | (x_test['Soil_Type31'] == 1) | (x_test['Soil_Type32'] == 1) | (x_test['Soil_Type33'] == 1) |
                                      (x_test['Soil_Type34'] == 1) | (x_test['Soil_Type36'] == 1) | (x_test['Soil_Type37'] == 1) | (x_test['Soil_Type38'] == 1) | (x_test['Soil_Type39'] == 1) | (x_test['Soil_Type40'] == 1), 1, 0)


# In[ ]:


X_train.shape


# 18 new features are added

# In[ ]:


plt.figure(figsize=(20,30))
plt.subplot(5, 2, 1)
fig = train.boxplot(column='Elevation')
fig.set_title('')
fig.set_ylabel('Elevation')
 
plt.subplot(5, 2, 2)
fig = train.boxplot(column='Aspect')
fig.set_title('')
fig.set_ylabel('Aspect')

plt.subplot(5, 2, 3)
fig = train.boxplot(column='Slope')
fig.set_title('')
fig.set_ylabel('Slope')
 
plt.subplot(5, 2, 4)
fig = train.boxplot(column='Horizontal_Distance_To_Hydrology')
fig.set_title('')
fig.set_ylabel('Horizontal_Distance_To_Hydrology')

plt.subplot(5, 2, 5)
fig = train.boxplot(column='Vertical_Distance_To_Hydrology')
fig.set_title('')
fig.set_ylabel('Vertical_Distance_To_Hydrology')
 
plt.subplot(5, 2, 6)
fig = train.boxplot(column='Horizontal_Distance_To_Roadways')
fig.set_title('')
fig.set_ylabel('Horizontal_Distance_To_Roadways')

plt.subplot(5, 2, 7)
fig = train.boxplot(column='Hillshade_9am')
fig.set_title('')
fig.set_ylabel('Hillshade_9am')
 
plt.subplot(5, 2, 8)
fig = train.boxplot(column='Hillshade_Noon')
fig.set_title('')
fig.set_ylabel('Hillshade_Noon')

plt.subplot(5, 2, 9)
fig = train.boxplot(column='Hillshade_3pm')
fig.set_title('')
fig.set_ylabel('Hillshade_3pm')
 
plt.subplot(5, 2, 10)
fig = train.boxplot(column='Horizontal_Distance_To_Fire_Points')
fig.set_title('')
fig.set_ylabel('Horizontal_Distance_To_Fire_Points')


# Clearly the columns other than elevation and aspect have outliers in them
# 
# i will be using 2 techniques for outliers handling
# 
# 1.top_coding and bottom_coding: in this the values which are out of the range are caped within it
# 
# 2.Discretisation: the intervals of all values are formed so that outliers could come in some interval
# 
# 1.Top_coding: Slope
# 
# 2.Bottom_coding:Hillshade_3pm
# 
# 3.Discretisation: all other columns

# Slope:
# 
# now i have to calculate the interquantile range(IQR) for the column

# In[ ]:


IQR = X_train.Slope.quantile(0.75) - X_train.Slope.quantile(0.25) #finding interquantile range
Lower_fence = X_train.Slope.quantile(0.25) - (IQR * 1.5)          #lower value
Upper_fence = X_train.Slope.quantile(0.75) + (IQR * 1.5)          #upper value
print('slope number outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
Slope = len(X_train[X_train.Slope>Upper_fence]) / np.float(len(X_train))
print('Number of Slope with values higher than {upperboundary}: {slope}'.format(upperboundary=Upper_fence, slope=Slope))


# the uppervalue of the quantile in slope is 40
# 
# now we have to cap all the values which are above 40 to 40

# In[ ]:


def top_code(df, variable, top):                            #function to top_code values
    return np.where(df[variable]>top, top, df[variable])
X_train['Slope']= top_code(X_train, 'Slope', 40)            #train_set is top_coded
x_test['Slope']= top_code(x_test,'Slope',40)  


# Bottom_coding : Hillshade_3pm

# In[ ]:


IQR2 = X_train.Hillshade_3pm.quantile(0.75) - X_train.Hillshade_3pm.quantile(0.25)
Lower_fence = X_train.Hillshade_3pm.quantile(0.25) - (IQR2 * 1.5)
Upper_fence = X_train.Hillshade_3pm.quantile(0.75) + (IQR2 * 1.5)
print('Hillshade_3pm number outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
Hillshades_3pm = len(X_train[X_train.Hillshade_3pm<Lower_fence]) / np.float(len(X_train))
print('Number of Hillshade_3pm with values lower than {lowerboundary}: {Hillshade_3pm}'.format(lowerboundary=Lower_fence, Hillshade_3pm=Hillshades_3pm))


# the lower value in inter quantile range is 14(approx)
# 
# we we have to cap the values lower than 14 to 14

# In[ ]:


def bottom_code(df, variable, bottom):                      #function to bottom values
    return np.where(df[variable]<bottom, bottom, df[variable])

#bottom coding Hillshade_3pm
X_train['Hillshade_3pm']=bottom_code(X_train,'Hillshade_3pm',14)
x_test['Hillshade_3pm']=bottom_code(x_test,'Hillshade_3pm',14)


# now we have top_coded and bottom coded respective columns
# 
# now i will use discretisation for the other columns which have outliers

# In[ ]:


def discretise(var,X_train,x_test):
#     #ploting figure before discritisation
#     fig = plt.figure()
#     fig = X_train.groupby([var])['Cover_Type'].mean().plot(figsize=(12,6))
#     fig.set_title('relationship between variable and target before discretisation')
#     fig.set_ylabel('Cover_Type')
    
    # find quantiles and discretise train set
    
    X_train[var], bins = pd.qcut(x=X_train[var], q=8, retbins=True, precision=3, duplicates='raise')
    x_test[var] = pd.cut(x = x_test[var], bins=bins, include_lowest=True)
    
    t1 = X_train.groupby([var])[var].count() / np.float(len(X_train))
    t3 = x_test.groupby([var])[var].count() / np.float(len(x_test))
    
    
    #plot to show distribution of values
#     temp = pd.concat([t1,t3], axis=1)
#     temp.columns = ['train', 'test']
#     temp.plot.bar(figsize=(12,6))
    
    #plot after discretisation
#     fig = plt.figure()
#     fig = X_train.groupby([var])['Cover_Type'].mean().plot(figsize=(12,6))
#     fig.set_title('Normal relationship between variable and target after discretisation')
#     fig.set_ylabel('Cover_Type')


# In[ ]:


try:
    X_train,x_test=discretise('Horizontal_Distance_To_Hydrology',X_train,x_test)
except TypeError:
    pass

try:
    X_train,x_test=discretise('Vertical_Distance_To_Hydrology',X_train,x_test)
except TypeError:
    pass

try:
    X_train,x_test=discretise('Horizontal_Distance_To_Roadways',X_train,x_test)
except TypeError:
    pass

try:
    X_train,x_test=discretise('Hillshade_9am',X_train,x_test)
except TypeError:
    pass

try:
    X_train,x_test=discretise('Hillshade_Noon',X_train,x_test)
except TypeError:
    pass

try:
    X_train,x_test=discretise('Horizontal_Distance_To_Fire_Points',X_train,x_test)
except TypeError:
    pass


# now thier are few missing values in the test set and we have to inpute them

# In[ ]:


def inpute(var,X_train,x_test):
    if x_test[var].isnull().sum()>0:
        x_test.loc[x_test[var].isnull(), var] = X_train[var].unique()[0]
        print("inputed column :: ",var)


# In[ ]:


for col in X_train.columns:
    if col !='Cover_Type':
        inpute(col,X_train,x_test)


# now the intervals are formed
# 
# before encoding them we have change their data type to object

# In[ ]:


for df in [X_train, x_test]:
    df.Horizontal_Distance_To_Hydrology = df.Horizontal_Distance_To_Hydrology.astype('O')
    df.Vertical_Distance_To_Hydrology = df.Vertical_Distance_To_Hydrology.astype('O')
    df.Horizontal_Distance_To_Roadways = df.Horizontal_Distance_To_Roadways.astype('O')
    df.Hillshade_9am = df.Hillshade_9am.astype('O')
    df.Hillshade_Noon = df.Hillshade_Noon.astype('O')
    df.Horizontal_Distance_To_Fire_Points = df.Horizontal_Distance_To_Fire_Points.astype('O')


# In[ ]:


def encode_categorical_variables(var, target):
        # make label to risk dictionary
        ordered_labels = X_train.groupby([var])[target].mean().to_dict()
        
        # encode variables
        X_train[var] = X_train[var].map(ordered_labels)
        x_test[var] = x_test[var].map(ordered_labels)


# In[ ]:


for var in ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Horizontal_Distance_To_Fire_Points']:
    print(var)
    encode_categorical_variables(var, 'Cover_Type')


# In[ ]:


y_train=train['Cover_Type']
X_train.drop(['Cover_Type','Id'],axis=1,inplace=True) #removing target and id column from train set
x_test.drop(['Id'],axis=1,inplace=True) 


# In[ ]:


print(X_train.shape,x_test.shape)


# ### Models

# In[ ]:


import lightgbm as lgb

from mlxtend.classifier import StackingCVClassifier

from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, 
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix


# ### Stacking

# In[ ]:


random_state = 1
random.seed(random_state)
np.random.seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

print('> Setting up classifiers...')
n_jobs = -1  # Use all processor cores to speed things up

ab_clf = AdaBoostClassifier(n_estimators=200,
                            base_estimator=DecisionTreeClassifier(
                                min_samples_leaf=3,
                                random_state=random_state),
                            random_state=random_state)

bg_clf = BaggingClassifier(n_estimators=200,
                           random_state=random_state)

gb_clf = GradientBoostingClassifier(n_estimators=400,
                                    min_samples_leaf=3,
                                    tol=0.1,
                                    verbose=0,
                                    random_state=random_state)

lg_clf = LGBMClassifier(n_estimators=400,
                        num_leaves=30,
                        verbosity=0,
                        random_state=random_state,
                        n_jobs=n_jobs)

rf_clf = RandomForestClassifier(n_estimators=885,
                                min_samples_leaf=3,
                                verbose=0,
                                random_state=random_state,
                                n_jobs=n_jobs)

xg_clf = XGBClassifier(n_estimators=400,
                       min_child_weight=3,
                       verbosity=0,
                       random_state=random_state,
                       n_jobs=n_jobs)

ensemble = [('ab', ab_clf),
            ('bg', bg_clf),
            ('gb', gb_clf),
            ('lg', lg_clf),
            ('rf', rf_clf),
            ('xg', xg_clf)]

stack = StackingCVClassifier(classifiers=[clf for label, clf in ensemble],
                             meta_classifier=rf_clf,
                             cv=5,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=random_state,
                             n_jobs=n_jobs)

# TODO: Find best parameters for each classifier

print('> Cross-validating classifiers...')
X_train = X_train.to_numpy()  # Converting to numpy matrices because...
x_test = x_test.to_numpy()  # ...XGBoost complains about dataframe columns

scores = dict()
for label, clf in ensemble:
    print('  -- Cross-validating {} classifier...'.format(label))
    score = cross_val_score(clf, X_train, y_train,
                            cv=5,
                            scoring='accuracy',
                            verbose=1,
                            n_jobs=n_jobs)
    scores[label] = score
    print('  -- {} : {:.3f} : {}'.format(label, np.mean(score), np.around(score, 3)))
    print()

print('> All cross-validation scores')
for label, score in scores.items():
    print('  -- {} : {:.3f} : {}'.format(label, np.mean(score), np.around(score, 3)))


# In[ ]:



print('> Fitting & predicting...')
stack = stack.fit(X_train, y_train)
prediction = stack.predict(x_test)


# In[ ]:


output = pd.DataFrame({'Id': test.Id,
                      'Cover_Type': prediction})
output.to_csv('sample_submission.csv', index=False)

