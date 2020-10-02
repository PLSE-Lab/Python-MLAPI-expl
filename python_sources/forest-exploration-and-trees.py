#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# This kernel is dedicated to extensive EDA of Forest Cover Type Challenge competition as well as feature engineering.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')


# ## Data overview

# In[ ]:


train.head()


# In[ ]:


train.shape, test.shape


# In[ ]:


train.describe(include='all')


# In[ ]:


train.Cover_Type.value_counts()


# There are no missing values and 55 features, but 40 of them share "Soil_Type" name. Maybe it is possible to do something about it. There are 7 classes which have the same number of samples in train.

# ## Feature analysis
# We saw some information about features, so let's now analyze them in more details.

# ### Elevation

# In[ ]:


plt.hist(train['Elevation']);
plt.title('Elevation distribution.');


# As we can see, distribution of values is almost normal.

# ### Slope and aspect

# In[ ]:


train.groupby('Cover_Type').Aspect.mean()


# In[ ]:


for col in ['Aspect', 'Slope']:
    sns.violinplot(data=train, x='Cover_Type', y=col)  
    plt.show()


# We can see that distribution of values doesn't differ too much between different cover types.

# In[ ]:


(train['Soil_Type7'] == 0).all(), (train['Soil_Type15'] == 0).all()


# Two columns have only zero values in train, so I drop them.

# In[ ]:


train.drop(['Soil_Type7', 'Soil_Type15'], axis=1, inplace=True)
test.drop(['Soil_Type7', 'Soil_Type15'], axis=1, inplace=True)


# In[ ]:


sns.pairplot(train[['Elevation', 'Aspect', 'Slope',
       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
       'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
       'Wilderness_Area4', 'Cover_Type']], hue='Cover_Type')


# In[ ]:


# great features from this kernel: https://www.kaggle.com/rohandx1996/pca-fe-data-viz-top-10
####################### Train data #############################################
train['HF1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
train['HF2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HR1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['HR2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['FR1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['FR2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
train['ele_vert'] = train.Elevation-train.Vertical_Distance_To_Hydrology

train['slope_hyd'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
train.slope_hyd=train.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
train['Mean_Amenities']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2 
####################### Test data #############################################
test['HF1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
test['HF2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HR1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['HR2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['FR1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['FR2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])
test['ele_vert'] = test.Elevation-test.Vertical_Distance_To_Hydrology

test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

#Mean distance to Amenities 
test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2


# We can see that some classes are clearly recognized with some combinations of features. It's time for modelling now.

# ## Modelling

# In[ ]:


feature = [col for col in train.columns if col not in ['Cover_Type','Id']]
X = train[feature]
X_test = test[feature]
y = train['Cover_Type']


# In[ ]:


forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
scores = cross_val_score(forest, X, y, scoring='accuracy', n_jobs=-1, cv=3)
print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))


# In[ ]:


etc = ExtraTreesClassifier(n_estimators=300, n_jobs=-1)
scores = cross_val_score(etc, X, y, scoring='accuracy', n_jobs=-1, cv=3)
print('Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))


# LGB requires labels to start from 0, while our labels start from 0. So I'll subtruct 1 while training and then I'll add 1 to the predictions.

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y-1, test_size=0.20, random_state=42)
params = {'learning_rate': 0.05, 'max_depth': 13, 'boosting': 'gbdt', 'objective': 'multiclass', 'num_class': 7,
          'metric': ['multi_logloss'], 'is_training_metric': True, 'seed': 19, 'num_leaves': 256, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
          'bagging_freq': 5, 'lambda_l1': 4, 'lambda_l2': 4, 'num_threads': 12}

model = lgb.train(params, lgb.Dataset(X_train, label=y_train),10000,
                           lgb.Dataset(X_valid, label=y_valid),
                           verbose_eval=100, early_stopping_rounds=100)


# In[ ]:


accuracy_score(y_valid, np.round(np.argmax(model.predict(X_valid), axis=1)).astype(int))


# In[ ]:


forest.fit(X, y);
etc.fit(X, y);


# In[ ]:


pred_forest = forest.predict(X_test)
pred_etc = etc.predict(X_test)
pred_ldb = np.round(np.argmax(model.predict(X_test), axis=1)).astype(int) + 1


# In[ ]:


sub['Cover_Type'] = np.round((pred_forest + pred_etc + pred_ldb) / 3).astype(int)
sub.to_csv("blend.csv", index=False)

