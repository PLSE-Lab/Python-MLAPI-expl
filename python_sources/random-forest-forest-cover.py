#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


df.columns


# ### Feature Engineering

# In[ ]:


# Given the horizontal & vertical distance to hydrology
# It will be more intuitive to obtain the euclidean distance
# sqrt{(verticaldistance)^2 + (horizontaldistance)^2}

from math import sqrt

df['Euclid_Distance_To_Hydrology'] = df.apply(lambda x: abs(sqrt(x['Horizontal_Distance_To_Hydrology']**2 +                             x['Vertical_Distance_To_Hydrology']**2)), axis=1)


# ### Basic Random Forest

# In[ ]:


cols = [i for i in df.columns if i not in ['Id', 'Cover_Type']]

predictor = df[cols] #exclude id & target columns
target = df['Cover_Type']

train_predictor, test_predictor, train_target, test_target = train_test_split(predictor, target, test_size=0.2)


# In[ ]:


model = RandomForestClassifier().fit(train_predictor, train_target)
predictions = model.predict(test_predictor)


# ### Accuarcy

# In[ ]:


metrics.accuracy_score(test_target, predictions)


# ### Feature Importance

# In[ ]:


df2= pd.DataFrame(model.feature_importances_, index=cols)
df2 = df2.sort_values(by=0,ascending=False)
df2.columns = ['feature importance']
df2


# We can see that __Soil_Type7, Soil_Type15__ have 0 low feature importance.
# 
# Will dump them in final model so that processing can be faster.

# ### Confusion Matrix

# In[ ]:


def forest(x):
    if x==1:
        return 'Spruce/Fir'
    elif x==2:
        return 'Lodgepole Pine'
    elif x==3:
        return 'Ponderosa Pine'
    elif x==4:
        return 'Cottonwood/Willow'
    elif x==5:
        return 'Aspen'
    elif x==6:
        return 'Douglas-fir'
    elif x==7:
        return 'Krummholz'

# Create pd Series for Original
Original = test_target.apply(lambda x: forest(x)).reset_index(drop=True)
Original.name = 'Original'

# Create pd Series for Predicted
Predicted = pd.DataFrame(predictions, columns=['Predicted'])
Predicted = Predicted['Predicted'].apply(lambda x: forest(x))


# In[ ]:


confusion = pd.crosstab(Original, Predicted)
confusion


# In[ ]:


plt.figure(figsize=(10, 5))
sns.heatmap(confusion,annot=True,cmap=sns.cubehelix_palette(8));


# ### Grid Search Hyper-Parameters

# In[ ]:


# dump columns with 0 importance
for i in ['Soil_Type7', 'Soil_Type15']:
    del predictor[i]


# In[ ]:


model = RandomForestClassifier()

# removed some parameter queries so can upload notebook faster
grid_values = {'n_estimators':[200],
                'max_features':[0.2,0.5],
                'max_depth':[60],
                "criterion": ["gini"]}
grid = GridSearchCV(model, param_grid = grid_values, cv=5, n_jobs=-1)
grid.fit(predictor, target)

print(grid.best_params_)
print(grid.best_score_)


# ### Rerun Model with Entire Train Set

# In[ ]:


model = RandomForestClassifier(n_estimators = grid.best_params_['n_estimators'],                                max_features = grid.best_params_['max_features'],                                max_depth = grid.best_params_['max_depth'],                                criterion = grid.best_params_['criterion'])
model.fit(predictor, target)


# ### Submission

# In[ ]:


test = pd.read_csv('../input/test.csv')


# In[ ]:


# dump columns with 0 importance
for i in ['Soil_Type7', 'Soil_Type15']:
    del test[i]


# In[ ]:


# Feature Engineering
test['Euclid_Distance_To_Hydrology'] = test.apply(lambda x: abs(sqrt(x['Horizontal_Distance_To_Hydrology']**2 +                               x['Vertical_Distance_To_Hydrology']**2)), axis=1)


# In[ ]:


cols = [i for i in cols if i not in ['Soil_Type7', 'Soil_Type15']]

predict_real_test = model.predict(test[cols])
submit = pd.concat([test['Id'], pd.Series(predict_real_test)],axis=1)
submit.columns = columns=['Id','Cover_Type']
submit.to_csv('submission.csv',index=False)

