#!/usr/bin/env python
# coding: utf-8

# ## Kaggle Forest Cover Type Prediction
# ### Logistic regression, Random Forest, and LightGBM

# [Competition](https://www.kaggle.com/c/forest-cover-type-prediction). 
# In this competition you are asked to predict the forest cover type (the predominant kind of tree cover) from strictly cartographic variables (as opposed to remotely sensed data). The actual forest cover type for a given 30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data. Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is in raw form (not scaled) and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.
# 
# features (more info on [this](https://www.kaggle.com/c/forest-cover-type-prediction/data) competition page):

# * Elevation - Elevation in meters
# * Aspect - Aspect in degrees azimuth
# * Slope - Slope in degrees
# * Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
# * Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
# * Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
# * Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
# * Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
# * Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
# * Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
# * Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
# * Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
# * Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation (target)

# **Import libs and load data**

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


train = pd.read_csv('../input/forest-cover-type-prediction/train.csv',
                   index_col='Id')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv',
                  index_col='Id')


# In[ ]:


train.head(1).T


# In[ ]:


train['Cover_Type'].value_counts()


# In[ ]:


def write_to_submission_file(predicted_labels, out_file,
                             target='Cover_Type', index_label="Id", init_index=15121):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(init_index, 
                                                  predicted_labels.shape[0] + init_index),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


# **Perform train-test split**

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(
    train.drop('Cover_Type', axis=1), train['Cover_Type'],
    test_size=0.3, random_state=17)


# **Test logistic regression**

# In[ ]:


logit = LogisticRegression(C=1, solver='lbfgs', max_iter=500,
                           random_state=17, n_jobs=4,
                          multi_class='multinomial')
logit_pipe = Pipeline([('scaler', StandardScaler()), 
                       ('logit', logit)])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'logit_pipe.fit(X_train, y_train)')


# In[ ]:


logit_val_pred = logit_pipe.predict(X_valid)


# In[ ]:


accuracy_score(y_valid, logit_val_pred)


# In[ ]:


first_forest = RandomForestClassifier(
    n_estimators=100, random_state=17, n_jobs=4)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'first_forest.fit(X_train, y_train)')


# In[ ]:


forest_val_pred = first_forest.predict(X_valid)


# In[ ]:


accuracy_score(y_valid, forest_val_pred)


# In[ ]:


pd.DataFrame(first_forest.feature_importances_,
             index=X_train.columns, columns=['Importance']).sort_values(
    by='Importance', ascending=False)[:10]


# In[ ]:


lgb_clf = LGBMClassifier(random_state=17)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lgb_clf.fit(X_train, y_train)')


# In[ ]:


accuracy_score(y_valid, lgb_clf.predict(X_valid))


# **1 stage of hyper-param tuning: tuning model complexity**

# In[ ]:


param_grid = {'num_leaves': [7, 15, 31, 63], 
              'max_depth': [3, 4, 5, 6, -1]}


# In[ ]:


grid_searcher = GridSearchCV(estimator=lgb_clf, param_grid=param_grid, 
                             cv=5, verbose=1, n_jobs=4)


# In[ ]:


grid_searcher.fit(X_train, y_train)


# In[ ]:


grid_searcher.best_params_, grid_searcher.best_score_


# In[ ]:


accuracy_score(y_valid, grid_searcher.predict(X_valid))


# **2 stage of hyper-param tuning: convergence:**

# In[ ]:


num_iterations = 200
lgb_clf2 = LGBMClassifier(random_state=17, max_depth=-1, 
                          num_leaves=63, n_estimators=num_iterations,
                          n_jobs=1)

param_grid2 = {'learning_rate': np.logspace(-3, 0, 10)}
grid_searcher2 = GridSearchCV(estimator=lgb_clf2, param_grid=param_grid2,
                               cv=5, verbose=1, n_jobs=4)
grid_searcher2.fit(X_train, y_train)
print(grid_searcher2.best_params_, grid_searcher2.best_score_)
print(accuracy_score(y_valid, grid_searcher2.predict(X_valid)))


# In[ ]:


final_lgb = LGBMClassifier(n_estimators=200, num_leaves=63,
                           learning_rate=0.2, max_depth=-1,
                         n_jobs=4)


# In[ ]:


get_ipython().run_cell_magic('time', '', "final_lgb.fit(train.drop('Cover_Type', axis=1), train['Cover_Type'])")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'lgb_final_pred = final_lgb.predict(test)')


# In[ ]:


write_to_submission_file(lgb_final_pred, 
                         'lgb_forest_cover_type.csv')


# **Kaggle Public LB accuracy - 0.76851.**

# **Feature importance:**

# In[ ]:


pd.DataFrame(final_lgb.feature_importances_,
             index=X_train.columns, columns=['Importance']).sort_values(
    by='Importance', ascending=False)[:10]

