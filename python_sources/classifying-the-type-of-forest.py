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
    test_size=0.3, random_state=101)


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


train.columns


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
for col in train.columns:
    train[col]=LabelEncoder().fit(train[col]).transform(train[col])
model= DecisionTreeClassifier(criterion= 'entropy',max_depth = 1)
AdaBoost= AdaBoostClassifier(base_estimator= first_forest, n_estimators= 400,learning_rate=1)
boostmodel= AdaBoost.fit(X_train, y_train)
y_predict= boostmodel.predict(X_valid)
accuracy_score(y_valid, y_predict)


# In[ ]:





# By simply executing AdaBoost on our RandomForest CLassifier we were increase our accuracy to 0.85, which is really good.

# In[ ]:


write_to_submission_file(y_predict,'final answer.csv')


# In[ ]:




