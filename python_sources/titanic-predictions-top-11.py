#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np 
import pandas as pd 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_selection import SelectKBest, f_classif



from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

from keras.utils import to_categorical


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


trainingData=pd.read_csv('/kaggle/input/titanic/train.csv')
testData=pd.read_csv('/kaggle/input/titanic/test.csv')
sampleSubmission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[ ]:


trainingData.info()


# In[ ]:


trainingData


# In[ ]:


features=trainingData.drop(columns="Survived")
y=trainingData[['Survived']]

X_train_full, X_test_full, y_train, y_test = train_test_split(features, y, test_size = 0.20)


# In[ ]:


categorical_columns=[column_name for column_name in X_train_full.columns if X_train_full[column_name].dtype=="object"]
numerical_columns=[column_name for column_name in X_train_full.columns if X_train_full[column_name].dtype in ["int64", "float64"]]


# In[ ]:


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])


# In[ ]:


#using XGBoost
modelXGB = XGBClassifier(n_estimators=200,learning_rate=0.5)

#build the pipeline
pipelineXGB = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', modelXGB)
                             ])


# In[ ]:


#now we can fit the pipeline
pipelineXGB.fit(X_train_full, y_train)


# In[ ]:


predictionsXGB = pipelineXGB.predict(X_test_full)


# In[ ]:


# Calculate MAE
maeXGB = mean_absolute_error(predictionsXGB, y_test)
print("Mean Absolute Error:" , maeXGB)


# In[ ]:


#using RandomForest
modelRF = RandomForestClassifier(n_estimators=100)

#build the pipeline
pipelineRF = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', modelRF)
                             ])


# In[ ]:


pipelineRF.fit(X_train_full,y_train)
predictionsRF=pipelineRF.predict(X_test_full)
# Calculate MAE
maeRF = mean_absolute_error(predictionsRF, y_test)
print("Mean Absolute Error:" , maeRF)


# In[ ]:


#make the predictions
predictionsTest=pipelineRF.predict(testData)


# In[ ]:


submission = pd.DataFrame({'PassengerId':testData['PassengerId'],'Survived':predictionsTest})


# In[ ]:


submission


# In[ ]:


submission.Survived.unique()


# In[ ]:


filename = 'Titanic predictions 2.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

