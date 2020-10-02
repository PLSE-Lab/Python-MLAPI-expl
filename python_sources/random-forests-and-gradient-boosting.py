#!/usr/bin/env python
# coding: utf-8

# #  Random Forests and Gradient Boosting on Mushroom Classification
# 
# This is my first kernel on kaggle. This kernel is designed to classify mushrooms as edible or non edible based on the various features given.
# I have used classification models:
# * Random Forest Classifier
# * Gradient Boosted Classifier
# 
# This dataset has only categorical features and as I don't have any domain knowledge about mushrooms, I've skipped feature engineering for now.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


df = pd.read_csv("../input/mushrooms.csv")


# In[ ]:


df.head()


# In[ ]:


abt = pd.get_dummies(df)


# In[ ]:


abt.head()


# In[ ]:


#Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Building everything
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix

#Saving the model
import pickle


# In[ ]:


df["class"].replace(["e", "p"], [1, 0], inplace= True)


# In[ ]:


# Create separate object for target variable
y = df["class"]
# Create separate object for input features
X = abt.drop(["class_e", "class_p"], axis= 1).astype(float)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)


# In[ ]:


print(len(X_train), len(X_test), len(y_train), len(y_test))


# In[ ]:


pipelines = {
    "rf": make_pipeline(StandardScaler(), RandomForestClassifier(random_state=123)),
    "gb": make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=123))
}


# In[ ]:


rf_hyperparameters = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_features": ["auto", "sqrt", 0.33]
}
gb_hyperparameters = {
    "gradientboostingclassifier__n_estimators": [100, 200],
    'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingclassifier__max_depth': [1, 3, 5]
}
hyperparameters = {"rf": rf_hyperparameters,
                   "gb": gb_hyperparameters}


# In[ ]:


# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    # Create cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv= 10, n_jobs= -1)
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')


# In[ ]:


for name, model in fitted_models.items():
    print(name, model.best_score_)


# In[ ]:


for name, model in fitted_models.items():
    print(name)
    print("-----------")
    pred = model.predict(X_test)
    print('Acc:', accuracy_score(y_test, pred))
    print("cm:\n", confusion_matrix(y_test, pred))


# In[ ]:


with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)


# ### Finished!
