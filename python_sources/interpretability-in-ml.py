#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pydotplus lime')


# In[ ]:


import lime
import lime.lime_tabular

from sklearn.tree import (
    ExtraTreeClassifier,
    DecisionTreeClassifier,
    export_graphviz
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from sklearn.externals.six import StringIO
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder
)
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt

from IPython.display import Image

import pydotplus

import seaborn as sns
import pandas as pd 
import numpy as np


# In[ ]:


df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')


# # Interpretability Examples
# 
# ## Dataset - UCI Heart Disease
# 
# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient.
# 
# ## Features
# 
# > 1. age
# > 2. sex
# > 3. chest pain type (4 values)
# > 4. resting blood pressure
# > 5. serum cholestoral in mg/dl
# > 6. fasting blood sugar > 120 mg/dl
# > 7. resting electrocardiographic results (values 0,1,2)
# > 8. maximum heart rate achieved
# > 9. exercise induced angina
# > 10. oldpeak = ST depression induced by exercise relative to rest
# > 11. the slope of the peak exercise ST segment
# > 12. number of major vessels (0-3) colored by flourosopy
# > 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# We will change the name of the columns to something that makes more sense:|

# In[ ]:


df.columns = [
    'age',
    'sex',
    'chest_pain_type',
    'resting_blood_pressure',
    'cholesterol',
    'fasting_blood_sugar',
    'rest_ecg',
    'max_heart_rate_achieved',
    'exercise_induced_angina',
    'st_depression',
    'st_slope',
    'num_major_vessels',
    'thalassemia',
    'target'
]


# # Exploratory Data Analysis
# 
# Just exploring a little bit the data to know how it looks like...

# In[ ]:


df.head(5)


# You can see that the categorial features have been transformed to integers, this will not help us when dealing with the model, so let's transform the features into its nominal values.

# In[ ]:


df['sex'][df['sex'] == 0] = 'female'
df['sex'][df['sex'] == 1] = 'male'

df['chest_pain_type'][df['chest_pain_type'] == 1] = 'typical angina'
df['chest_pain_type'][df['chest_pain_type'] == 2] = 'atypical angina'
df['chest_pain_type'][df['chest_pain_type'] == 3] = 'non-anginal pain'
df['chest_pain_type'][df['chest_pain_type'] == 4] = 'asymptomatic'

df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

df['rest_ecg'][df['rest_ecg'] == 0] = 'normal'
df['rest_ecg'][df['rest_ecg'] == 1] = 'ST-T wave abnormality'
df['rest_ecg'][df['rest_ecg'] == 2] = 'left ventricular hypertrophy'

df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'
df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'

df['st_slope'][df['st_slope'] == 1] = 'upsloping'
df['st_slope'][df['st_slope'] == 2] = 'flat'
df['st_slope'][df['st_slope'] == 3] = 'downsloping'

df['thalassemia'][df['thalassemia'] == 1] = 'normal'
df['thalassemia'][df['thalassemia'] == 2] = 'fixed defect'
df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'


# In[ ]:


df.head(5)


# In[ ]:


print(f"{df['target'].sum()/df.shape[0]:.2f} is the proportion of healthy and unhealthy patients")


# Let's try visualizing the data to see if there are some features that already look interesting...

# In[ ]:


sns.set(style="whitegrid")


# In[ ]:


sns.barplot(x='sex', y='age', data=df, hue='target')


# In[ ]:


sns.barplot(x='sex', y='st_depression', data=df, hue='target')


# Because we have categorical data we must convert them to a 'vector' of values...

# In[ ]:


df = pd.get_dummies(df, drop_first=True)


# In[ ]:


X = df.drop('target', axis=1)
y = df['target']


# Splitting data for training/testing

# In[ ]:


seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), 
    df['target'], 
    test_size=0.2, 
    stratify=df['target'], 
    random_state=seed
)


# # Building models and trying to interpret them...

# ## Decision Tree
# 
# Decision trees are widely used in the medicine for structured data because they can be easily transformed into a set of if-else statements and, therefore, are very easy to interpret.
# 
# C4.5 and CART are some widely used implementations of Decision Trees.
# 
# Because we are only focusing on building 'toy' models we will not focus on the metrics (like using Cross-Validation to optimize some parameters).

# In[ ]:


tree = DecisionTreeClassifier(random_state=seed, max_depth=3)  # we set max_depth to 3 using a rule of thumb for visualization
tree.fit(X_train, y_train)


# In[ ]:


y_pred = tree.predict(X_test)


# In[ ]:


target_labels = ['healthy', 'unhealthy']

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, columns=target_labels, index=target_labels)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})


# In[ ]:


print(classification_report(y_test, y_pred))


# ### Let's visualize how the decision tree looks like

# In[ ]:


dot_data = StringIO()

export_graphviz(
    tree,
    out_file=dot_data,
    filled=True,
    rounded=True, 
    class_names=target_labels, 
    special_characters=True, 
    feature_names=df.drop('target', axis=1).columns
)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())


# ## What if we want to learn about feature importance?
# 
# Let's use an extreme tree model (more complex than a decision tree) to learn more about feature importance!

# # Extreme Tree
# 
# Extreme Trees remember Random Forests with two main differences:
# 
# 1. There is no bootstraping of data, that is, no data is replaced back when sampling
# 2. The trees do not look for optimal splits, they select the splits randomly

# In[ ]:


xtra_tree = ExtraTreeClassifier(random_state=seed, max_depth=3)
xtra_tree.fit(X_train, y_train)


# In[ ]:


y_pred = xtra_tree.predict(X_test)


# In[ ]:


target_labels = ['healthy', 'unhealthy']

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, columns=target_labels, index=target_labels)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


df_feature_importance = pd.DataFrame({"Feature": X_train.columns, "Feature importance": xtra_tree.feature_importances_})


# In[ ]:


plt.figure(figsize=(16, 6))
sns.barplot(
    x='Feature importance',
    y='Feature',
    data=df_feature_importance.sort_values(by='Feature importance', ascending=False)
)


# ### What if I want to understand local predictions?
# 
# Here we can see the global effect of features, but what if we want to explain a local prediction?

# ## We can use LIME!
# 
# LIME explain local predictions using:
# 
# 1. explainable model - to explain the black box model we need to use a model that we can understand
# 2. perturbations - LIME generates new instances perturbating the instance in interest (it uses a Gaussian distribution)
# 3. kernel - estimate proximity to the instance in insterest

# In[ ]:


explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, 
    feature_names=X_train.columns, 
    discretize_continuous=False,
    mode='classification'
)


# In[ ]:


instance = X_test.iloc[0]
exp = explainer.explain_instance(instance.to_numpy(), xtra_tree.predict_proba)


# In[ ]:


print(f"True value: {y_test.iloc[0]}")


# In[ ]:


exp.show_in_notebook(show_table=True, show_all=True)


# #### References:
# 
# Lime-Python: https://github.com/marcotcr/lime
# 
# "Why Should I Trust You?": Explaining the Predictions of Any Classifier: https://arxiv.org/abs/1602.04938
# 
# Extremely randomized trees: https://link.springer.com/article/10.1007/s10994-006-6226-1

# ## To Do
# 
# SHAP values!
