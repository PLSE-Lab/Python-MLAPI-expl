#!/usr/bin/env python
# coding: utf-8

# # Introduction and imports
# Welcome to the notebook introducing Trelawney, a unified Python API for interpretation of Machine Learning Model. For more information about this package, you can find a Medium article about it here (https://medium.com/@antoine.redier2/introducing-trelawney-a-unified-python-api-for-interpretation-of-machine-learning-models-6fbc0a1fd6e7)
# 
# 
# The point of this kernel is not to achieve the best performance but to demonstate how to explain your model's prediction with the Trelawney package, with both overall importance of feature and local explanation of a single prediction.
# 
# In the first section of this notebook, we built the model to predict the survival of a given passenger on the Titanic. In the second section, we show you how to use the Trelawney package, both for global and local interpretation

# In[ ]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install trelawney')


# In[ ]:


train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()


# # 1. Modelling 

# This first section creates a model to predict whether a certain passenger will survive or not.

# ## 1.1 Feature engineering 

# In[ ]:


train = train.drop(["Name", "Ticket", "Fare"],axis=1)
test = test.drop(["Name", "Ticket", "Fare"],axis=1)


# In[ ]:


train.head()


# In[ ]:


train.isna().sum()


# In[ ]:


#fill the missing cabin values with mode
train["Cabin"] = train["Cabin"].fillna(str(train["Cabin"].mode().values[0]))
test["Cabin"] = test["Cabin"].fillna(str(test["Cabin"].mode().values[0]))


# In[ ]:


train["Cabin"] = train["Cabin"].apply(lambda x:str(x).replace(' ','')if ' ' in str(x) else str(x))
test["Cabin"] = test["Cabin"].apply(lambda x:str(x).replace(' ','')if ' ' in str(x) else str(x))


# In[ ]:


train["Deck"] = train["Cabin"].str.slice(0,1)
test["Deck"] = test["Cabin"].str.slice(0,1)


# In[ ]:


train = train.drop(["Cabin"],axis=1)
test = test.drop(["Cabin"],axis=1)


# In[ ]:


def impute_median(series):
    return series.fillna(series.median())


# In[ ]:


train.Age = train.Age.transform(impute_median)
test.Age = test.Age.transform(impute_median)


# In[ ]:


train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train['Is_Married'] = np.where(train['SibSp']==1, 1, 0)
test['Is_Married'] = np.where(test['SibSp']==1, 1, 0)

train.head()


# In[ ]:


train["Family_Size"] = train.SibSp + train.Parch
test["Family_Size"] = test.SibSp + test.Parch

train.head()


# In[ ]:


train['Elderly'] = np.where(train['Age']>=50, 1, 0)
train.head()


# In[ ]:


#Split the data set into independent(x) and dependent (y) data sets

y = train["Survived"].values.reshape(-1, 1)
x = train.iloc[:, 2:12]
x_test  = test.drop("PassengerId",axis=1).copy()


# In[ ]:


x.dtypes


# In[ ]:


x_test.dtypes


# In[ ]:


from collections import Counter


# In[ ]:


##### encode the categorical data values
from sklearn.preprocessing import LabelEncoder


labelEncoder_Y = LabelEncoder()
x.iloc[:,1] = labelEncoder_Y.fit_transform(x.iloc[:, 1].values)
x_test.iloc[:,1] = labelEncoder_Y.transform(x_test.iloc[:, 1].values)

x.iloc[:,5] = labelEncoder_Y.fit_transform(x.iloc[:, 5].values)
x_test.iloc[:,5] = labelEncoder_Y.transform(x_test.iloc[:, 5].values)

x.iloc[:,6] = labelEncoder_Y.fit_transform(x.iloc[:, 6].values)
x_test.iloc[:,6] = labelEncoder_Y.transform(x_test.iloc[:, 6].values)


# In[ ]:


x.dtypes


# In[ ]:


x_test.dtypes


# ## 1.2 Data Preprocessing

# In[ ]:


#split the data set
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y , test_size=0.25, random_state=42)


# In[ ]:


#scale the data(feature scaling)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = pd.DataFrame(sc.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
x_val = pd.DataFrame(sc.fit_transform(x_val), columns=x_val.columns, index=x_val.index)
y_train = pd.DataFrame(y_train, index=x_train.index)
y_val = pd.DataFrame(y_val, index=x_val.index)


# In[ ]:


x_train.shape, x_val.shape, y_train.shape, y_val.shape


# ## 1.3 Modeling with Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# In[ ]:


model = RandomForestClassifier(min_samples_leaf=3, max_depth=5, n_estimators=100)


# In[ ]:


model.fit(x_train, y_train)
print(metrics.classification_report(y_val, model.predict(x_val)))


# # 2. Model interpretation (Trelawney explanations start here)
# 
# ## 2.1 Global explanation
# 
# The first kind of analysis we can do to interpret our model is to provide a global explanation of it, namely which features are more influential for the model overall. Trelawney provides two types of global explainers:
# 
# - Shap which will use the Shap package to check which features influence each prediction and aggregate them
# - Surrogate explainer : This technique uses an interpretable model (Single Tree for us) to mimic the outputs of our black box model
# 
# ### 2.1.1 SHAP

# In[ ]:


from trelawney.shap_explainer import ShapExplainer

explainer = ShapExplainer()
explainer.fit(model, x_train, y_train)


# Feature importance of the model, according to SHAP

# In[ ]:


feature_importance_graph = explainer.graph_feature_importance(x_val)
feature_importance_graph.update_layout(title='Shap Feature Importance')
feature_importance_graph.show()


# ### 2.1.2 Surogate explainer

# In[ ]:


from trelawney.surrogate_explainer import SurrogateExplainer
from sklearn.tree import DecisionTreeClassifier

explainer = SurrogateExplainer(DecisionTreeClassifier(max_depth=4))
explainer.fit(model, x_train, y_train)


# **surogate decision tree**

# In[ ]:


from IPython.display import Image
explainer.plot_tree(out_path='./tree_viz')
Image('./tree_viz.png', width=1000, height=500)


# In[ ]:


explainer.adequation_score()


# here we can see that the surogate tree (that you can see in the graph) erxplains ~95% of predictions.
# By default the `adequation_score` metric uses accuracy but you can choose which ever suits you best:

# In[ ]:


from sklearn import metrics


# In[ ]:


explainer.adequation_score(metric=metrics.roc_auc_score)


# ## 2.2 Local explanations
# 
# The second type of explanations you can do are local explanations. Here we will try to understand specific predictions for a given observation of our model (rather than understanding the model as a whole)
# 
# For this we have two explainers available to us again:
# 
# - LIME, this uses the LIME package that creates local explainable models that approximate a model around a specifc prediction to understand how the model came to that prediction
# - SHAP, we can use the SHAP method again but instead of aggregating the SHAP values on a dataset, we use the values for the prediction that interest us

# In[ ]:


y_pred = pd.DataFrame(model.predict_proba(x_val)[:, 1], index=x_val.index)


# In[ ]:


most_probable = y_pred.idxmax()
biggest_false_positive = (y_pred - y_val).idxmax()
biggest_false_negative = (y_pred - y_val).idxmin()


# ## 2.2.1 lime

# In[ ]:


from trelawney.lime_explainer import LimeExplainer


# In[ ]:


explainer = LimeExplainer()
explainer.fit(model, x_train, y_train)


# the next three graph are the contribution of each feature to specific predictions using the LIME explainer:

# In[ ]:


x_val.loc[most_probable, :]


# In[ ]:


lime_explanation_graph = explainer.graph_local_explanation(x_val.loc[most_probable, :])
lime_explanation_graph.update_layout(title='Lime individual prediction interpretation')
lime_explanation_graph.show()


# In[ ]:


x.loc[biggest_false_positive, :]


# In[ ]:


lime_explanation_graph = explainer.graph_local_explanation(x_val.loc[biggest_false_positive, :])
lime_explanation_graph.update_layout(title='Lime individual prediction interpretation')
lime_explanation_graph.show()


# In[ ]:


x.loc[biggest_false_negative, :]


# In[ ]:


lime_explanation_graph = explainer.graph_local_explanation(x_val.loc[biggest_false_negative, :])
lime_explanation_graph.update_layout(title='Lime individual prediction interpretation')
lime_explanation_graph.show()


# ## 2.2.2 SHAP

# In[ ]:


from trelawney.shap_explainer import ShapExplainer

explainer = ShapExplainer()
explainer.fit(model, x_train, y_train)


# the next three graph are the contribution of each feature to the same three predictions using the SHAP explainer:

# In[ ]:


shap_explanation_graph = explainer.graph_local_explanation(x_val.loc[most_probable, :])
shap_explanation_graph.update_layout(title='SHAP individual prediction interpretation')
shap_explanation_graph.show()


# In[ ]:


shap_explanation_graph = explainer.graph_local_explanation(x_val.loc[biggest_false_positive, :])
shap_explanation_graph.update_layout(title='SHAP individual prediction interpretation')
shap_explanation_graph.show()


# In[ ]:


shap_explanation_graph = explainer.graph_local_explanation(x_val.loc[biggest_false_negative, :])
shap_explanation_graph.update_layout(title='SHAP individual prediction interpretation')
shap_explanation_graph.show()

