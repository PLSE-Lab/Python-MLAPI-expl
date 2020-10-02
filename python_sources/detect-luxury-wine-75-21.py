#!/usr/bin/env python
# coding: utf-8

# ### Work in progress...

# ### 1. Import

# In[ ]:


import os


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[ ]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier


# ### 2. Read data

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')


# In[ ]:


df = df.drop(['Unnamed: 0'], axis=1)


# In[ ]:


print('Shape: ', df.shape)
df.head()


# ### 3. Visualization
# 
# Note:
# I did visualization with [Tableau Public](https://public.tableau.com)

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1580309606698' style='position: relative'>\n    <noscript>\n        <a href='https:&#47;&#47;www.kaggle.com&#47;zynicide&#47;wine-reviews#winemag-data-130k-v2.csv'>\n            <img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Wi&#47;WineDashboard_15802909636010&#47;WineDashboard&#47;1_rss.png' style='border: none' />\n        </a>\n    </noscript>\n    <object class='tableauViz'  style='display:none;'>\n        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> \n        <param name='embed_code_version' value='3' /> \n        <param name='site_root' value='' />\n        <param name='name' value='WineDashboard_15802909636010&#47;WineDashboard' />\n        <param name='tabs' value='yes' /><param name='toolbar' value='yes' />\n        <param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Wi&#47;WineDashboard_15802909636010&#47;WineDashboard&#47;1.png' /> \n        <param name='animate_transition' value='yes' />\n        <param name='display_static_image' value='yes' />\n        <param name='display_spinner' value='yes' />\n        <param name='display_overlay' value='yes' />\n        <param name='display_count' value='yes' />\n    </object></div>                \n    <script type='text/javascript'>                    \n        var divElement = document.getElementById('viz1580309606698');                    \n        var vizElement = divElement.getElementsByTagName('object')[0];                    \n        if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='1500px';vizElement.style.maxWidth='100%';vizElement.style.minHeight='850px';vizElement.style.maxHeight=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='1500px';vizElement.style.maxWidth='100%';vizElement.style.minHeight='850px';vizElement.style.maxHeight=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.minHeight='1300px';vizElement.style.maxHeight=(divElement.offsetWidth*1.77)+'px';}                     \n        var scriptElement = document.createElement('script');                    \n        scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    \n        vizElement.parentNode.insertBefore(scriptElement, vizElement);                \n    </script>")


# ### 4. Data preparation

# In[ ]:


df.head()


# **About task:**
# 
# I will learn model for binary classification:
# * wine with point >89 - **Luxury Wine**
# * wine with point <=89 - **Avg wine**
# 
# As 1st step, we will use the next columns:
# * country
# * price
# * province
# * variety

# In[ ]:


# create target

df.loc[df['points'] <= 89, 'points'] = 0
df.loc[df['points'] > 89, 'points'] = 1


# In[ ]:


# count of target

print('Avg wine: ', len(df.loc[df['points']==0]))
print('Luxury Wine: ', len(df.loc[df['points']==1]))


# In[ ]:


# select columns

columns = ['country', 'price', 'province', 'variety', 'points']
data = df[columns]


# In[ ]:


data.head()


# In[ ]:


data.isna().sum()


# In[ ]:


# replace nan values with average of columns

data.price = data.price.fillna(data.price.mean())


# In[ ]:


# replace nan values with Unknown

data = data.fillna('Unknown')


# In[ ]:


data.isna().sum() 


# In[ ]:





# In[ ]:


# one hot encoding

country = pd.get_dummies(data['country'], prefix = 'country')
province = pd.get_dummies(data['province'], prefix = 'province')
variety = pd.get_dummies(data['variety'], prefix = 'variety')


# In[ ]:


target = data['points']
data = data['price']
data = pd.concat([data, country], axis=1)
data = pd.concat([data, province], axis=1)
data = pd.concat([data, variety], axis=1)


# In[ ]:


data.shape


# In[ ]:


# Split data into random train and test subsets

X_train, X_test, Y_train, Y_test = train_test_split(
    data, 
    target, 
    test_size=0.33, 
    random_state=42
)


# ### Modeling

# In[ ]:


def report(model):
    Y_pred = model.predict(X_test)
    return print(classification_report(Y_test, Y_pred))


# ###### Logistic Regression

# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

acc_log = round(logreg.score(X_test, Y_test) * 100, 2)
acc_log


# In[ ]:


report(logreg)


# In[ ]:


coeff_df = pd.DataFrame(data.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# ###### Linear SVC

# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

acc_linear_svc = round(linear_svc.score(X_test, Y_test) * 100, 2)
acc_linear_svc


# In[ ]:


report(linear_svc)


# ### Decision Tree

# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
acc_decision_tree


# In[ ]:


report(decision_tree)


# ### Random Forest

# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)
acc_random_forest


# In[ ]:


report(random_forest)


# ### AdaBoostClassifier

# In[ ]:


bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(X_train, Y_train)

acc_bdt = round(bdt.score(X_test, Y_test) * 100, 2)
acc_bdt


# In[ ]:


report(bdt)


# ### GradientBoostingClassifier

# In[ ]:


clf_gb = GradientBoostingClassifier(n_estimators=100, 
                                 max_depth=1, 
                                 random_state=0)
clf_gb.fit(X_train, Y_train)

acc_clf_gb = round(clf_gb.score(X_test, Y_test) * 100, 2)
acc_clf_gb


# In[ ]:


report(clf_gb)


# ### MLPClassifier

# In[ ]:


mlp = MLPClassifier(solver='lbfgs', 
                    alpha=1e-5, 
                    hidden_layer_sizes=(21, 2), 
                    random_state=1)

mlp.fit(X_train, Y_train)

acc_mlp = round(mlp.score(X_test, Y_test) * 100, 2)
acc_mlp


# In[ ]:


report(mlp)


# ### Evalution models

# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 
              'Random Forest',   
              'Linear SVC', 
              'Decision Tree',
             'AdaBoostClassifier', 'GradientBoostingClassifier'],
    'Score': [acc_log, 
              acc_random_forest,   
              acc_linear_svc, acc_decision_tree,
             acc_bdt, acc_clf_gb]})
models.sort_values(by='Score', ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




