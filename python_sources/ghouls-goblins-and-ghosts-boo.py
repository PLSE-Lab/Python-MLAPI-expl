#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl

import os
print(os.listdir("../input"))


# > ### Load data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


test_id = test['id'] # save for submission
del train['id']
del test['id']


# > ### Check unique possible classes

# In[ ]:


train['type'].unique(), train['color'].unique()


# > ### Some data analysis

# In[ ]:


sns.violinplot(x='bone_length', y='type', data=train)


# We can see that **Ghouls** have the longest bones, while **Goblins** and **Ghosts** have shorter ones.

# In[ ]:


sns.boxplot(x='hair_length', y='type', data=train)


# **Ghouls** also have the longest hair from the three classes

# In[ ]:


sns.pairplot(train)


# One can notice that creatures that have **more soul** also tend to have **longer bones** and **longer hair** *:D*

# > ### One-Hot encode color

# In[ ]:


from category_encoders import OneHotEncoder

encoder = OneHotEncoder(cols=['color'], use_cat_names=True)

train = encoder.fit_transform(train)
test = encoder.fit_transform(test)


# In[ ]:


train.head()


# > ### Label Encode target type

# In[ ]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(train['type'])

print(encoder.classes_)

train['type_no'] = encoder.transform(train['type'])


# In[ ]:


train.head()


# In[ ]:


sns.heatmap(train.corr(), xticklabels=list(train), yticklabels=list(train))


# The *heat map* also suggests that **having more** *soul* correlates with **higher** *hair* and *bone* lengths.

# > ### Data preparation for training and predictions

# In[ ]:


target = train['type_no'] # for visualizations
target_string = train['type'] # for final predictions

del train['type']
del train['type_no']

target.head()


# In[ ]:


from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(train, target, test_size=0.2, random_state=42)


# > ### Visualizing decision boundaries

# In[ ]:


from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools

def decisions(classifier, features):
    classifier.fit(train_data[features], train_target)
    ax = plot_decision_regions(test_data[features].values, test_target.values, clf=classifier, legend=2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['Ghost', 'Ghoul', 'Goblin'], framealpha=0.3, scatterpoints=1)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()
    
def grid_decisions(classifiers, classifier_names, features):
    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(10,8))

    for clf, lab, grd in zip(classifiers,classifier_names, itertools.product([0, 1], repeat=2)):
        clf.fit(train_data[features], train_target)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(test_data[features].values, test_target.values, clf=clf, legend=2)
        handles, labels = fig.get_legend_handles_labels()
        fig.legend(handles, ['Ghost', 'Ghoul', 'Goblin'], framealpha=0.3, scatterpoints=1)
        plt.title(lab)

    plt.show()


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

decisions(DecisionTreeClassifier(), ['hair_length', 'bone_length'])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

clfs = [RandomForestClassifier(), AdaBoostClassifier(), SVC(), KNeighborsClassifier()]
labels = ['Random Forest', 'Ada Boost', 'Support Vector', 'K-Neighbors']

grid_decisions(clfs, labels, ['hair_length', 'bone_length'])


# In[ ]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= 'https://cdn-images-1.medium.com/max/1600/1*JZbxrdzabrT33Yl-LrmShw.png', width=750, height=750)


# >

# > ### Evaluate initial accuracy of chosen models

# In[ ]:


train.head()


# In[ ]:


from sklearn.metrics import accuracy_score

clfs = [RandomForestClassifier(), AdaBoostClassifier(), SVC(), KNeighborsClassifier()]
labels = ['Random Forest', 'Ada Boost', 'Support Vector', 'K-Neighbors']

for model, name in zip(clfs, labels):
    model.fit(train_data, train_target)
    predictions = model.predict(test_data)
    print('{} accuracy is: {}'.format(name, accuracy_score(test_target, predictions)))


# While rerunning above cell multiple times the **SVC** seems to be the most performant, with **Ada Boost** coming in second most of the time.

# > ### Tweaking hyperparameters with Cross-Validation

# In[ ]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= 'https://scikit-learn.org/stable/_images/grid_search_cross_validation.png', width=500, height=500)


# In[ ]:


from sklearn.model_selection import GridSearchCV

params = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1]
}

grid_search = GridSearchCV(SVC(), params, cv=5)

grid_search.fit(train, target)

grid_search.best_params_


# > ### A look at what's important for our model

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

importance_model = SVC(C=10, gamma=0.1, probability=True)
importance_model.fit(train, target)

perm = PermutationImportance(importance_model, random_state=42).fit(test_data, test_target)
eli5.show_weights(perm, feature_names=test_data.columns.tolist())


# In[ ]:


import shap

data_for_prediction = test_data.iloc[0]

k_explainer = shap.KernelExplainer(importance_model.predict_proba, train_data)
k_shap_values = k_explainer.shap_values(data_for_prediction)

shap.initjs()
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)


# In[ ]:


test_target.iloc[0], target_string.iloc[0]


# > ### Fitting final model on all of the training data

# In[ ]:


model = SVC(C=10, gamma=0.1)
model.fit(train, target_string)


# > ### Make final predictions, aggregate predicted types with ids and export as CSV

# In[ ]:


predictions = model.predict(test)
predictions[:10]


# In[ ]:


submission = pd.DataFrame({'id': test_id, 'type': predictions})
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)

