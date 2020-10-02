#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Dataset

# ## **0. Before we begin**

# Please  **comment** or **upvote** this kernel.

# ### Kernel goals:
# 
# * Data exploration
# * Find important features for L1-regularized Logistic regression
# * Propose correct scoring metrics for this dataset
# * Fight off overly-optimistic score
# * Compare results of various classifiers

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# ## **1. Data exploration**

# In[ ]:


sns.set(style='whitegrid')


# In[ ]:


data = pd.read_csv('../input/heart.csv')
print(F"Null values? {data.isnull().values.any()}")


# **Attribute Information:**
# > 1. **age** - age
# > 2. **sex** - (1 = male; 0 = female)
# > 3. **cp** - chest pain type (4 values) 
# > 4. **trestbps** - resting blood pressure 
# > 5. **chol** - serum cholestoral in mg/dl 
# > 6. **fbs** - fasting blood sugar > 120 mg/dl
# > 7. **restecg** - resting electrocardiographic results (values 0,1,2)
# > 8. **thalach** - maximum heart rate achieved 
# > 9. **exang** - exercise induced angina 
# > 10. **oldpeak** -  ST depression induced by exercise relative to rest 
# > 11. **slope** - the slope of the peak exercise ST segment 
# > 12. **ca** -  number of major vessels (0-3) colored by flourosopy 
# > 13. **thal** - 3 = normal; 6 = fixed defect; 7 = reversable defect

# ### Dataset sample

# In[ ]:


data.head()


# ### Number of examples per class

# In[ ]:


plt.figure(figsize=(7, 5))
count_per_class = [len(data[data['target'] == 0]),len(data[data['target'] == 1])]
labels = [0, 1]
colors = ['yellowgreen', 'lightblue']
explode = (0.05, 0.1)
plt.pie(count_per_class, explode=explode, labels=labels, 
        colors=colors,autopct='%4.2f%%',shadow=True, startangle=45)
plt.title('Examples per class')
plt.axis('equal')
plt.show()


# Classes are well balanced!

# ### Gender shares in dataset

# In[ ]:


plt.figure(figsize=(7, 5))
count_per_class = [len(data[data['sex'] == 0]),len(data[data['sex'] == 1])]
labels = ['Female', 'Male']
colors = ['lightgreen', 'gold']
explode = (0.05, 0.1)
plt.pie(count_per_class, explode=explode, labels=labels, 
        colors=colors,autopct='%4.2f%%',shadow=True, startangle=70)
plt.title('Gender shares')
plt.axis('equal')
plt.show()


# ### Age-sex distribution

# In[ ]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(data['age'], data['sex'], shade=True)
plt.title('Age-sex density estimate')
plt.subplot(1, 2, 2)
sns.distplot(data['age'])
plt.title('Age distribution')
plt.show()


# ### Serum cholestoral per class distribution

# In[ ]:


plt.figure(figsize=(8, 6))
sns.distplot(data[data.target == 0]['chol'], label='without heart disease')
sns.distplot(data[data.target == 1]['chol'], label='with heart disease')
plt.xlabel('serum cholestoral in mg/dl')
plt.title('serum cholestoral per class')
plt.legend()
plt.show()


# ### Maximum heart rate achieved per class distribution

# In[ ]:


plt.figure(figsize=(8, 6))
sns.distplot(data[data.target == 0]['thalach'], label='without heart disease')
sns.distplot(data[data.target == 1]['thalach'], label='with heart disease')
plt.title('maximum heart rate achieved per class')
plt.xlabel('maximum heart rate achieved')
plt.legend()
plt.show()


# ### Features heatmap

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, linewidths=2, cmap="YlGnBu")
plt.show()


# ### Resting blood pressure per class

# In[ ]:


data.groupby('target')['trestbps'].describe()


# In[ ]:


ax2 = sns.jointplot("target", "trestbps", data=data, kind="reg", color='r')
ax2.set_axis_labels('target','resting blood pressure')
plt.show()


# In[ ]:


X = data.values[:, :13]
y = data.values[:, 13]


# ## **2. Feature importances (for L1-regularized Logistic Regression)**

# In[ ]:


import eli5
from sklearn.linear_model import LogisticRegression
from eli5.sklearn import PermutationImportance

logistic_regression = LogisticRegression(penalty='l1')
logistic_regression.fit(X, y)
perm_imp = PermutationImportance(logistic_regression, random_state=42).fit(X, y)
eli5.show_weights(perm_imp, feature_names = data.columns.tolist()[:13])


# **Model interpretation:** We can see that the number of major vessels colored by fluoroscopy and chest pain type are the most important features for correct classification.

# ## **3. Appropriate metric? Recall!**

# **Q:** *Why Recall?* <br/>
# **A:** Our classifier should be sensitive to false negatives. For this dataset, false negative is a person that has heart disease but our classifier decided that the person does not have any heart problems. In other words, classifier said that the ill person is healthy. On the other side, false positive is a person that does not have any heart diseases and our classifier decided that person is ill. In that case, the person will run more tests and conclude it does not have any heart problems.

# ## **4. Nested cross-validation (way to fight off overly-optimistic score)**

# Nested cross-validation is used to train a model in which hyperparameters also need to be optimized. I've used it to fight off overly-optimistic scores.<br/>
# More about nested cross-validation:<br/>
# https://www.elderresearch.com/blog/nested-cross-validation <br/>
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

# In[ ]:


from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score

def nested_kfold_cv(model, param_grid, X, y, outer_metric=accuracy_score,
                    scoring='accuracy' , k1=10, k2=3, verbose = 1, n_jobs=3, shuffle=True):
    scores = []
    estimators = []
    kf = KFold(n_splits=k1, shuffle=shuffle)
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=k2,
                                   verbose=verbose, n_jobs=n_jobs, scoring=scoring)
        grid_search.fit(X=X_train, y=y_train)
        estimator = grid_search.best_estimator_
        estimators.append(estimator)
        estimator.fit(X_train, y_train)
        scores.append(outer_metric(estimator.predict(X_test), y_test))
    return estimators, scores


# ## **5. Classification**

# ### **5.1. AdaBoost**

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, confusion_matrix


# In[ ]:


tree_model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    random_state=42
)


# In[ ]:


tree_params = {
    'n_estimators': [25, 50, 75]
}
estimators, tree_scores = nested_kfold_cv(tree_model, tree_params, X, y, outer_metric=recall_score,
                                     scoring='f1' , k1=10, k2=5, verbose = 0, n_jobs=4, shuffle=True)


# In[ ]:


print(f"Average recall: {np.mean(tree_scores)}")


# ### **5.2. SVM**

# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC


# In[ ]:


svm_model = Pipeline(steps=[
    ('standard_scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif)), # params: k
    ('svm', SVC(kernel='rbf', random_state=42)) # params: gamma, C
])


# In[ ]:


svm_grid = {
    'feature_selection__k': [10, 12, 13],
    'svm__C': [3, 5, 10, 15, 20, 25, 30, 35],
    'svm__gamma': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    
}
estimators, svm_scores = nested_kfold_cv(svm_model, svm_grid, X, y, outer_metric=recall_score,
                                     scoring='f1' , k1=10, k2=5, verbose = 0, n_jobs=4, shuffle=True)


# In[ ]:


print(f"Average recall: {np.mean(svm_scores)}")


# ### **5.3. Logistic Regression** 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


log_model = Pipeline(steps=[
    ('feature_selection', SelectKBest(f_classif)), # params: k
    ('log', LogisticRegression()) # params:  C
])


# In[ ]:


log_grid = {
    'log__C': [0.01, 0.1, 0.5, 1, 3, 5],
    'feature_selection__k': [5, 9, 10, 12, 13],
}
estimators, lr_scores = nested_kfold_cv(log_model, log_grid, X, y, outer_metric=recall_score,
                                     scoring='f1' , k1=10, k2=5, verbose = 0, n_jobs=4, shuffle=True)


# In[ ]:


print(f"Average recall: {np.mean(lr_scores)}")


# ### **5.4. KNN**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn_model = Pipeline(steps=[
    ('standard_scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(weights='distance')) # params: n_neighbors
])


# In[ ]:


knn_grid = {
    'knn__n_neighbors': [3, 5, 7, 10, 12, 15, 17, 20],
}
estimators, knn_scores = nested_kfold_cv(knn_model, knn_grid, X, y, outer_metric=recall_score,
                                     scoring='f1' , k1=10, k2=5, verbose = 0, n_jobs=4, shuffle=True)


# In[ ]:


print(f"Average recall: {np.mean(knn_scores)}")


# ### **5.5. Neural network**

# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


nn_model = Pipeline(steps=[
    ('standard_scaler', StandardScaler()),
    ('nn', MLPClassifier(max_iter=400)) # params:
])


# In[ ]:


nn_grid = {
    'nn__solver': ['adam', 'lbfgs']
}
estimators, nn_scores = nested_kfold_cv(nn_model, nn_grid, X, y, outer_metric=recall_score,
                                     scoring='f1' , k1=10, k2=5, verbose = 0, n_jobs=4, shuffle=True)


# In[ ]:


print(f"Average recall: {np.mean(nn_scores)}")


# ## **6. Classification results overview**

# In[ ]:


results = pd.DataFrame({'KNN': knn_scores, 'Logistic regression': lr_scores, 'SVC': svm_scores, 'AdaBoost': tree_scores, 'Neural network': nn_scores})
results.boxplot(figsize=(8, 6))


# 
# ### _**Thanks for reading!**_
