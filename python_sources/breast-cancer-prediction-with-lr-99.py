#!/usr/bin/env python
# coding: utf-8

# # Breast cancer prediction with Logistic Regression
# 
# ### Author
# Piotr Tynecki  
# Last edition: May 4, 2018
# 
# ### About the Breast Cancer Wisconsin Diagnostic Dataset
# Breast Cancer Wisconsin Diagnostic Dataset (WDBC) consists of features which were computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. Those features describe the characteristics of the cell nuclei found in the image.
# 
# ![Diagnosing Breast Cancer from Image](https://kaggle2.blob.core.windows.net/datasets-images/180/384/3da2510581f9d3b902307ff8d06fe327/dataset-cover.jpg)
# 
# This dataset has 569 instances: 212 - Malignant and 357 - Benign. It consists of 31 attributes including the class attribute. The attributes description is ten real-valued features which are computed for each cell nucleus. These features include: Texture, Radius, Perimeter, Smoothness, Area, Concavity, Compactness, Symmetry, Concave points and Fractal dimension.
# 
# In this document I demonstrate an automated methodology to predict if a sample is benign or malignant.

# In[1]:


import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


# ### Step 1: Exploratory Data Analysis (EDA)
# EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. It let us to summarize data main characteristics.

# In[2]:


breast_cancer = pd.read_csv('../input/data.csv')
breast_cancer.head()


# In[3]:


breast_cancer.info()


# In[4]:


breast_cancer.shape


# In[5]:


breast_cancer.describe()


# In[6]:


breast_cancer.groupby('diagnosis').size()


# #### Data quality checks

# In[7]:


breast_cancer.isnull().sum()


# In[8]:


for field in breast_cancer.columns:
    amount = np.count_nonzero(breast_cancer[field] == 0)
    
    if amount > 0:
        print('Number of 0-entries for "{field_name}" feature: {amount}'.format(
            field_name=field,
            amount=amount
        ))


# ### Step 2: Feature Engineering

# In[9]:


# Features "id" and "Unnamed: 32" are not useful 
feature_names = breast_cancer.columns[2:-1]
X = breast_cancer[feature_names]
# "diagnosis" feature is our class which I wanna predict
y = breast_cancer.diagnosis


# #### Transforming the prediction target

# In[10]:


class_le = LabelEncoder()
# M -> 1 and B -> 0
y = class_le.fit_transform(breast_cancer.diagnosis.values)


# #### Correlation Matrix
# A matrix of correlations provides useful insight into relationships between pairs of variables.

# In[11]:


sns.heatmap(
    data=X.corr(),
    annot=True,
    fmt='.2f',
    cmap='RdYlGn'
)

fig = plt.gcf()
fig.set_size_inches(20, 16)

plt.show()


# ### Step 3: Automated Logistic Regression performance evaluation with Pipeline and GridSearchCV 
# 
# For this case study I decided to use [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier.
# 
# #### Model Parameter Tuning
# [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) returns the set of parameters which have an imperceptible impact on model evaluation. Model parameter tuning with other steps like data standardization, principal component analysis (PCA) and cross-validation splitting strategy can be automated by [Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) class.
# 
# #### Data standardization
# [Standardize features](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) by removing the mean and scaling to unit variance.
# 
# ####  Principal component analysis (PCA)
# The main goal of a PCA analysis is to identify patterns in data. PCA aims to detect the correlation between variables. If a strong correlation between variables exists, the attempt to reduce the dimensionality only makes sense.

# Let's start with defining the Pipeline instance. In this case I used `StandardScaler` for preprocesing, `PCA` for feature selection and `LogisticRegression` for classification.

# In[12]:


pipe = Pipeline(steps=[
    ('preprocess', StandardScaler()),
    ('feature_selection', PCA()),
    ('classification', LogisticRegression())
])


# Next, I needed to prepare attributes with values for above steps which wanna to check by the model parameter tuning process: 5 values for `C` (regularization strength for classifier) and the generator of numbers for `n_components` (number of components to keep for feature selector).

# In[13]:


c_values = [0.1, 1, 10, 100, 1000]
n_values = range(2, 31)
random_state = 42


# Next, I needed to prepare supported combinations for classifier parameters including above attributes. In LogisticRegression case I stayed with two scenarios.

# In[14]:


log_reg_param_grid = [
    {
        'feature_selection__random_state': [random_state],
        'feature_selection__n_components': n_values,
        'classification__C': c_values,
        'classification__penalty': ['l1'],
        'classification__solver': ['liblinear'],
        'classification__multi_class': ['ovr'],
        'classification__random_state': [random_state]
    },
    {
        'feature_selection__random_state': [random_state],
        'feature_selection__n_components': n_values,
        'classification__C': c_values,
        'classification__penalty': ['l2'],
        'classification__solver': ['liblinear', 'newton-cg', 'lbfgs'],
        'classification__multi_class': ['ovr'],
        'classification__random_state': [random_state]
    }
]


# Next, I needed to prepare cross-validation splitting strategy object with `StratifiedKFold` and passed it with others to `GridSearchCV`. In that case for evaluation I used `accuracy` metric.

# In[15]:


strat_k_fold = StratifiedKFold(
    n_splits=10,
    random_state=42
)

log_reg_grid = GridSearchCV(
    pipe,
    param_grid=log_reg_param_grid,
    cv=strat_k_fold,
    scoring='accuracy'
)

log_reg_grid.fit(X, y)

# Best LogisticRegression parameters
print(log_reg_grid.best_params_)
# Best score for LogisticRegression with best parameters
print('Best score for LogisticRegression: {:.2f}%'.format(log_reg_grid.best_score_ * 100))

best_params = log_reg_grid.best_params_


# #### Model evaluation
# 
# Finally, all of the best parameters values were passed to new feature selection and classifier instances.

# In[16]:


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.20
)

std_scaler = StandardScaler()

X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

pca = PCA(
    n_components=best_params.get('feature_selection__n_components'),
    random_state=best_params.get('feature_selection__random_state')
)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print(pca.explained_variance_ratio_)
print('\nPCA sum: {:.2f}%'.format(sum(pca.explained_variance_ratio_) * 100))

log_reg = LogisticRegression(
    C=best_params.get('classification__C'),
    penalty=best_params.get('classification__penalty'),
    solver=best_params.get('classification__solver'),
    multi_class=best_params.get('classification__multi_class'),
    random_state=best_params.get('classification__random_state'),
)

log_reg.fit(X_train_pca, y_train)

log_reg_predict = log_reg.predict(X_test_pca)
log_reg_predict_proba = log_reg.predict_proba(X_test_pca)[:, 1]

print('LogisticRegression Accuracy: {:.2f}%'.format(accuracy_score(y_test, log_reg_predict) * 100))
print('LogisticRegression AUC: {:.2f}%'.format(roc_auc_score(y_test, log_reg_predict_proba) * 100))
print('LogisticRegression Classification report:\n\n', classification_report(y_test, log_reg_predict))
print('LogisticRegression Training set score: {:.2f}%'.format(log_reg.score(X_train_pca, y_train) * 100))
print('LogisticRegression Testing set score: {:.2f}%'.format(log_reg.score(X_test_pca, y_test) * 100))


# #### Confusion Matrix
# 
# Also known as an Error Matrix, is a specific table layout that allows visualization of the performance of an algorithm. The table have two rows and two columns that reports the number of False Positives (FP), False Negatives (FN), True Positives (TP) and True Negatives (TN). This allows more detailed analysis than accuracy.

# In[17]:


outcome_labels = sorted(breast_cancer.diagnosis.unique())

# Confusion Matrix for LogisticRegression
sns.heatmap(
    confusion_matrix(y_test, log_reg_predict),
    annot=True,
    xticklabels=outcome_labels,
    yticklabels=outcome_labels
)


# #### Receiver Operating Characteristic (ROC)
# 
# [ROC curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

# In[18]:


# ROC for LogisticRegression
fpr, tpr, thresholds = roc_curve(y_test, log_reg_predict_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for LogisticRegression')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# #### F1-score after 10-fold cross-validation

# In[19]:


strat_k_fold = StratifiedKFold(
    n_splits=10,
    random_state=42
)

std_scaler = StandardScaler()

X_std = std_scaler.fit_transform(X)
X_pca = pca.fit_transform(X_std)

fe_score = cross_val_score(
    log_reg,
    X_pca,
    y,
    cv=strat_k_fold,
    scoring='f1'
)

print("LogisticRegression: F1 after 10-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    fe_score.mean() * 100,
    fe_score.std() * 2
))


# ### Final step: Conclusions
# 
# After the application of data standardization, staying with 16 numbers of componens for feature selector and tuning the classifier parameters I achieved the following results:
# 
# * Accuracy: 99%
# * F1-score: 99%
# * Precision: ~100%
# * Recall: ~99%
# 
# I would love to knows your comments and other tuning proposals for that study case.
