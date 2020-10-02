#!/usr/bin/env python
# coding: utf-8

# # Predict diabetes diagnosis for Pima Female Indians with Logistic Regression
# 
# ### Author
# Piotr Tynecki  
# Last edition: March 29, 2018
# 
# ### About the Pima Indian Diabetes Dataset
# The Pima Indian Diabetes Dataset consists of information on 768 of women population: 268 tested positive and 500 tested negative instances coming from a population near Phoenix, Arizona, USA. Tested positive and tested negative indicates whether the patient is diabetic or not, respectively. Each instance is comprised of 8 attributes, which are all numeric. These data contain personal health data as well as results from medical examinations.
# 
# The detailed attributes in the dataset are listed below:
# 
# * Number of times pregnant (*Pregnancies*)
# * Plasma glucose concentration at 2h in an oral glucose tolerance test (*Glucose*)
# * Diastolic blood pressure (*BloodPressure*)
# * Triceps skin fold thickness (*SkinThickness*)
# * 2-h serum insulin (*Insulin*)
# * Body Mass Index (*BMI*)
# * Diabetes pedigree function (*DiabetesPedigreeFunction*)
# * Age (*Age*)
# * Class variable (*Outcome*)

# In[1]:


import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV


# ### Step 1: Exploratory Data Analysis (EDA)
# EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. It let us to summarize data main characteristics.

# In[2]:


diabetes = pd.read_csv('../input/diabetes.csv')
diabetes.head()


# In[3]:


diabetes.info()


# In[4]:


diabetes.shape


# In[5]:


diabetes.describe()


# In[6]:


diabetes.groupby('Outcome').size()


# In[7]:


# Detailed distribution of the features in the dataset
sns.pairplot(data=diabetes, hue='Outcome')
plt.show()


# #### Data quality checks

# In[8]:


diabetes.isnull().sum()


# In[9]:


# Display how many 0 value each feature have
for field in diabetes.columns[:8]:
    print('Number of 0-entries for "{field_name}" feature: {amount}'.format(
        field_name=field,
        amount=np.count_nonzero(diabetes[field] == 0)
    ))


# We could replace 0 values to mean values for each features (excluding Pregnancies field) but it has bad effect on metrics (accuracy, F1-score) at least. So, that's why the code below is commented.

# In[10]:


# features_with_zeros = diabetes.columns[1:-1]
    
# diabetes[features_with_zeros] = diabetes[features_with_zeros].replace(0, np.nan)
# diabetes[features_with_zeros] = diabetes[features_with_zeros].fillna(diabetes.mean())


# ### Step 2: Feature Engineering

# In[11]:


feature_names = diabetes.columns[:8]
feature_names


# In[12]:


X = diabetes[feature_names]
y = diabetes.Outcome


# #### Correlation Matrix
# A matrix of correlations provides useful insight into relationships between pairs of variables.

# In[13]:


sns.heatmap(
    data=X.corr(),
    annot=True,
    fmt='.2f',
    cmap='RdYlGn'
)

fig = plt.gcf()
fig.set_size_inches(10, 8)

plt.show()


# #### Recursive Feature Elimination with Cross Validation
# The goal of [Recursive Feature Elimination (RFE)](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) is to select features by feature ranking with recursive feature elimination.
# 
# For more confidence of features selection I used K-Fold Cross Validation with [Stratified k-fold](http://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold).

# In[14]:


# I temporarily removed a few Glucose, BloodPressure and BMI rows with 0 values for better RFE result
diabetes_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
diabetes_mod.shape


# In[15]:


X_mod = diabetes_mod[feature_names]
y_mod = diabetes_mod.Outcome

strat_k_fold = StratifiedKFold(
    n_splits=10,
    random_state=42
)

logreg_model = LogisticRegression()

rfecv = RFECV(
    estimator=logreg_model,
    step=1,
    cv=strat_k_fold,
    scoring='accuracy'
)
rfecv.fit(X_mod, y_mod)

plt.figure()
plt.title('RFE with Logistic Regression')
plt.xlabel('Number of selected features')
plt.ylabel('10-fold Crossvalidation')

# grid_scores_ returns a list of accuracy scores
# for each of the features selected
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()

print('rfecv.grid_scores_: {grid_scores}'.format(grid_scores=rfecv.grid_scores_))

# support_ is another attribute to find out the features
# which contribute the most to predicting
new_features = list(filter(
    lambda x: x[1],
    zip(feature_names, rfecv.support_)
))

print('rfecv.support_: {support}'.format(support=rfecv.support_))

# Features are the most suitable for predicting the response class
new_features = list(map(operator.itemgetter(0), new_features))
print('\nThe most suitable features for prediction: {new_features}'.format(new_features=new_features))


# ### Step 3: Data standardization
# [Standardize features](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) by removing the mean and scaling to unit variance.

# In[16]:


# Features chosen based on RFECV result
best_features = [
    'Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction'
]

X = StandardScaler().fit_transform(X[best_features])


# In[17]:


# Split your data into training and testing (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.20
)


# #### Principal component analysis (PCA)
# The main goal of a PCA analysis is to identify patterns in data. PCA aims to detect the correlation between variables. If a strong correlation between variables exists, the attempt to reduce the dimensionality only makes sense.

# In[18]:


pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(pca.explained_variance_ratio_)
print('PCA sum: {:.2f}%'.format(sum(pca.explained_variance_ratio_) * 100))


# ### Step 4: Evaluating the performance of Logistic Regression model
# For this case study I decided to use [LogisticRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier as a beginning of my Data Science trip.
# 
# Model Parameter Tuning with [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) returns the set of parameters which have an imperceptible impact on model evaluation.  
#   
# You can check it by yourself:

# In[19]:


from sklearn.model_selection import GridSearchCV

c_values = list(np.arange(1, 100))

param_grid = [
    {
        'C': c_values,
        'penalty': ['l1'],
        'solver': ['liblinear'],
        'multi_class': ['ovr'],
        'random_state': [42]
    },
    {
        'C': c_values,
        'penalty': ['l2'],
        'solver': ['liblinear', 'newton-cg', 'lbfgs'],
        'multi_class': ['ovr'],
        'random_state': [42]
    }
]

grid = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=strat_k_fold,
    scoring='f1'
)
grid.fit(X, y)

# Best LogisticRegression parameters
print(grid.best_params_)
# Best score for LogisticRegression with best parameters
print('Best score: {:.2f}%'.format(grid.best_score_ * 100))


# #### Model learning

# In[20]:


log_reg = LogisticRegression(
    # Parameters chosen based on GridSearchCV result
    C=1,
    multi_class='ovr',
    penalty='l2',
    solver='newton-cg',
    random_state=42
)
log_reg.fit(X_train, y_train)

log_reg_predict = log_reg.predict(X_test)
log_reg_predict_proba = log_reg.predict_proba(X_test)[:, 1]


# #### Model evaluation

# In[21]:


print('Accuracy: {:.2f}%'.format(accuracy_score(y_test, log_reg_predict) * 100))
print('AUC: {:.2f}%'.format(roc_auc_score(y_test, log_reg_predict_proba) * 100))
print('Classification report:\n\n', classification_report(y_test, log_reg_predict))
print('Training set score: {:.2f}%'.format(log_reg.score(X_train, y_train) * 100))
print('Testing set score: {:.2f}%'.format(log_reg.score(X_test, y_test) * 100))


# #### Confusion Matrix
# 
# Also known as an Error Matrix, is a specific table layout that allows visualization of the performance of an algorithm. The table have two rows and two columns that reports the number of False Positives (FP), False Negatives (FN), True Positives (TP) and True Negatives (TN). This allows more detailed analysis than accuracy.

# In[22]:


outcome_labels = sorted(diabetes.Outcome.unique())

sns.heatmap(
    confusion_matrix(y_test, log_reg_predict),
    annot=True,
    xticklabels=outcome_labels,
    yticklabels=outcome_labels
)


# #### Receiver Operating Characteristic (ROC)
# 
# [ROC curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

# In[23]:


fpr, tpr, thresholds = roc_curve(y_test, log_reg_predict_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# #### F1-score after 10-fold cross-validation

# In[24]:


strat_k_fold = StratifiedKFold(
    n_splits=10,
    random_state=42
)

X_pca = pca.transform(X)

fe_score = cross_val_score(
    log_reg,
    X_pca,
    y,
    cv=strat_k_fold,
    scoring='f1'
)

print("F1 after 10-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    fe_score.mean() * 100,
    fe_score.std() * 2
))


# ### Final step: Conclusions
# 
# After the application of data standardization, Recursive Feature Elimination (RFE) and Principal Component Analysis (PCA) I achieved the following results:
# 
# * Accuracy: ~84%
# * F1-score: 83%
# * Precision: 84%
# * Recall: 84%
# 
# From my observations it could be one of the highest score for Logistic Regression in Kaggle but the results from F1-score after 10-fold cross-validation really bugging me.
# 
# I would love to knows your comments and other tuning proposals for that study case.
