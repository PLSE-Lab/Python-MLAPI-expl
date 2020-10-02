#!/usr/bin/env python
# coding: utf-8

# ## Heart Disease analysis and model

# ### Import libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
plt.style.use('seaborn')
sns.set(font_scale=2)
pd.set_option('display.max_columns', 500)


# ### Utility functions

# In[ ]:


def plot_kde_hist_for_numeric(df, col, target):
    fig, ax = plt.subplots(1,4 , figsize=(16, 8))
    
    sns.kdeplot(df.loc[df[target] == 0, col], ax=ax[0], label='no heart disease')
    sns.kdeplot(df.loc[df[target] == 1, col], ax=ax[0], label='has heart disease')
    
    df.loc[df[target] == 0, col].hist(ax=ax[1], bins=10)
    df.loc[df[target] == 1, col].hist(ax=ax[1], bins=10)
    
    df.loc[df[target] == 0, col].plot.box(ax=ax[2])
    df.loc[df[target] == 1, col].plot.box(ax=ax[3])
    
    
    plt.suptitle(col, fontsize=30)
    ax[0].set_title('KDE plot')
    ax[1].set_title('Histogram')
    ax[1].legend(['no heart disease', 'has heart disease'])
    
    ax[2].set_title('no heart disease')
    ax[3].set_title('has heart disease')
    plt.show()
    
def corr_heatmap(df):
    correlations = df.corr()
    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})
    plt.show()

def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


# ## Variables definition

#    ** 1. age**: The person's age in years.<br>
#    ** 2. sex**: The person's sex (1 = male, 0 = female)<br>
#    ** 3. cp**: The chest pain experienced (Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic)<br>
#    ** 4. trestbps**: The person's resting blood pressure (mm Hg on admission to the hospital). <br>
#    ** 5. chol**: The person's cholesterol measurement in mg/dl. <br>
#    ** 6. fbs**: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)<br>
#    ** 7. restecg**: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)<br>
#    ** 8. thalach**: The person's maximum heart rate achieved.<br>
#    ** 9. exang**: Exercise induced angina (1 = yes; 0 = no)<br>
#    ** 10. oldpeak**: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here).<br>
#    ** 11. slope**: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)<br>
#    ** 12. ca**: The number of major vessels (0-3). <br>
#    ** 13. thal**: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)<br>
#    ** 14. target**: Heart disease (0 = no, 1 = yes)<br>

# ### Read the dataset

# In[ ]:


get_ipython().run_line_magic('time', "df = pd.read_csv('../input/heart.csv')")


# In[ ]:


print(df.shape)


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.dtypes


# ## Check the target

# In[ ]:


df['target'].value_counts().plot.bar()
plt.title('Target')
plt.show()


# ### Checking missing

# In[ ]:


get_ipython().run_cell_magic('time', '', 'df.isnull().values.any()')


# ### Exploratory data analysis

# #### Sex variable

# In[ ]:


sns.countplot(x='sex', hue='target',data=df)
plt.show()


# We have relatively more female with heart disease in the dataset. Does that mean that more women tend to have heart disease than men? 

# #### cp variable: The chest pain experienced (Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic)

# In[ ]:


sns.countplot(x='cp', hue='target',data=df)
plt.show()


# Chest pain such as atypical angina and non-anginal pain tend to be present more in people with heart disease

# #### The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)

# In[ ]:


sns.countplot(x='fbs', hue='target',data=df)
plt.show()


# People with heart disease tend to have a fasting blood sugar less than 120 mg/dl

# ### restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)

# In[ ]:


sns.countplot(x='restecg', hue='target',data=df)
plt.show()


# People with heart disease tend to have a ST-T wave abnormality

# #### exang: Exercise induced angina (1 = yes; 0 = no)

# In[ ]:


sns.countplot(x='exang', hue='target',data=df)
plt.show()


# ### slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)

# In[ ]:


sns.countplot(x='slope', hue='target',data=df)
plt.show()


# ### ca: The number of major vessels (0-3).

# In[ ]:


sns.countplot(x='ca', hue='target',data=df)
plt.show()


# ### thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)

# In[ ]:


sns.countplot(x='thal', hue='target', data=df)
plt.show()


# ### Age

# In[ ]:


plot_kde_hist_for_numeric(df, 'age', target='target')


# ## Male

# In[ ]:


male_df = df[df['sex'] == 1]
plot_kde_hist_for_numeric(male_df, 'age', target='target')


# ## Female

# In[ ]:


female_df = df[df['sex'] == 0]
plot_kde_hist_for_numeric(female_df, 'age', target='target')


# ### trestbps: The person's resting blood pressure (mm Hg on admission to the hospital). 

# In[ ]:


plot_kde_hist_for_numeric(df, 'trestbps', target='target')


# ### chol: The person's cholesterol measurement in mg/dl.

# In[ ]:


plot_kde_hist_for_numeric(df, 'chol', target='target')


# ### thalach: The person's maximum heart rate achieved.

# In[ ]:


plot_kde_hist_for_numeric(df, 'thalach', target='target')


# ### oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot.

# In[ ]:


plot_kde_hist_for_numeric(df, 'oldpeak', target='target')


# ### Correlations

# In[ ]:


corr_heatmap(df)


# ## Train

# In[ ]:


array = df.values
X = array[:,0:12].astype(float)
Y = array[:,13]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
    test_size=validation_size, random_state=seed)


# In[ ]:


num_folds = 10
seed = 7
scoring =  'accuracy'


# In[ ]:


models = []
models.append(( 'LR' , LogisticRegression()))
models.append(( 'LDA' , LinearDiscriminantAnalysis()))
models.append(( 'KNN' , KNeighborsClassifier()))
models.append(( 'CART' , DecisionTreeClassifier()))
models.append(( 'NB' , GaussianNB()))
models.append(( 'SVM' , SVC()))


# In[ ]:


results = []
names = []
for name, model in models:
  kfold = KFold(n_splits=num_folds, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)


# In[ ]:


# Compare Algorithms
fig = plt.figure(figsize=(16, 8))
fig.suptitle( 'Algorithm Comparison' )
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ### Standardize the data

# In[ ]:


pipelines = []
pipelines.append(( 'ScaledLR' , Pipeline([( 'Scaler' , StandardScaler()),( 'LR' ,
    LogisticRegression())])))
pipelines.append(( 'ScaledLDA' , Pipeline([( 'Scaler' , StandardScaler()),( 'LDA' ,
    LinearDiscriminantAnalysis())])))

pipelines.append(( 'ScaledKNN' , Pipeline([( 'Scaler' , StandardScaler()),( 'KNN' ,
    KNeighborsClassifier())])))
pipelines.append(( 'ScaledCART' , Pipeline([( 'Scaler' , StandardScaler()),( 'CART' ,
    DecisionTreeClassifier())])))
pipelines.append(( 'ScaledNB' , Pipeline([( 'Scaler' , StandardScaler()),( 'NB' ,
    GaussianNB())])))
pipelines.append(( 'ScaledSVM' , Pipeline([( 'Scaler' , StandardScaler()),( 'SVM' , SVC())])))
results = []
names = []
for name, model in pipelines:
  kfold = KFold(n_splits=num_folds, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)


# In[ ]:


# Compare Algorithms
fig = plt.figure(figsize=(16, 8))
fig.suptitle( 'Scaled Algorithm Comparison' )
ax = fig.add_subplot(111 )
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ### Tuning SVM

# In[ ]:


# Tune scaled SVM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = [ 'linear' ,  'poly' ,  'rbf' ,  'sigmoid' ]
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_[ 'mean_test_score' ]
stds = grid_result.cv_results_[ 'std_test_score' ]
params = grid_result.cv_results_[ 'params' ]
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# Logistic regression seems to perform better than SVM. Let's try ensemble models

# In[ ]:


# ensembles
ensembles = []
ensembles.append(( 'AB' , AdaBoostClassifier()))
ensembles.append(( 'GBM' , GradientBoostingClassifier()))
ensembles.append(( 'RF' , RandomForestClassifier()))
ensembles.append(( 'ET' , ExtraTreesClassifier()))
results = []
names = []
for name, model in ensembles:
  kfold = KFold(n_splits=num_folds, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
  
  
  
  


# In[ ]:


fig = plt.figure(figsize=(16, 8))
fig.suptitle( 'Ensembles models' )
ax = fig.add_subplot(111 )
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Logistic regression still win

# ### Data augmentation

# In[ ]:


X_t, y_t = augment(X, Y, t=20)


# In[ ]:


X_train, X_validation, Y_train, Y_validation = train_test_split(X_t, y_t,
    test_size=validation_size, random_state=seed)


# In[ ]:


models = []
models.append(( 'LR' , LogisticRegression()))
models.append(( 'LDA' , LinearDiscriminantAnalysis()))
models.append(( 'KNN' , KNeighborsClassifier()))
models.append(( 'CART' , DecisionTreeClassifier()))
models.append(( 'NB' , GaussianNB()))
models.append(( 'SVM' , SVC()))


# In[ ]:


results = []
names = []
for name, model in models:
  kfold = KFold(n_splits=num_folds, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)


# Data augmentation has increased the accuracy of our model

# In[ ]:


# Compare Algorithms
fig = plt.figure(figsize=(16, 8))
fig.suptitle( 'Algorithm Comparison' )
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ### Standardize data

# In[ ]:


pipelines = []
pipelines.append(( 'ScaledLR' , Pipeline([( 'Scaler' , StandardScaler()),( 'LR' ,
    LogisticRegression())])))
pipelines.append(( 'ScaledLDA' , Pipeline([( 'Scaler' , StandardScaler()),( 'LDA' ,
    LinearDiscriminantAnalysis())])))

pipelines.append(( 'ScaledKNN' , Pipeline([( 'Scaler' , StandardScaler()),( 'KNN' ,
    KNeighborsClassifier())])))
pipelines.append(( 'ScaledCART' , Pipeline([( 'Scaler' , StandardScaler()),( 'CART' ,
    DecisionTreeClassifier())])))
pipelines.append(( 'ScaledNB' , Pipeline([( 'Scaler' , StandardScaler()),( 'NB' ,
    GaussianNB())])))
pipelines.append(( 'ScaledSVM' , Pipeline([( 'Scaler' , StandardScaler()),( 'SVM' , SVC())])))
results = []
names = []
for name, model in pipelines:
  kfold = KFold(n_splits=num_folds, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)


# SVM wins with more standardized data

# In[ ]:


# Compare Algorithms
fig = plt.figure(figsize=(16, 8))
fig.suptitle( 'Scaled Algorithm Comparison' )
ax = fig.add_subplot(111 )
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# ### Tune SVM

# In[ ]:


# Tune scaled SVM
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = [ 'linear' ,  'poly' ,  'rbf' ,  'sigmoid' ]
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_[ 'mean_test_score' ]
stds = grid_result.cv_results_[ 'std_test_score' ]
params = grid_result.cv_results_[ 'params' ]
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[ ]:


# ensembles
ensembles = []
ensembles.append(( 'AB' , AdaBoostClassifier()))
ensembles.append(( 'GBM' , GradientBoostingClassifier()))
ensembles.append(( 'RF' , RandomForestClassifier()))
ensembles.append(( 'ET' , ExtraTreesClassifier()))
results = []
names = []
for name, model in ensembles:
  kfold = KFold(n_splits=num_folds, random_state=seed)
  cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
  
  


# In[ ]:


fig = plt.figure(figsize=(16, 8))
fig.suptitle( 'Ensembles models' )
ax = fig.add_subplot(111 )
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# GBM wins with more data

# In[ ]:





# In[ ]:





# In[ ]:




