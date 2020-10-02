#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# for python 2.x and 3.x support
from __future__ import division, print_function, unicode_literals

# computation libraries used
import pandas as pd
import numpy as np

#### graphing libraries ####
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
############################


# sklearn for ML
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score

# mjd to norml time conversion
from datetime import datetime, timedelta


# What categories does this project fall in?

# # Q.1. What does the data look like?

# In[ ]:


sloan = pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')


# In[ ]:


sloan.head()


# ## Datatypes present

# In[ ]:


sloan.info()


# No null values present in any column

# ## Understanding the significance of features present

#  1. ~~objid~~  
#  Unique ID given to an object by the data collection process
#  
#  2. **ra**  
#  Right ascention. Analogous to longitude on Earth
#  
#  3. **dec**  
#  Distance of object from horizon. usually measured in degrees. Analogous to latitude on Earth
#  
#  4. **u, g, r, i, z**  
#  Wavelength filters in the telescope Ultraviolet, Green, Red, Near-infrared, Infrared
#  
#  5. ~~run~~  
#  ID of observed strip in the run
#  
#  6. ~~rerun~~  
#  Reprocessing ID of the run strip
#  
#  7. ~~camcol~~  
#  ID given to a part of a run
#  
#  8. ~~field~~  
#  Photo member of a camcol. It is 2048x1498 pixels wide
#  
#  9. ~~specobjid~~  
#  UID for an object generated using few other fields
#  
#  10. **redshift**  
#  Stretching of light waves due to presence of relative velocity between two bodies
#  
#  11. ~~plate~~  
#  Telescope contains 6 plates. This is a unique identifier
#  
#  12. **mjd**  
#  Modified Julian Date  
#  > Julian Date begins from January 1 4713 BC  
#  > MJD adds 24,00,000.5 days to it and begins from November 17 1858
#  
#  13. ~~fiberid~~  
#  Optical fiber identifier, which brings light towards the slit

# ## The 5-number summary

# In[ ]:


sloan.describe()


# In[ ]:


sloan.drop(columns=['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid', 'fiberid', 'plate'], inplace=True)


# In[ ]:


sloan.head()


# # Q2. What insights can the data provide? (EDA)

# ## Univariate Analysis

# How are the objects positioned in the sky?

# In[ ]:


f, axes = plt.subplots(2, 1, figsize=(15, 10))
sns.boxplot(y='class', x='ra', data=sloan, ax=axes[0])
sns.boxplot(y='class', x='dec', data=sloan, ax=axes[1])


# Find Out: Why are the stellar objects positioned this way? (HINT Look for Baryon Acoustic Oscillation)

# Is the redshift representative of any properties of the objects?

# In[ ]:


f, ax = plt.subplots(figsize=(15, 10))
sns.catplot(y='redshift', x='class', data=sloan, ax=ax)


# Stars have the lowest average redshift, followed by Galaxies and then Quasars. What do the distributions of *u, g, r, i, z* filters look like?

# In[ ]:


f, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
c = ['STAR', 'GALAXY', 'QSO']

for ax_id in range(3):
    sns.distplot(sloan.loc[sloan['class']==c[ax_id],'u'], hist=False, color='purple', ax=axes[ax_id], label='u')
    sns.distplot(sloan.loc[sloan['class']==c[ax_id],'g'], hist=False, color='blue', ax=axes[ax_id], label='g')
    sns.distplot(sloan.loc[sloan['class']==c[ax_id],'r'], hist=False, color='green', ax=axes[ax_id], label='r')
    sns.distplot(sloan.loc[sloan['class']==c[ax_id],'i'], hist=False, color='red', ax=axes[ax_id], label='i')
    sns.distplot(sloan.loc[sloan['class']==c[ax_id],'z'], hist=False, color='grey', ax=axes[ax_id], label='z')
    axes[ax_id].set(xlabel=c[ax_id], ylabel='Intensity')


# <pre>
# Green    filter o/p goes in to Blue  part of the image  
# Red      filter o/p goes in to Green part of the image  
# Infrared filter o/p goes in to Red   part of the image  
# </pre>

# In[ ]:


f, axes = plt.subplots(5, 1, figsize=(16, 20))
c = ['u','g', 'r', 'i', 'z']

for idx, cls in enumerate(c):
    sns.boxplot(y='class', x=cls, data=sloan, ax=axes[idx])


# Can we visualise the discovery timeline for various objects?

# In[ ]:


# MJD starts at 17th November 1858, midnight
_MJD_BASE_TIME_ = datetime.strptime('17/11/1858 00:00', '%d/%m/%Y %H:%M')

def convertMJD(x=0):
    return _MJD_BASE_TIME_ + timedelta(days=x)


# In[ ]:


timeline_stars  = sloan.loc[sloan['class']=='STAR'  , 'mjd']
timeline_galaxy = sloan.loc[sloan['class']=='GALAXY', 'mjd']
timeline_qso    = sloan.loc[sloan['class']=='QSO'   , 'mjd']


# In[ ]:


f, ax = plt.subplots(figsize=(16, 10))
sns.distplot(timeline_stars , hist=False, label='STAR'  , ax=ax)
sns.distplot(timeline_galaxy, hist=False, label='GALAXY', ax=ax)
sns.distplot(timeline_qso   , hist=False, label='QSO'   , ax=ax)


# Find Out: What is the reason for the sudden spike in Galaxy discovery around MJD 52000 (or 1st April 2001) ?

# ## Multivariate Analysis

# Correlation
# * What is Pearson correlation?  

# In[ ]:


sns.pairplot(sloan, hue='class')


# Can we say that hotter objects emit more of every wavelength?

# In[ ]:


sns.pairplot(sloan[['u','g','r','i','z','class']], hue='class')


# *ugriz* correlation looks in accordance with expected physical behaviour - Hotter objects emit more of every wavelength. Can we represent the relationship between these variables in a more efficient manner?

# In[ ]:


f, axes = plt.subplots(1, 3, figsize=(16, 5))

star_corr = sloan.loc[sloan['class']=='STAR', ['u','g','r','i','z']].corr()
galaxy_corr = sloan.loc[sloan['class']=='GALAXY', ['u','g','r','i','z']].corr()
qso_corr = sloan.loc[sloan['class']=='QSO', ['u','g','r','i','z']].corr()

msk = np.zeros_like(star_corr)
msk[np.triu_indices_from(msk)] = True

sns.heatmap(star_corr, cmap='RdBu_r', mask=msk, ax=axes[0])
sns.heatmap(galaxy_corr, cmap='RdBu_r', mask=msk, ax=axes[1])
sns.heatmap(qso_corr, cmap='RdBu_r', mask=msk, ax=axes[2])


# This also tells us that all wavelength radiations are strongly correlated... except *u*. Why?
# 
# Find Out: What can be possible reasons for low *u* correlation?

# How are the objects positioned in space?

# In[ ]:


f, ax = plt.subplots(figsize=(16, 10))
sns.scatterplot(x='ra', y='dec', hue='class', data=sloan)


# Redshift gives us the distance between us and the object. We can incorporate it to get a more representative view

# In[ ]:


lbl = LabelEncoder()
cls_enc = lbl.fit_transform(sloan['class'])

g = go.Scatter3d(
    x=sloan['ra'], y=sloan['dec'], z=sloan['redshift'],
    mode='markers',
    marker=dict(
        color=cls_enc,
        opacity=0.5,
    )
)

g_data = [g]

layout = go.Layout(margin=dict(
    l=0, r=0, b=0, t=0
))

figure = go.Figure(data=g_data, layout=layout)

iplot(figure, filename='3d-repr-redshift')


# # Q3. Can we learn from the data? (Fitting models)

# ## Data Preparation

# In[ ]:


sloan.drop(columns=['mjd'], inplace=True)


# In[ ]:


lbl_enc = LabelEncoder()
sloan['class'] = lbl_enc.fit_transform(sloan['class'])


# In[ ]:


sloan.head()


# In[ ]:


X = sloan.drop(columns=['class'])
y = sloan['class']


# In[ ]:


strat_split = StratifiedShuffleSplit(n_splits=1, train_size=0.9, random_state=42)

for train_index, test_index in strat_split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_test.shape


# In[ ]:


strat_split_val = StratifiedShuffleSplit(n_splits=1, train_size=0.75, random_state=42)

for train_index, val_index in strat_split.split(X_train, y_train):
    X_train, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train, y_val = y_train.iloc[train_index], y_train.iloc[val_index]


# In[ ]:


X_train.shape


# In[ ]:


y_train.shape


# In[ ]:


X_val.shape


# In[ ]:


y_val.shape


# In[ ]:


scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)


# In[ ]:


X_train_scaled.head()


# In[ ]:


y_val.value_counts()


# In[ ]:


lbl_enc.classes_


# ## Models

# ### K Nearest Neighbors

# Cost
# 
# Euclidean Distance

# In[ ]:


knn = KNeighborsClassifier(n_jobs=-1)


# In[ ]:


knn.fit(X_train_scaled, y_train)


# In[ ]:


accuracy_score(y_val, knn.predict(X_val_scaled))


# In[ ]:


confusion_matrix(y_val, knn.predict(X_val_scaled))


# ### Logistic Regression

# Predict (Closed Form)
# 
# $$\hat p = \sigma(\theta^T \cdot X)$$
# $$\hat y = 
# \begin{cases}
# 0, & \text{if $\hat p \le$ 0.5} \\
# 1, & \text{if $\hat p \geq$ 0.5}
# \end{cases}
# $$
# 
# where
# 
# $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
# 
# Cost
# 
# $$
# c(\theta) =
# \begin{cases}
# -\log(\hat y),  & \text{if y = 1} \\
# -\log(1 - \hat y), & \text{if y = 0}
# \end{cases}
# $$

# In[ ]:


log_reg = LogisticRegression(n_jobs=-1)


# In[ ]:


log_reg.fit(X_train, y_train)


# In[ ]:


accuracy_score(y_val, log_reg.predict(X_val))


# In[ ]:


confusion_matrix(y_val, log_reg.predict(X_val))


# ### SGD Classifier

# Cost Derivative
# 
# $$
# \frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^m (\sigma(\theta^T \cdot x^{(i)}) - y^{(i)})x^{(i)}_{j}
# $$
# 
# Training
# 
# $$
# \theta_{j+1} = \theta_j - \eta \nabla MSE(\theta)
# $$

# In[ ]:


sgd_cls = SGDClassifier(n_jobs=-1)


# In[ ]:


sgd_cls.fit(X_train_scaled, y_train)


# In[ ]:


accuracy_score(y_val, sgd_cls.predict(X_val_scaled))


# In[ ]:


confusion_matrix(y_val, sgd_cls.predict(X_val_scaled))


# ### SVC

# Minimise
# 
# $$\frac{1}{2} w^T \cdot w$$
# 
# given
# 
# $$t(w^T\cdot x + b) \geq 1$$

# In[ ]:


svc_cls = SVC()


# In[ ]:


svc_cls.fit(X_train_scaled, y_train)


# In[ ]:


accuracy_score(y_val, svc_cls.predict(X_val_scaled))


# In[ ]:


confusion_matrix(y_val, svc_cls.predict(X_val_scaled))


# ### Decision Tree Classifier

# Split on highest information gain

# In[ ]:


tree = DecisionTreeClassifier()


# In[ ]:


tree.fit(X_train, y_train)


# In[ ]:


accuracy_score(y_val, tree.predict(X_val))


# In[ ]:


confusion_matrix(y_val, tree.predict(X_val))


# ### Ensembles

# #### Random Forest Classifier

# In[ ]:


rf = RandomForestClassifier(n_jobs=-1, oob_score=True)


# In[ ]:


rf.fit(X_train, y_train)


# In[ ]:


rf.oob_score_


# In[ ]:


accuracy_score(y_val, rf.predict(X_val))


# In[ ]:


confusion_matrix(y_val, rf.predict(X_val))


# #### GradientBoosting Classifier

# In[ ]:


xgb_cls = GradientBoostingClassifier()


# In[ ]:


xgb_cls.fit(X_train, y_train)


# In[ ]:


accuracy_score(y_val, xgb_cls.predict(X_val))


# In[ ]:


confusion_matrix(y_val, xgb_cls.predict(X_val))


# #### Extremely Randomised Trees

# In[ ]:


etree = ExtraTreesClassifier(oob_score=True, n_jobs=-1, bootstrap=True)


# In[ ]:


etree.fit(X_train, y_train)


# In[ ]:


accuracy_score(y_val, etree.predict(X_val))


# In[ ]:


etree.oob_score_


# In[ ]:


confusion_matrix(y_val, etree.predict(X_val))


# # Q4. Can we verify the model performances? (Cross Validation)

# Selected models
# 1. ExtraTreesClassifier
# 2. GradientBoostingClassifier
# 3. DecisonTreeClassifier
# 4. RandomForestClassifier

# In[ ]:


def display_scores(scores):
    print(scores)
    print('Mean: {}'.format(scores.mean()))
    print('Std: {}'.format(scores.std()))


# ## ExtraTreesClassifier CV

# In[ ]:


etree_scores = cross_val_score(etree, X_train, y_train, cv=10, n_jobs=-1)


# In[ ]:


display_scores(etree_scores)


# ## GradientBoostingClassifier CV

# In[ ]:


xgb_cls_scores = cross_val_score(xgb_cls, X_train, y_train, cv=10, n_jobs=-1)


# In[ ]:


display_scores(xgb_cls_scores)


# ## DecisionTreeClassifier CV

# In[ ]:


tree_scores = cross_val_score(tree, X_train, y_train, cv=10, n_jobs=-1)


# In[ ]:


display_scores(tree_scores)


# ## RandomForestClassifier CV

# In[ ]:


rf_scores = cross_val_score(rf, X_train, y_train, cv=10, n_jobs=-1)


# In[ ]:


display_scores(rf_scores)


# Random Forest and Gradient Boosting perform equally well. We will select Random Forest for parameter tuning

# # Q5. Is it possible to fine-tune our model to the problem? (Hyper-parameter tuning)

# In[ ]:


param_grid_rf = {
    'criterion': ['gini', 'entropy'],
    'max_features': [0.5, 0.75, 0.9, 'auto'],
    'min_samples_leaf': [1, 2, 3, 4],
    'n_estimators': [5, 10, 20, 50, 75, 100]
}


# In[ ]:


cv_rf = GridSearchCV(rf, param_grid_rf, n_jobs=-1, refit=True, verbose=1)


# In[ ]:


cv_rf.fit(X_train, y_train)


# In[ ]:


cv_rf.best_params_


# In[ ]:


cv_rf.best_score_


# In[ ]:


final_model = cv_rf.best_estimator_


# # Q6. Can we predict new instances? (Deployment and Prediction)

# Looking at test set

# In[ ]:


accuracy_score(y_test, final_model.predict(X_test))


# In[ ]:


confusion_matrix(y_test, final_model.predict(X_test))


# Precision, Recall and F1 score

# In[ ]:


precision_score(y_test, final_model.predict(X_test), average='weighted')


# In[ ]:


recall_score(y_test, final_model.predict(X_test), average='weighted')


# In[ ]:


f1_score(y_test, final_model.predict(X_test), average='weighted')


# http://skyserver.sdss.org/dr14/en/tools/search/sql.aspx
# 
# SELECT TOP 20 s.specObjID, s.ra, s.dec, s.z, s.class FROM SpecObjAll as s WHERE s.dec >= 0 AND s.dec <= 1.1

# # Pipeline

# ## Frame problem and look at the big picture

# ## Setup data sources

# ## Explore data

# ## Prepare data

# ## Short-list promising models

# ## Fine-tune the system

# ## Communicate your results

# ## Deploy

# In[ ]:




