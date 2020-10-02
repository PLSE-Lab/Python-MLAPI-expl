#!/usr/bin/env python
# coding: utf-8

# # Heart Disease UCI

# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.
# 
# Can we find a corrleation between the features and the presence of heart dieases?

# ## Features descriptions
# 
# Attribute Information: 
# - 1. age: age
# - 2. sex: sex
# - 3. cp: chest pain type (4 values) 
# - 4. trespbps: resting blood pressure 
# - 5. chol: serum cholestoral in mg/dl 
# - 6. fbs: fasting blood sugar > 120 mg/dl
# - 7. restecg: resting electrocardiographic results (values 0,1,2)
# - 8. thalach: maximum heart rate achieved 
# - 9. exang: exercise induced angina 
# - 10. oldpeak: ST depression induced by exercise relative to rest 
# - 11. slope: the slope of the peak exercise ST segment 
# - 12. ca: number of major vessels (0-3) colored by flourosopy 
# - 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# ## Imports and configurations

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
from mlxtend.classifier import StackingClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
defaultcolor = '#66ccff'
pd.options.display.float_format = '{:.2f}'.format
rc={'savefig.dpi': 75, 'figure.autolayout': False, 'figure.figsize': [12, 8], 'axes.labelsize': 18,   'axes.titlesize': 18, 'font.size': 18, 'lines.linewidth': 2.0, 'lines.markersize': 8, 'legend.fontsize': 16,   'xtick.labelsize': 16, 'ytick.labelsize': 16}
sns.set(style='ticks',rc=rc)
sns.set_palette('husl')


# ## Overall look at the data trying to find any quick insight

# In[ ]:


df = pd.read_csv('../input/heart.csv')
df.head()


# In[ ]:


df.describe()


# Let's try to get quick insights about the data

# In[ ]:


fig, ax = plt.subplots()
sns.countplot(df.target, ax=ax)
for i,p in enumerate(ax.patches):
    ax.annotate('{:.2f}%'.format(df['target'].value_counts().apply(lambda x: 100*x/df['target'].value_counts().sum())[i]), (p.get_x()+0.32, p.get_height()+1)).set_fontsize(15)
ax.set_ylabel("")
ax.set_xlabel("")
ax.set_title("Target distribution");


# We have a farely well distrubited dataset so we won't have to worry to much with the model "remebering" the train targets.

# In[ ]:


fig, ax = plt.subplots(figsize=[15,15])
df.hist(ax=ax, bins=30, color='b');


# In[ ]:


fig, ax = plt.subplots(figsize=[20,15])
sns.heatmap(df.corr(), ax=ax, cmap='Blues', annot=True);
ax.set_title("Pearson correlation coefficients", size=20);


# ## Exploratory data analysis

# ### Age distribution

# In[ ]:


fig, ax = plt.subplots()
df.groupby(['age', 'target']).size().reset_index().pivot(index='age', columns='target', values=0).fillna(0).plot.bar(stacked=True, ax=ax)
ax.set_title("Distribution of the target according to the age")
ax.set_xlabel("");


# First thing that can be noticed is that for ages between 40 and 50 the proportion of target=1 is pretty high comparing to pepole with ages ranging from 57 to 67. 

# In[ ]:


fig, ax = plt.subplots()
sns.scatterplot(x='age', y='thalach', data=df[df.target==1], color='b', ax=ax)
sns.scatterplot(x='age', y='thalach', data=df[df.target==0], color='r', ax=ax)
ax.legend(['1', '0']);


# We can see we have a good separtion here between 1 and 0, which is good for our model.

# ### Target and cp

# In[ ]:


sns.heatmap(df.groupby(['exang', 'cp']).size().reset_index().pivot(columns='exang', index='cp', values=0), cmap='Blues', fmt='g', annot=True);


# ### Slope and thalach correlation

# In[ ]:


fig, ax = plt.subplots()
sns.boxplot(x='slope', y='thalach', data=df, ax=ax);
ax.set_title("Thalach distribution by slope values");


# Looks like we don't have a big difference between 0 and 1 but when slope=2 the thalach distribution get's narrower and higher.

# ### Thalach and cp

# In[ ]:


sns.violinplot(x='cp', y='thalach', data=df);


# ### Slope and oldpeak

# In[ ]:


sns.boxplot(x='slope', y='oldpeak', data=df);


# We can see a clear correlation between these two features, why is that?
# 
# After searching around in the internet we can found out that both of them are metrics to evaluate the ST segment and having an low oldpeak means you probably will have a high slope.
# 
# There is a lot of bilbiography on the internet about those features. 

# Here is a figure representing the ST segment.

# <img src='https://www.teachingmedicine.com/media/lessons/images/Screen%20Shot%202014-06-01%20at%204_12_09%20PM.png'></img>
# 
# <p>Extracted from <a href='https://www.teachingmedicine.com/Lesson.aspx?l_id=139'>teachingmedicine</a></p>

# ## Baseline models

# #### Let's first split into training and testing and create a dictionary with the most common models

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df.target, test_size=0.2, random_state=56)


# In[ ]:


models = {
    'CART': DecisionTreeClassifier(),
    'SVC': SVC(probability=True),
    'XGB': XGBClassifier(n_jobs=-1),
    'GNB': GaussianNB(),
    'LDA': LinearDiscriminantAnalysis(),
    'LR': LogisticRegression(),
    'KNN': KNeighborsClassifier()
}


# #### Some useful functions

# In[ ]:


def cv_report(models, X, y):
    results = []
    for name in models.keys():
        model = models[name]
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print("Accuracy: %.3f (+/- %.3f) [%s]" %(scores.mean(), scores.std(), name))


# In[ ]:


cv_report(models, X_train, y_train)


# As we can see the baseline models already had a good performance, let's try to improve them.

# #### Hyperparameters tunneling using gridsearch

# ##### XGBoost

# In[ ]:


xgb_params = {
    'max_depth': [2,3,4],
    'n_estimators': [50, 100, 400, 1000],
    'learning_rate': [0.1, 0.01, 0.05]
}

xg_grid = GridSearchCV(models['XGB'], xgb_params, cv=5)
models['XGB_Grid'] = xg_grid


# In[ ]:


cv_report(models, X_train, y_train)


# ##### Logistic Regression

# In[ ]:


lr_params = [{
                'penalty': ['l2'],
                'C': (0.1, 0.5, 1.0, 1.5, 2.0),
                'solver': ['newton-cg', 'lbfgs', 'sag'],
                'max_iter': [50, 100, 200, 500]
            },
            {
                'penalty': ['l1', 'l2'],
                'C': (0.1, 0.5, 1.0, 1.5, 2.0),
                'solver': ['liblinear', 'saga']
            }
]

lr_grid = GridSearchCV(models['LR'], lr_params, cv=5)
models['LR_Grid'] = lr_grid


# In[ ]:


cv_report(models, X_train, y_train)


# Let's use the tunneled LR for predicting in the test set

# In[ ]:


models['LR_Grid'].fit(X_train, y_train)


# In[ ]:


predictions = models['LR_Grid'].predict(X_test)


# In[ ]:


print("Accuracy of the model: {:.2f}%".format(100*accuracy_score(predictions, y_test)))


# In[ ]:


fig, ax = plt.subplots()
ax.set_title("Confusion Matrix")
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap='Blues');

