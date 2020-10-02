#!/usr/bin/env python
# coding: utf-8

# # **Introduction and uploading data**

# ![](https://img.webmd.com/dtmcms/live/webmd/consumer_assets/site_images/articles/health_tools/did_you_know_this_could_lead_to_heart_disease_slideshow/493ss_thinkstock_rf_heart_illustration.jpg)<br>**Hi, everyone! That's my analysis and classification for Heart Disease UCI.** Here you can find general analysis, comparison between different variables and investigation of features importance. If you find my notebook interesting and helpful, please **UPVOTE.** Enjoy the analysis :)

# **Import packages**

# In[ ]:


# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='ticks', rc={'figure.figsize':(15, 10)})

# machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve


# **Acquire data**

# In[ ]:


data = pd.read_csv("../input/heart-disease-uci/heart.csv")


# # **Exploratory Data Analysis (EDA)**

# **Let's learn some info about our data.** For that I create a function which can show us missing ratio, distincts, skewness, etc.

# In[ ]:


data.head()


# In[ ]:


def detailed_analysis(df):
  obs = df.shape[0]
  types = df.dtypes
  counts = df.apply(lambda x: x.count())
  nulls = df.apply(lambda x: x.isnull().sum())
  distincts = df.apply(lambda x: x.unique().shape[0])
  missing_ratio = (df.isnull().sum() / obs) * 100
  uniques = df.apply(lambda x: [x.unique()])
  skewness = df.skew()
  kurtosis = df.kurt()
  print('Data shape:', df.shape)

  cols = ['types', 'counts', 'nulls', 'distincts', 'missing ratio', 'uniques', 'skewness', 'kurtosis']
  details = pd.concat([types, counts, nulls, distincts, missing_ratio, uniques, skewness, kurtosis], axis=1)

  details.columns = cols 
  dtypes = details.types.value_counts()
  print('________________________\nData types:\n', dtypes)
  print('________________________')

  return details


# In[ ]:


details = detailed_analysis(data)
details


# In[ ]:


data.describe()


# Wow, our data is totally clear, so we can visualize some things

# **Target value distribution**

# In[ ]:


values = data.target.value_counts()
indexes = values.index

sns.barplot(indexes, values)


# **Pair plot between all variables**

# In[ ]:


sns.pairplot(data=data, vars=data.columns.values[:-1], hue='target')


# **Analysis of different chest types and their influence to the target value**
# <br>Types of pain:
# - Value 0: typical angina
# - Value 1: atypical angina
# - Value 2: non-anginal pain
# - Value 3: asymptomatic

# Here we can see that people with the same chest pain have almost the same age regardless of the sex

# In[ ]:


sns.barplot(x='cp', y='age', data=data, hue='sex', ci=None)


# **Relationship between chest pain and different variables separated by target value.**
# 1. Here we can find out that fbs has significantly various values which are dependent on the chest pain 
# 2. Resting ecg results with normal values mean that patient hasn't heart diseases (exception: asymptomatic chest pain, value 3)
# 3. If exang is 1 a patient must be healthy (exception: asymptomatic chest pain, value 3)
# 4. If oldpeak is high a patient must be healthy (exception: asymptomatic chest pain, value 3)
# 5. It's better if slope has low value (again asymptomatic chest pain as an exception)
# 6. High number of ca (major vessels) is always great
# 7. It's good when thal nearly equals 3

# In[ ]:


fig = plt.figure(figsize=(20, 25))
palettes = ['Greens', 'Purples', 'YlOrRd', 'RdBu', 'BrBG', 'cubehelix'] * 2

for x in range(10):
    fig1 = fig.add_subplot(5, 2, x+1)
    sns.barplot(x='cp', y=data.columns.values[x+3], data=data, hue='target', ci=None, palette=palettes[x])


# **Correlation heatmap**

# In[ ]:


correlation = data.corr()

fig = plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, center=1, cmap='RdBu')


# **Relationship between slope and oldpeak**
# <br><br>This plot confirms our statement that lower slope is better. According to the jointplot lower slope values have higher oldpeak values which mean our patient is healthy

# In[ ]:


sns.jointplot(x='slope', y='oldpeak', data=data, height=10)


# **Violin plots for all variables**
# <br><br>Here we can investigate things about features importance too. If plots for 0 and 1 are the same it means that correlation is low. Moreover we can see smooth values distribution for each variable

# In[ ]:


fig = plt.figure(figsize=(20, 25))
palettes = ['Greens', 'Purples', 'YlOrRd', 'RdBu', 'BrBG', 'cubehelix'] * 2

for x in range(12):
    fig1 = fig.add_subplot(6, 2, x+1)
    sns.violinplot(x='target', y=data.columns.values[x], data=data, palette=palettes[x])


# **SelectKBest**
# <br><br>Finally for EDA we're gonna check the best features using SelectKBest

# In[ ]:


X = data.drop('target', axis=1)
y = data.target

selector = SelectKBest(score_func=chi2, k=5)
fitted = selector.fit(X, y)
features_scores = pd.DataFrame(fitted.scores_)
features_columns = pd.DataFrame(X.columns)

best_features = pd.concat([features_columns, features_scores], axis=1)
best_features.columns = ['Feature', 'Score']
best_features.sort_values(by='Score', ascending=False, inplace=True)
best_features


# # **Model**

# **Let's split our data to test and train**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print('Input train shape', X_train.shape)
print('Output train shape', y_train.shape)
print('Input test shape', X_test.shape)
print('Output test shape', y_test.shape)


# **Now we're gonna scale our data**

# In[ ]:


sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

type(X_train), type(X_test)


# **So, we can test some classification algorithms on our data. Also we create a DataFrame to collect scores**

# In[ ]:


scores = pd.DataFrame(columns=['Model', 'Score'])


# **Also we define a function to show additional metrics (Confusion Matrix and ROC Curve)**

# In[ ]:


def show_metrics():
    fig = plt.figure(figsize=(25, 10))

    # Confusion matrix
    fig.add_subplot(121)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)

    # ROC Curve
    fig.add_subplot(122)

    ns_probs = [0 for _ in range(len(y_test))]
    p_probs = model.predict_proba(X_test)[:, 1]

    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, p_probs)

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='o', label='Logistic')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend()
    plt.show()


# **Logistic Regression**

# In[ ]:


model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)
scores = scores.append({'Model': 'Logistic Regression', 'Score': accuracy}, ignore_index=True)


# In[ ]:


show_metrics()


# **Support Vector Classifier (SVC)**

# In[ ]:


model = SVC(probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)
scores = scores.append({'Model': 'SVC', 'Score': accuracy}, ignore_index=True)


# In[ ]:


show_metrics()


# **Random Forest Classifier**

# In[ ]:


model = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [50, 100, 200, 300], 'max_depth': [2, 3, 4, 5]}, cv=4)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100, model.best_params_)
scores = scores.append({'Model': 'Random Forest', 'Score': accuracy}, ignore_index=True)


# In[ ]:


show_metrics()


# **Gradien Boosting Classifier**

# In[ ]:


model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)
scores = scores.append({'Model': 'Gradient Boosting', 'Score': accuracy}, ignore_index=True)


# In[ ]:


show_metrics()


# **Extra Trees Classifier**

# In[ ]:


model = GridSearchCV(estimator=ExtraTreesClassifier(), param_grid={'n_estimators': [50, 100, 200, 300], 'max_depth': [2, 3, 4, 5]}, cv=4)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)
scores = scores.append({'Model': 'Extra Trees', 'Score': accuracy}, ignore_index=True)


# In[ ]:


show_metrics()


# **K-Neighbors Classifier**

# In[ ]:


model = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': [1, 2, 3]}, cv=4)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)
scores = scores.append({'Model': 'K-Neighbors', 'Score': accuracy}, ignore_index=True)


# In[ ]:


show_metrics()


# **Gaussian Naive Bayes**

# In[ ]:


model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)
scores = scores.append({'Model': 'Gaussian NB', 'Score': accuracy}, ignore_index=True)


# In[ ]:


show_metrics()


# **Decision Tree Classifier**

# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)
scores = scores.append({'Model': 'Decision Tree', 'Score': accuracy}, ignore_index=True)


# In[ ]:


show_metrics()


# XGB Classifier

# In[ ]:


model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)
scores = scores.append({'Model': 'XGB Classifier', 'Score': accuracy}, ignore_index=True)


# In[ ]:


show_metrics()


# **Finally, let's review our scores**

# In[ ]:


scores.sort_values(by='Score', ascending=False)


# In[ ]:


sns.lineplot(x='Model', y='Score', data=scores)


# **Top-3 are Random Forest, K-Neighbors and Extra Trees.**

# # Tuning and Ensemble Stacking

# **Ok, now let's tune XGBoost Classifier and try to get better score.** We select our params and model. We'll tune it gradually to save time. **At first we tune max_depth and min_child_weight**

# In[ ]:


params = {
  'max_depth': range(2, 8, 2),
  'min_child_weight': range(1, 8, 2)
  }

xgb = XGBClassifier(learning_rate=0.1, n_estimators=200, objective='binary:logistic',
                    silent=True, nthread=4, max_depth=6, min_child_weight=1, tree_method='gpu_hist',
                    gamma=0, subsample=1, colsample_bytree=1, scale_pos_weight=1, seed=228)

grid_search = GridSearchCV(xgb, params, n_jobs=2, cv=5, scoring='accuracy', verbose=1)

grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_, grid_search.best_score_


# Let's go deeper now

# In[ ]:


params = {
  'max_depth': [3, 4, 5],
  'min_child_weight': [4, 5, 6]
  }

xgb = XGBClassifier(learning_rate=0.1, n_estimators=200, objective='binary:logistic',
                    silent=True, nthread=4, max_depth=4, min_child_weight=5, tree_method='gpu_hist',
                    gamma=0, subsample=1, colsample_bytree=1, scale_pos_weight=1, seed=228)

grid_search = GridSearchCV(xgb, params, n_jobs=2, cv=5, scoring='accuracy', verbose=1)

grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_, grid_search.best_score_


# In[ ]:


params = {
  'min_child_weight': [1, 4, 6, 7, 8, 10, 12]
  }

xgb = XGBClassifier(learning_rate=0.1, n_estimators=200, objective='binary:logistic',
                    silent=True, nthread=4, max_depth=3, min_child_weight=6, tree_method='gpu_hist',
                    gamma=0, subsample=1, colsample_bytree=1, scale_pos_weight=1, seed=228)

grid_search = GridSearchCV(xgb, params, n_jobs=2, cv=5, scoring='accuracy', verbose=1)

grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_, grid_search.best_score_


# **Now let's tune gamma**

# In[ ]:


params = {
  'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
  }

xgb = XGBClassifier(learning_rate=0.1, n_estimators=200, objective='binary:logistic',
                    silent=True, nthread=4, max_depth=3, min_child_weight=6, tree_method='gpu_hist',
                    gamma=0, subsample=1, colsample_bytree=1, scale_pos_weight=1, seed=228)

grid_search = GridSearchCV(xgb, params, n_jobs=1, cv=5, scoring='accuracy', verbose=1)

grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_, grid_search.best_score_


# **Tune subsample and colsample_bytree**

# In[ ]:


params = {
  'subsample': [0.6, 0.7, 0.8, 0.9],
  'colsample_bytree': [0.6, 0.7, 0.8, 0.9]
}

xgb = XGBClassifier(learning_rate=0.1, n_estimators=200, objective='binary:logistic',
                  silent=True, nthread=4, max_depth=3, min_child_weight=6, tree_method='gpu_hist',
                  gamma=0, subsample=1, colsample_bytree=1, scale_pos_weight=1, seed=228)

grid_search = GridSearchCV(xgb, params, n_jobs=1, cv=5, scoring='accuracy', verbose=1)

grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_, grid_search.best_score_


# In[ ]:


params = {
    'subsample': [0.55, 0.6, 0.65],
    'colsample_bytree': [0.65, 0.7, 0.75]
  }

xgb = XGBClassifier(learning_rate=0.1, n_estimators=200, objective='binary:logistic',
                    silent=True, nthread=4, max_depth=3, min_child_weight=6, tree_method='gpu_hist',
                    gamma=0, subsample=0.6, colsample_bytree=0.7, scale_pos_weight=1, seed=228)

grid_search = GridSearchCV(xgb, params, n_jobs=1, cv=5, scoring='accuracy', verbose=1)

grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_, grid_search.best_score_


# **Tune regularization parameters**

# In[ ]:


params = {
  'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]
  }

xgb = XGBClassifier(learning_rate=0.1, n_estimators=200, objective='binary:logistic',
                    silent=True, nthread=4, max_depth=3, min_child_weight=6, tree_method='gpu_hist',
                    gamma=0, subsample=0.6, colsample_bytree=0.7, scale_pos_weight=1, seed=228, reg_alpha=0)

grid_search = GridSearchCV(xgb, params, n_jobs=1, cv=5, scoring='accuracy', verbose=1)

grid_search.fit(X_train, y_train)


# In[ ]:


grid_search.best_params_, grid_search.best_score_


# **Now let's perform our new model**

# In[ ]:


grid_search.best_estimator_


# In[ ]:


model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0,
              learning_rate=0.001, max_delta_step=0, max_depth=3,
              min_child_weight=6, missing=None, n_estimators=5000, n_jobs=1,
              nthread=4, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=228,
              silent=True, subsample=0.6, tree_method='gpu_hist', verbosity=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy * 100)
scores = scores.append({'Model': 'XGB Classifier', 'Score': accuracy}, ignore_index=True)


# **Actually, I don't know why the tuned model performs worse than the standart one** even though we have better score while tuning. So, I'm probably gonna fix this later, but now we're gonna do **ensemble stacking**

# In[ ]:


# Here soon

