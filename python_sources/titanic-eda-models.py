#!/usr/bin/env python
# coding: utf-8

# <img src='https://cdn.radiofrance.fr/s3/cruiser-production/2018/10/bfca8681-cbe1-4396-b725-b3bf51e6e68e/870x489_titanic_uppa_photoshot_maxnewsfr040608.jpg' width='800'>

# # Data Description

# >* PassengerId: type should be integers 
# >* Survived: Survived or Not 
# >* Pclass: Class of Travel
# >* Name: Name of Passenger
# >* Sex: Gender
# >* Age: Age of Passengers
# >* SibSp: Number of Sibling/Spouse aboard
# >* Parch: Number of Parent/Child aboard
# >* Ticket: Ticket number
# >* Fare: Fare for ticket
# >* Cabin: Cabin Number
# >* Embarked: The port in which a passenger has embarked. C - Cherbourg, S - Southampton, Q = Queenstown

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)
import seaborn as sns
plt.style.use("fivethirtyeight")
get_ipython().run_line_magic('matplotlib', 'inline')

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')

import os
os.listdir('../input')


# In[ ]:


df = pd.read_csv("../input/train.csv")
df.sample(5)


# In[ ]:


df.info()


# In[ ]:


null_sum = pd.DataFrame(df.isnull().sum(), columns=['Sum'])
null_percent = pd.DataFrame((df.isnull().sum()/df.shape[0])*100, columns=['Percent'])
total = pd.concat([null_sum, null_percent], axis=1)
total.sort_values(['Sum', 'Percent'], ascending=False)


# >* Cabin and Age columns has most null values

# In[ ]:


''' Droping Cabin column and filling Age column null values with mean '''
df.drop('Cabin', axis=1, inplace=True)
df["Age"] = df["Age"].fillna(value=df.Age.mean())


# In[ ]:


def pie_plot(cnts, colors, title):
    labels = cnts.index
    values = cnts.values
    
    trace = go.Pie(labels=labels,
                   values=values,
                   title=title,
                   textinfo='value',
                   hoverinfo='label+percent',
                   hole=.4,
                   textposition='inside',
                   marker=dict(colors=colors,
                               line=dict(color='#000000', width=2)
                              )
                  )
    layout = go.Layout(hovermode='closest')
    fig = go.Figure(data=[trace], layout=layout)
    return py.iplot(fig)


# In[ ]:


pie_plot(df['Embarked'].value_counts(), colors=['yellow','orange', 'cyan'], title='Embarked')


# >* Most passengers got on board from Southampton

# In[ ]:


''' Filling null values with most common value '''
df["Embarked"] = df["Embarked"].fillna("S")


# # How many people Survived?

# In[ ]:


pie_plot(df['Survived'].value_counts(), colors=['gold','cyan'], title='Survived?')


# >* 38.4% of people Survived

# # Who Survived more Male or Female?

# In[ ]:


plt.figure(figsize=(10,10))
plt.style.use('ggplot')
sns.countplot(df['Survived'], hue=df['Sex'], palette='plasma')
plt.title("Survived vs Sex")
plt.show()


# >* Females Survived the most

# # Which class survived the most?

# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(x='Survived', hue='Pclass', data=df, palette='inferno')
plt.title('Survived vs Pclass')
plt.show()


# >* First Class Passengers Survived the most

# # Exploring relations

# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
sns.boxplot(x="Pclass", y="Age", data=df)
plt.title('Pclass vs Age')

plt.subplot(1,2,2)
sns.boxplot(x="Survived", y="Fare", data=df)
plt.title('Survived vs Fare')

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()

plt.figure(figsize=(15, 8))
sns.boxplot(x='Survived', y='Age', data=df);


# >* Higher Age passengers are mostly in First Class
# >* As First Class Passengers have survived the most there is a direct relation with Fare
# >* Higher Age passengers survived the most which is because most higher age passengers travelled in first class and most first class passengers Survived

# In[ ]:


pd.DataFrame(df.corr()['Survived'].sort_values(ascending=False))


# >* Fare, Parch are quite correlated with survived

# In[ ]:


plt.figure(figsize=(10,10), dpi=100)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm');


# # Feature Engineering

# In[ ]:


''' Extracting prefix from names '''
df["Title"] = df["Name"].str.extract("([A-Za-z]+)\.", expand=False)
df["Title"].unique()


# In[ ]:


''' Correcting spelling mistakes in prefixes and naming rarely used prefixes with Rare'''
df["Title"] = df["Title"].replace(["Don", "Rev", "Dr", "Major", "Lady", 
                                         "Sir", "Col", "Capt", "Countess", "Jonkheer"], "Rare")

df["Title"] = df["Title"].replace("Mlle", "Mrs")
df["Title"] = df["Title"].replace("Ms", "Miss")
df["Title"] = df["Title"].replace("Mme", "Mrs")

df[["Title", "Survived"]].groupby("Title", as_index=False).mean()


# In[ ]:


'''Mapping male as 1 and Female as 0'''
df["Sex"] = df["Sex"].map({"male": 1, "female":0})


# In[ ]:


'''Bining the values of Fare'''
df["Fareband"] = pd.qcut(df["Fare"], 4)
df[["Fareband", "Survived"]].groupby("Fareband", as_index=False).mean().sort_values(by="Fareband", ascending=True)


# In[ ]:


df.loc[df["Fare"] <= 7.91, "Fare"] = 0
df.loc[(df["Fare"] > 7.91) & (df["Fare"] <= 14.454), "Fare"] = 1
df.loc[(df["Fare"] > 14.454) & (df["Fare"] <= 31.0), "Fare"] = 2
df.loc[df["Fare"] > 31.0, "Fare"] = 3
df["Fare"] = df["Fare"].astype("int")


# In[ ]:


'''Bining the values of Age'''
df["Ageband"] = pd.cut(df["Age"], 4)
df[["Ageband", "Survived"]].groupby("Ageband", as_index=False).mean().sort_values(by="Ageband", ascending=True)


# In[ ]:


df.loc[df["Age"] <= 20.315, "Age"] = 0
df.loc[(df["Age"] > 20.315) & (df["Age"] <= 40.21), "Age"] = 1
df.loc[(df["Age"] > 40.21) & (df["Age"] <= 60.105), "Age"] = 2
df.loc[(df["Age"] > 60.105) & (df["Age"] <= 80.0), "Age"] = 3
df["Age"] = df["Age"].astype("int")


# In[ ]:


'''Creating isAlone Feature to see if the person is alone or not'''
df["isAlone"] = df["SibSp"] + df["Parch"] + 1
df["isAlone"].loc[df["isAlone"] > 1] = 0
df["isAlone"].loc[df["isAlone"] == 1] = 1


# # Who Survived?

# In[ ]:


def horizontal_bar_chart(cnt_srs, color):
    trace = go.Bar(
        y = cnt_srs.index[::-1],
        x = cnt_srs.values[::-1],
        showlegend = False,
        orientation = 'h',
        marker = dict(
            color = color,
        ),
    )
    return trace

cnt_srs = df.groupby('Title')["Survived"].agg(['count', 'mean'])
cnt_srs.columns = ["count", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace_1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace_2 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')


cnt_srs = df.groupby('Age')["Survived"].agg(['count', 'mean'])
cnt_srs.columns = ["count", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace_3 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(71, 58, 131, 0.6)')
trace_4 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(71, 58, 131, 0.6)')


cnt_srs = df.groupby('Fare')["Survived"].agg(['count', 'mean'])
cnt_srs.columns = ["count", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace_5 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(246, 78, 139, 0.6)')
trace_6 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(246, 78, 139, 0.6)')


fig = tools.make_subplots(3, 2, vertical_spacing=0.04,
                         subplot_titles=["Count-Title", "Mean-Title",
                                         "Count-Ageband", "Mean-Ageband",
                                         "Count-Fareband", "Mean-Fareband"])

fig.append_trace(trace_1, 1, 1)
fig.append_trace(trace_2, 1, 2)
fig.append_trace(trace_3, 2, 1)
fig.append_trace(trace_4, 2, 2)
fig.append_trace(trace_5, 3, 1)
fig.append_trace(trace_6, 3, 2)

fig['layout'].update(height=1200, width=1200, paper_bgcolor='rgb(233, 233, 233)', title="Survived by Columns")
py.iplot(fig, filename="survived-by-cols")


# >* Count and Mean Survived by title, ageband, fareband

# # Imbalanced ?

# In[ ]:


df['Survived'].value_counts()


# >* target column is imbalanced
# >* We have handled this later with resampling

# In[ ]:


plt.figure(figsize=(10,10))
sns.countplot(df['Survived'], palette='Set1');


# # Preprocessing

# In[ ]:


df.drop(['PassengerId', 'Ageband', 'Fareband', 'Name', 'Ticket'], axis=1, inplace=True)


# In[ ]:


'''One Hot Encoding'''
objects = df.select_dtypes(include=['object'])
objects = pd.get_dummies(objects, drop_first=True)
df.drop(df.select_dtypes(include=['object']), axis=1, inplace=True)
df = pd.concat([df, objects], axis=1)
df.head()


# In[ ]:


X = df.drop('Survived', axis=1)
y = df["Survived"]


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn import svm
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[ ]:


df_train, df_test, Y_train, Y_test = train_test_split(X, y , test_size=0.2, random_state=101)


# # Resampling

# <img src = "https://raw.githubusercontent.com/rafjaa/machine_learning_fecib/master/src/static/img/resampling.png" width='800'>

# * Undersampling
# >* Some samples of majority class are removed till it becomes equal to the samples of minority class
# 
# * Oversampling
# >* Copies of monority class samples are made till it becomes equal to the samples of majority class

# In[ ]:


'''OverSampling'''
X = pd.concat([df_train, Y_train], axis=1)

not_survived = X[X.Survived==0]
survived = X[X.Survived==1]

over_sampled = resample(survived, 
                        replace=True, # Samples with replacement
                        n_samples=len(not_survived), # Number of samples
                        random_state=27)

over_sampled = pd.concat([not_survived, over_sampled])

over_sampled.Survived.value_counts()


# In[ ]:


plt.figure(figsize=(8,8))
sns.countplot(over_sampled['Survived'], palette='Set2');


# In[ ]:


over_train = over_sampled.drop('Survived', axis=1)
y_over_train = over_sampled['Survived']

lg = LogisticRegression()
lg.fit(over_train, y_over_train)
lg_pred = lg.predict(df_test)
print('OverSampling Accuracy:', metrics.accuracy_score(Y_test, lg_pred))


# In[ ]:


'''UnderSampling'''

under_sampled = resample(not_survived,
                         replace=False, # sample without replacement
                         n_samples=len(survived), # Number of Samples
                         random_state=27
                        )
under_sampled = pd.concat([survived, under_sampled])

under_sampled.Survived.value_counts()


# In[ ]:


plt.figure(figsize=(8,8))
sns.countplot(under_sampled['Survived'], palette='Set1');


# In[ ]:


X_train = under_sampled.drop('Survived', axis=1)
y_train = under_sampled['Survived']

lg = LogisticRegression()
lg.fit(X_train, y_train)
lg_pred = lg.predict(df_test)
print('UnderSampling Accuracy:', metrics.accuracy_score(Y_test, lg_pred))


# # SMOTE

# SMOTE (Synthetic Minority Oversampling TEchnique) consists of synthesizing elements for the minority class, based on those that already exist. It works randomly picking a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.

# <img src="https://raw.githubusercontent.com/rafjaa/machine_learning_fecib/master/src/static/img/smote.png" width='800'>

# In[ ]:


X = df.drop('Survived', axis=1)
y = df["Survived"]

smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X, y)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X_sm, y_sm , test_size=0.2, random_state=101)


# In[ ]:


lg = LogisticRegression()
lg.fit(x_train, y_train)
lg_pred = lg.predict(x_test)
print('SMOTE Accuracy:', metrics.accuracy_score(y_test, lg_pred))


# * OverSampling gives the best result so we will be using oversampled data

# # Logistic Regression

# In[ ]:


x_train = over_train
y_train = y_over_train
x_test = df_test
y_test = Y_test


# In[ ]:


folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

lr = LogisticRegression()

gs_Log = GridSearchCV(lr, param_grid=params, cv=folds, n_jobs=-1, scoring='accuracy', return_train_score=True, 
                      verbose=1)

gs_Log.fit(x_train, y_train)


# In[ ]:


gs_Log.best_params_


# In[ ]:


gs_results = pd.DataFrame(gs_Log.cv_results_)
gs_results


# In[ ]:


plt.plot(gs_results['param_C'], gs_results['mean_test_score'])
plt.plot(gs_results['param_C'], gs_results['mean_train_score'])
plt.legend(["Test Score", "Train Score"], loc="lower right")
plt.xscale("log")


# # SVM

# In[ ]:


svc = SVC()

'''Finding right params'''
params = {"C": [0.01, 0.1, 1, 10, 100],
          "gamma": [1e-1, 1e-2, 1e-3, 1e-4],
          "kernel" : ["linear", "rbf", "sigmoid", "poly"]}

svc_gs = GridSearchCV(svc, param_grid=params, cv=folds, scoring="accuracy", n_jobs=-1, return_train_score=True, 
                      verbose=1)

svc_gs.fit(x_train, y_train)


# In[ ]:


svc_gs.best_params_


# In[ ]:


svc_results = pd.DataFrame(svc_gs.cv_results_)
svc_results.head()


# In[ ]:


svc_results['param_C'] = svc_results['param_C'].astype('int')

fig, ax = plt.subplots(2,2, figsize=(12,10))
gamma_01 = svc_results.loc[(svc_results['param_gamma'] == 0.1) & (svc_results['param_kernel'] == 'rbf')]
sns.lineplot(gamma_01['param_C'], gamma_01['mean_test_score'], ax=ax[0,0])
sns.lineplot(gamma_01['param_C'], gamma_01['mean_train_score'], ax=ax[0,0])
ax[0,0].set_title("Gamma=0.1")
ax[0,0].set_ylim([0.50, 1.0])
ax[0,0].set_xscale("log")
ax[0,0].legend(["Test Score", "Train Score"], loc="lower right")


gamma_01 = svc_results.loc[(svc_results['param_gamma'] == 0.01) & (svc_results['param_kernel'] == 'rbf')]
ax[0,1].plot(gamma_01['param_C'], gamma_01['mean_test_score'])
ax[0,1].plot(gamma_01['param_C'], gamma_01['mean_train_score'])
ax[0,1].set_title("Gamma=0.01")
ax[0,1].set_ylim([0.50, 1.0])
ax[0,1].set_xscale("log")
ax[0,1].legend(["Test Score", "Train Score"], loc="lower right")


gamma_01 = svc_results.loc[(svc_results['param_gamma'] == 0.001) & (svc_results['param_kernel'] == 'rbf')]
ax[1,0].plot(gamma_01['param_C'], gamma_01['mean_test_score'])
ax[1,0].plot(gamma_01['param_C'], gamma_01['mean_train_score'])
ax[1,0].set_title("Gamma=0.001")
ax[1,0].set_ylim([0.50, 1.0])
ax[1,0].set_xscale("log")
ax[1,0].legend(["Test Score", "Train Score"], loc="lower right")


gamma_01 = svc_results.loc[(svc_results['param_gamma'] == 0.0001) & (svc_results['param_kernel'] == 'rbf')]
ax[1,1].plot(gamma_01['param_C'], gamma_01['mean_test_score'])
ax[1,1].plot(gamma_01['param_C'], gamma_01['mean_train_score'])
ax[1,1].set_title("Gamma=0.0001")
ax[1,1].set_ylim([0.50, 1.0])
ax[1,1].set_xscale("log")
ax[1,1].legend(["Test Score", "Train Score"], loc="lower right")

plt.tight_layout()


# In[ ]:


final_svc = SVC(C=1, gamma=0.1, kernel="rbf")
final_svc.fit(x_train, y_train)


# In[ ]:


'''Feature Importance'''
perm = PermutationImportance(final_svc, random_state=1).fit(x_train, y_train)
eli5.show_weights(perm, feature_names=x_train.columns.tolist())


# # Different Algorithms

# In[ ]:


kfold = KFold(5, random_state=10)
accuracy = []

classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN','Decision Tree','Naive Bayes','Random Forest']

models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(C=1),
        KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),GaussianNB(),
        RandomForestClassifier(n_estimators=100)]

for model in models:
    cv_result = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    accuracy.append(cv_result.mean())
new_models_df = pd.DataFrame({"CV Mean": accuracy}, index=classifiers)
new_models_df


# >* Radial SVM gives the best score

# # Features Impact

# In[ ]:


import shap 
rf = RandomForestClassifier(n_estimators=1000, max_depth=2)
rf.fit(x_train, y_train)


# In[ ]:


exp = shap.TreeExplainer(rf, x_train)
shap_values = exp.shap_values(x_test)

shap.summary_plot(shap_values[1], x_test)


# In[ ]:


shap.initjs()
shap.force_plot(exp.expected_value[0], shap_values[0], x_test)


# # Reference:-
# 
# >* https://www.kaggle.com/shahules/tackling-class-imbalance

# # UpVote if this was helpful
# >* I would be updating it more 
