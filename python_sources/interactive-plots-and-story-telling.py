#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#First-Steps" data-toc-modified-id="First-Steps-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>First Steps</a></span></li><li><span><a href="#Data-Preparation-and-Cleaning" data-toc-modified-id="Data-Preparation-and-Cleaning-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Preparation and Cleaning</a></span><ul class="toc-item"><li><span><a href="#Drop-Unique-Columns" data-toc-modified-id="Drop-Unique-Columns-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Drop Unique Columns</a></span></li><li><span><a href="#Missing-Data" data-toc-modified-id="Missing-Data-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Missing Data</a></span></li></ul></li><li><span><a href="#Data-Exploration" data-toc-modified-id="Data-Exploration-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Exploration</a></span><ul class="toc-item"><li><span><a href="#Embarked" data-toc-modified-id="Embarked-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Embarked</a></span></li><li><span><a href="#Fare" data-toc-modified-id="Fare-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Fare</a></span></li></ul></li><li><span><a href="#Prediction-Models" data-toc-modified-id="Prediction-Models-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Prediction Models</a></span><ul class="toc-item"><li><span><a href="#More-Data-Preparation" data-toc-modified-id="More-Data-Preparation-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>More Data Preparation</a></span><ul class="toc-item"><li><span><a href="#Imputing-Missing-Age-values" data-toc-modified-id="Imputing-Missing-Age-values-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Imputing Missing Age values</a></span></li><li><span><a href="#Standardizing-numerical-values" data-toc-modified-id="Standardizing-numerical-values-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Standardizing numerical values</a></span></li><li><span><a href="#One-hot-encoding-categorial-values" data-toc-modified-id="One-hot-encoding-categorial-values-5.1.3"><span class="toc-item-num">5.1.3&nbsp;&nbsp;</span>One hot encoding categorial values</a></span></li><li><span><a href="#X,-y-preparation-and-train-test-splitting" data-toc-modified-id="X,-y-preparation-and-train-test-splitting-5.1.4"><span class="toc-item-num">5.1.4&nbsp;&nbsp;</span>X, y preparation and train-test splitting</a></span></li></ul></li><li><span><a href="#K-Nearest-Neighbour" data-toc-modified-id="K-Nearest-Neighbour-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>K Nearest Neighbour</a></span></li><li><span><a href="#Support-Vector-Machine-(SVD)" data-toc-modified-id="Support-Vector-Machine-(SVD)-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Support Vector Machine (SVD)</a></span></li><li><span><a href="#Random-Forest-Classifier-(Ensemble-of-Decision-trees)" data-toc-modified-id="Random-Forest-Classifier-(Ensemble-of-Decision-trees)-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Random Forest Classifier (Ensemble of Decision trees)</a></span></li><li><span><a href="#Ensemble-of-previously-trained-classifiers" data-toc-modified-id="Ensemble-of-previously-trained-classifiers-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Ensemble of previously trained classifiers</a></span></li></ul></li></ul></div>

# # Introduction

#     This notebook is for practice purposes, as a participation in a competition on Kaggle.
#     Link to the competition: https://www.kaggle.com/c/titanic
# 
#     The objectives are: 
#         1 - Create interactive plots using with and without using cufflinks.
#         2 - To create a survival prediction model

# # First Steps

# In[ ]:


#Import common modules
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.graph_objs as go
# init_notebook_mode(connected=True)
import plotly
# plotly.tools.set_credentials_file(username='YazanSh', api_key='2GxEQaYp9s6UJAc0bpY5')
# cf.go_offline()


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/train.csv')
data.head()


# In[ ]:


data.columns.to_list()


# In[ ]:


train = data.copy()


# # Data Preparation and Cleaning

# ## Drop Unique Columns

# In[ ]:


train.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


# ## Missing Data

# In[ ]:


missing = pd.DataFrame(train.count(), columns=['Count'])
missing['Missing Values'] = 891 - missing['Count']
missing['%Missing Values'] = (missing['Missing Values'] / 891) * 100
missing.drop('Count', axis=1, inplace=True)
missing = missing[missing['Missing Values'] != 0]
missing


# In[ ]:


sns.set(rc={'figure.figsize':(11.7, 8.27)})
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()


#     Too many missing values for 'Cabin' => Drop the column.
#     
# 
#     Few missing 'Embarked' values => Drop the rows with missing values.
# 
# 
#     Age is possibly an influencing factor for survival, so we need to keep it => Impute missing values.

# In[ ]:


train.drop(['Cabin'], axis=1, inplace=True)


# In[ ]:


train.dropna(axis=0, subset=['Embarked'], inplace=True)


#     As for Age, we will impute it later when exploring the feature, the idea is to find a pattern to better impute the age.

# # Data Exploration

#     We'll Explore the data by visualizing and then noting patterns and relationships between different   
#     variables to try and get an out of the box view as mush as possible, later this can help us impute the    'Age' 
#     which is first in order alphabetically but we'll leave it last to get a better understanding and impute it 
#     better than just replacing missing values with the median.

# ## Embarked

#         Passengers on the Titanic embarked from three locations: Cherbourg, Queenstown and Southampton.
#         Let's explore where the passengers embarked from and is there any obvious influence for survival.

# In[ ]:


# Preparing Data for plots
#General
train.replace({'S':'Southampton', 'C':'Cherbourg', 'Q':'Queenstown'}, inplace=True)
embarked_points = ['Cherbourg', 'Queenstown', 'Southampton']
#Total
embarked_count_total = train.groupby(['Embarked'], as_index=True, sort=True).count()['Survived'].to_list()
#Sruvived
embarked_count_survived = train[train['Survived'] == 1].groupby(['Embarked'], as_index=True, sort=True).count()['Survived']
embarked_count_died = train[train['Survived'] == 0].groupby(['Embarked'], as_index=True, sort=True).count()['Survived']
#Sex
embarked_count_male = train[train['Sex'] == 'male'].groupby(['Embarked'], as_index=True, sort=True).count()['Sex']
embarked_count_female = train[train['Sex'] == 'female'].groupby(['Embarked'], as_index=True, sort=True).count()['Sex']
#Passenger Class
embarked_count_Pclass1 = train[train['Pclass'] == 1].groupby(['Embarked'], as_index=True, sort=True).count()['Pclass']
embarked_count_Pclass2 = train[train['Pclass'] == 2].groupby(['Embarked'], as_index=True, sort=True).count()['Pclass']
embarked_count_Pclass3 = train[train['Pclass'] == 3].groupby(['Embarked'], as_index=True, sort=True).count()['Pclass']


# Preparing figure objects
#Total Pie Chart
embarked_pie = go.Pie(labels=embarked_points, values=embarked_count_total)
embarked_pie_layout = go.Layout(title='Embarked')
embarked_pie_fig = go.Figure(data = [embarked_pie], layout=embarked_pie_layout)
#Survived Stacked Bar Chart
embarked_bar_died = go.Bar(x=embarked_points, y=embarked_count_died, name="Didn't Survive")
embarked_bar_survived = go.Bar(x=embarked_points, y=embarked_count_survived, name="Survived")
embarked_bar_layout = go.Layout(barmode='stack', title='Embark Points and Survival')
embarked_bar_fig = go.Figure(data=[embarked_bar_died, embarked_bar_survived], layout=embarked_bar_layout)
#Age Box plots
Q_age = go.Box(y=train[train['Embarked'] == 'Queenstown']['Age'], name='Queenstown')
C_age = go.Box(y=train[train['Embarked'] == 'Cherbourg']['Age'], name='Cherbourg')
S_age = go.Box(y=train[train['Embarked'] == 'Southampton']['Age'], name='Southampton')
age_box_layout = go.Layout(title='Age amongst Embark Points')
age_box_fig = go.Figure(data=[C_age, Q_age, S_age ], layout=age_box_layout)
#Sex Stacked Bar Chart
embarked_bar_female = go.Bar(x=embarked_points, y=embarked_count_female, name='Female')
embarked_bar_male = go.Bar(x=embarked_points, y=embarked_count_male, name='Male')
embarked_sex_layout = go.Layout(title='Embarked and Sex', barmode='stack')
embarked_sex_fig = go.Figure(data=[embarked_bar_female, embarked_bar_male], layout=embarked_sex_layout)
#Fare Box Plot
Q_fare = go.Box(y=train[train['Embarked'] == 'Queenstown']['Fare'], name='Queenstown')
C_fare = go.Box(y=train[train['Embarked'] == 'Cherbourg']['Fare'], name='Cherbourg')
S_fare = go.Box(y=train[train['Embarked'] == 'Southampton']['Fare'], name='Southampton')
fare_box_layout = go.Layout(title='Embarkment and Fares')
fare_box_fig = go.Figure(data=[C_fare, Q_fare, S_fare ], layout=fare_box_layout)

#PClass Grouped Bar Plot
embarked_bar_Pclass1 = go.Bar(x=embarked_points, y=embarked_count_Pclass1, name='Upper Class')
embarked_bar_Pclass2 = go.Bar(x=embarked_points, y=embarked_count_Pclass2, name='Middle Class')
embarked_bar_Pclass3 = go.Bar(x=embarked_points, y=embarked_count_Pclass3, name='Lower Class')
embarked_Pclass_layout = go.Layout(title='Embarkment and Passenger Class')
embarked_Pclass_fig = go.Figure(data = [embarked_bar_Pclass1, embarked_bar_Pclass2, embarked_bar_Pclass3], layout=embarked_Pclass_layout)


# Plotting
iplot(embarked_pie_fig, filename='Titanic Embarked', )
iplot(embarked_bar_fig)
iplot(age_box_fig)
iplot(embarked_sex_fig)
iplot(fare_box_fig)
iplot(embarked_Pclass_fig)


#     1.Most passengers embarked from Southampton, then Cherbourg followed by Queenstown .
#  
#     2.Passengers who embarked from Southampton and Queenstown had less survivals, although the survival rate is 
#     relatively low for Southampton passengers, but what is unique is that Cherbourg passengers had more 
#     survivors!
#  
#     3.The median of ages for all embark points are somehow similar, although Southampton passengers have more 
#     elderly.
#     
#     4.Sex distribution amongst embark points looks reasonable where number of males are always higher..
#     
#     5.Fare distribution and passengers' classes tells a story for each embarkment point : 
#         Cherbourg: A median as high as 29.7$ can be misleading, this embarkment point had 66 lower class and
#     85 upper class with 17 of middle class, this tells us that it's almost a 50-50 split between the 
#     rich and the poor, not overlooking a rocketing value of 512$ and multiple fares in the range of 
#         200$.
#         
#         Queenstown: A low median of 7.5$ for fares, the distribution of passenger classes reinforce the fact
#         that most of the passengers from this point were less fortunate. Although there's two upper class
#         and 3 middle class passengers.
#         
#         Southampton: The point that most passengers embarked from, with very high variance, but we can tell
#         that the majority of the passengers were at the lower class.
#         
# 

# ## Fare

# In[ ]:


#Preparing the data
#General
all_fares = train['Fare']
#Survival
fares_survived = train[train['Survived'] == 1]['Fare']
fares_died = train[train['Survived'] == 0]['Fare']


#Preparing figures
#General
all_fares_hist = go.Histogram(x=all_fares, marker={'color':'#35477d'})
all_fares_hist_layout = go.Layout(title='Fares')
all_fares_hist_fig = go.Figure(data=[all_fares_hist], layout=all_fares_hist_layout)
#Survival
fares_survived_box = go.Box(y=fares_survived, name='Survived', marker={'color':'#0939CF'})
fares_died_box = go.Box(y=fares_died, name="Didn't Survive", marker={'color':'#CF1E09'})
fares_survival_layout = go.Layout(title='Fares and Survival')
fares_survival_fig = go.Figure(data=[fares_died_box, fares_survived_box], layout=fares_survival_layout)


#Plotting
iplot(all_fares_hist_fig)
iplot(fares_survival_fig)


#     1. The distribution of fares shows that most fares paid were as low as (5-15)$
#     
#     2. It's evident that survival is linked with a higher 'Fare' median, although this is not to be generalized since 
#     some of the passengers who didn't had paid fares as high as 260$

# # Prediction Models 

# ## More Data Preparation

# ### Imputing Missing Age values

# In[ ]:


from sklearn.preprocessing import Imputer
imputer = Imputer()
train['Age'] = imputer.fit_transform(train['Age'].to_numpy().reshape(-1,1))


# ### Standardizing numerical values

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train['Fare'] = scaler.fit_transform(train['Fare'].to_numpy().reshape(-1,1))
train['Age'] = scaler.fit_transform(train['Age'].to_numpy().reshape(-1,1))


# ### One hot encoding categorial values

# In[ ]:


encoded_sex =  pd.get_dummies(train['Sex'])
encoded_class = pd.get_dummies(train['Pclass'], prefix='Pclass')
encoded_embark = pd.get_dummies(train['Embarked'], prefix='embarked_from')
train = pd.concat([train, encoded_sex, encoded_class, encoded_embark], axis=1)

try:
    train.drop('Sex', axis=1, inplace=True)
except:
    pass
try:
    train.drop('Pclass', axis=1, inplace=True)
except:
    pass
try:
    train.drop('Embarked', axis=1, inplace=True)
except:
    pass


# ### X, y preparation and train-test splitting

# In[ ]:


y = train['Survived']
X = train.drop('Survived', axis=1)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# ## K Nearest Neighbour

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
## Grid search to find optimal number of neighbours
from sklearn.model_selection import GridSearchCV
search = GridSearchCV(estimator=KNN, param_grid={'n_neighbors':range(1,15)}, scoring='precision', n_jobs=-1, cv=10)
search.fit(X_train, y_train)
KNN_optimum = search.best_estimator_


# In[ ]:


## KNN with grid search result
predictions = KNN_optimum.predict(X_test)
from sklearn.metrics import classification_report
KNN_report = classification_report(y_test, predictions)
print(KNN_report)


# ## Support Vector Machine (SVD)

# In[ ]:


from sklearn.svm import SVC
SVM = SVC()
SVM.fit(X_train, y_train)
SVM_predictions = SVM.predict(X_test)
print(classification_report(y_test, SVM_predictions))


# ## Random Forest Classifier (Ensemble of Decision trees)
# 
# 
# 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train, y_train)
RF_predictions = RF.predict(X_test)
print(classification_report(y_test, RF_predictions))


# ## Ensemble of previously trained classifiers

# In[ ]:


from sklearn.ensemble import VotingClassifier
VC = VotingClassifier(estimators=[('rf', RF), ('svm', SVM), ('knn', KNN_optimum)])


# In[ ]:


VC.fit(X_train, y_train)
VC_predictions = VC.predict(X_test)
print(classification_report(y_test, VC_predictions))


# <span style='font-size:24px'>To no suprise, an ensemble voting classifier yielded the best result</span>

# In[ ]:





# In[ ]:




