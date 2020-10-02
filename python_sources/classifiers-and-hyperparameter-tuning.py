#!/usr/bin/env python
# coding: utf-8

# # In this kernel, i'm trying to create a classifier with higher accuracy.
# 
# Most of the information of the data can be viewed here http://archive.ics.uci.edu/ml/datasets/heart+disease
# 
# The following are the steps:
# 1. **Data importing and analysis**: Simple analysis by overviewing the data, info, mean values, null values, shape etc
# 2. **Data Visualization**: Viewing the values in the data to understand the distribution
# 3. **Model Creation**: Creating a classifier model with appropriate parameters
# 4. **Hyperparameter tuning**: Tuning the hyperparameters to achieve better accuracy

# # Data Analysis

# In[242]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[243]:


data = pd.read_csv('../input/heart.csv')


# In[244]:


data.head()


# In[245]:


data.info()


# In[246]:


data.describe()


# In[247]:


data.isnull().sum()


# ## Dataset explanation for unknown terms
# * cp - chest pain type 
# * trestbps - resting blood pressure (in mm Hg on admission to the hospital) 
# * chol - serum cholestoral in mg/dl 
# * fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
# * restecg - resting electrocardiographic results 
# * thalach - maximum heart rate achieved 
# * exang - exercise induced angina (1 = yes; 0 = no) 
# * oldpeak - ST depression induced by exercise relative to rest 
# * slope - the slope of the peak exercise ST segment 
# * ca - number of major vessels (0-3) colored by flourosopy 
# * thal - 3 = normal; 6 = fixed defect; 7 = reversable defect 
# * target - have disease or not (1=yes, 0=no)

# Setting up the style for the visualizaiton. You can also view the available styles by
# #plt.style.available

# # Data Visualization

# In[248]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(10,10))


# In[249]:


data['cp'].value_counts().plot(kind='bar', title='Chest Pain Types',align='center');


# #### * Type 0: typical angina
# #### * Type 1: atypical angina
# #### * Type 2: non-anginal pain
# #### * Type 3: asymptomatic

# In[250]:


data['fbs'].value_counts().plot(kind='bar', title='Fast Blood Sugar (fbs)',align='center');


# ### Angina is a type of chest pain caused by reduced blood flow to the heart. Considering the dataset, it is given as exercise induce angina, so we must look for the age factor

# In[251]:


data['exang'].value_counts().plot(kind='bar', title='Exercise induced angina',align='center');


# In[252]:


plt.bar(data['age'],data['oldpeak'],color='r')
plt.xlabel('Age')
plt.ylabel('ST Depression Peak')


# ### Looks like the ST Depression peak occurs mostly for people aged between 55 to 65

# In[253]:


sns.distplot(data['thalach'],bins=20,hist=True,)


# #### Most of the thalach values are between 150-160

# In[254]:


fig,ax = plt.subplots()
ax.scatter(data['age'],data['chol'],c=data.age)
ax.set_xlabel('Age')
ax.set_ylabel('Cholestrol Level')
ax.set_title('Age vs Cholestrol Level')


# ### as the age increases, the cholestrol level also increases

# In[255]:


data['slope'].value_counts().plot(kind='bar', title='Slope of the peak',align='center');


# * Value 0: upsloping 
# * Value 1: flat 
# * Value 2: downsloping 

# In[256]:


data['target'].value_counts().plot(kind='bar', title='Has the disease or not?',align='center');
plt.xlabel('Yes or No')
plt.ylabel('Count')


# #### Using pairplot, we can try to look at the correlation between the features

# In[ ]:


sns.pairplot(data)


# Since we have categorical columns, it is good if we split them using get_dummies or one hot encoder, so that our model accuracy could be improved

# In[257]:


cp_categories = pd.get_dummies(data['cp'], prefix = "cp")
thal_categroies = pd.get_dummies(data['thal'], prefix = "thal")
Slope_categories = pd.get_dummies(data['slope'], prefix = "slope")
data.drop(['cp','thal','slope'],axis=1,inplace=True)


# In[258]:


data = pd.concat([data,cp_categories,thal_categroies,Slope_categories], axis = 1)


# In[259]:


data.head()


# # Model Creation

# In[260]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


# ### Instantiate the models

# In[261]:


xtreme = xgb.XGBClassifier(learning_rate=0.1,n_estimators=100)
knn = KNeighborsClassifier(n_neighbors=2)
rf = RandomForestClassifier(random_state=1,n_estimators=1000)
lr = LogisticRegression()
dt = DecisionTreeClassifier(criterion='gini',max_depth=8)
svc = SVC(random_state=1)

#ada = AdaBoostClassifier(n_)
#model = VotingClassifier(estimators=[lr,knn,rf])


# Set the X and y values. 
# 
# 'Target' is our dependent variable (target variable), rest all are features

# In[262]:


y = data['target']
X = data.drop(['target'],axis=1)


# In[277]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[278]:


names = ['KNN','LogisticRegression','Decision Tree','Gradient Boost','Random Forest','SVM']
scores = []
def accuracy(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))


# In[279]:


models = [knn, lr, dt, xtreme, rf, svc]
for i in models:
    accuracy(i,X_train, y_train, X_test,y_test)


# In[280]:


scores


# In[281]:


plt.plot(names,scores)
plt.xticks(rotation=90)


# # Hyperparameter tuning

# In[282]:


from sklearn.model_selection import GridSearchCV


# ### KNN

# In[283]:


neighbors = np.arange(1,20)
parameters = {'n_neighbors':neighbors}


# In[284]:


knn_grid = GridSearchCV(estimator=knn,param_grid=parameters,cv=5)
#knn_grid.fit(X_train,y_train)


# In[285]:


knn_grid.fit(X_train,y_train)


# In[286]:


knn_grid.best_params_


# In[287]:


y_pred = knn_grid.predict(X_test)


# In[288]:


knn_new = accuracy_score(y_test,y_pred)


# ### Random Forest

# In[289]:


max_fea = np.random.randint(1,11) #select a random int value
m_split = np.random.randint(2, 11)


# In[290]:


params = {"max_depth": [1,8,None], #just gave numbers i felt that would be good. More numbers we give, more time it would take
             "max_features": [max_fea],
              "min_samples_split": [m_split],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
#No of iterations: 3 * 1 * 1 * 2 * 2 * no of folds.


# In[291]:


rf_grid = GridSearchCV(estimator=rf,param_grid=params,cv=3)


# In[292]:


rf_grid.fit(X_train,y_train)


# In[293]:


rf_grid.best_params_


# In[294]:


y_pred = rf_grid.predict(X_test)
rf_new = accuracy_score(y_test,y_pred)


# ### XGBoost

# In[295]:


params = {
        'learning_rate': [0.1,0.05,0.01],
        'min_child_weight': [1, 5, 10],
        'gamma': [1,5],
        'max_depth': [3, 4, 5],
        'n_estimator':[100,1000]
        }


# In[296]:


xg_grid = GridSearchCV(estimator=xtreme,param_grid=params,cv=3)
xg_grid.fit(X_train,y_train)


# In[297]:


xg_grid.best_params_


# In[298]:


y_pred = xg_grid.predict(X_test)
xgb_new = accuracy_score(y_test,y_pred)


# ### Logistic Regression

# In[299]:


penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
params = dict(C=C, penalty=penalty)


# In[300]:


lr_grid = GridSearchCV(estimator=lr,param_grid=params,cv=3)
lr_grid.fit(X_train,y_train)


# In[301]:


lr_grid.best_params_


# In[302]:


y_pred = lr_grid.predict(X_test)
lr_new = accuracy_score(y_test,y_pred)


# ### SVM

# In[303]:


Cs =[0.01]
gammas = [0.001, 0.01]
kernels = ['linear', 'rbf']
params = {'C': Cs, 'gamma' : gammas,'kernel':kernels}


# In[304]:


grid = GridSearchCV(estimator=svc,param_grid=params,cv=3)
grid.fit(X_train,y_train)


# In[305]:


grid.best_params_


# In[306]:


y_pred = grid.predict(X_test)
svm_new = accuracy_score(y_test,y_pred)


# ## Decision Trees

# In[307]:


max_dep = np.arange(3, 10)
cri = ['gini','entropy']

params= {'max_depth': max_dep,'criterion':cri}


# In[308]:


dt_grid = GridSearchCV(estimator=dt,param_grid=params,cv=3)


# In[309]:


dt_grid.fit(X_train,y_train)


# In[310]:


dt_grid.best_params_


# In[311]:


y_pred = dt_grid.predict(X_test)
dt_new = accuracy_score(y_test,y_pred)


# In[312]:


names = ['KNN','LogisticRegression','Decision Tree','Gradient Boost','Random Forest','SVM']


# In[313]:


new_scores= [knn_new,lr_new,dt_new,xgb_new,rf_new,svm_new]


# In[314]:


fig, ax = plt.subplots()
ax.plot(names,scores,marker = 'o',markersize=15)
ax.plot(names,new_scores,marker='o',markersize=15)
ax.legend(['Old Scores','New Scores'])
fig.set_size_inches(18.5, 10.5)


# ## Now we can look at the model with the best accuracy. 
# 
# ## Ofcourse we can try creating and tuning other classifiers. Also, we can improve the accuracy by adding more params during the cross validation. I just wanted to share my learning experience of hyperparameter tuning. 
# 
# ## Thanks. I would appreciate your inputs and comments

# In[ ]:




