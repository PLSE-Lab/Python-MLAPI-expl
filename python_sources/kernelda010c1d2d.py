#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


from xgboost import plot_importance


# In[5]:


import seaborn as sns


# In[6]:


import os


# In[7]:


# importing libraries
from sklearn.preprocessing import StandardScaler as ss


# In[8]:


#dimentional reductionality
from sklearn.decomposition import PCA


# In[9]:


#libraries for data splitting
from sklearn.model_selection import train_test_split


# In[10]:


#libraries for model pipelining
from sklearn.pipeline import Pipeline


# In[11]:


from sklearn.pipeline import make_pipeline


# In[12]:


#libraries for model parameter tuning
from sklearn.model_selection import GridSearchCV


# In[13]:


from sklearn.model_selection import RandomizedSearchCV


# In[14]:


from bayes_opt import BayesianOptimization


# In[15]:


from xgboost.sklearn import XGBClassifier


# In[16]:


from sklearn.model_selection import cross_val_score


# In[17]:


import eli5


# In[18]:


from eli5.sklearn import PermutationImportance


# In[19]:


#miscaleneous Libraries
import time


# In[20]:


import random


# In[21]:


import gc


# In[22]:


from scipy.stats import uniform


# In[23]:


import os


# In[24]:


#Importing libraries for performance measures
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve


# In[25]:


#loading dataset


# In[26]:


pd.set_option("display.max_columns", 100)


# In[27]:


os.chdir("../input")


# In[28]:


data=pd.read_csv("winequalityN.csv")


# In[29]:


#getting to know the data


# In[30]:


data.head()


# In[31]:


data.info()


# In[32]:


data.shape


# In[33]:


data.isnull().sum()


# In[ ]:





# In[34]:


data.dropna(axis=0, inplace=True)


# In[35]:


data.isnull().sum()


# In[36]:


data.shape


# In[37]:


data.head(3)


# In[38]:


data.describe()


# In[39]:


data.corr()


# In[40]:


plt.figure(figsize=(12, 12))
sns.heatmap(data.corr(),annot=True,vmin=-1,cmap='YlGnBu')


# In[41]:


sns.countplot(x = data.quality, data=data, hue='type', palette="rocket")


# In[42]:


fig = plt.figure(figsize=(24,10))
features = ["total sulfur dioxide", "residual sugar", "volatile acidity", "total sulfur dioxide", "chlorides", "fixed acidity", "citric acid","sulphates"]

for i in range(8):
    ax1 = fig.add_subplot(2,4,i+1)
    sns.barplot(x='quality', y=features[i],data=data, hue='type', palette='rocket')


# In[ ]:





# In[43]:


#split data as predictors and target


# In[44]:


X= data.iloc[ : , 1:14]


# In[45]:


X.head(4)


# In[46]:


y=data.iloc[:, 0]


# In[47]:


y.head(4)


# In[48]:


y=y.map({'white':1, 'red':0})


# In[49]:


y.dtype


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, shuffle=True)


# In[51]:


Xgb_pipelist = [('ss', ss() ),
            ('pca', PCA()),
            ('xg',  XGBClassifier(silent = False,
                                  n_jobs=2)        # Specify other parameters here
            )
            ]


# In[52]:


Xgb_pipeline=Pipeline(Xgb_pipelist)


# In[53]:


#parameter tuning

#grid search

parameters = {'xg__learning_rate':  [0.4, 0.05],
              'xg__n_estimators':   [100,  150],
              'xg__max_depth':      [3,5],
              'pca__n_components' : [5,7]
              }       


# In[ ]:





# In[54]:


#    Create Grid Search object first with all necessary

grid_search = GridSearchCV(Xgb_pipeline,
                   parameters,         
                   n_jobs = 2,         
                   cv =2 ,             
                   verbose =2,      
                   scoring = ['accuracy', 'roc_auc'],  
                   refit = 'roc_auc'   
                   )


# In[55]:


#fitting the data
start = time.time()
grid_search.fit(X_train, y_train)   
end = time.time()
(end - start)/60 


# In[56]:


f"Best score: {grid_search.best_score_} "


# In[57]:


f"Best parameter set {grid_search.best_params_}"


# In[58]:


plt.bar(grid_search.best_params_.keys(), grid_search.best_params_.values(), color='b')
plt.xticks(rotation=10)


# In[59]:


y_pred = grid_search.predict(X_test)
y_pred


# In[60]:


accuracy = accuracy_score(y_test, y_pred)


# In[61]:


f"Accuracy: {accuracy * 100.0}"


# In[62]:


Xgb_pipelist = [('sts', ss() ),
            ('pca', PCA()),
            ('xg',  XGBClassifier(silent = False,
                                  n_jobs=2)        # Specify other parameters here
            )
            ]


# In[63]:


Xgb_pipeline=Pipeline(Xgb_pipelist)


# In[64]:


#random Search 

parameter_random = {'xg__learning_rate':  uniform(0, 1),
              'xg__n_estimators':   range(50,100),
              'xg__max_depth':      range(3,5),
              'pca__n_components' : range(5,7)}


# In[65]:


random_search = RandomizedSearchCV(Xgb_pipeline,
                        param_distributions=parameter_random,
                        scoring= ['roc_auc', 'accuracy'],
                        n_iter=15,          
                        verbose = 3,
                        refit = 'roc_auc',
                        n_jobs = 2,          
                        cv = 2               
                        )


# In[66]:


start = time.time()
random_search.fit(X_train, y_train)
end = time.time()
(end - start)/60


# In[67]:


f"Best score: {random_search.best_score_} "


# In[68]:


f"Best parameter set: {random_search.best_params_} "


# In[69]:


plt.bar(random_search.best_params_.keys(), random_search.best_params_.values(), color='y')
plt.xticks(rotation=10)


# In[70]:


y_pred = random_search.predict(X_test)
y_pred


# In[71]:


accuracy = accuracy_score(y_test, y_pred)
f"Accuracy: {accuracy * 100.0}"


# In[72]:


parameter_set = {
           'learning_rate':  (0, 1),                 
           'n_estimators':   (50,100),               
           'max_depth':      (3,5),                 
           'n_components' :  (5,7)          
            }


# In[73]:


def xg_eval(learning_rate,n_estimators, max_depth,n_components):
    #  Make pipeline. Pass parameters directly here
    pipe_xg1 = make_pipeline (ss(),                        # Why repeat this here for each evaluation?
                              PCA(n_components=int(round(n_components))),
                              XGBClassifier(
                                           silent = False,
                                           n_jobs=2,
                                           learning_rate=learning_rate,
                                           max_depth=int(round(max_depth)),
                                           n_estimators=int(round(n_estimators))
                                           )
                             )

    # Now fit the pipeline and evaluate
    cv_result = cross_val_score(estimator = pipe_xg1,
                                X= X_train,
                                y = y_train,
                                cv = 2,
                                n_jobs = 2,
                                scoring = 'f1'
                                ).mean()             # take the average of all results


    #  Finally return maximum/average value of result
    return cv_result


# In[74]:


bayesian_opt = BayesianOptimization(
                             xg_eval,     
                             parameter_set  
                             )


# In[75]:


start = time.time()


# In[76]:


bayesian_opt.maximize(init_points=5,
               n_iter=15,        
               )


# In[77]:


f"Best parameter set: {bayesian_opt.max} "


# In[78]:


bayesian_opt.max.values()


# In[79]:


for features in bayesian_opt.max.values(): 
    print(features)


# In[80]:


features


# In[81]:


plt.bar(features.keys(), features.values(), color='r')
plt.xticks(rotation=10)


# In[82]:


#Fitting parameters into our model and Feature Importance


# In[83]:


#Model with parameters of grid search
model_gs = XGBClassifier(
                    learning_rate = grid_search.best_params_['xg__learning_rate'],
                    max_depth = grid_search.best_params_['xg__max_depth'],
                    n_estimators=grid_search.best_params_['xg__n_estimators']
                    )


# In[84]:


#Model with parameters of random search
model_rs = XGBClassifier(
                    learning_rate = random_search.best_params_['xg__learning_rate'],
                    max_depth = random_search.best_params_['xg__max_depth'],
                    n_estimators=random_search.best_params_['xg__n_estimators']
                    )


# In[85]:


#Model with parameters of bayesian optimization
model_bo = XGBClassifier(
                    learning_rate = int(features['learning_rate']),
                    max_depth = int(features['max_depth']),
                    n_estimators=int(features['n_estimators'])
                    )


# In[86]:


start = time.time()
model_gs.fit(X_train, y_train)
model_rs.fit(X_train, y_train)
model_bo.fit(X_train, y_train)


# In[87]:


y_pred_gs = model_gs.predict(X_test)
y_pred_rs = model_rs.predict(X_test)
y_pred_bo = model_bo.predict(X_test)


# In[88]:


accuracy_gs = accuracy_score(y_test, y_pred_gs)
accuracy_rs = accuracy_score(y_test, y_pred_rs)
accuracy_bo = accuracy_score(y_test, y_pred_gs)


# In[89]:


print("Grid search Accuracy: "+str(accuracy_gs))
print("Grid search Accuracy: "+str(accuracy_rs))
print("Bayesian Optimization Accuracy: "+str(accuracy_bo))


# In[90]:


model_gs.feature_importances_
model_rs.feature_importances_
model_bo.feature_importances_
plot_importance(model_gs)
plot_importance(model_rs)
plot_importance(model_bo)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




