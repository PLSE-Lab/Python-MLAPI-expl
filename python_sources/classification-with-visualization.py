#!/usr/bin/env python
# coding: utf-8

# In[58]:


import os
print("Data:\n",os.listdir("../input"))


# In[59]:


import pandas as pd
data_dir = '../input/breast-cancer-wisconsin.data.txt'
labels = ['id', 'ClumpThickness', 'CellSize', 'CellShape',
         'MarginalAdhesion','EpithelialCellSize','BareNuclei',
         'BlandChromatin','NormalNuclei','Mitoses','Classes']
data = pd.read_csv(data_dir)
data.columns = labels
data.set_index('id', inplace=True)
data.head()


# In[60]:


data.shape


# In[61]:


data.describe()


# In[62]:


data.info()


# In[63]:


pd.value_counts(data['Classes']) # 2-benign 4-malignant


# ### search for missing values

# In[64]:


data.isnull().sum()


# ## search for correlation

# In[65]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[66]:


data.hist(bins=20, figsize=(20,15))
plt.show()


# In[67]:


corr_matrix = data.corr()
sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
plt.show()


# In[68]:


corr_matrix['Classes'].sort_values(ascending=False)


# In[69]:


data.groupby('BareNuclei').count()


# In[70]:


data['BareNuclei'] = data['BareNuclei'].replace('?', 1)


# In[71]:


# Convert to int 
data['BareNuclei'] = data['BareNuclei'].astype(int)
data.info()


# In[72]:


## Left 'classes' column 
classes = data['Classes']
drop_classes_data = data.drop('Classes',axis=1)
drop_classes_data.head()


# In[73]:


# Normalize data from 0 - 1 
from sklearn.preprocessing import StandardScaler

after_scaler_data = StandardScaler().fit_transform(drop_classes_data)


# In[74]:


after_scaler_data


# In[75]:


breast_data = pd.DataFrame(data=after_scaler_data, columns=labels[1:-1])
breast_data.head()


# ## Split data

# In[76]:


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
breast_data['Classes'] = classes.values
breast_data = shuffle(breast_data)
breast_data.head()


# In[77]:


x_train, x_test, y_train, y_test = train_test_split(breast_data.drop('Classes',axis=1), breast_data['Classes'],test_size=0.2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# ## Select and Train a Model

# ### LogisticRegression

# In[78]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)


# In[79]:


# Measure accuracy 
from sklearn.metrics import f1_score
breast_cancer_predict = log_reg.predict(x_test)
log_f1 = f1_score(y_test, breast_cancer_predict, pos_label=2)
log_f1


# In[80]:


from sklearn.metrics import classification_report

log_class_rep = classification_report(y_test, breast_cancer_predict, labels=[2,4],digits=4)
print(log_class_rep)


# ### DecisionTreeClassifier

# In[81]:


from sklearn.tree import DecisionTreeClassifier

dec_tree_class = DecisionTreeClassifier()
dec_tree_class.fit(x_train,y_train)


# In[82]:


breast_cancer_predict_tree = dec_tree_class.predict(x_test)
tree_class_rep = classification_report(y_test, breast_cancer_predict_tree, labels=[2,4],digits=4)
print(tree_class_rep)


# ### RandomForestClassifier

# **without gridsearch**

# In[83]:


from sklearn.ensemble import RandomForestClassifier

rand_for_class = RandomForestClassifier()
rand_for_class.fit(x_train,y_train)


# In[84]:


forest_predict = rand_for_class.predict(x_test)
forest_class_rep = classification_report(y_test, forest_predict, labels=[2,4],digits=4)
print(forest_class_rep)


# **with gridsearch**

# In[85]:


from sklearn.model_selection import GridSearchCV


# In[86]:


param_grid_forest = [
    {'n_estimators':[5,10,15,20],'max_depth':[None,2,5,10]},
    {'min_samples_split':[0.1,0.2,0.5],'min_samples_leaf':[1,2]}
]


# In[87]:


forest_grid = GridSearchCV(rand_for_class, param_grid_forest,cv=5, scoring='v_measure_score')
forest_grid.fit(x_train,y_train)


# In[88]:


forest_grid.best_params_


# In[89]:


grid_forest_predict = forest_grid.predict(x_test)
grid_forest_raport = classification_report(y_test, grid_forest_predict, labels=[2,4],digits=4)
print(grid_forest_raport)


# ## SVC

# **without gridsearch**

# In[90]:


from sklearn.svm import SVC

svc_clf = SVC()
svc_clf.fit(x_train,y_train)


# In[91]:


svc_predict = svc_clf.predict(x_test)
svc_predict_rep = classification_report(y_test, svc_predict, labels=[2,4],digits=4)
print(svc_predict_rep)


# **with gridsearch**

# In[95]:


param_grid_svc = [
    {'C':[0.5,1,2],'kernel':['linear','poly','rbf'],'gamma':['auto'],
    'degree':[2,3,4]}
]


# In[96]:


svc_grid = GridSearchCV(svc_clf, param_grid_svc, cv=5, scoring='v_measure_score')
svc_grid.fit(x_train, y_train)


# In[97]:


svc_grid.best_params_


# In[98]:


svc_grid_predict = svc_grid.predict(x_test)
svc_grid_rep = classification_report(y_test, svc_grid_predict,labels=[2,4],digits=4)
print(svc_grid_rep)


# ### AdaBoostClassifier

# In[99]:


from sklearn.ensemble import AdaBoostClassifier

ada_boost = AdaBoostClassifier()
ada_boost.fit(x_train,y_train)


# In[100]:


ada_predict = ada_boost.predict(x_test)
ada_predict_rep = classification_report(y_test, ada_predict, labels=[2,4],digits=4)
print(ada_predict_rep)


# ### Using Cross-Validation

# In[101]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(dec_tree_class, x_train, y_train,cv=10)
scores.mean(), scores.std()


# ### Grid Search

# In[102]:


param_grid = [
    {'max_depth':[1,2,5,10]}
]


# In[103]:


dec_tree = DecisionTreeClassifier()
grid_search = GridSearchCV(dec_tree, param_grid, cv=5, scoring='v_measure_score')
grid_search.fit(x_train,y_train)


# In[104]:


grid_search.best_params_


# In[105]:


grid_search.best_estimator_


# In[106]:


grid_tree_predict = grid_search.predict(x_test)
grid_raport = classification_report(y_test, grid_tree_predict, labels=[2,4],digits=4)
print(grid_raport)


# In[107]:


dec_tree.get_params().keys()


# ### RandomizedSearchCV

# In[108]:


from sklearn.model_selection import RandomizedSearchCV

param_distributions = {'max_depth':[None,1,2,5,10],
                       'min_samples_leaf':[0.1,0.4,0.5],
                       'splitter':['best','random'],
                       'max_leaf_nodes':[None,2,5,10]
                       }
n_iter_search = 4
rand_search = RandomizedSearchCV(estimator=dec_tree,
                                 param_distributions=param_distributions,
                                 n_iter=n_iter_search)


# In[109]:


rand_search.fit(x_train,y_train)


# In[110]:


rand_search.best_params_


# In[112]:


rand_tree_predict = rand_search.predict(x_test)
rand_raport = classification_report(y_test, rand_tree_predict, labels=[2,4],digits=4)
print(rand_raport)


# In[ ]:




