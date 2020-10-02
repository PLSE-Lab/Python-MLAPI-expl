#!/usr/bin/env python
# coding: utf-8

# <font size="5">Prediction of Breast Cancer using KNN with 99% accuracy</font>
# 
# 
# By using the Breast Cancer Wisconsin (Diagnostic) dataset. We create a KNN classifier that can predict the patients of Breast Cancer with accuracy of 99%.

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# **Read the csv file** 

# In[ ]:


data = pd.read_csv('../input/data.csv', index_col=False)
data.head(5)


# **Shape of Data**

# In[ ]:


print(data.shape)


# In[ ]:


print(data.columns)


# **Describe the dataset**

# In[ ]:


print(data.describe)


# In our dataset column name 'diagnosis' is our target value. Now replace M=1 & B=0

# In[ ]:


data['diagnosis']=data['diagnosis'].apply(lambda x: '1' if x=='M' else '0')


# In[ ]:


data.head(10)


# In[ ]:


unique_diagnosis=data.groupby('diagnosis').size()
print(unique_diagnosis)


# * * * * 

# In[ ]:


unique_id=data.groupby('id').size()
print(unique_id)


# <font size="5">Data Cleaning</font>
# 
# 
# Column 'Id' and 'Unnamed:32' have no too much impact on the prediction the Breast Cancer Patients. So remove both columns 

# In[ ]:


data.drop(['id','Unnamed: 32'],axis=1)


# In[ ]:


data.plot(kind='density',layout=(5,7),subplots=True,sharex=False, legend=False, fontsize=1)
plt.show()


# **Data Visulization**
# 
# we plot density plot we can determine shape of our features which represents gaussian distribution 

# In[ ]:


data.plot(kind='density', subplots=True, layout=(5,7), sharex=False, legend=False, fontsize=1)
plt.show()


# In[ ]:


data.hist(sharex=False,layout=(5,7) ,sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()


# In[ ]:


y=data.diagnosis
x=data.iloc[:,1:32]


# **Heat map**
# 
# It looks like there is also some structure in the order of the attributes. The diagonal suggests that attributes that are next to each other are generally more correlated with each other. The black patches also suggest some moderate negative correlation the further
# attributes are away from each other in the ordering. 

# In[ ]:


import seaborn as sns
r = x.corr()
fig, ax = plt.subplots(figsize=(20,20))         # Sample figsize in inches
sns.heatmap(r, annot=True, linewidths=.5, ax=ax)
#sns.heatmap(r, annot = True)
plt.show()


# **Validation Dataset**
# 
# It is a good idea to use a validation hold-out set. This is a sample of the data that we hold back from our analysis and modeling. We use it right at the end of our project to confirm the accuracy of our final model. It is a smoke test that we can use to see if we messed up and to give us confidence on our estimates of accuracy on unseen data. We will use 70% of the dataset for modeling and hold back 30% for test data.

# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=7)


# We will use 10-fold cross validation. The dataset is not too small and this is
# a good standard test harness conguration. We will evaluate algorithms using the accuracy
# metric. This is a gross metric that will give a quick idea of how correct a given model is. More
# useful on binary classication problems like this one.

# In[ ]:


seed = 7
scoring = 'accuracy'
models=[]
models.append(('KNN', KNeighborsClassifier()))
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=5, random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# **Standardize Data**
# 
# We suspect that the differing distributions of the raw data may be negatively impacting the skill of some of the algorithm. Let's evaluate the same algorithm with a standardized copy of the dataset. This is where the data is transformed such that each attribute has a mean value of zero and a standard deviation of one. We also need to avoid data leakage when we transform the data. A good way to avoid leakage is to use pipelines that standardize the data and build the model for each fold in the cross validation test harness. That way we can get a fair estimation of how each model with standardized data might perform on unseen data.

# In[ ]:


pipelines=[]
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=5, random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# **Tuning KNN**

# In[ ]:


scaler = StandardScaler().fit(x_train)
print(scaler)
rescaledX = scaler.transform(x_train)
neighbors = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier(weights='uniform', algorithm='auto',p=2, metric='minkowski')
kfold = KFold(n_splits=5, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# **Finalize Model**

# In[ ]:


scaler = StandardScaler().fit(x_train)
rescaledX = scaler.transform(x_train)
model = KNeighborsClassifier(n_neighbors=13,weights='distance', algorithm='auto',p=2, metric='minkowski')
kfold = KFold(n_splits=5, random_state=seed)
model.fit(rescaledX, y_train)
rescaledValidationX = scaler.transform(x_test)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(y_test, predictions))    
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

