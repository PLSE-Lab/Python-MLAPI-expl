#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split


# In[ ]:


df = pd.read_csv( " C\Users\User\Downloads\LAB 1\train1.csv " )
df.head()


# In[ ]:


df.info()


# In[ ]:


missing_count = df.isnull().sum()
missing_count[missing_count > 0]


# In[ ]:


df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)
df_dtype_nunique.columns = ["dtype","unique"]
df_dtype_nunique


# In[ ]:


df.fillna(value=df.mean(),inplace=True)
df.head()


# In[ ]:


df.isnull().any().any()


# In[ ]:


df.columns


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(data=df.corr(),cmap='Blues',annot=True)


# In[ ]:


numerical_features = [ 'feature1','feature2','feature3','feature4', 'feature5',
       'feature6','feature7','feature8','feature9','feature10','feature11']
categorical_features = ['type']
X_train = df[numerical_features+categorical_features]
y_train = df["rating"]


# In[ ]:


#plt.figure(figsize=(20,20))
#sns.heatmap(data=df.corr(),cmap='Blues',annot=True)


# In[ ]:


type_code = {'old':0,'new':1}
X_train['type'] = X_train['type'].map(type_code)

#One-hot
#X_train = pd.get_dummies(data=X_train,columns=['type'])


# In[ ]:


#X_train.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler


X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.20,random_state = 0)


scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
#X_val[numerical_features] = scaler.transform(X_val[numerical_features])  

# It is important to scale tain and val data separately because val is supposed to be unseen data on which we test our models. If we scale them together, data from val set will also be considered while calculating mean, median, IQR, etc
#We donot want to train model on our dataset so ,we just want to do the preprocessing while we want to do the preprocessing as well as train our model on our training set . 
#X_train[numerical_features].head()


# In[ ]:


#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import mean_squared_error
#from math import sqrt
#regressor=LinearRegression()
#regressor.fit(X_train,y_train)
#pred=regressor.predict(X_val)
#rms=sqrt(mean_squared_error(y_val,pred))
#print(rms)
#from sklearn.ensemble import RandomForestRegressor
#rf = RandomForestRegressor(max_features='sqrt',min_samples_leaf=1,max_depth=100, min_samples_split=2,n_estimators=10000)
#bootstrap=True,max_features='auto',min_samples_leaf=1,max_depth=100,min_samples_split=2,n_estimators=1000
#rf.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_features='sqrt',min_samples_leaf=1,max_depth=100, min_samples_split=2,n_estimators=10000)
rf.fit(X_train, y_train)

#def evaluate(model, test_features, test_labels):
#    predictions = model.predict(test_features)
#    errors = abs(predictions - test_labels)
#    mape = 100 * np.mean(errors / test_labels)
#    accuracy = 100 - mape
#    print('Model Performance')
#    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
#    return accuracy
#base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
#base_model.fit(X_train,y_train)
#base_accuracy = evaluate(base_model, X_val, y_val)

#pre=rf.predict(X_val)
#rms = sqrt(mean_squared_error(y_val, pre))
#print(rms) 

#from sklearn.tree import DecisionTreeRegressor
#model = DecisionTreeRegressor()
#model.fit(X_train,y_train)
#pred=model.predict(X_val)
#rms=sqrt(mean_squared_error(y_val,pred))
#print(rms)

#from sklearn.neighbors import KNeighborsClassifier
#classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
#classifier.fit(X_train,y_train)
#pred=classifier.predict(X_val)
#rms=sqrt(mean_squared_error(y_val,pred))
#print(rms)


# In[ ]:


y_pred=rf.predict(X_val)


# In[ ]:


from sklearn.model_selection import GridSearchCV
parameters=[{'bootstrap': [False],
 'max_depth': [100,150],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [ 4,6],
 'min_samples_split': [ 5, 10],
 'n_estimators': [ 3000, 4000]}]
rf_random=GridSearchCV(estimator=rf,param_grid=parameters,cv=3,n_jobs=-1)
rf_random.fit(X_train,y_train)


# In[ ]:


rf_random.best_params_


# In[ ]:


df1=pd.read_csv('test1.csv')
df1.fillna(value=df.mean(),inplace=True)


# In[ ]:


X_test_numerical_features = ['feature1','feature2','feature3','feature4','feature5',
       'feature6','feature7','feature8','feature9','feature10','feature11']
X_test_categorical_features = ['type']
X_test = df1[X_test_numerical_features+X_test_categorical_features]

type_code = {'old':0,'new':1}
X_test['type'] = X_test['type'].map(type_code)

#X_test = pd.get_dummies(data=X_test,columns=['type'])
#X_test.head()

scaler = StandardScaler()
X_test[X_test_numerical_features] = scaler.fit_transform(X_test[X_test_numerical_features])

X_test[X_test_numerical_features].head()


# In[ ]:


pred1=rf.predict(X_test)
pred1


# In[ ]:


"""from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
#TODO
clf = RandomForestRegressor()        #Initialize the classifier object

parameters = {'n_estimators':[40,50,60,70,80]}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(clf,parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train

best_clf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

unoptimized_predictions = (clf.fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions
optimized_predictions = best_clf.predict(X_val)        #Same, but use the best estimator

acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model
acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model

print("Accuracy score on unoptimized model:{}".format(acc_unop))
print("Accuracy score on optimized model:{}".format(acc_op))"""


# In[ ]:


df1['rating']=np.array(pred1)
df1.head()


# In[ ]:


#df1.round({'rating': 0})


# In[ ]:


out=df1[['id','rating']]
out=out.round({'rating': 0})
out.head()


# In[ ]:


out.to_csv('submit_eval_lab_one37.csv',index='True')

