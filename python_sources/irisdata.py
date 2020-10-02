#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


# Helper libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.datasets import load_iris


# In[ ]:


data = load_iris()


# In[ ]:


features = data.data
target = data.target


# In[ ]:


data1 = pd.DataFrame(features, columns = ['sepal_length','sepal_width','petal_length','petal_width'])
data2 = pd.DataFrame(target, columns = ['class'])


# In[ ]:


df = pd.concat([data1,data2], axis = 1)


# In[ ]:


df.describe(include='all')


# In[ ]:


df.info()


# In[ ]:


# Let start Data Processing which include
# missing values
# outlier detection
# uniform columns
# normalization
# one hot encoding of categorical columns
# check correlation
# add/remove attributes


# 

# In[ ]:


# missing values
df.isnull().sum()


# In[ ]:


#outlier detection in dependent variable
df['class'].plot.box()
# no outlier


# 

# In[ ]:


# check the coliumn values are uniform 
# because the column values are numerical 
# so no need to check uniformation of values
df.columns


# In[ ]:


# apply normalization 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[ ]:


# should we normalize the dataset?
# lets check first
from scipy.stats import skew

# check value for dependent variable
value = skew(df['class'])
if value == 0:
    print(f'no need to apply normalization skew value is {value}')
else:
    print(f'apply normalization skew value is {value}')
    
# check for independent variables
value2 = skew(df)
print(value2)
# [ 0.31175306  0.31576711 -0.27212767 -0.10193421  0. ]
# now we should apply normalization to independent variables

# NOTE: 
# apply log1p to dependent variable for normalization
# apply boxcox to independent variable for normalization

from sklearn.preprocessing import power_transform
df_features = df.drop('class', axis = 1)
normalized_features = power_transform(df_features, method='box-cox') 

# pipe = Pipeline({
#     ('std_scaler',StandardScaler())
# })
# clean_data = pipe.fit(df).transform(df)


# In[ ]:


skew(normalized_features)
normalize_features_df = pd.DataFrame(normalized_features, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])


# 

# In[ ]:


clean_data = pd.concat([normalize_features_df,df['class']], axis = 1)


# In[ ]:


clean_data
# no need to apply one hot encoding because there is no categorical data


# In[ ]:


sns.heatmap(clean_data.corr())


# In[ ]:


# we need to drop sepal_width column
# clean_data.drop('sepal_width', axis = 1, inplace=True)


# In[ ]:


sns.heatmap(clean_data.corr())
clean_data


# In[ ]:


from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier 


# In[ ]:


# data splitting into train and test
from sklearn.model_selection import train_test_split
train,test = train_test_split(clean_data, test_size = 0.2, random_state = 7)


# In[ ]:


train_x = train.iloc[:,:-1]
train_y = train.iloc[:,-1]
test_x = test.iloc[:,:-1]
test_y = test.iloc[:,-1]


# In[ ]:


train_x
# train_y


# In[ ]:


# now apply machine learning algorithms
# Spot Check Algorithms

scoring = 'accuracy'
seed = 7
models = [] 
models.append(('Logistic R', LogisticRegression(solver ='liblinear', multi_class ='ovr'))) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('RA', RandomForestClassifier())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('DTree', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC(gamma ='auto')))
models.append(('xgboost', XGBClassifier()))

# evaluate each model in turn 
results = [] 
names = [] 

for name, model in models: 
	kfold = model_selection.KFold(n_splits = 10, random_state = seed) 
	cv_results = model_selection.cross_val_score( 
			model, train_x, train_y, cv = kfold, scoring = scoring) 
	results.append(cv_results) 
	names.append(name) 
	msg = "% s: % f (% f)" % (name, cv_results.mean(), cv_results.std()) 
	print(msg) 


# In[ ]:


# Compare Algorithms 
fig = plt.figure() 
fig.suptitle('Algorithm Comparison') 
ax = fig.add_subplot(111) 
plt.boxplot(results) 
ax.set_xticklabels(names) 
plt.show() 

# we should pick Support vector Machine, because it is giving 98% accuracy

from sklearn.model_selection import GridSearchCV


# In[ ]:


svm = SVC()
svm.fit(train_x,train_y)
svm.score(train_x,train_y)
predicted_y = svm.predict(train_x)
# predicted_y


# In[ ]:


# we should pick Support vector Machine, because it is giving 98% accuracy
# # now we will (Hyper parameter tuning)fine tune our model

tuned_parameters = [
    {'kernel': ['rbf'], 
    'gamma': [1e-3, 1e-4],
    'C': [1, 10, 100, 1000],
    'gamma': [1e-3, 1e-4],
     'degree':[1,2,3,4],
    'coef0':[0.0,0.2,0.1],
    'shrinking' : [True,False],
    'probability':[True,False]
    },
    {'kernel': ['linear'], 
    'C': [1, 10, 100, 1000]
    },
    {'kernel': ["poly"],
     'gamma': [1e-3, 1e-4],
     'degree':[1,2,3,4],
    'coef0':[0.0,0.2,0.1],
    'shrinking' : [True,False],
    'probability':[True,False]
    },
]

GSV = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='accuracy')
GSV.fit(train_x,train_y)


# In[ ]:


means = GSV.cv_results_['mean_test_score']
stds = GSV.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, GSV.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))


# In[ ]:


print(GSV.best_params_)


# In[ ]:


# now check confusion matrix and classification report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

predict_y = GSV.predict(test_x)
print(classification_report(test_y,predict_y))
print(confusion_matrix(test_y,predict_y))


# In[ ]:




