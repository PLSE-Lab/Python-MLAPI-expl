#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data=pd.read_csv('../input/banking.csv')


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.describe()


# In[ ]:


data.corr()


# ## Visualization

# In[ ]:



sns.countplot(x=data['y'])
plt.xlabel('Subscribed for Term deposit')
labels=["Didn't open term deposit","Open term deposit"]


# In[ ]:


sns.pairplot(data=data,hue='y')


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(x='education',hue='y',data=data)
plt.tight_layout()


# In[ ]:


sns.countplot(x='marital',hue='y',data=data)


# In[ ]:


sns.countplot(x='housing',hue='y',data=data)


# In[ ]:


sns.countplot(x='loan',hue='y',data=data)


# In[ ]:


sns.countplot(x='poutcome',hue='y',data=data)


# In[ ]:


pd.crosstab(data.marital,data.y).plot(kind='bar')
plt.xlabel('Marital status')
plt.ylabel('Proporation of customers')


# In[ ]:


data.education[data.education=='basic.4y']='Basic'
data.education[data.education=='basic.6y']='Basic'
data.education[data.education=='basic.9y']='Basic'
data.education[data.education=='high.school']='High School'
data.education[data.education=='illiterate']='Illiterate'
data.education[data.education=='professional.course']='Professional Course'
data.education[data.education=='university.degree']='University Degree'
data.education[data.education=='unknown']='Unknown'


# Select numeric and categorical dtypes

# In[ ]:


numeric_dtype=data.select_dtypes(exclude='object')


# In[ ]:


sns.heatmap(numeric_dtype.corr(),annot=True)
plt.title('Correlation Matrix')


# Create dummy variables

# In[ ]:


cat_col=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']


# In[ ]:


for col in cat_col:
    data=pd.concat([data.drop(col,axis=1),pd.get_dummies(data[col],prefix=col,prefix_sep='-',drop_first=True)],axis=1)


# In[ ]:


data.columns


# Splitting the data

# In[ ]:


X=data.drop('y',axis=1)
y=data['y']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score,classification_report

lr=LogisticRegression()
lr.fit(X_train,y_train)
y_predl=lr.predict(X_test)
probs=lr.predict_proba(X_test)
prob=probs[:,1]


auc=roc_auc_score(y_test,prob)
print('AUC score is',auc)


# In[ ]:


fpr,tpr,threshols=roc_curve(y_test,prob)
plt.plot([0,1],[0,1],linestyle='-')
plt.plot(fpr,tpr,marker='.')


# In[ ]:


cm=confusion_matrix(y_test,y_predl)
sns.heatmap(cm,annot=True)


# In[ ]:


print('Classification Report',classification_report(y_test,y_predl))


# Cross Validation

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


score=cross_val_score(lr,X,y,scoring='accuracy',cv=10)


# In[ ]:


print('After cross validation,Accuracy is',score.mean()*100)


# Feature Extraction with RFE

# In[ ]:


from sklearn.feature_selection import RFE

rfe=RFE(lr,12)
fit=rfe.fit(X_train,y_train)
selected_feature=fit.support_
print('No. of features %d',fit.n_features_)
print('Selected Features %d',fit.support_)
print('Ranking Features %d',fit.ranking_)


# In[ ]:


col_to_drop=[]
for i in range(len(X.columns)-1):
    if selected_feature[i] == False:
        col_to_drop.append(i)
        
X_train.drop(X.iloc[:, col_to_drop], axis=1, inplace = True)
X_test.drop(X.iloc[:, col_to_drop], axis=1, inplace = True)


# In[ ]:


X_train.columns


# In[ ]:


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


models=[SVC(class_weight='balanced'),
        KNeighborsClassifier(),
        LogisticRegression(class_weight='balanced'),
        
]
scores=pd.DataFrame(columns=['Model','Accuracy','F1-score','Precision','Recall'])
for model in models:
    classifier=model.fit(X_train,y_train)
    name=str(classifier).rsplit('()',1)
    accuracy = np.average(cross_val_score(classifier, X_test, y_test, scoring= "accuracy"))
    
    f1 = np.average(cross_val_score(classifier, X_test, y_test, scoring= "f1_weighted"))
    
    precision = np.average(cross_val_score(classifier, X_test, y_test, scoring='precision_weighted'))
    
    recall = np.average(cross_val_score(classifier, X_test, y_test, scoring='recall_weighted'))
    
    scores = scores.append({'Model': name,'Accuracy': accuracy,'F1-score': f1,
                             'Precision': precision, 'Recall': recall}, ignore_index=True)







# In[ ]:


scores.set_index('Model')


# In[ ]:


scores.plot(kind='bar',title='Scores')


# Grid search with KNN

# In[ ]:


from sklearn.model_selection import GridSearchCV
param={'n_neighbors':[3,4,6,7,8,9,11,15],'weights':['uniform','distance'],'metric':['euclidean','manhattan']}
grid=GridSearchCV(KNeighborsClassifier(),param_grid=param,cv=8,n_jobs=-1)
gs_results=grid.fit(X_train,y_train)
print('Best Score',gs_results.best_score_)
print('Best Estimator',gs_results.best_estimator_)
print('Best Param',gs_results.best_params_)


# Conclusion :- In this dataset I applied Logistic Regression algorithm with cross validation.And I found that there is a slight increase in accuracy.After that I did RFE feature selection method and tried modelling with SVC,KNN ,Logistic & KNN with grid search.Among these KNN performs well with an accuracy of 89%.

# Suggest me to improve better classification accuracy
