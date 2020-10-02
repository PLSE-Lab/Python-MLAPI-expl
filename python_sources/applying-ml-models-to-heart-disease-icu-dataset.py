#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import display
from sklearn import metrics
hrt= pd.read_csv('../input/heart.csv')
hrt.head(10)


# In[2]:


hrt.info()


# In[3]:


hrt.isnull().sum() #checking for null data


# In[ ]:





# In[4]:


#preliminary figures to help understand the gist of the dataset
sns.barplot(x='sex',y='chol',data=hrt,estimator=np.std) #1 for male and 0 for female


# In[5]:


sns.barplot(x='exang',y='chol',data=hrt,estimator=np.std)
#exang= exercise-induced angina
#1 = yes & 0 = no


# In[6]:


sns.boxplot(x='sex',y='chol',data=hrt,hue='exang').set_title('Sex vs Cholestrol levels & excercise induced angina')


# In[7]:


sns.stripplot(x='thalach',y='chol',data=hrt,hue='exang',dodge=True,jitter=True).set_title('Maximum heart rate vs Cholestrol levels & excercise induced angina')


# In[8]:


sns.lmplot(x='thalach',y='trestbps',data=hrt,hue='sex',markers=['o','v'],scatter_kws={'s':100})


# In[9]:


label_quality=LabelEncoder()


# In[10]:


hrt['exang']= label_quality.fit_transform(hrt['exang'])


# In[11]:


hrt['exang'].value_counts()


# In[12]:


ax=sns.countplot(hrt['exang'])
ax.set_title('exercise-induced angina')


# In[14]:


#separate the dataset as response variable(dependent variable) and feature variable(independent variable)
X= hrt.drop('exang', axis=1) #use all columns minus 'exang' as independent variables
y=hrt['exang'] #use 'exang' as dependent variable
#upper case 'X' and lowercase 'y' is the common-use


# In[15]:


#Train and test splitting of the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
#returns 4 variables


# In[16]:


#applying standard scaling to get optimized results
sc= StandardScaler()
'''this levels the playing field so columns with big values (e.g.chol) 
dont get a prefrence over columns with smaller values (e.g. oldpeak)'''
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)


# In[45]:


#Random Forest Classifier
rfc= RandomForestClassifier(n_estimators=200) #how many models do I need/trees in the forest 
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)


# In[46]:


#Let's see how the model performs
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test ,pred_rfc))
#good at predicting when exercise-induced angina doesn't happen but not good at predicting when it does.


# In[19]:


#SVM Classifier
clf=svm.SVC()
clf.fit(X_train,y_train)
pred_clf=clf.predict(X_test)


# In[20]:


#Let's see how the model performs
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test ,pred_clf))
#worse percision than RFC


# In[27]:


# Neural Networks
mlpc= MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=200)
#3 layers of 11 nodes #more hidden layers= more resources but can lead to overfitting #normal amount of iterations are 200
mlpc.fit(X_train,y_train)
pred_mlpc=mlpc.predict(X_test)


# In[28]:


#Let's see how the model performs
print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test ,pred_mlpc))
#better percision for predicting when exercise-induced angina doesn't happen, but much worse at predicting when it should


# # Testing the same models after excluding certain independent variables

# In[47]:


X= hrt.drop(['exang','sex','oldpeak','target'], axis=1)#use all columns minus 'exang', 'sex', 'oldpeak', and 'target' as independent variables
y=hrt['exang'] #use 'exang' as dependent variable


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[49]:


sc= StandardScaler()
'''this levels the playing field so columns with big values (e.g.chol) 
dont get a prefrence over columns with smaller values (e.g. oldpeak)'''
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)


# In[50]:


#Random Forest Classifier
rfc= RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc=rfc.predict(X_test)


# In[51]:


#Let's see how the model performs
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test ,pred_rfc))
#worse than initial


# In[52]:


#SVM Classifier
clf=svm.SVC()
clf.fit(X_train,y_train)
pred_clf=clf.predict(X_test)


# In[53]:


#Let's see how the model performs
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test ,pred_clf))
#slightly better percision than inital


# In[56]:


# Neural Networks
mlpc= MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=200)
#3 layers of 11 nodes
mlpc.fit(X_train,y_train)
pred_mlpc=mlpc.predict(X_test)


# In[57]:


#Let's see how the model performs
print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test ,pred_mlpc))
#better percision than initial


# In[ ]:


'''after dropping the independent variables sex, oldpeak, and target, 
the SVM and Neural network models both performed slightly better than before,
but the Random Forest model performed slightly worse'''

