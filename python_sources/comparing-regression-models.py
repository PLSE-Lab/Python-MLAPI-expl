#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/HR_comma_sep.csv')
data.head()


# In[ ]:


#cloumn information
column = data.columns.tolist()
print(column)


# In[ ]:


#column data types
data.dtypes


# In[ ]:


#null check
data.isnull().any()


# In[ ]:


#co-relation matrix
sns.heatmap(data.corr(), vmax=1, square=True)
#co-relation matrix shown unrerlated columns


# In[ ]:


#Defining features and target for this dataset based on co-relation
features = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'sales', 'salary',]
target = ['left']


# In[ ]:


#pre-processing of data on salary/sales colmuns converting in float/int basically
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
data['salary'] = label_encoder.fit_transform(data['salary'])
data['salary'].head()
data['sales'] = label_encoder.fit_transform(data['sales'])
data['sales'].head()


# In[ ]:


from sklearn.model_selection import train_test_split # to split the data into two parts
#splitting data set into training and test data set in 0.7/0.3
train, test = train_test_split(data,test_size=0.30)


# In[ ]:


#Fill the training and test data with require information
X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]


# In[ ]:


#training/test data analysis
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score


# In[ ]:


Classifiers = [
  #  LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200),
    AdaBoostClassifier(),
    GaussianNB()]


# In[ ]:


ModelAccuracy=[]
RegModel=[]
for classifier in Classifiers:
    fit = classifier.fit(X_train ,y_train)
    score_train = fit.score(X_test, y_test)
    score_test = fit.score(X_train, y_train)
    pred=fit.predict(X_test)
    accuracy = accuracy_score(pred,y_test)
    ModelAccuracy.append(accuracy)
    RegModel.append(classifier.__class__.__name__)
    print("Accuracy of "+classifier.__class__.__name__+"is Training score: ",score_train)
    print("Accuracy of "+classifier.__class__.__name__+"is Testing score: ",score_test)
    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy)) 
    print('\n')


# In[ ]:


Index = [1,2,3,4,5,6]
plt.bar(Index,ModelAccuracy)
plt.xticks(Index, RegModel,rotation=45)
plt.ylabel('Model Accuracy')
plt.xlabel('Regression Model')
plt.title('Accuracies of regression Models')

