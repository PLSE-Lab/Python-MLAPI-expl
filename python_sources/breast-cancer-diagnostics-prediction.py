#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# ## Load the Data

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin/breast cancer.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df.describe().T


# In[ ]:


df.diagnosis.unique()


# In[ ]:


df['diagnosis'].value_counts()


# In[ ]:


sns.countplot(df['diagnosis'], palette='husl')


# ## clean and prepare the data

# In[ ]:


df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})
df.head()


# In[ ]:


df.isnull().sum()

#def diagnosis_value(diagnosis):
    if diagnosis == 'M':
        return 1
    else:
        return 0

#df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)
# In[ ]:


df.corr()


# radius_mean , perimeter _mean, area_mean have a high correlation with malignant tumor

# In[ ]:


plt.hist(df['diagnosis'], color='g')
plt.title('Plot_Diagnosis (M=1 , B=0)')
plt.show()


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)


# In[ ]:


# generate a scatter plot matrix with the "mean" columns
cols = ['diagnosis',
        'radius_mean', 
        'texture_mean', 
        'perimeter_mean', 
        'area_mean', 
        'smoothness_mean', 
        'compactness_mean', 
        'concavity_mean',
        'concave points_mean', 
        'symmetry_mean', 
        'fractal_dimension_mean']

sns.pairplot(data=df[cols], hue='diagnosis', palette='rocket')


# almost perfectly linear patterns between the radius, perimeter and area attributes are hinting at the presence of multicollinearity between these variables. (they are highly linearly related)
# Another set of variables that possibly imply multicollinearity are the concavity, concave_points and compactness.
Multicollinearity is a problem as it undermines the significance of independent varibales and we fix it 
by removing the highly correlated predictors from the model
Use Partial Least Squares Regression (PLS) or Principal Components Analysis, regression methods that cut the number 
of predictors to a smaller set of uncorrelated components.
# In[ ]:


# Generate and visualize the correlation matrix
corr = df.corr().round(2)

# Mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set figure size
f, ax = plt.subplots(figsize=(20, 20))

# Define custom colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.tight_layout()

we can verify the presence of multicollinearity between some of the variables. 
For instance, the radius_mean column has a correlation of 1 and 0.99 with perimeter_mean and area_mean columns, respectively.
This is because the three columns essentially contain the same information, which is the physical size of the observation
(the cell). 
Therefore we should only pick ONE of the three columns when we go into further analysis.Another place where multicollienartiy is apparent is between the "mean" columns and the "worst" column.
For instance, the radius_mean column has a correlation of 0.97 with the radius_worst column.
# also there is multicollinearity between the attributes compactness, concavity, and concave points. So we can choose just ONE
# out of these, I am going for Compactness.

# In[ ]:


# first, drop all "worst" columns
cols = ['radius_worst', 
        'texture_worst', 
        'perimeter_worst', 
        'area_worst', 
        'smoothness_worst', 
        'compactness_worst', 
        'concavity_worst',
        'concave points_worst', 
        'symmetry_worst', 
        'fractal_dimension_worst']
df = df.drop(cols, axis=1)

# then, drop all columns related to the "perimeter" and "area" attributes
cols = ['perimeter_mean',
        'perimeter_se', 
        'area_mean', 
        'area_se']
df = df.drop(cols, axis=1)

# lastly, drop all columns related to the "concavity" and "concave points" attributes
cols = ['concavity_mean',
        'concavity_se', 
        'concave points_mean', 
        'concave points_se']
df = df.drop(cols, axis=1)

# verify remaining columns
df.columns


# In[ ]:


# Draw the heatmap again, with the new correlation matrix
corr = df.corr().round(2)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.tight_layout()


# ## Building Model
# 

# In[ ]:


X=df.drop(['diagnosis'],axis=1)
y = df['diagnosis']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)


# ### Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()

X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)


# ### Models and finding out the Best one

# #### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,prediction1)
cm


# In[ ]:


sns.heatmap(cm,annot=True)
plt.savefig('h.png')


# In[ ]:


TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]
print('Testing Accuracy:',(TP+TN)/(TP+TN+FN+FP))


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,prediction1)


# #### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()
model2=dtc.fit(X_train,y_train)
prediction2=model2.predict(X_test)
cm2= confusion_matrix(y_test,prediction2)


# In[ ]:


cm2


# In[ ]:


accuracy_score(y_test,prediction2)


# #### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
model3 = rfc.fit(X_train, y_train)
prediction3 = model3.predict(X_test)
confusion_matrix(y_test, prediction3)


# In[ ]:


accuracy_score(y_test, prediction3)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction3))


# In[ ]:


print(classification_report(y_test, prediction1))

print(classification_report(y_test, prediction2))


# #### K Nearest Neighbor (K NN)
# #### Support Vector Machine
# #### Naive Bayes

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


# In[ ]:


models=[]

models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


# evaluate each model

results =[]
names=[]
for name , model in models:
    kfold=KFold(n_splits=10, random_state=40)
    cv_results= cross_val_score(model, X_train, y_train, cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    
    msg= '%s:, %f, (%f)' % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:


# make predictions on test datasets

SVM = SVC()
SVM.fit(X_train, y_train)
predictions= SVM.predict(X_test)
print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


# ##### We are getting the best accuracy with SVM which is 96.4%  , the model is predicting with 96% accuracy on our test data
# 

# TP :112 cases are  correctly identified 
# TN :53 are correctly rejected
# FN : 3 are incorrectly rejected and 
# FP : 3 are incorrectly identified
