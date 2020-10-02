#!/usr/bin/env python
# coding: utf-8

# # Pima indianas Diabetes Database
# 
# Hello, today i am going to  estimate the liklihood of person to be diabetic, this is my first MachineLearning real application away from the theory.
#  
# 

# ## Prepare , load the data 
# 
# first things first, import the modules needed

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# disable warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


filename = '../input/diabetes.csv'
data=pd.read_csv(filename)


# let's take a very quick look at our data

# In[ ]:


print(data.columns) # to know all the features(variables) we got in our data


# In[ ]:


print(data.head()) #the first 5 rows


#  ## First things first: analysing the outcome
#  
#  by the outcome we mean whether the person is diabetic or no (1 = yes; 0 = no)
# 
# descriptive statistics summary :
# 
#  
#  

# In[ ]:


data['Outcome'].describe()


# now the histogram of the outcome : 
# 

# In[ ]:


data['Outcome'].hist(figsize=(7,7))


# we can easily see that  the Non-Diabetic persons is more than the diabetics by what it seems the half
# that's mean accuracy is no more fine way to mesure how well our models doing.
# 

# #  ' Diabetes ' feature Relationships !!!!
# 
# **Diabetes** love Aged,ppl with high level of Glucose  " based from what I learned from high School", 
# So let's start with them, by that i mean there relation with Diabetes.
# 
# 

# In[ ]:



#borrowed from my friend Ayoub Benaissa.
def plot_diabetic_per_feature(data, feature):
    grouped_by_Outcome = data[feature].groupby(data["Outcome"])
    diabetic_per_feature = pd.DataFrame({"Sick": grouped_by_Outcome.get_group(1),
                                        "Not Sick": grouped_by_Outcome.get_group(0),
                                        })
    hist = diabetic_per_feature.plot.hist(bins=60, alpha=0.6)
    hist.set_xlabel(feature)
    plt.show()
    


# let's start with the age :

# In[ ]:


plot_diabetic_per_feature(data, "Age")


# *Intersting*, as u see, The older the persons,The higher the number of diabetics.
# ofc that's based from the dataset ' my doctor friend told me there's 2 types of Diabetes, so maybe this data concern the one who target the olders"
# 

# In[ ]:


plot_diabetic_per_feature(data, "Glucose")


# Same ... the more Glucose you have to more likley to have Diabetes, i guess that will be pretty obvious for you  if you know bit of biology.
# We notice the odd 0 information, let's confirm: 
# 
# 

# In[ ]:


print(data["Glucose"].min())


# that true, I don't think that a person can have 0 Glucose and still alive, so i'll delete it later + It also may effect the accuracy of the model.
# 
# 
# What about the **BMI** ( it's basically the weight of the person with kg) :
# 

# In[ ]:


plot_diabetic_per_feature(data, "BMI")


# very reasonable results, same notes from the 2 above, we can also notice the 0 odd info so we'll delete it late for the same reasons.

# # Think beyond the box
# let's think beyond the box, how is that? 
#     with :
# * Correlation matrix
# * Scatter plots between the most correlated variables
# 

# **Correlation matrix  : **
# 

# In[ ]:


import seaborn as sns #the librery we'll use for the job xD

corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cbar=True, annot=True, square=True, vmax=.8);


# Alright, let's break this matrix down to some few notes : 
# * Glucose, Age and  BMI are the most Correlated features with the 'Outcome'
# * Bloodpressure, SkinThikness have tiny Correlation with the outcome, hummm !
# * check how the SkinThikness and BMI Correlated, make me think of rolling it out since mose of the fat ppl tends to have thicc skin
# * Age with Pregnancies are the most Correlated features
# * Insulin with Glucuse ' BIOLOGY  :) "
# * DiabetesPedigreeFunction bit Correlated with most of them ' I am not sure with feature really mean"
# * finnaly SkinThikness with Insulin, that's odd !

# ### Scatter plots between 'Outcome' and correlated variables

# In[ ]:


sns.set()
cols = ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
sns.pairplot(data[cols], size = 2.5)
plt.show();


# opss! that's a huge amount of graphs to analyse but let's carry on :
# * bloodpressure and age tend to have a relation and that's kinda obviose since most of aged ppl have bloodpressure
# * Glucose and insulin have very strong relation makes me think about deleting the inslin futuere
# * in pregnancy/age we notice some kind of a liniar line in the right bottom 
# 
# that's it, no need to repeat the past notes ' even tho i did '
# 
# 

# # Outliers 
# Outliers is also something that we should be aware of. Why? Because outliers can markedly affect our models.
# 
# let's see what we can do :

# In[ ]:


data.min()


# It's okay to have 0 Pregnancies, for the rest, we'll delete the rows containing the 0 values:
# 

# In[ ]:


data = data.drop(data[data['Glucose'] == 0].index)
data = data.drop(data[data['SkinThickness'] == 0].index) # even it will be deleted xD
data = data.drop(data[data['BloodPressure'] == 0].index) #same
data = data.drop(data[data['BMI'] == 0].index)
data = data.drop(data[data['Insulin'] == 0].index)

print(data.min()) # let's check



# *excellent*, NEXT !!

# # Preparing the data/Selecting a model
# 
# since our data is only about 700 row of data, I'll go with k-fold cross validation (better that test/train in accuracy, but take more  computation time)

# In[ ]:


#just the libreries we need
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split 

from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import accuracy_score, make_scorer,f1_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


#preparing the data
cols = ['Pregnancies','Glucose','DiabetesPedigreeFunction','Insulin','BMI','Age']

Y=data['Outcome']
#rescaledX = StandardScaler().fit_transform(data[cols])
#X=pd.DataFrame(data = rescaledX, columns= cols)
X=data[cols]

# I deleted BloodPressure and Skinthikness
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, random_state = 25, test_size = 0.2)




svm1 = svm.SVC(kernel='linear')
svm2 = svm.SVC(kernel='rbf') 
lr = LogisticRegression()
rf = RandomForestClassifier()
knn=KNeighborsClassifier()
models = {"Logistic Regression": lr,"Random Forest": rf, "svm linear": svm1 , "svm rbf": svm2,"KNeighborsClassifier": knn }
l=[]
for model in models:
    l.append(make_pipeline(Imputer(),  models[model]))
#Finally get the cross-validation scores
i=0

for Classifier in l:    
    accuracy = cross_val_score(Classifier,X_train,Y_train,scoring='accuracy',cv=10)
    print("===", [*models][i] , "===")
    print("accuracy = ",accuracy)
    print("accuracy.mean = ", accuracy.mean())
    print("accuracy.variance = ", accuracy.var())
    i=i+1
    print("")
    


# our outcome (1 and 0) isn't balanced, but not very much so I think it's okay to use accuracy to defince the best models then compare them by F1 score
# 
# notes:
# * I split the data into test/train sets , I did the cv on the train test then I'll pick the best models and test them in the test set ( overfitting avoiding      level:99999 )  and compare them by f1 score / recall / precision
# * Logistic Regression got the higher accuracy mean and  low variance 
# * Random Forest and  svm linear  both seems to be good models
# * svm rbf get around 0 variance but I don't actually care since it has pretty low accuracy.mean (checking its table all the values are less than 0.7) 
# 
# I'll go with  Logistic Regression,  Random Forest and  svm linear:

# In[ ]:



lr = LogisticRegression()
lr.fit(X_train,Y_train)
predictions = lr.predict(X_test)
sns.heatmap(confusion_matrix(Y_test, predictions), annot=True, cmap="YlGn")
plt.title(' LogisticRegression ')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print("the f1 score for Logistic Regression is :",(f1_score(Y_test, predictions, average="macro")))
print("the precision score is :",(precision_score(Y_test, predictions, average="macro")))
print("the recall score is :",(recall_score(Y_test, predictions, average="macro")))   


# In[ ]:



rf = RandomForestClassifier()
rf.fit(X_train,Y_train)
predictions = rf.predict(X_test)
sns.heatmap(confusion_matrix(Y_test, predictions), annot=True, cmap="YlGn")
plt.title(' Random Forest Classifier ')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print("the f1 score for Random Forest Classifier is :",(f1_score(Y_test, predictions, average="macro")))
print("the precision score is :",(precision_score(Y_test, predictions, average="macro")))
print("the recall score is :",(recall_score(Y_test, predictions, average="macro")))   


# In[ ]:


svm = svm.SVC(kernel='linear')
svm.fit(X_train,Y_train)
predictions = svm.predict(X_test)
sns.heatmap(confusion_matrix(Y_test, predictions), annot=True, cmap="YlGn")
plt.title(' SVM kernel(linear) ')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print("the f1 score for SVM linear is :",(f1_score(Y_test, predictions, average="macro")))
print("the precision score is :",(precision_score(Y_test, predictions, average="macro")))
print("the recall score is :",(recall_score(Y_test, predictions, average="macro")))   


# Random Forest and support vector machine (linear) both have pretty good f1 score but I'll go with the **Random forest** since it have  higher **fscore** and lower ** false positive values**, and it can be more tuned later.
# 
# 
# **thanks for reading  and to the next version/kernels **
# 
# 
# 

# In[ ]:





# In[ ]:



    


# In[ ]:




