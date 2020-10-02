#!/usr/bin/env python
# coding: utf-8

# This notebook aims to explore the dataset and evaluate various diabetes mellitus forecasting algorithms. 

# <h1>Importing and Cleaning Dataset</h1>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mlt
from sklearn.model_selection import cross_val_score


# In[ ]:


Data=pd.read_csv("../input/diabetes.csv")
Data=Data.dropna(thresh=9)


# For purpose of predictive modelling the data is split into test and training set. 

# In[ ]:


M_Data=Data
Outcome=M_Data['Outcome']
M_Data.drop('Outcome',axis=1,inplace=True)


# Split the data into Positive and Negative Examples for purpose of exploratory analysis

# In[ ]:


Data=pd.read_csv("../input/diabetes.csv")
Positives=Data[Data['Outcome']==1]
Negatives=Data[Data['Outcome']==0]


# <h2>Attributes</h2>
# 
# **Pregnancies:** Number of times pregnant
# 
# **Glucose:** Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# **BloodPressure:** Diastolic blood pressure (mm Hg)
# 
# **SkinThickness:** Triceps skin fold thickness (mm)
# 
# **Insulin:** 2-Hour serum insulin (mu U/ml)
# 
# **BMI:** Body mass index (weight in kg/(height in m)^2)
# 
# **DiabetesPedigreeFunction:** Diabetes pedigree function
# 
# **Age:** Age (years)
# 
# **Outcome:** Class variable (0 or 1)

# <h1> Exploratory Analysis</h1>

# In[ ]:


#Import data from csv file
Data.sample(frac=0.1).head(n=5)


# In[ ]:


Data.describe()


# In[ ]:


#For curve fitting
from scipy import stats


# Upon splitting the dataset into Positive and Negative examples , we analyze any visible trends in the positive and negative examples. The below are plots of variables that show trends in relationship with respect to the outcome (postive or negative).  The plots show below only depict variables with a visible relation. 
# The **red** graphs indicate **positive examples** , while **green** graphs indicate **negative examples**. 

# <h2> Positive Examples </h2>

# In[ ]:


fig, ax =plt.subplots(1,3)
sns.distplot(Positives['Pregnancies'],rug=True,kde=False,color='r',fit=stats.gamma,ax=ax[0])
sns.distplot(Positives['BloodPressure'],rug=True,kde=False,color='r',fit=stats.gamma,ax=ax[1])
sns.distplot(Positives['Age'],rug=True,kde=False,color='r',fit=stats.gamma,ax=ax[2])
fig.show()


# <h2>Negative Examples</h2>

# In[ ]:


fig, ax =plt.subplots(1,3)
sns.distplot(Negatives['Pregnancies'],rug=True,kde=False,color='g',fit=stats.gamma,ax=ax[0])
sns.distplot(Negatives['BloodPressure'],rug=True,kde=False,color='g',fit=stats.gamma,ax=ax[1])
sns.distplot(Negatives['Age'],rug=True,kde=False,color='g',fit=stats.gamma,ax=ax[2])
fig.show()


# From the above it could be noted that there is little between having Diabetes to pregnancies. While age and blood pressure have correlation with chances of diabetes. 

# <h3>Correlation between variables</h3>

# In[ ]:


Corr=Data[Data.columns].corr()
sns.heatmap(Corr,annot=True)


# <h1>Predictive Modelling </h1>

# In[ ]:


Data.Outcome.value_counts()


# In[ ]:


(500/float(len(Data)))*100


# The majority of samples in the data set are negatives with 500 examples makiing 65.104% of the data. If we were to simply predict negative for every value we would get a accuracy of 65.104% on this dataset. So it would be the baseline accuracy for evaluating learning algorithms.

# We are going to evaluate Naive Bayes , SVM linear and rbf kernel and Knn algorithms on cross validation data of K=5.

# In[ ]:


#Import the required libraries for machie learning algorithms

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree


# <h3>Naive Bayes</h3>

# In[ ]:


#Naive Bayes Algorithm 

gNB=GaussianNB()
scores=cross_val_score(gNB, M_Data,Outcome, cv=5)
print("Accuracy: ",scores.mean())


# <h3>K Nearest Neighbors</h3>

# In[ ]:



scores=[]
for i in range(1,31):
    neigh=KNeighborsClassifier(n_neighbors=i)
    scores.append(cross_val_score(neigh,M_Data,Outcome,cv=5).mean())
    
max_a=0
k_max=0

for i in range(0,30):
    
    if(scores[i]>=max_a):
        
        max_a=scores[i]
        
        if(i>k_max):
                
            k_max=i
        
print("K is maximum in Knn for ",k_max," with a accuracy of ",max_a)       
 


# <h3>Support Vector Machine (SVM)</h3>

# <h3>Linear Kernel</h3>

# In[ ]:


clf=svm.SVC(kernel='linear')
print("Accuracy: ",cross_val_score(clf, M_Data,Outcome, cv=5).mean())


# <h3>RBF Kernel</h3>

# In[ ]:


clf_r=svm.SVC(kernel='rbf')
print("Accuracy: ",cross_val_score(clf_r,M_Data,Outcome, cv=5).mean())


# <h3>Decision Tree</h3>

# In[ ]:


from sklearn import tree
cl=tree.DecisionTreeClassifier()
print("Accuracy: ",cross_val_score(cl,M_Data,Outcome, cv=5).mean())


# <h3> Random Forest Classifier</h3>

# In[ ]:


Rf=RandomForestClassifier()
print("Accuracy: ",cross_val_score(Rf,M_Data,Outcome, cv=5).mean())


# <h3>AdaBoost</h3>

# In[ ]:


AB=AdaBoostClassifier()
print("Accuracy: ",cross_val_score(AB,M_Data,Outcome, cv=5).mean())


# <h3>Artificial Neural Networks (ANN)</h3>

# In[ ]:


from keras.models import Sequential 
from keras.layers import Dense,Activation
from sklearn.model_selection import StratifiedKFold


# In[ ]:


Numpy_Matrix=M_Data.as_matrix()
Numpy_Outcome=Outcome.as_matrix()


# In[ ]:


from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout

K_fold=StratifiedKFold(n_splits=4,shuffle=True)
cv_scores=[]

for train,test in K_fold.split(M_Data,Outcome):
    
        model=Sequential()
        model.add(Dense(10,activation='relu',input_dim=8))
        model.add(BatchNormalization())
        
        model.add(Dense(12,activation='relu',input_dim=8))
        model.add(BatchNormalization())
        
        model.add(Dense(12,activation='relu'))
        model.add(BatchNormalization())
        
        model.add(Dense(12,activation='relu'))
        model.add(BatchNormalization())
        
        
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(Numpy_Matrix[train],Numpy_Outcome[train], epochs=280, batch_size=32)
        scores=model.evaluate(Numpy_Matrix[test],Numpy_Outcome[test])
        cv_scores.append(scores[1]*100)


# In[ ]:


def Mean(Scores):
    Sum=0
    for i in Scores:
        
        Sum+=i
        
    return(Sum/len(Scores))


# In[ ]:


print(Mean(cv_scores))


# The learning algorithm algorithm with the best accuracy is linear SVM.

# Lets further try to find if we could improve the model by eliminating irrelevant features.  by digging up the correlation matrix.

# In[ ]:


Corr.mean()


# In[ ]:


Data_P=Data
Data_P.drop('Pregnancies',axis=1,inplace=True)
Data_P.drop('Outcome',axis=1,inplace=True)
clf=svm.SVC(kernel='linear')
print("Accuracy: ",cross_val_score(clf,Data_P,Outcome,cv=5).mean())


# It is founded that eliminating pregnancies feature resulted in a better accuracy. However ,  feature Diabetes Pedigree Function and both DPF and pregnancies resulted in same accuracy.  L

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(Data_P,Outcome,test_size=0.5,random_state=0)
clf.fit(x_train,y_train)
Conf=confusion_matrix(y_test,clf.predict(x_test))
sns.heatmap(Conf,annot=True,)


# In[ ]:


print(classification_report(y_test, clf.predict(x_test)))

