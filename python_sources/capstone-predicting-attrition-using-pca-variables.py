#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df=pd.read_csv("../input/HR_Employee_Attrition_Data.csv")


# In[ ]:


df.head()


# # Checking Nulls

# In[ ]:


df.isnull().any()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# # Taking the numeric Colums for analysis

# In[ ]:


df_n=df.select_dtypes(include=['number'])


# In[ ]:


df_n.head()


# # Finding outliers

# In[ ]:



fig,axes =plt.subplots(6,5, figsize=(20, 30))
ax=axes.ravel()
i=0

for column in df_n.columns:
    sns.boxplot(data=df_n[column],ax=ax[i])
    ax[i].set_xlabel(list(df_n.columns)[i])
    i +=1
plt.show()    


# In[ ]:


#MonthlyIncome,NunCompaniesWorked,StockOptionLevel,TotalWorkingYears,TrainingTimesLastYear,YearsAtCompany,YearsInCurrentRole,
#YearsSinceLastPromotion,YearsWithCurrManager


# # Finding Categorical Variable

# In[ ]:


df.describe(include='object')


# In[ ]:


df_object=df.select_dtypes(include=['object'])


# In[ ]:


df_object.head()


# # Dropping Dependant Variable

# In[ ]:


df_object.drop("Attrition",axis=1,inplace=True)


# In[ ]:


df_object.head()


# In[ ]:


df_object_FU=df_object


# # Getting Dummies or Hot encoding for Categorical Variable

# In[ ]:


for column in list(df_object.columns):
    df_tmp=pd.get_dummies(df_object[column])
    df_object=pd.concat([df_object,df_tmp], axis=1)
    


# In[ ]:


df_object.head()


# In[ ]:


df_object.drop(list(df_object_FU.columns),axis=1,inplace=True)


# In[ ]:


df_object.head()


# # Concatinating Numeric variable and Dummy for Categorical Variavle

# In[ ]:


df_C=pd.concat([df_n,df_object],axis=1)


# In[ ]:


df_C.head()


# # Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_C_scaled=scaler.fit_transform(df_C)


# In[ ]:


df_C_scaled.shape


# # Pricipal Componant Analysis

# In[ ]:


from sklearn.decomposition import PCA
pca=PCA(n_components=5) 
x_pca=pca.fit_transform(df_C_scaled) 
#let's check the shape of X_pca array
print ("shape of x_pca", x_pca.shape)


# In[ ]:


x_pca


# # Variation Covered by PCA components

# In[ ]:


ex_variance=np.var(x_pca,axis=0)
ex_variance


# In[ ]:


ex_variance_ratio = ex_variance/np.sum(ex_variance)
print (ex_variance_ratio)


# # Cumulative Variance

# In[ ]:


Cuml_Var=[]
a=0
for var in ex_variance_ratio:
    a +=var
    Cuml_Var.append(a*100)
Cuml_Var


# In[ ]:


PCA_var=pd.DataFrame({"PCA":['PC1','PC2','PC3','PC4','PC5'],'Varience':ex_variance_ratio*100,"Cuml_Variance":Cuml_Var})


# In[ ]:


PCA_var


# # PCA Data Frame

# In[ ]:


df_C_pca=pd.DataFrame(x_pca[:,0:4], columns=['PC1','PC2','PC3','PC4']) 
#Taking 4 PCs as by using 4 PCs we can cover 86 percent variance in Dataset


# In[ ]:


df_C_pca.head()


# # Feature Selection

# In[ ]:


X=df_C_pca.values


# In[ ]:


y=df.Attrition.values


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


X_test


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)


# # Prediction

# In[ ]:


y_pred=logreg.predict(X_test)


# In[ ]:


y_pred


# # Accuracy Calculation

# In[ ]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Accuracy:","\n",metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:","\n",metrics.classification_report(y_test, y_pred))


# # Applying SVM

# In[ ]:


from sklearn.svm import SVC
svc=SVC() #Default hyperparameters
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Accuracy:","\n",metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:","\n",metrics.classification_report(y_test, y_pred))

