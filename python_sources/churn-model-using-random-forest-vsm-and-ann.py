#!/usr/bin/env python
# coding: utf-8

# # Objectives
# 
# I am trying to build a churn model using three ML algorithms (Random Forest, SVM , and ANN). I am building as initial trial all models with default parameters without any tuning or Cross Validation. Will Check which is the best model for next steps of tuning
# 

# In[ ]:



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.metrics import roc_curve , auc
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Load Data
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# We observe SeniorCitizen Should be categorical variables, but comes as int64. Will convert it back to categorical

# In[ ]:


df.SeniorCitizen.unique()


# In[ ]:


#Convert to Categorical variable
df.SeniorCitizen= df.SeniorCitizen.apply(lambda x : 'No' if x == 0 else 'Yes')


# In[ ]:


#Check Type after conversion
df.SeniorCitizen.unique()


# Another Observation that TotalCharges is continues variables and comes as object. Will convert to numeric format

# In[ ]:


df['TotalCharges_new']= pd.to_numeric(df.TotalCharges,errors='coerce')


# In[ ]:


#Check NULL values after the conversion
df.loc[pd.isna(df.TotalCharges_new),'TotalCharges']


# In[ ]:


#Fill 11 Missing values from the original column
TotalCharges_Missing=[488,753,936,1082,1340,3331,3826,4380,5218,6670,6754]
df.loc[pd.isnull(df.TotalCharges_new),'TotalCharges_new']=TotalCharges_Missing


# In[ ]:


#We are good to replace old columns with the new numerical column
df.TotalCharges=df.TotalCharges_new
df.drop(['customerID','TotalCharges_new'],axis=1,inplace=True)
df.info()


# Now will check all categorical variables levels

# In[ ]:


df.dtypes=='object'
categorical_var=[i for i in df.columns if df[i].dtypes=='object']
for z in categorical_var:
    print(df[z].name,':',df[z].unique())


# There are some variables has value 'No Internet Service' that equivalent to 'No'. Will merge both values

# In[ ]:


Dual_features= ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for i in Dual_features:
    df[i]=df[i].apply(lambda x: 'No' if x=='No internet service' else x)
#Remove No Phones Service that equivilent to No for MultipleLines
df.MultipleLines=df.MultipleLines.apply(lambda x: 'No' if x=='No phone service' else x)


# In[ ]:


#Check levels or all Categorical Variables
for z in [i for i in df.columns if df[i].dtypes=='object']:
    print(df[z].name,':',df[z].unique())


# # Data Analysis and Visualizations
# 
# First will analyze continues variables against Churn variable

# In[ ]:


continues_var=[i for i in df.columns if df[i].dtypes !='object']
fig , ax = plt.subplots(1,3,figsize=(15,5))
for i , x in enumerate(continues_var):
    ax[i].hist(df[x][df.Churn=='No'],label='Churn=0',bins=30)
    ax[i].hist(df[x][df.Churn=='Yes'],label='Churn=1',bins=30)
    ax[i].set(xlabel=x,ylabel='count')
    ax[i].legend()


# We can see a real impact of all continues variables on Churn specially Tenue(Loyal Customers Stay)
# 
# Will Check now Box Plot for more explorations

# In[ ]:


fig , ax = plt.subplots(1,3,figsize=(15,5))
for i , xi in enumerate(continues_var):
    sns.boxplot(x=df.Churn,y=df[xi],ax=ax[i],hue=df.gender)
    ax[i].set(xlabel='Churn',ylabel=xi)
    ax[i].legend()


# Now it is more clear the impact of Continues Variables on Churn , We can see minimal impact of Gender
# 
# Now will convert to check regarding Categorical Variables

# In[ ]:


#Remove Churn Variable for Analysis
categorical_var_NoChurn= categorical_var[:-1]


# In[ ]:


#Count Plot all Categorical Variables with Hue Churn
fig , ax = plt.subplots(4,4,figsize=(20,20))
for axi , var in zip(ax.flat,categorical_var_NoChurn):
    sns.countplot(x=df.Churn,hue=df[var],ax=axi)


# - We cannot see a real Impact of gender
# - Seniors are less loyalty
# - Partners are more loyal
# - Dependents are more loyal
# - Customers does not have multiplelines are more loyal
# - Customer are not happy with Optical Fiber and Leaving with rate of other internet services
# - Customers with month-to-month contract are more willing to leave than people with contracts
# - Paperless customers are more willing to leave that paper billing
# - Customer pay using electronic check is more willing to leave
# 
# I Can conclude that mostly customers are suffering from the services , and specially advances customers who are using paperless billing and electronic payment. Some variables has no real impact of Churn but as a first trial for the model i will include all variables, should remove variables in the tuning phase

# ## Categorical Variables Encoding
# 
# For logistics variables(2 classes) will encode using Label Encoder , For Variables has more than 2 classes will use get_dummies function 

# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
for x in [i for i in df.columns if len(df[i].unique())==2]:
    print(x, df[x].unique())
    df[x]= label_encoder.fit_transform(df[x])


# In[ ]:


#Check Variables after Encoding
[[x, df[x].unique()] for x in [i for i in df.columns if len(df[i].unique())<10]]


# In[ ]:


#Encode Variables with more than 2 Classes
df= pd.get_dummies(df, columns= [i for i in df.columns if df[i].dtypes=='object'],drop_first=True)
  


# In[ ]:


#Check Variables after Encoding
[[x, df[x].unique()] for x in [i for i in df.columns if len(df[i].unique())<10]]


# Variables Looks good now and we are ready for data splitting and scaling
# 
# # Data Scaling and Splitting

# In[ ]:


#Create Features DataFrame
X=df.drop('Churn',axis=1)
#Create Target Series
y=df['Churn']
#Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[ ]:


#Scale Data
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_train=pd.DataFrame(X_train,columns=X.columns)
X_test=sc.transform(X_test)


# In[ ]:


#Check Data after Scaling
X_train.head()


# # Applying ML Models
# 
# ## Random Forest Model
# 
# Will start by train Random forest model using default parameters and all variables and get initial results

# In[ ]:


#Apply RandomForest Algorethm
random_classifier= RandomForestClassifier()
random_classifier.fit(X_train,y_train)


# In[ ]:


y_pred= random_classifier.predict(X_test)


# In[ ]:


#Classification Report
print(classification_report(y_test,y_pred))


# In[ ]:


#Confusion Matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
          xticklabels=['No','Yes'],
          yticklabels=['No','Yes'] )
plt.xlabel('true label')
plt.ylabel('predicted label')


# Result are not bad as a start. Recall, and Precision of Churn='Yes' is not that good. We need to check features importance for the next tuning

# In[ ]:


#get features Importances
xx= pd.Series(random_classifier.feature_importances_,index=X.columns)
xx.sort_values(ascending=False)


# We need to use this list in next tuning of the model
# 
# Finally will draw ROC curve for the model

# In[ ]:


y_pred_proba=random_classifier.predict_proba(X_test)[:,1]


# In[ ]:


fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc=auc(fpr,tpr)
#Now Draw ROC using fpr , tpr
plt.plot([0, 1], [0, 1], 'k--',label='Random')
plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Random Forest ROC curve')
plt.legend(loc='best')


# ## SVM Model
# 
# Now will run SVM model and compare

# In[ ]:


svm_classifier= SVC(probability=True)
svm_classifier.fit(X_train,y_train)


# In[ ]:


#Predict
y_pred_svm= svm_classifier.predict(X_test)
#Classification Report
print(classification_report(y_test,y_pred_svm))


# In[ ]:


#Confusion Matrix
mat_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(mat_svm.T, square=True, annot=True, fmt='d', cbar=False,
          xticklabels=['No','Yes'],
          yticklabels=['No','Yes'] )
plt.xlabel('true label')
plt.ylabel('predicted label')


# SVM results is a little better than Random Forest. But not a huge improvement
# 
# Finally will draw ROC Curve for this model
# 

# In[ ]:


y_pred_svm_proba=svm_classifier.predict_proba(X_test)[:,1]
#ROC Curve
fpr_svm, tpr_svm, _svm = roc_curve(y_test, y_pred_svm_proba)
roc_auc=auc(fpr_svm,tpr_svm)
#Now Draw ROC using fpr , tpr
plt.plot([0, 1], [0, 1], 'k--',label='Random')
plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('SVM ROC curve')
plt.legend(loc='best')


# ## ANN Model
# 
# Lastly, will build Artificial Neural Network (ANN) Model which theoretically should bring best result.
# Will build 2 Hidden Layers with 12 Nodes , Using Variate Function , and Output layer with one Node using Sigmoid Function. Will not run Cross Validation as a first run and will use Adam as optimizer with 100 epochs
# 

# In[ ]:


#Initiate ANN Classifier
ann_classifier= Sequential()
X.shape


# In[ ]:


#Adding Hidden Layer1
ann_classifier.add(Dense(12,activation='relu',kernel_initializer='uniform',input_dim=23))
#Adding Hidden Layer2
ann_classifier.add(Dense(12,activation='relu',kernel_initializer='uniform'))
#Adding output Layer
ann_classifier.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))
#Compile them Model
ann_classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


ann_classifier.summary()


# In[ ]:


get_ipython().run_line_magic('time', 'ann_classifier.fit(X_train,y_train,batch_size=10,epochs=100)')


# In[ ]:


#Get Prediction Proba
y_pred_ann_proba= ann_classifier.predict(X_test)


# In[ ]:


#Convert Prediction to Int
y_pred_ann= (y_pred_ann_proba>.5).astype('int')


# In[ ]:


#Priint Classification Report
print(classification_report(y_test,y_pred_ann))


# In[ ]:


#Confusion Matrix
mat_ann = confusion_matrix(y_test, y_pred_ann)
sns.heatmap(mat_ann.T, square=True, annot=True, fmt='d', cbar=False,
          xticklabels=['No','Yes'],
          yticklabels=['No','Yes'] )
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[ ]:


#Roc Curve
fpr_ann,tpr_ann,_ann=roc_curve(y_test,y_pred_ann_proba)
roc_auc=auc(fpr_ann,tpr_ann)
#Now Draw ROC using fpr , tpr
plt.plot([0, 1], [0, 1], 'k--',label='Random')
plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' %roc_auc)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')


# We can see a big improvement with ANN comparing with other models.
# 
# ## Next Step
# 
# I will review deeply all variables, and start tune ANN models for better results
# 
# 
# I hope this Kernel is useful. Happy to receive your comments, questions, and advises
# 
# 

# In[ ]:




