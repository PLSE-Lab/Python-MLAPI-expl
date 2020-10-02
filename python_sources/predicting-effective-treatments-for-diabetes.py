#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.


# ## Given were the 5 datasets,we would like to work on the EDA and Feature Engineering of data separately and then merge them together for bringing out the insights from the data and for predictive analysis

# **Importing the dataset Admission Details** 

# In[ ]:


admission = pd.read_excel('../input/admission_details.xlsx')


# In[ ]:


admission.head(2)


# In[ ]:


admission.info()


# #### Checking for any NULL values

# In[ ]:


admission.isnull().values.any()


# In[ ]:


admission.payer_code.value_counts()


# In[ ]:


admission.medical_specialty.value_counts()


# ### We can observe that, Payer code and medical speciality have more than 50% of the missing data, and prefer to drop those features.

# In[ ]:


admission = admission.drop(['medical_specialty','payer_code'],axis = 1)


# In[ ]:


admission.time_in_hospital.value_counts()


# In[ ]:


admission.columns


# In[ ]:


admission.head(2)


# In[ ]:


admission.shape


# ### Loading the Diabetes data of the patients

# In[ ]:


diabetic = pd.read_csv('../input/diabetic_data.csv')


# In[ ]:


diabetic.columns


# In[ ]:


diabetic.info()


# In[ ]:


diabetic.readmitted.value_counts()


# In[ ]:


diabetic.diabetesMed.value_counts()


# In[ ]:


diabetic.change.value_counts()


# #### Checking for NULL values

# In[ ]:


diabetic.isnull().values.any()


#  #### Custom encoding for the 23 Drug Features
# 

# In[ ]:


Diabetic_transform=diabetic.replace(['No','Steady','Up','Down'],[0,1,1,1])
Diabetic_transform.set_index('encounter_id',inplace=True)


# In[ ]:


Diabetic_transform.head()


# In[ ]:


Diabetic_transform.sum(axis=1).value_counts()


# # Patients are Given at max a combination of 6 drugs for treating diabetes

# In[ ]:


#diabetic.set_index('encounter_id',inplace=True)


# ### Feature Engineering - Creating a new feature "Treatments"

# **1. When the value of Insuin is '1' , creating the classes "insulin" & "io" (insulin + others )********

# In[ ]:


i1 = Diabetic_transform[Diabetic_transform['insulin']==1].sum(axis = 1).replace([1,2,3,4,5,6],['insulin','io','io','io','io','io'])


# In[ ]:


i1.value_counts()


# **2. When the value of Insuin is '0' , creating the classes "others" & "no med"**

# In[ ]:


i0=Diabetic_transform[Diabetic_transform['insulin']==0].sum(axis=1).replace([0,1,2,3,4,5,6],['no med','other','other','other','other','other','other'])


# In[ ]:


i0.value_counts()


# In[ ]:


treatments=pd.concat([i1,i0])
treatments = pd.DataFrame({'treatments':treatments})


# In[ ]:


treatments.head()


# **Adding the new feature to the Diabetic Dataframe**

# In[ ]:


diabetic=diabetic.join(treatments,on='encounter_id') #setting index as encounter_id


# In[ ]:


diabetic.head(2)


# In[ ]:


diabetic.shape


# In[ ]:


diabetic.columns


# ### Importing Lab Sessions Dataset

# In[ ]:


lab_sessions = pd.read_excel('../input/Lab-session.xlsx')


# In[ ]:


lab_sessions.columns


# In[ ]:


lab_sessions.info()


# In[ ]:


lab_sessions.columns


# ### Checking for null values

# In[ ]:


lab_sessions.isnull().values.any()


# In[ ]:


lab_sessions.shape


# ### Importing the Patient Details Dataset

# In[ ]:


patient_details = pd.read_excel('../input/Paitent_details.xlsx')


# In[ ]:


patient_details.columns


# In[ ]:


patient_details.isnull().values.any()


# **Since weight has more than 50% missing values, we tend to drop that feature**

# In[ ]:


patient_details = patient_details.drop(['weight'],axis = 1)


# In[ ]:


patient_details.race.value_counts()


# **We can observe that the "Race" Feature has some missing values**

# **Missing value Imputation using MODE for Race Feature as most of the people in the Dataset are Caucasian**

# ##### 1. Replacing the ? with NaN's

# In[ ]:


patient_details['race']=patient_details.race.replace('?',np.nan)


# ##### 2. Filling the NaN's with the mode

# In[ ]:


patient_details['race'].fillna(patient_details['race'].mode()[0], inplace=True)


# In[ ]:


patient_details.race.isnull().sum()


# In[ ]:


patient_details.shape


# # Concatinatin the Dataframes "Admission" , "Diabetic" ,"Patient_details" & "Lab_sessions"

# In[ ]:


print("Admission" , admission.shape)
print("Diabetic" ,diabetic.shape)
print("Lab Sessions",lab_sessions.shape)
print("Patient_details",patient_details.shape)


# In[ ]:


data = pd.concat([patient_details,admission,lab_sessions,diabetic],axis=1)


# In[ ]:


data.shape


# In[ ]:


#data = pd.read_csv('Final_Diabetes_withallrows.csv')


# In[ ]:


#data[['admission_source_id','time_in_hospital']] = admission[['admission_source_id','time_in_hospital']]


# In[ ]:


#data[['num_lab_procedures','num_medication','number_outpatient','number_emergency','number_inpatient','num_procedures']] = lab_sessions[['num_lab_procedures','num_medications',
                                                                                                                        #'number_outpatient','number_emergency','number_inpatient','num_procedures']]


# In[ ]:


data.columns


# # Storing Encounter_id and Patient_nbr columns in a seperate DataFrame

# In[ ]:


df1=data.iloc[:,:2]

df1.head()


# # Dropping all Duplicate columns from DataFrame

# In[ ]:


df2=data.drop(['encounter_id','patient_nbr'],axis=1)

df2.head()


# # Concatenation of df1 and df2 Dataframes

# In[ ]:


data_final=pd.concat([df1,df2],axis=1)

data_final.shape


# In[ ]:


data_final.head()


# **Filtering the records of patients having Diabetes**

# In[ ]:



data_diamed_yes=data_final[data_final.diabetesMed=='Yes']
data_diamed_yes.shape


# **Considering records of Diabetic Patients who didn't Readmit**

# In[ ]:


data_readmit_no=data_diamed_yes[data_diamed_yes.readmitted=='NO']
data_readmit_no.shape


# **Excluding the patients who are Dead and are in Hospice**

# In[ ]:


data_new=data_readmit_no[~data_readmit_no.discharge_disposition_id.isin([11,13,14,19,20])]
data_new.shape


# ****Choosing the records with treatments Insulin and Insulin + other ( w.r.t Problem Statement)**

# In[ ]:


data_model=data_new[data_new.treatments!='other']


# In[ ]:


data_model.info()


# In[ ]:


data_model.shape


# In[ ]:


data_model.head().T


# In[ ]:


#data_cat = data_model.select_dtypes(include=['object']).copy()
data_model.treatments.value_counts()


# **Since Treatments column is the combination of the 23 drug features,we will be dropping them**

# In[ ]:


data_model = data_model.drop(['metformin',
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone'],axis = 1)
data_model.shape


# In[ ]:


data_model.columns


# In[ ]:


data_model.num_procedures.plot(kind='hist')
plt.xlabel("No.of Lab Procedures")


# In[ ]:


import seaborn as sns
sns.barplot(data_model.discharge_disposition_id)


# In[ ]:


data_model.num_medications.plot(kind='hist')
plt.xlabel("No.of Medications")


# # Here the features which contains numeric values are of type Discrete Quantitative and has a finite set of values. Discrete data can be both Quantitative and Qualitative. So treating outliers in this dataset is not possible

# **One hot encoding the nominal categorical values**

# In[ ]:


data_onehot = pd.get_dummies(data_model, columns=['race', 'gender','max_glu_serum', 'A1Cresult', 'change',
       'diabetesMed', 'readmitted'])


# **Label Encoding the AGE(ordinal) categorical column**

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[ ]:


data_onehot['age']=data_onehot.apply(le.fit_transform)


# In[ ]:


data_onehot[['discharge_disposition_id','admission_type_id', 'admission_source_id','num_lab_procedures',
       'num_procedures','time_in_hospital',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient']]=  data_model[['discharge_disposition_id','admission_type_id','admission_source_id','num_lab_procedures',
       'num_procedures','time_in_hospital',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient']]


# In[ ]:


data_onehot.shape


# **Feature selection**

# In[ ]:


data_onehot.info()


# In[ ]:


data_onehot.head()


# In[ ]:


#data_onehot = data_onehot.drop(['Unnamed: 0'],axis = 1)


# In[ ]:


data_onehot.shape


# In[ ]:


data_onehot.columns


# # Chi-Square Test of Independence

# In[ ]:


import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)


# In[ ]:


data_onehot['dummyCat'] = np.random.choice([0, 1], size=(len(data_onehot),), p=[0.5, 0.5])

data_onehot.dummyCat.value_counts()


# In[ ]:


#Initialize ChiSquare Class
cT = ChiSquare(data_onehot)

#Feature Selection
testColumns = ['encounter_id', 'patient_nbr', 'age', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
       'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient',
        'race_AfricanAmerican', 'race_Asian', 'race_Caucasian',
       'race_Hispanic', 'race_Other', 'gender_Female', 'gender_Male',
       'max_glu_serum_>200', 'max_glu_serum_>300', 'max_glu_serum_None',
       'max_glu_serum_Norm', 'A1Cresult_>7', 'A1Cresult_>8', 'A1Cresult_None',
       'A1Cresult_Norm', 'change_Ch', 'change_No', 'diabetesMed_Yes',
       'readmitted_NO']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="treatments" ) 


# In[ ]:


from scipy.stats import chisquare,chi2_contingency

cat_col = []
chi_pvalue = []
chi_name = []

def chi_sq(i):
    ct = pd.crosstab(data_onehot['treatments'],data_onehot[i])
    chi_pvalue.append(chi2_contingency(ct)[1])
    chi_name.append(i)

for i in testColumns:
    chi_sq(i)

chi_data = pd.DataFrame()
chi_data['Pvalue'] = chi_pvalue
chi_data.index = chi_name

plt.figure(figsize=(11,8))
plt.title('P-Values of Chisquare with ''Treatments'' as Target Categorical Attribute',fontsize=16)
x = chi_data.Pvalue.sort_values().plot(kind='barh')
x.set_xlabel('P-Values',fontsize=15)
x.set_ylabel('Independent Categorical Attributes',fontsize=15)
plt.show()


# **Feature Selection using Random Forest Algorithm**

# In[ ]:


# Import `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Isolate Data, class labels and column values
X = data_onehot.drop(['treatments'],axis=1)
Y = data_onehot['treatments']
Y=Y.replace(['insulin','io'],[0,1])
names = data_onehot.columns.values


# In[ ]:


X.head()


# In[ ]:


Y.head()


# In[ ]:


Y.shape


# In[ ]:


# Build the model
rfc = RandomForestClassifier()

# Fit the model
rfc.fit(X, Y)


# In[ ]:


#Finding the feature importance using Random Forest
feature_imp=pd.DataFrame({'Features':X.columns,'Importance':rfc.feature_importances_})
feature_imp.sort_values(by = 'Importance',ascending=True)


# In[ ]:


data_onehot.columns


# ## Out of Two techniques for Feature Selection, Chi-Square Test of Independence seems to be more efficient when compared to Random Forest. So dropping the Features which are labelled as not important predictor by Chi-Square test
# 

# In[ ]:


X = data_onehot.drop(['encounter_id','patient_nbr','age','num_lab_procedures','number_outpatient','number_emergency',
                      'race_Asian','race_Other','diabetesMed_Yes','max_glu_serum_>200','A1Cresult_>8','A1Cresult_Norm',
                      'readmitted_NO','dummyCat','treatments'],axis=1)
X.info()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ## Baseline model

# In[ ]:


y_p=[]
for i in range(y_test.shape[0]):
    y_p.append(y_test.mode()[0])#Highest class is assigned to a list which is compared with ytest
len(y_p) 


# In[ ]:


y_pred=pd.Series(y_p)


# In[ ]:


accuracy_score(y_test,y_pred)


# #### Baseline Accuracy is around 54.4%

# ### Below Built every model is a base model so there is furthur possibilty that a model can perform better by tuning it.

# # Logistic Regression

# In[ ]:


#Logistic Regression
m1=LogisticRegression()
m1.fit(X_train,y_train)
y_pred_lr=m1.predict(X_test)
Train_Score_lr = m1.score(X_train,y_train)
Test_Score_lr = accuracy_score(y_test,y_pred_lr)


# In[ ]:


print('Training Accuracy is:',Train_Score_lr)
print('Testing Accuracy is:',Test_Score_lr)
print(classification_report(y_test,y_pred_lr))


# #### The training accuracy and testing accuracy of model is almost similar which helps us to understand that model is neither overfitting nor underfitting

# In[ ]:


from sklearn.metrics import roc_curve, auc
fpr,tpr, _ = roc_curve(y_test,y_pred_lr)
roc_auc_lr = auc(fpr, tpr)

print('Auc for Logistic Regression is:',roc_auc_lr)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# ## KNN Classifier
# 

# In[ ]:


m2 = KNeighborsClassifier()
m2.fit(X_train,y_train)
y_pred_knn = m2.predict(X_test)
Train_Score_knn = m2.score(X_train,y_train)
Test_Score_knn = accuracy_score(y_test,y_pred_knn)


# In[ ]:


print('Training Accuracy is :',Train_Score_knn)
print('Testing Accuracy is:',Test_Score_knn)
print(classification_report(y_test,y_pred_knn))


# #### The training accuracy when compared to testing accuracy of model is more which helps us to understand that model is overfitting

# In[ ]:


fpr,tpr, _ = roc_curve(y_test,y_pred_knn)
roc_auc_knn = auc(fpr, tpr)

print('Auc for KNN is:',roc_auc_knn)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# ## Bernoulli Naives Bayes

# In[ ]:


m3=BernoulliNB()
m3.fit(X_train,y_train)
y_pred_bnb=m3.predict(X_test)
Train_Score_bnb = m3.score(X_train,y_train)
Test_Score_bnb = accuracy_score(y_test,y_pred_bnb)


# In[ ]:


print('Training Accuracy :',Train_Score_bnb)
print('Testing Accuracy  :',Test_Score_bnb)
print(classification_report(y_test,y_pred_bnb))


# #### The training accuracy and testing accuracy of model is almost similar which helps us to understand that model is neither overfitting nor underfitting

# In[ ]:


fpr,tpr, _ = roc_curve(y_test,y_pred_bnb)
roc_auc_bnb = auc(fpr, tpr)

print('Auc for Bernoulli Naive Bayes is:',roc_auc_bnb)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# # Decision Tree

# In[ ]:


m4 = DecisionTreeClassifier()
m4.fit(X_train,y_train)
y_pred_dt=m4.predict(X_test)
Train_Score_dt = m4.score(X_train,y_train)
Test_Score_dt = accuracy_score(y_test,y_pred_dt)


# In[ ]:


print('Training Accuracy :',Train_Score_dt)
print('Testing Accuracy :',Test_Score_dt)
print(classification_report(y_test,y_pred_dt))


# #### The training accuracy when compared to testing accuracy of model is more which helps us to understand that model is overfitting

# In[ ]:


fpr,tpr, _ = roc_curve(y_test,y_pred_dt)
roc_auc_dt = auc(fpr, tpr)

print('Auc for Decision Tree is:',roc_auc_dt)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# # Random Forest

# In[ ]:


m5 = RandomForestClassifier()
m5.fit(X_train,y_train)
y_pred_rf=m5.predict(X_test)
Train_Score_rf = m5.score(X_train,y_train)
Test_Score_rf = accuracy_score(y_test,y_pred_rf)


# In[ ]:


print('Training Accuracy :',Train_Score_rf)
print('Testing Accuracy :',Test_Score_rf)
print(classification_report(y_test,y_pred_rf))


# #### The training accuracy when compared to testing accuracy of model is more which helps us to understand that model is overfitting

# In[ ]:


fpr,tpr, _ = roc_curve(y_test,y_pred_rf)
roc_auc_rf = auc(fpr, tpr)

print('Auc for Random Forest is:',roc_auc_rf)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# # Let us tune the hyper parameters of Non-parametric models ( Decision Tree and KNN ) using GridSearch

# In[ ]:


# GridSearchCV to find optimal max_depth
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 3

# parameters to build the model on
parameters = {'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ["entropy", "gini"]}

# instantiate the model
dtree = DecisionTreeClassifier(random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
tree.fit(X_train, y_train)


# In[ ]:


tree.best_params_


# # Building a decision tree model with tuned parameters

# In[ ]:


m6 = DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=100,min_samples_split=50)
m6.fit(X_train,y_train)
y_pred_tdt=m6.predict(X_test)
Train_Score_tdt = m6.score(X_train,y_train)
Test_Score_tdt = accuracy_score(y_test,y_pred_tdt)


# In[ ]:


print('Training Accuracy :',Train_Score_tdt)
print('Testing Accuracy  :',Test_Score_tdt)
print(classification_report(y_test,y_pred_tdt))


# #### The training accuracy and testing accuracy of model is almost similar which helps us to understand that model is neither overfitting nor underfitting

# In[ ]:


fpr,tpr, _ = roc_curve(y_test,y_pred_tdt)
roc_auc_tdt = auc(fpr, tpr)

print('Auc for Tuned Decision Tree is:',roc_auc_tdt)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# In[ ]:


#Gridsearch CV to find Optimal K value for KNN model
grid = {'n_neighbors':np.arange(1,50)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,grid,cv=3)
knn_cv.fit(X_train,y_train)


print("Tuned Hyperparameter k: {}".format(knn_cv.best_params_))


# In[ ]:


m7 = KNeighborsClassifier(n_neighbors=45)
m7.fit(X_train,y_train)
y_pred_tknn=m7.predict(X_test)
Train_Score_tknn = m7.score(X_train,y_train)
Test_Score_tknn = accuracy_score(y_test,y_pred_tknn)


# In[ ]:


print('Training Accuracy :',Train_Score_tknn)
print('Testing Accuracy  :',Test_Score_tknn)
print(classification_report(y_test,y_pred_tknn))


# #### The training accuracy when compared to testing accuracy of model is more which helps us to understand that model is overfitting

# In[ ]:


fpr,tpr, _ = roc_curve(y_test,y_pred_tknn)
roc_auc_tknn = auc(fpr, tpr)

print('Auc for Tuned KNN is:',roc_auc_tknn)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# ### Hyper Tuning the Random Forest Model

# In[ ]:


parameter={'n_estimators':np.arange(1,101)}
gs = GridSearchCV(m5,parameter,cv=3)
gs.fit(X_train,y_train)
gs.best_params_


# In[ ]:


m8 = RandomForestClassifier(n_estimators=71)
m8.fit(X_train,y_train) 
y_pred_trf=m8.predict(X_test)
Train_Score_trf = m8.score(X_train,y_train)
Test_Score_trf = accuracy_score(y_test,y_pred_trf)


# In[ ]:


print('Training Accuracy :',Train_Score_trf)
print('Testing Accuracy  :',Test_Score_trf)
print(classification_report(y_test,y_pred_trf))


# #### The training accuracy when compared to testing accuracy of model is more which helps us to understand that model is overfitting

# In[ ]:


fpr,tpr, _ = roc_curve(y_test,y_pred_trf)
roc_auc_trf = auc(fpr, tpr)

print('Auc for Tuned Random Forest is:',roc_auc_trf)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# # CatBoost

# In[ ]:


data_model.treatments.replace(['insulin','io'],[0,1],inplace = True)


# In[ ]:


data_model.head().T


# In[ ]:


a = data_model.drop(['age','diabetesMed','readmitted','treatments'],axis=1)
b = data_model.treatments


# In[ ]:


a.dtypes


# In[ ]:


cate_features_index = np.where(a.dtypes != int)[0]


# In[ ]:


xtrain,xtest,ytrain,ytest = train_test_split(a,b,train_size=.70,random_state=2)


# In[ ]:


from catboost import CatBoostClassifier, Pool,cv
#let us make the catboost model, use_best_model params will make the model prevent overfitting
model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)


# In[ ]:


model.fit(xtrain,ytrain,cat_features=cate_features_index,eval_set=(xtest,ytest))


# In[ ]:


#show the model test acc, but you have to note that the acc is not the cv acc,
#so recommend to use the cv acc to evaluate your model!
print('the test accuracy is :{:.6f}'.format(accuracy_score(ytest,model.predict(xtest))))
test_score_catboost = accuracy_score(ytest,model.predict(xtest))
print("the train accuracy is :",model.score(xtrain,ytrain))
train_score_catboost = model.score(xtrain,ytrain)


# In[ ]:


fpr,tpr, _ = roc_curve(ytest,model.predict(xtest))
roc_auc_cb = auc(fpr, tpr)

print('Auc for Cat Boost is:',roc_auc_cb)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# In[ ]:


model.predict(xtest)


# In[ ]:


Model_Scores=pd.DataFrame({'Models':['Logistic Regression','KNN','Bernauli Naives Bayes','Decision Tree','Random Forest','Tuned Decison Tree','Tuned KNN','Tuned Random Forest','Cat Boost'],
             'Training Accuracy':[Train_Score_lr,Train_Score_knn,Train_Score_bnb,Train_Score_dt,Train_Score_rf,Train_Score_tdt,Train_Score_tknn,Train_Score_trf,train_score_catboost],
             'Testing Accuracy':[Test_Score_lr,Test_Score_knn,Test_Score_bnb,Test_Score_dt,Test_Score_rf,Test_Score_tdt,Test_Score_tknn,Test_Score_trf,test_score_catboost],
                'AUC':[roc_auc_lr,roc_auc_knn,roc_auc_bnb,roc_auc_dt,roc_auc_rf,roc_auc_tdt,roc_auc_tknn,roc_auc_trf,roc_auc_cb]})

Model_Scores.sort_values(by=('Testing Accuracy'),ascending=False)


# # Logistic Regression gives us the Best Test Accuracy so we choose Logistic Regression as our BenchMark model

# # Applying Boosting Technique on Benchmark Model (Logistic Regression)

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
bslr=AdaBoostClassifier(base_estimator=LogisticRegression())
bslr.fit(X_train,y_train)


# In[ ]:


y_pred_blr=bslr.predict(X_test)
Train_Score_bslr = bslr.score(X_train,y_train)
Test_Score_bslr = accuracy_score(y_test,y_pred_blr)


# In[ ]:


print('Training Accuracy :',Train_Score_bslr)
print('Testing Accuracy  :',Test_Score_bslr)
print(classification_report(y_test,y_pred_blr))


# In[ ]:


fpr,tpr, _ = roc_curve(y_test,y_pred_blr)
roc_auc_bslr = auc(fpr, tpr)

print('Auc for Boosted Logistic Regression is:',roc_auc_bslr)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# In[ ]:


bglr=BaggingClassifier(base_estimator=LogisticRegression())
bglr.fit(X_train,y_train)


# In[ ]:


y_pred_bglr=bglr.predict(X_test)
Train_Score_bglr = bglr.score(X_train,y_train)
Test_Score_bglr = accuracy_score(y_test,y_pred_blr)


# In[ ]:


print('Training Accuracy :',Train_Score_bglr)
print('Testing Accuracy  :',Test_Score_bglr)
print(classification_report(y_test,y_pred_bglr))


# In[ ]:


fpr,tpr, _ = roc_curve(y_test,y_pred_bglr)
roc_auc_bglr = auc(fpr, tpr)

print('Auc for Bagged Logistic Regression is:',roc_auc_bglr)
plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.show()


# # Comparing Benchmark model with Boosted Logistic Regression

# In[ ]:


Model_Scores=pd.DataFrame({'Models':['Logistic Regression','Boosted Logistic Regression','Cat Boost'],
             'Training Accuracy':[Train_Score_lr,Train_Score_bslr,train_score_catboost],
             'Testing Accuracy':[Test_Score_lr,Test_Score_bslr,test_score_catboost],
                    'AUC':[roc_auc_lr,roc_auc_bslr,roc_auc_cb]})


# In[ ]:


Model_Scores.sort_values(by='Testing Accuracy',ascending=False)


# # After applying boosted technique on Logistic Regression still the best accuracy is given by the Benchmark Model

# # Performing Stacking technique on the models

# In[ ]:


from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
stacked = VotingClassifier(estimators=[('Logistic Regression',m1),('KNN',m2),('Naive Bayes',m3),('Decision Tree',m4),
                                      ('RandomForest',m5),('Tuned Decision Tree',m6),('Tuned KNN',m7),
                                       ('Tuned Random Forest',m8),('Boosted Logistic Regression',bslr)],voting='hard')


# In[ ]:


for model, name in zip([m1,m2,m3,m4,m5,m6,m7,m8,bslr,stacked],['Logistic Regression','KNN','Naive Bayes','Decision Tree','RandomForest',
                                                               'Tuned Decision Tree','Tuned KNN','Tuned Random Forest',
                                                               'Boosted Logistic Regression','stacked']):
    scores=cross_val_score(model,X,Y,cv=5,scoring='accuracy')
    print('Accuarcy: %0.02f (+/- %0.4f)(%s)'%(scores.mean(),scores.var(),name))


# # Gradient Boosting Algorithm to check the accuracy

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbdt=GradientBoostingClassifier(n_estimators=150,random_state=2)
gbdt.fit(X_train,y_train)


# In[ ]:


y_pred_gbdt=gbdt.predict(X_test)
Train_Score_gbdt = gbdt.score(X_train,y_train)
Test_Score_gbdt = accuracy_score(y_test,y_pred_gbdt)


# In[ ]:


print('Training Accuracy :',Train_Score_gbdt)
print('Testing Accuracy  :',Test_Score_gbdt)
print(classification_report(y_test,y_pred_gbdt))


# # CV method to check the bias and variance error

# In[ ]:


models=[]
models.append(('Logistic_Regression',m1))
models.append(('KNN',m2))
models.append(('Bernoulli_NB',m3))
models.append(('Decison Tree',m4))
models.append(('Random Forest',m5))
models.append(('Tuned Decision Tree',m6))
models.append(('Tuned KNN',m7))
models.append(('Tuned Random Forest',m8))
models.append(('Bagged Logistic Regression',bglr))
models.append(('Boosted Logistic regression',bslr))


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
results=[]
names=[]
scoring='accuracy'
for name,model in models:
    kfold=KFold(n_splits=5,random_state=2)
    cv_results=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)"%(name,np.mean(cv_results),cv_results.var())
    print(msg)
#boxplot alogorithm comparision
fig=plt.figure(figsize=(16,8))
fig.suptitle('Algorithm Comparision')
ax=fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[ ]:


from sklearn.metrics import log_loss

log_loss(y_test, y_pred_lr, eps=1e-15)


# # Conclusion

# **As per occum's razor rule and also by considering bias-variance trade off, base logistic regression stands out to be the best model with 76 percent accuracy**
# 
