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

import matplotlib.pyplot as plt #Visualization Library
import seaborn as sns #Visualization Library


# Before starting with the project and using some  machine learning stuff to solve problem, it is very important to know about the importance of each variable which represents each characteristic that helps inturns to achieve our objective (or its Garbage In and Garbage Out).
# 
# Our approach for solving the problem will be answering to three questions mainly 1.What is the Objective? 2. Why it is important to achieve objectve? 3. How to achieve Objective?
# 
# - What is the Objective ?
# 
#     The objective of this  is to identify which treatment is effective among Solo Insulin or Insulin Combined with other drugs for the diabetic patients and building a predictive model to predict effective treatment for a diabetic patient.
#     
# - Why it is important to achieve objective?
# 
#     Since,From many years people are suffering from the diabeties but the diabeties is only controlled when right medication is given to the patients identifying the charateristics of a patients and treating accordingly.
#     
# - How to achieve the objective ?
#         For this analysis the data is collected from 130 US hospitals over 10 years(1999-2008).It includes over 50 features representing patient and hospital outcomes.Here one can find the solution to this complex problems by using Machine Learnig.
#         
# 
# Let's start our analysis by understanding each and every input variable ......
# 
# From the dataset the following attributes are used for identifying unique patinets.
# 
# - Patinent_nbr : This attribute is used to identify  unique patientsadmitted into the hospitals.
# - Encounter_id : This attribute gives information of Unique identifier of an patient encounter. 
# 

# # EXPLORATORY DATA ANALYSIS

# In[ ]:


patient_details = pd.read_excel('../input/Paitent_details.xlsx')
patient_details.head()


# 1. > The data above consits of patients details who joins into the hospital in simple terms demographics like race, gender, age and weight of patients of each patient which plays crucial role in prescribing suitable treatment.

# In[ ]:


patient_details.info()


# In[ ]:


print(patient_details.race.value_counts())
print('Missing values in race :',(2273/101766)*100)
patient_details.race.value_counts().plot(kind='bar')


# From the above graph can infer that caucasian(white people) count is more because data is collected from US.

# ### Missing Values Imputation for race attribute[](http://)

# In[ ]:


patient_details.race.replace('?',np.nan,inplace=True) # Replacing all '?' with null values
patient_details.race.fillna(patient_details.race.mode()[0],inplace=True) # Replacing all null values with mode
patient_details.isnull().any() # Checking if any null values are present in the data.
print('Since data is collected from US assuming that 2% missing values are caucasian race people and imputing it with caucasian')
patient_details.race.value_counts()


# In[ ]:


print(patient_details.gender.value_counts())
patient_details.gender.value_counts().plot(kind='bar')


# From the above attribute can observe that patients female encounter count is more than male encounter count.

# In[ ]:


print(patient_details.age.value_counts())
patient_details.age.value_counts().plot(kind='bar')


# From the above graph clearly see that 50-90 age group of people are admitting into the hospital.

# In[ ]:


print(patient_details.weight.value_counts())
print('Missing Values in weight:',(98569/101766)*100)


# Since weight column has almost 97% missing values and weight is given as bins,if the data has missing values greater than 50% generally its better to drop the variable because imputation of values(>50%) may show wrong results.

# ## Diagnosis Session

# In[ ]:


Diagnosis_session = pd.read_excel('../input/Diagnosis_session.xlsx')
Diagnosis_session.head()


# In[ ]:


Diagnosis_session.info()


# - From the above data the attributes diag_1,diag_2,diag_3 are codes of treatments given to a patient for each encounter.
# - Number_diagnosis attributes gives the information about number of diagnosis taken each paient. 

# ## Admission Details of patients

# In[ ]:


admission_details = pd.read_excel('../input/admission_details.xlsx')
admission_details.head()


# In[ ]:


admission_details.info()


# - admission_type_id : Integer identifier corresponding to 9 distinct values, for example, emergency, urgent, elective, newborn, and not available i.e which says how patient admitted into hospital.
# - discharge_disposition_id : Integer identifier corresponding to 29 distinct values, for example, discharged to home, expired, and not available i.e which says that how a patient is discharged from the hospital.
# - admission_source_id : Integer identifier corresponding to 21 distinct values, for example, physician referral, emergency room, and transfer from a hospital i.e from where patient got admitted into the hospital.
# - time_in_hospital : number of days between admission and discharge
# - medical_speciality : identifier of a specialty of the admitting physician, corresponding to 84 distinct values, for example, cardiology, internal medicine, family\general practice, and surgeon

# In[ ]:


print(admission_details.admission_type_id.value_counts())
admission_details.admission_type_id.value_counts().plot(kind='bar')


# - 53990 out of 101766 patients are admitting into the hospitals with admission type as 1 
# - 37,000 out of 101766 patients are admitting into the hospitals with admission type as 2 and 3

# In[ ]:


print(admission_details.discharge_disposition_id.value_counts())


# In[ ]:


print(admission_details.discharge_disposition_id.value_counts().plot(kind='bar'))


# * From the above bar chart it is witnessed that 60% of patients are "discharged to home" which is indicated as disposition id 1.
# - Similarly disposition id 2 indicates discharged/transferred to another short term hospital.
# - Disposition id 3 indicates Discharged/transferred to SNF.

# In[ ]:


print(admission_details.payer_code.value_counts())
print('Missing Values in payer code :',(40256/101766)*100)


# In[ ]:


admission_details.payer_code.value_counts().plot(kind='bar')


# From the above visualization and value counts which is clear that 40% are missing values.

# In[ ]:


print(admission_details.medical_specialty.value_counts())


# In[ ]:


print('Missing Values in payer code :',(49949/101766)*100)


# ## Diabetic Data

# In[ ]:


diabetic_data = pd.read_csv('../input/diabetic_data.csv')
diabetic_data.head()


# In[ ]:


diabetic_data.info()


# - The above diabetic Data says about different types of drugs used to treat the patients which consists of four values No,Steady,Up,Down which indicate dosage of drug given to a patient and No indicate drug is not given to the patient.
# - Two attributes which indicates the tests taken by each patient they are max_glu_selrum test and A1c result.
# - Change : This attribute indicates two values ch,No Ch indicates if there is any change in medication.
# - Diabetes Med: This attribute has two values Yes,No which says that a patient is given any medication for diabeties or not.
# - Readmission : This attribute tell us about wheather a patient is readmitting into the hospital.

# ### Creating new attribute which helps us to identify type of treatment given to each patient

# In[ ]:


treat=diabetic_data.iloc[:,3:26].replace(['Steady','Up','Down','No'],[1,1,1,0])
treat.set_index(diabetic_data.encounter_id,inplace=True)
print(treat.sum(axis=1).value_counts())


# #### From the above snipet we can infer combination of drugs given to the patient as a part of treatment.
# - Intrestingly we can see that only few patients have given more combination of drugs.

# In[ ]:


print('insulin based treatments ',treat[treat['insulin']==1].sum(axis=1).value_counts())

print('insulin is not used for treating diabeties',treat[treat['insulin']==0].sum(axis=1).value_counts())


# In[ ]:


i_p=treat[treat['insulin']==1].sum(axis=1).replace([1,2,3,4,5,6],['insulin','io','io','io','io','io'])
i_a=treat[treat['insulin']==0].sum(axis=1).replace([0,1,2,3,4,5,6],['NoMed','other','other','other','other','other','other'])
treatments=pd.concat([i_p,i_a])
treatments = pd.DataFrame({'treatments':treatments})


# In[ ]:


diabetic_data=diabetic_data.join(treatments,how='inner',on='encounter_id')
diabetic_data.head()


# - According to the given objective there is no varible which tells us about what kind of treatment is given to a patients but we have information of drugs given to a patients by using those 23 drug column created a new column which simply represents all 23 column information in one column named 'Treatments' (Treatment Column represents all 23 drug column information).

# In[ ]:


#Dropping all drug details because all information been represented in one column which results in redundancy
diabetic_data.drop(['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone'],axis=1,inplace=True)


# ## Lab Sessions

# In[ ]:


lab_session = pd.read_excel('../input/Lab-session.xlsx')
lab_session.head()


# - num_lab_procedures : Number of lab tests performed during the encounter
# - Num_procedures  : Number of procedures (other than lab tests) performed during the encounter
# - num_medications : Number of distinct generic names administered during the encounter
# - number_outpatients : Number of outpatient visits of the patient in the year preceding the encounter
# - number_emergency   : Number of emergency visits of the patient in the year preceding the encounter 
# - number_inpatient   : Number of inpatient visits of the patient in the year preceding the encounter

# In[ ]:


lab_session.info()


# # Merging individual DataFrames consists details of patients into one DataFrame

# In[ ]:


data=pd.concat([patient_details,admission_details,Diagnosis_session,lab_session,diabetic_data],axis=1)


# In[ ]:


data.info()


# In[ ]:


# Removing all duplicate columns present in dataset after merging.
data_final=data.T.drop_duplicates().T


# In[ ]:


#From the initial analysis the below three columns have large missing values and diag_1,diag_2,diag_3 are codes which are mostly not useful in our analysis.
data_final.drop(['weight','payer_code','medical_specialty','diag_1','diag_2','diag_3'],axis=1,inplace=True)


# In[ ]:


data_final.head()


# In[ ]:


data_final.info()


# # FINDING SOLO INSULIN OR INSULIN CONJUNCTION WITH OTHER DRUGS ARE MORE EFFECTIVE FOR DIABETIC PATIENTS FROM THE DATA.

# In[ ]:


eff_diab_data=data_final[(data_final['diabetesMed']=='Yes')&(data_final['readmitted']=='NO')&(data_final['treatments'].isin(['insulin','io']))&(~data_final.discharge_disposition_id.isin([11,13,14,19,20]))]
print(eff_diab_data.shape)
eff_diab_data.head()


# #### Before achieving objective arranging data accordingly for simple and easy analysis.
# 
# * From the given data diabetesMed attribute is "YES" when a diabetic medication is given to treatment and similary "NO" means that no diabetic medication is given to a patient.
#     - So, according to the given objective considering records who are taking diabetic medication.
# * According to objective need to suggest effective medication,the word effective means the patient should have some effect after taking particular treatment.
#     - Made an assumption that if patient taking a diabetic medication and not readmitting into the hospital is effective treatment for that patient and considering records with reamitted as "NO".
# * Objective is to find effective treatment among solo insulin or insulin combined with other drugs.
#     - As we have three treatments given to a patient but according to objective only require two treatments.
# * Removing some records with having discharge disposition id as 11,13,14,19,20 because the patients having discharge disposition id are either dead or hospice which is not use in our analysis.

# In[ ]:


eff_diab_data.info()


# In[ ]:


# Total Number of patients who are taking diabetic treatment
data_final[~data_final.discharge_disposition_id.isin([11,13,14,19,20])].treatments.value_counts()


# In[ ]:


# Diabetic Patients taking treatment and not readmitting into hospital which means treatment is effective.
eff_diab_data.treatments.value_counts()


# In[ ]:


print('Readmission Rate of Patients given Solo Insulin ',int(abs(1-(14675/29864))*100),'%')
print('Readmission Rate of patients given Insulin combined with other Drugs',int(abs(1-(12145/23100))*100),'%')


# ### Here effectivity of medication is calculated by 
# 
#                         (Patinets taking "X" treatment and not readmitting into the hospital)/(Total No.of patients treated with "X" treatment)
#                         
# - An intresting observation from the above snippet is despite of giving Solo Insulin to more patients but the patients readmission rate is high for patients taking Solo Insulin Treatment.
# - Important inference from the data is Insulin combined with other drugs is more effective for patients.

# ## PREPROCESSING DATA FOR BUILING A MODEL WHICH CAN PREDICT AN EFFECTIVE TREATMENT FOR A DIABETIC PATIENT
# 
# - Using eff_diab_data for modelling because as disscussed earlier assumption of effective treatment is satisfied when a patient taking diabetic medication and not readmitting into the hospital.
#    - Note: This data is after removing the patients who are dead and hospice.

# In[ ]:


eff_diab_data.head()


# In[ ]:


eff_diab_data.info()


# In[ ]:


# First step is converting age column to numeric values using label encoding since data is ordinal and values should maintain order.
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
eff_diab_data['age'] = le.fit_transform(eff_diab_data['age'])


# In[ ]:


# Coverted age categorical data into numerical values
eff_diab_data.age.value_counts().plot(kind='bar')


# In[ ]:


#Second Step is converting treatment column with numerical values
eff_diab_data.treatments.replace(['insulin','io'],[0,1],inplace=True)


# In[ ]:


# Conveted treatment column with numerical values 0:Insulin,1: Insulin combination with other drugs
eff_diab_data.treatments.value_counts().plot(kind='bar')


# In[ ]:


eff_diab_data.head()


# In[ ]:


eff_diab_data.info()


# In[ ]:


# Converting numeric colums to interger data type since data is in object data type
eff_diab_data[['encounter_id', 'patient_nbr','admission_type_id', 'discharge_disposition_id', 'admission_source_id','time_in_hospital', 'number_diagnoses', 'num_lab_procedures','num_procedures','num_medications', 'number_outpatient','number_emergency', 'number_inpatient']]=eff_diab_data[['encounter_id', 'patient_nbr','admission_type_id', 'discharge_disposition_id', 'admission_source_id','time_in_hospital', 'number_diagnoses', 'num_lab_procedures','num_procedures','num_medications', 'number_outpatient','number_emergency', 'number_inpatient']].astype('int64')


# In[ ]:


# Creating new columns using get dummies for nominal data which helps in intrepretability of the model.
data_model=pd.get_dummies(eff_diab_data)
data_model.head()


# Since don't have any domain expertise identifying the potential input variables is a challenging part.
# - Since the data is mostly of input of categorical data and output column is categorical it is suggested to perform chisquare test to check the statistical importance of each variable to choose potential input variables. 
# 
# Let's Perform CHI-SQUARE TEST OF INDEPENDENCE

# ## CHI-SQUARE TEST OF INDEPENDENCE

# In[ ]:


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


#Introducing some random numbers and checking weather the test is performing correctly or not on given data
data_model['dummyCat'] = np.random.choice([0, 1], size=(len(data_model),), p=[0.5, 0.5])
data_model.dummyCat.value_counts()


# In[ ]:


#Initialize ChiSquare Class
cT = ChiSquare(data_model)

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
       'readmitted_NO','dummyCat']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="treatments" )


# It is evident that the randomly introduced dummy cat is not an imporatant input variable and in chisquare test it results to discard the varible from  model.
# - Since there is no Domain Expertise taking all input variables as per statistical importance

# # Model Building for predicting effective diabetic treatment based on characteristics of patient

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import SVC


# In[ ]:


X = data_model.drop(['encounter_id','patient_nbr','age','num_lab_procedures','number_outpatient','number_emergency',
                      'race_Asian','race_Other','diabetesMed_Yes','max_glu_serum_>200','A1Cresult_>8','A1Cresult_Norm',
                      'readmitted_NO','dummyCat','treatments','dummyCat'],axis=1)
y=data_model['treatments']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ## BaseLine Accuracy

# In[ ]:


y_p=[]
for i in range(y_test.shape[0]):
    y_p.append(y_test.mode()[0])#Highest class is assigned to a list which is compared with ytest
y_pred=pd.Series(y_p)
print('BaseLine Accuracy :',accuracy_score(y_test,y_pred))


# Baseline accuracy given by the data is 54.3% and if model is build then the model should have high accuracy than baseline accuracy 

# ## Model Building using Logistic Regression

# In[ ]:


model_lr = LogisticRegression(solver='liblinear')
model_lr.fit(X_train,y_train)
y_pred_lr = model_lr.predict(X_test)


# In[ ]:


traning_acc_lr = model_lr.score(X_train,y_train)
testing_acc_lr = accuracy_score(y_test,y_pred_lr)
print('............. LOGISTIC REGRESSION METRICS ...............')
print('Training Accuracy :',traning_acc_lr)
print('Testing Accuracy  :',testing_acc_lr)
print(confusion_matrix(y_test,y_pred_lr))
print(classification_report(y_test,y_pred_lr))


# ## K- NEAREST NEIGHBOURS

# In[ ]:


model_knn = KNeighborsClassifier()
model_knn.fit(X_train,y_train)
y_pred_knn = model_knn.predict(X_test)


# In[ ]:


traning_acc_knn = model_knn.score(X_train,y_train)
testing_acc_knn = accuracy_score(y_test,y_pred_knn)
print('............. K-Nearest Neighbours METRICS ...............')
print('Training Accuracy :',traning_acc_knn)
print('Testing Accuracy  :',testing_acc_knn)
print(confusion_matrix(y_test,y_pred_knn))
print(classification_report(y_test,y_pred_knn))


# > ## Bernoulli Naive Bayes

# In[ ]:


model_bnb = BernoulliNB()
model_bnb.fit(X_train,y_train)
y_pred_bnb = model_bnb.predict(X_test)


# In[ ]:


traning_acc_bnb = model_bnb.score(X_train,y_train)
testing_acc_bnb = accuracy_score(y_test,y_pred_bnb)
print('Training Accuracy :',traning_acc_bnb)
print('Testing Accuracy  :',testing_acc_bnb)
print(confusion_matrix(y_test,y_pred_bnb))
print(classification_report(y_test,y_pred_bnb))


# ## Decision Tree 

# In[ ]:


model_dt = DecisionTreeClassifier()
model_dt.fit(X_train,y_train)
y_pred_dt = model_dt.predict(X_test)


# In[ ]:


traning_acc_dt = model_dt.score(X_train,y_train)
testing_acc_dt = accuracy_score(y_test,y_pred_dt)
print('............. Decision Tree METRICS ...............')
print('Training Accuracy :',traning_acc_dt)
print('Testing Accuracy  :',testing_acc_dt)
print(confusion_matrix(y_test,y_pred_dt))
print(classification_report(y_test,y_pred_dt))


# ## Random Forest

# In[ ]:


model_rf = RandomForestClassifier()
model_rf.fit(X_train,y_train)
y_pred_rf = model_rf.predict(X_test)


# In[ ]:



traning_acc_rf = model_rf.score(X_train,y_train)
testing_acc_rf = accuracy_score(y_test,y_pred_rf)
print('............. Random Forest METRICS ...............')
print('Training Accuracy :',traning_acc_rf)
print('Testing Accuracy  :',testing_acc_rf)
print(confusion_matrix(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))


# # Hyperparameter Tuning 

# ### Tunned KNN

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# In[ ]:


#Gridsearch CV to find Optimal K value for KNN model
grid = {'n_neighbors':np.arange(1,50)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,grid,cv=3)
knn_cv.fit(X_train,y_train)
print("Tuned Hyperparameter k: {}".format(knn_cv.best_params_))


# In[ ]:


model_tknn = KNeighborsClassifier(n_neighbors=39)
model_tknn.fit(X_train,y_train)
y_pred_tknn=model_tknn.predict(X_test)


# In[ ]:


traning_acc_tknn = model_tknn.score(X_train,y_train)
testing_acc_tknn = accuracy_score(y_test,y_pred_tknn)
print('............. Tunned K Nearest Neighbours METRICS ...............')
print('Training Accuracy :',traning_acc_tknn)
print('Testing Accuracy  :',testing_acc_tknn)
print(confusion_matrix(y_test,y_pred_tknn))
print(classification_report(y_test,y_pred_tknn))


# ### Tunned Decision Tree

# In[ ]:


# GridSearchCV to find optimal max_depth
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
tree.best_params_


# In[ ]:


model_tdt = DecisionTreeClassifier(criterion='entropy',max_depth=5,min_samples_leaf=50,min_samples_split=50)
model_tdt.fit(X_train,y_train)
y_pred_tdt=model_tdt.predict(X_test)


# In[ ]:


training_acc_tdt = model_tdt.score(X_train,y_train)
testing_acc_tdt = accuracy_score(y_test,y_pred_tdt)
print('............. Tunned Decision Tree METRICS ...............')
print('Training Accuracy :',training_acc_tdt)
print('Testing Accuracy  :',testing_acc_tdt)
print(confusion_matrix(y_test,y_pred_tdt))
print(classification_report(y_test,y_pred_tdt))


# ## Tunned Random Forest

# In[ ]:


rfc=RandomForestClassifier(random_state=42)
parameter={'n_estimators':np.arange(1,101)}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=parameter, cv= 3)
CV_rfc.fit(X_train, y_train)


# In[ ]:


CV_rfc.best_params_


# In[ ]:


model_trf = RandomForestClassifier(n_estimators=59)
model_trf.fit(X_train,y_train) 
y_pred_trf = model_trf.predict(X_test)


# In[ ]:


training_acc_trf = model_trf.score(X_train,y_train)
testing_acc_trf = accuracy_score(y_test,y_pred_trf)
print('............. Tunned Random Forest METRICS ...............')
print('Training Accuracy :',training_acc_trf)
print('Testing Accuracy  :',testing_acc_trf)
print(confusion_matrix(y_test,y_pred_trf))
print(classification_report(y_test,y_pred_trf))


# ## SVM Classifier

# In[ ]:


model_svc = SVC(kernel='linear')
model_svc.fit(X_train,y_train)
y_pred_svc = model_svc.predict(X_test)


# In[ ]:


training_acc_svc = model_svc.score(X_train,y_train)
testing_acc_svc = accuracy_score(y_test,y_pred_svc)
print('............. Support Vector Classifier METRICS ...............')
print('Training Accuracy :',training_acc_svc)
print('Testing Accuracy  :',testing_acc_svc)
print(confusion_matrix(y_test,y_pred_svc))
print(classification_report(y_test,y_pred_svc))


# ## Models Comparision

# In[ ]:


models_com = pd.DataFrame({'Model':['Logistic Regression','K-Nearest Neighbours','Bernoulli Naive Bayes','Decision Tree','Random Forest','Tunned K-Nearest Neighbours','Tunned Decision Tree','Tunned Random Forest','SVM'],
                           'Training Accuracy':[traning_acc_lr,traning_acc_knn,traning_acc_bnb,traning_acc_dt,traning_acc_rf,traning_acc_tknn,training_acc_tdt,training_acc_trf,training_acc_svc],
                           'Testing Accuracy':[testing_acc_lr,testing_acc_knn,testing_acc_bnb,testing_acc_dt,testing_acc_rf,testing_acc_tknn,testing_acc_tdt,testing_acc_trf,testing_acc_svc]})
models_com.sort_values(by='Testing Accuracy',ascending=False)


# In[ ]:


plt.figure(figsize=[25,8])
plt.plot(models_com.Model, models_com['Testing Accuracy'], label='Testing Accuracy')
plt.plot(models_com.Model, models_com['Training Accuracy'], label='Training Accuracy')
plt.legend()
plt.title('Model Comparision',fontsize=20)
plt.xlabel('Models',fontsize=30)
plt.ylabel('Accuracy',fontsize=30)
plt.xticks(models_com.Model)
plt.grid()
plt.show()


# From the above Models comparision Graph Comparing Training and Testing Accuracy of differnt models and found that Logistic Regression is performing better and is generalised model since the training and testing accuracy are almost similar.
# 
# - From  Occam's razor rule it's the idea that the simplest and most direct solution should be preferred,So in this case simple model and given best accuracy is given by Logistic Regression.
# 
# - So,Logistic Regression will be our final model for predicting effective treatments of diabetic patients with 75% accuracy  and giving least number of False Negative values compared to other models.
# 
# - From this 75% Accuracy we can infer that there are 75% for a patient who are taking predicted treatment will not readmit into the hospital and thus if patient taking a treatment and not admitted into the hopital then it an effective treatment.

# In[ ]:




