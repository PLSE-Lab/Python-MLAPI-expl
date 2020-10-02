#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings # Ignoreing all the warning
warnings.filterwarnings("ignore")


# In[2]:


# Loading all the library
import os
import numpy
import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as pyplot
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the data 

# In[3]:


file_list = os.listdir('../input')
file_list[0]
Data = pd.read_csv(r'../input/'+file_list[0])
Data.columns # # column Name of Data


# In[4]:


Data.head(10)


# # Count of Target variable

# In[5]:


Data['target'].value_counts() # Get the value_counts for Heart Diseases class and not Heart Diseases 


#  ##### There is no imbalance in Data set so no need of downsampling

# In[6]:


class_dist_df = pd.DataFrame(Data['target'].value_counts()).reset_index(drop=True)# Reseting the index and get the counts for each class 
class_dist_df['class']= ['HD','WHD'] # HD Means Heart diseases and WHD means NO diseases
sns.barplot(y = 'target', x = class_dist_df['class'], data=class_dist_df) # Plotting Class frequency vs Class
pyplot.title('Class Frequency Vs Class Name')
pyplot.show()


# # Data anlaysis and Feature Selection
# 
# 

# #### AGE ANALYSIS

# In[7]:


DF_HD = Data[Data['target']==1] # Dataframe of Heart diseases
DF_WHD = Data[Data['target']==0] # Dataframe of Without Heart diseases


# In[8]:


DF_HD.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
DF_WHD.columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']


# In[9]:


DF_WHD['age'].describe() #This will output the basic statsitics of age in population


# In[10]:


# Hist of Age
DF =DF_HD.append(DF_WHD) 
pd.crosstab(DF['age'],DF['target']).plot(kind="bar",figsize=(20,6))
pyplot.title('Heart Disease Frequency for Ages')
pyplot.xlabel('Age')
pyplot.ylabel('Frequency')
pyplot.savefig('heartDiseaseAndAges.png')
pyplot.show()


# #### After analysis histrogram with age we are not able say anything about Heart diseases

# #### Gender analysis 
# 

# In[77]:


print ('male count with the heart dieases ='), DF_HD[DF_HD['sex']==1]['sex'].value_counts() # 1 means Male in Sex Column


# In[78]:


print ('male count without the heart dieases ='), DF_WHD[DF_WHD['sex']==1]['sex'].count() # 0 means Female in Sex column


# In[79]:


print ('Female count with the heart dieases ='), len(DF_HD[DF_HD['sex']==0]['sex'])


# In[14]:


print ('Female count without the heart dieases ='), len(DF_WHD[DF_WHD['sex']==0]['sex'])


# In[15]:


Gender_HD = pd.DataFrame(DF_HD['sex'].value_counts()).reset_index() # Value counts for sex with Heart diseases
Gender_WHD = pd.DataFrame(DF_WHD['sex'].value_counts()).reset_index() # Value counts for sex with withot Heart diseases
Gender_HD['class'] = "HD"
Gender_WHD['class'] = "WHD"
Gender_HD['index'] =['Male','Female']
Gender_WHD['index'] =['Male','Female']
Gender_DF = Gender_HD.append(Gender_WHD)
Gender_DF.columns = ['Category','Gender_count','class']
sns.barplot(y='Gender_count', x='Category', data=Gender_DF, hue='class')
pyplot.title('Gender Frequency vs Gender_Name')
pyplot.show()


# #### Female population are more tend to Heart dieases in the population 
# 

# #### combine 2 Features gender and male

# In[16]:


print ('male age with heart dieases ='), DF_HD[DF_HD['sex']==1]['age'].describe()
print ('male age without heart dieases ='), DF_WHD[DF_WHD['sex']==1]['age'].describe()
print ('Female age without heart dieases ='), DF_WHD[DF_WHD['sex']==0]['age'].describe()
print ('Female age without heart dieases ='), DF_HD[DF_HD['sex']==0]['age'].describe()


# #### Chest Pain Type Analysis

# In[17]:


DF_HD['cp'].value_counts() # Get value counts for each class 


# In[18]:


DF_WHD['cp'].value_counts() # Get value counts for each class 


# In[19]:


cp_dist_df = pd.DataFrame(DF_HD['cp'].value_counts()).reset_index() # Value counts for CP with Heart diseases
cp_dist_Wdf = pd.DataFrame(DF_WHD['cp'].value_counts()).reset_index()# Value counts for CP without Heart diseases
cp_dist_df['class'] = "HD"
cp_dist_Wdf['class'] = "WHD"
cp_dist_df_copy = cp_dist_df.copy()
cp_dist_df_copy = cp_dist_df_copy.append(cp_dist_Wdf)
cp_dist_df_copy.columns = [u'index', u'cp_frequency', u'class']
sns.barplot(y='cp_frequency', x='index', data=cp_dist_df_copy, hue='class')
pyplot.show()


# #### 0 means less possibilty of heart attack
# #### 1 means more possibity of heart attack
# #### 2 means more possibity of heart attack
# #### 3 means more or less possibity of heart attack confused..

# #### The resting blood Pressure 

# In[20]:


DF_HD['trestbps'].describe() # Basic statsitics for trestbps for Heart Diseases cases


# In[21]:


DF_WHD['trestbps'].describe() # Basic statsitics for trestbps for without Heart Diseases cases


# In[22]:


pyplot.plot(DF[DF['target']==0]['trestbps'].values,'ro',label='WHD') # Heart Diseases cases vs trestbps
pyplot.plot(DF[DF['target']!=0]['trestbps'].values,'bo',label='HD') # Without Heart Diseases cases vs trestbps
pyplot.xlabel('Index')
pyplot.ylabel('trestbps')
pyplot.title('Heart_disease VS trestbps')
pyplot.legend()
pyplot.show()


# In[23]:


# Plot between age vs trestbps
pyplot.plot(DF_WHD['age'],DF_WHD['trestbps'],'bo')
pyplot.plot(DF_HD['age'],DF_HD['trestbps'],'ro')
pyplot.title('age VS trestbps')
pyplot.ylabel('age')
pyplot.xlabel('trestbps')
pyplot.show()


# #### No descision Boundary to classify between of heart disease and Not a heart disease with help of  trestbps
# 

# #### chol analysis(serum cholestoral in mg/dl)

# In[24]:


DF_HD['chol'].describe()# Basic statsitics for chol for Heart Diseases cases


# In[25]:


DF_WHD['chol'].describe() # Basic statsitics for chol for without Heart Diseases cases


# In[26]:


# Plot of target variable with chol level
pyplot.plot(DF[DF['target']==0]['chol'].values,'ro',label='WHD')
pyplot.plot(DF[DF['target']!=0]['chol'].values,'bo',label='HD')
pyplot.xlabel('Dummy_Index')
pyplot.ylabel('Chol_values')
pyplot.title('Heart_dieases VS Chol')
pyplot.legend()
pyplot.show()


# In[27]:


# Plot of age value with chol level
pyplot.plot(DF_WHD['age'],DF_WHD['chol'],'bo',label='WHD')
pyplot.plot(DF_HD['age'],DF_HD['chol'],'ro',label='HD')
pyplot.title('age VS chol')
pyplot.ylabel('age')
pyplot.xlabel('chol')
pyplot.legend()
pyplot.show()


# #### No descision Boundary to classify between of heart disease and Not a heart disease with help of  chol

# #### FBS Analysis(fasting blood sugar)

# In[28]:


DF_HD['fbs'].value_counts() # Value counts for categorical data (FBS) for heart disease


# In[29]:


DF_WHD['fbs'].value_counts() # Value counts for categorical data (FBS) for without heart disease


# In[30]:


fbs_dist_df = pd.DataFrame(DF_HD['fbs'].value_counts()).reset_index()
fbs_dist_Wdf = pd.DataFrame(DF_WHD['fbs'].value_counts()).reset_index()
fbs_dist_df['class'] = "HD"
fbs_dist_Wdf['class'] = "WHD"
fbs_dist_df_copy = fbs_dist_df.copy()
fbs_dist_df_copy = fbs_dist_df_copy.append(fbs_dist_Wdf)
fbs_dist_df_copy.columns = [u'index', u'fbs_frequency', u'class']
sns.barplot(y='fbs_frequency', x='index', data=fbs_dist_df_copy, hue='class')
pyplot.title('fbs_frequency vs class ')
pyplot.show()


# #### Not providing much information about classification of Heart disease and Non Heart disease

# #### Restecg analysis(resting electrocardiographic results)

# In[31]:


DF_HD['restecg'].value_counts()  # Value counts for categorical data (restecg) for heart disease


# In[32]:


DF_WHD['restecg'].value_counts() # Value counts for categorical data (restecg) for without heart disease


# In[33]:


restecg_dist_df = pd.DataFrame(DF_HD['restecg'].value_counts()).reset_index() # value counts for restecg for Heart diseases Cases
restecg_dist_Wdf = pd.DataFrame(DF_WHD['restecg'].value_counts()).reset_index()# value counts for restecg for without Heart diseases Cases
restecg_dist_df['class'] = "HD"
restecg_dist_Wdf['class'] = "WHD"
restecg_dist_df_copy = restecg_dist_df.copy()
restecg_dist_df_copy = restecg_dist_df_copy.append(restecg_dist_Wdf)
restecg_dist_df_copy.columns = [u'index', u'restecg_frequency', u'class']
sns.barplot(y='restecg_frequency', x='index', data=restecg_dist_df_copy, hue='class')
pyplot.title('restecg_frequency vs index')
pyplot.show()


# #### 1.Restecg with type 1 have more probablity of Heart dieases 
# #### 2.Restecg with type 0 have less probablity of Heart dieases 
# #### 3.Restecg with type 2 have less probablity of Heart dieases 

# ####  thalach(maximum heart rate achieved)

# In[34]:


DF_HD['thalach'].describe() # Basic statstics for thalach for Heart diseases 


# In[35]:


DF_WHD['thalach'].describe() # Basic statstics for thalach for without Heart diseases 


# In[36]:


# Plot between target vs thalach
pyplot.plot(DF[DF['target']==0]['thalach'].values,'ro',label='WHD')
pyplot.plot(DF[DF['target']!=0]['thalach'].values,'bo',label='HD')
pyplot.xlabel('Dummy_Index')
pyplot.ylabel('thalach')
pyplot.title('thalach vs Heart_dieases')
pyplot.legend()
pyplot.show()


# #### exang(exercise induced angina)

# In[37]:


DF_HD['exang'].value_counts()  # Value counts for categorical data (exang) for heart disease


# In[38]:


DF_WHD['exang'].value_counts() # Value counts for categorical data (exang) for without heart disease


# In[39]:


exang_dist_df = pd.DataFrame(DF_HD['exang'].value_counts()).reset_index()
exang_dist_Wdf = pd.DataFrame(DF_WHD['exang'].value_counts()).reset_index()
exang_dist_df['class'] = "HD"
exang_dist_Wdf['class'] = "WHD"
exang_dist_df_copy = exang_dist_df.copy()
exang_dist_df_copy = exang_dist_df_copy.append(exang_dist_Wdf)
exang_dist_df_copy.columns = ['index','exang_frequency','class']
sns.barplot(y='exang_frequency', x='index', data=exang_dist_df_copy, hue='class')
pyplot.title('exang_frequency vs index')
pyplot.show()


# #### More probablity of heart disease for Type 0 exang
# #### More probablity of heart disease for Type 1 exang
# 

# #### oldpeak(ST depression induced by exercise relative to rest)

# In[40]:


DF_HD[u'oldpeak'].describe() # Basic statstics for oldpeak for Heart diseases 


# In[41]:


DF_WHD[u'oldpeak'].describe() # Basic statstics for oldpeak for without Heart diseases 


# In[42]:


pyplot.plot(DF[DF['target']==0]['oldpeak'].values,'ro',label='WHD')
pyplot.plot(DF[DF['target']!=0]['oldpeak'].values,'bo',label='HD')
pyplot.xlabel('Dummy_Index')
pyplot.ylabel('oldpeak')
pyplot.title('oldpeak vs Heart_dieases')
pyplot.legend()
pyplot.show()


# #### slope(the slope of the peak exercise ST segment)

# In[43]:


DF_WHD[u'slope'].value_counts() # Value counts for categorical data (slope) for heart disease


# In[44]:


DF_HD[u'slope'].value_counts()  # Value counts for categorical data (slope) for without heart disease


# In[45]:


slope_dist_df = pd.DataFrame(DF_HD['slope'].value_counts()).reset_index()
slope_dist_Wdf = pd.DataFrame(DF_WHD['slope'].value_counts()).reset_index()
slope_dist_df['class'] = "HD"
slope_dist_Wdf['class'] = "WHD"
slope_dist_df_copy = slope_dist_df.copy()
slope_dist_df_copy = slope_dist_df_copy.append(slope_dist_Wdf)
slope_dist_df_copy.columns = ['index','slope_frequency','class']
sns.barplot(y='slope_frequency', x='index', data=slope_dist_df_copy, hue='class')
pyplot.show()


# #### Slope 0 means Very less difference
# #### Slope 1 Means less probability  for Heart Disease
# #### Slope 2 Means high probability  for Heart Disease

# #### Ca(number of major vessels (0-3) colored by flourosopy)

# In[46]:


DF_HD['ca'].value_counts() # Value counts for categorical data (ca) for heart disease


# In[47]:


DF_WHD['ca'].value_counts() # Value counts for categorical data (ca) for without heart disease


# In[48]:


ca_dist_df = pd.DataFrame(DF_HD['ca'].value_counts()).reset_index()
ca_dist_Wdf = pd.DataFrame(DF_WHD['ca'].value_counts()).reset_index()
ca_dist_df['class'] = "HD"
ca_dist_Wdf['class'] = "WHD"
ca_dist_df_copy = ca_dist_df.copy()
ca_dist_df_copy = ca_dist_df_copy.append(ca_dist_Wdf)
ca_dist_df_copy.columns =['index','ca_frequency','class']
sns.barplot(y='ca_frequency', x='index', data=ca_dist_df_copy, hue='class')
pyplot.title('ca_frequency vs index')
pyplot.show()


# #### Ca with value 0 means high probability of Heart disease
# #### Ca with value 1 means low probability of Heart disease
# #### Ca with value 2 means low probability of Heart disease
# #### Ca with value 3 means low probability of Heart disease
# #### Ca with value 4 means very less difference in  positive and Negative class

# #### Thal(reversable defect)

# In[49]:


DF_HD['thal'].value_counts()# Value counts for categorical data (thal) for heart disease


# In[50]:


DF_WHD['thal'].value_counts()# Value counts for categorical data (thal) for without heart disease


# In[51]:


thal_dist_df = pd.DataFrame(DF_HD['thal'].value_counts()).reset_index()
thal_dist_Wdf = pd.DataFrame(DF_WHD['thal'].value_counts()).reset_index()
thal_dist_df['class'] = "HD"
thal_dist_Wdf['class'] = "WHD"
thal_dist_df_copy = thal_dist_df.copy()
thal_dist_df_copy = thal_dist_df_copy.append(thal_dist_Wdf)
thal_dist_df_copy.columns = ['index','thal_frequency','class']
sns.barplot(y='thal_frequency', x='index', data=thal_dist_df_copy, hue='class')
pyplot.show()


# #### thal with value 2 have more probabilty of Heart disease
# #### thal with value 3 have less probabilty of Heart disease
# #### thal with value 1 very less difference in positive and Negative class
# #### thal with value 0 no difference positive and Negative class

# # Forming Feature Dataframe

# In[52]:


Features = ['thal','ca','slope','exang','restecg','trestbps','cp','sex'] # we will choose only these based on above analysis


# In[53]:


Data[Features].head(5) # Print the prepared Data


# In[54]:


# This Function will convert categorical data to numerical data and put the data of different category into different column
def one_hot_coding(Data,column_name):
    thal_value = Data[column_name].tolist()
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(thal_value)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    new_df = pd.DataFrame(onehot_encoded)
    new_df = rename_column(new_df,column_name)
    return new_df


# In[55]:


# This function rename the columns created by above column 
def rename_column(new_df,column_name):
    col_list = []
    for i in range(0,len(new_df.columns)):
        col_list.append(column_name+'_'+str(i))
    new_df.columns = col_list
    return new_df
        


# In[56]:


# converting the categorical data to numerical data
new_df_thal = one_hot_coding(Data,'thal')
new_df_ca = one_hot_coding(Data,'ca')
new_df_slope = one_hot_coding(Data,'slope')
new_df_exang = one_hot_coding(Data,'exang')
new_df_restecg = one_hot_coding(Data,'restecg')
new_df_cp = one_hot_coding(Data,'cp')
new_df_sex = one_hot_coding(Data,'sex')
new_df_thalach = Data['thalach']
new_df_oldpeak = Data['oldpeak']


# In[57]:


# Merging all the feature Dataframe into single Dataframe
Merged_df = pd.concat([new_df_thal, new_df_ca,new_df_slope,new_df_exang,new_df_restecg,new_df_cp,new_df_sex,new_df_thalach,new_df_oldpeak], axis=1)


# In[58]:


# Normalizing the numerical data and bring them in range 0 to 1
Merged_df['thalach'] = (Merged_df['thalach'] - np.min(Merged_df['thalach'])) / (np.max(Merged_df['thalach']) - np.min(Merged_df['thalach']))
Merged_df['oldpeak'] = (Merged_df['oldpeak'] - np.min(Merged_df['oldpeak'])) / (np.max(Merged_df['oldpeak']) - np.min(Merged_df['oldpeak']))


# In[59]:


(Merged_df.columns)


# In[60]:


# Divide the data into input and Output data 
Merged_df['Output_variable'] = Data['target']
Input_DF = Merged_df.drop(['Output_variable'],axis =1)


# In[61]:


# Divide the data into train and test data sets 
X_train, X_test, y_train, y_test = train_test_split(Input_DF, Merged_df['Output_variable'], test_size=0.20, random_state=42)


# In[62]:


len(X_train)


# In[63]:


len(X_test)


# # model selection

# In[64]:


# Intialization of classifier 
classifiers =[]
model1 = LogisticRegression()
classifiers.append(model1)
model2 = SVC()
classifiers.append(model2)
model3 = DecisionTreeClassifier()
classifiers.append(model3)
model4 = RandomForestClassifier()
classifiers.append(model4)
model5 = AdaBoostClassifier()
classifiers.append(model5)


# In[65]:


# List of models 
model_name = ['LogisticRegression','Support Vector Machine','DecisionTreeClassifier','RandomForestClassifier','AdaBoostClassifier']
Training_score ,Testing_score,TP,FP,FN,Precision,Recall,classifiers_list = [],[],[],[],[],[],[],[]


# In[66]:


# Running for differnent classifier and Save scores for different classfiers into model
for i in range(0,len(classifiers)):
    clf = classifiers[i]
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    classifiers_list.append(model_name[i])
    Training_score.append(clf.score(X_train,y_train))
    Testing_score.append(clf.score(X_test,y_test))
    TP.append(cm[1][1])
    FP.append(cm[0][1])
    FN.append(cm[1][0])
    Precision.append( precision_score(y_test,y_pred))
    Recall.append(recall_score(y_test,y_pred))
    
Score_DF = pd.DataFrame()
Score_DF['classifiers'] = classifiers_list
Score_DF['Training_score'] = Training_score
Score_DF['Testing_score'] = Testing_score
Score_DF['True_positive'] = TP
Score_DF['False_positive'] = FP
Score_DF['False_negative'] = FN
Score_DF['Precision'] = Precision
Score_DF['Recall'] = Recall
Score_DF


# # Hyperparameterzation

# # For Different Value of C and L2 Penalty

# In[67]:


# Since from above LogisticRegression was performing best between among the model and we will try with logistic model with different value of C ()
c =[0.0001,0.001,0.01,0.1,1,10,20,30,40,50] # c is inverse of Regularization Coefficient
Training_score ,Testing_score,TP,FP,FN,Precision,Recall,classifiers_list = [],[],[],[],[],[],[],[]
for i in range(0,len(c)):
    clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=c[i], fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='warn', max_iter=100, multi_class='warn', verbose=0, warm_start=False, n_jobs=None)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    Training_score.append(clf.score(X_train,y_train))
    Testing_score.append(clf.score(X_test,y_test))
    TP.append(cm[1][1])
    FP.append(cm[0][1])
    FN.append(cm[1][0])
    Precision.append( precision_score(y_test,y_pred))
    Recall.append(recall_score(y_test,y_pred))

Score_DF = pd.DataFrame()
Score_DF['C value'] = c
Score_DF['Training_score'] = Training_score
Score_DF['Testing_score'] = Testing_score
Score_DF['True_positive'] = TP
Score_DF['False_positive'] = FP
Score_DF['False_negative'] = FN
Score_DF['Precision'] = Precision
Score_DF['Recall'] = Recall
Score_DF


# In[68]:


# plot accuracy vs Regularization Coefficient
pyplot.plot(c,Score_DF['Testing_score'],'r-',label='Testing_Accuracy')
pyplot.plot(c,Score_DF['Training_score'],'b-',label='Trainig_Accuracy')
pyplot.xlabel('Regularization Coefficient')
pyplot.ylabel('Accuracy')
pyplot.legend()
axes = pyplot.gca()
axes.set_ylim([0.70,1])
pyplot.legend()
pyplot.show()


# In[69]:


# plot scores(Precision,Recall) vs Regularization Coefficient
pyplot.plot(c,Score_DF['Precision'],'g-',label='Precision')
pyplot.plot(c,Score_DF['Recall'],'y-',label='Recall')
pyplot.xlabel('Regularization Coefficient')
pyplot.ylabel('scores')
pyplot.legend()
pyplot.show()


# # For Different Value of C and L1 Penalty

# In[70]:


# we will try with L1 Penalty and different value of Regularization Coefficient
c =[0.1,1,10,20,30,40,50]
Training_score ,Testing_score,TP,FP,FN,Precision,Recall,classifiers_list = [],[],[],[],[],[],[],[]
for i in range(0,len(c)):
    clf = LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=c[i], fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='warn', max_iter=100, multi_class='warn', verbose=0, warm_start=False, n_jobs=None)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    Training_score.append(clf.score(X_train,y_train))
    Testing_score.append(clf.score(X_test,y_test))
    TP.append(cm[1][1])
    FP.append(cm[0][1])
    FN.append(cm[1][0])
    Precision.append( precision_score(y_test,y_pred))
    Recall.append(recall_score(y_test,y_pred))

Score_DF = pd.DataFrame()
Score_DF['C value'] = c
Score_DF['Training_score'] = Training_score
Score_DF['Testing_score'] = Testing_score
Score_DF['True_positive'] = TP
Score_DF['False_positive'] = FP
Score_DF['False_negative'] = FN
Score_DF['Precision'] = Precision
Score_DF['Recall'] = Recall
Score_DF


# In[71]:


# plot accuracy vs Regularization Coefficient
pyplot.plot(c,Score_DF['Testing_score'],'r-',label='Testing_Accuracy')
pyplot.plot(c,Score_DF['Training_score'],'b-',label='Trainig_Accuracy')
pyplot.xlabel('Regularization Coefficient')
pyplot.ylabel('Accuracy')
pyplot.legend()
axes = pyplot.gca()
axes.set_ylim([0.70,1])
pyplot.legend()
pyplot.show()


# In[72]:


# plot scores(Precision,Recall) vs Regularization Coefficient
pyplot.plot(c,Score_DF['Precision'],'g-',label='Precision')
pyplot.plot(c,Score_DF['Recall'],'y-',label='Recall')
pyplot.xlabel('Regularization Coefficient')
pyplot.ylabel('scores')
pyplot.legend()
pyplot.show()

