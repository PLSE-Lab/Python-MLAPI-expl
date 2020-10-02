#!/usr/bin/env python
# coding: utf-8

# # THANKS FOR CLICKING !!!!
# 
# ##  What are you going to learn with this Kernel?
# 
#  - Categorical to Continuous
#  - One Method To treat IMBALANCED MODEL
#  - Machine Learning Best Model Random Forest Classifier   
#  - ROC curve
#  - How to understand the problem and see which is the best model for your Dependent Variable
#  - Precision, Recall, F1, Avg_total Analysis
#  
# 
# ##  Bank Marketing
# 
# 
# **Abstract:** 
# The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).
# 
# **Data Set Information:**
# The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
# 
# ###  Attribute Information:
# 
# ####  Bank client data:
# 
#  - Age (numeric)
#  - Job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')
#  - Marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown' ; note: 'divorced' means divorced or widowed)
#  - Education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school',  'illiterate', 'professional.course', 'university.degree', 'unknown')
#  - Default: has credit in default? (categorical: 'no', 'yes', 'unknown')
#  - Housing: has housing loan? (categorical: 'no', 'yes', 'unknown')
#  - Loan: has personal loan? (categorical: 'no', 'yes', 'unknown')
# 
#     
# ####  Related with the last contact of the current campaign:
# 
#  - Contact: contact communication type (categorical:
#    'cellular','telephone')
#  - Month: last contact month of year (categorical: 'jan', 'feb', 'mar',
#    ..., 'nov', 'dec')
#  - Day_of_week: last contact day of the week (categorical:
#    'mon','tue','wed','thu','fri')
#  - Duration: last contact duration, in seconds (numeric). Important
#    note: this attribute highly affects the output target (e.g., if
#    duration=0 then y='no'). Yet, the duration is not known before a call
#    is performed. Also, after the end of the call y is obviously known.
#    Thus, this input should only be included for benchmark purposes and
#    should be discarded if the intention is to have a realistic
#    predictive model.
# 
#     
# ####  Other attributes:
# 
#  - Campaign: number of contacts performed during this campaign and for
#    this client (numeric, includes last contact)
#  - Pdays: number of days that passed by after the client was last
#    contacted from a previous campaign (numeric; 999 means client was not
#    previously contacted)
#  - Previous: number of contacts performed before this campaign and for
#    this client (numeric)
#  - Poutcome: outcome of the previous marketing campaign (categorical:
#    'failure','nonexistent','success')
# 
#     
# ####  Social and economic context attributes
#  - Emp.var.rate: employment variation rate - quarterly indicator
#    (numeric)
#  - Cons.price.idx: consumer price index - monthly indicator (numeric)
#  - Cons.conf.idx: consumer confidence index - monthly indicator
#    (numeric)
#  - Euribor3m: euribor 3 month rate - daily indicator (numeric)
#  - Nr.employed: number of employees - quarterly indicator (numeric)
# 
# ####  Output variable (desired target):
# 
#  - y - has the client subscribed a term deposit? (binary: 'yes', 'no')
# 
#      
# ###  Source:
# 
#  - Dataset from : http://archive.ics.uci.edu/ml/datasets/Bank+Marketing#
# 

# In[1]:


# Importing Data Analysis Librarys
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


bank = pd.read_csv('../input/bank-additional-full.csv', sep = ';')
#Converting dependent variable categorical to dummy
y = pd.get_dummies(bank['y'], columns = ['y'], drop_first = True)
yA = y
#y = y.drop('yes', axis = 1)


# # 1. Bank client data Analysis and Categorical Treatment [1/4]
# - To make things more clear, i'm going to creat a new datasets that contains just this part of data

# In[3]:


bank_client = bank.iloc[: , 0:7]
bank_client.head()


# ## 1.1. CONVERT COLUMNS TO CONTINUOUS

# In[4]:


# Label encoder order is alphabetical
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
bank_client['job']      = labelencoder_X.fit_transform(bank_client['job']) 
bank_client['marital']  = labelencoder_X.fit_transform(bank_client['marital']) 
bank_client['education']= labelencoder_X.fit_transform(bank_client['education']) 
bank_client['default']  = labelencoder_X.fit_transform(bank_client['default']) 
bank_client['housing']  = labelencoder_X.fit_transform(bank_client['housing']) 
bank_client['loan']     = labelencoder_X.fit_transform(bank_client['loan']) 


# In[5]:


#function to creat group of ages, this helps because we have 78 differente values here
def age(dataframe):
    dataframe.loc[dataframe['age'] <= 32, 'age'] = 1
    dataframe.loc[(dataframe['age'] > 32) & (dataframe['age'] <= 47), 'age'] = 2
    dataframe.loc[(dataframe['age'] > 47) & (dataframe['age'] <= 70), 'age'] = 3
    dataframe.loc[(dataframe['age'] > 70) & (dataframe['age'] <= 98), 'age'] = 4
           
    return dataframe

age(bank_client);


# In[6]:


bank_client.head()


# # 2. Related with the last contact of the current campaign [2/4]
# - Treat categorical, see those values
# - group continuous variables if necessary
# 

# In[7]:


# Slicing DataFrame to treat separately, make things more easy
bank_related = bank.iloc[: , 7:11]
bank_related.head()


# ## 2.1 CONVERT COLUMNS TO CONTINUOUS

# In[8]:


# Label encoder order is alphabetical
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
bank_related['contact']     = labelencoder_X.fit_transform(bank_related['contact']) 
bank_related['month']       = labelencoder_X.fit_transform(bank_related['month']) 
bank_related['day_of_week'] = labelencoder_X.fit_transform(bank_related['day_of_week']) 


# In[9]:


def duration(data):

    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration']    = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration']   = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 644.5), 'duration'] = 4
    data.loc[data['duration']  > 644.5, 'duration'] = 5

    return data
duration(bank_related);


# In[10]:


bank_related.head()


# # 3. Social and economic context attributes [3/4]

# In[11]:


bank_se = bank.loc[: , ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
bank_se.head()


# # 4. Other attributes [4/4]

# In[12]:


bank_o = bank.loc[: , ['campaign', 'pdays','previous', 'poutcome']]
bank_o.head()


# In[13]:


bank_o['poutcome'].replace(['nonexistent', 'failure', 'success'], [1,2,3], inplace  = True)


# # --------------  5. Model, Imbalance Dataset, ROC, Evaluate   --------------

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import resample


# In[15]:


bank_final= pd.concat([bank_client, bank_related, bank_se, bank_o], axis = 1)
bank_final = bank_final[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                     'contact', 'month', 'day_of_week', 'duration', 'emp.var.rate', 'cons.price.idx', 
                     'cons.conf.idx', 'euribor3m', 'nr.employed', 'campaign', 'pdays', 'previous', 'poutcome']]
bank_final.shape


# ### 5.1. IMBALANCED DATASET FIX
# 

# In[16]:


#IMBALANCED DATASET FIX
bank_final1 = pd.concat([bank_final, y], axis = 1)
df_majority = bank_final1[bank_final1['yes'] == 0]
df_minority = bank_final1[bank_final1['yes'] == 1]


# In[17]:


#IMBALANCED DATASET FIX
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    
                                 n_samples= int(4640*2),    
                                 random_state=123) 


# In[18]:


#IMBALANCED DATASET FIX
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
y_new = df_upsampled['yes']


# ### 5.2. Train Test DATASET FIX
# 

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(df_upsampled.drop('yes', axis = 1), y_new, test_size = 0.1942313295, random_state = 101)
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# ### 5.3. StandardScaler
# 

# In[20]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# ### 5.4. Random Forest Classifier

# In[21]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200, n_jobs=2, random_state = 12)#criterion = entopy,gini
rfc.fit(X_train, y_train)
rfcpred = rfc.predict(X_test)
RFCCV = (cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=2, scoring = 'accuracy').mean())


# In[22]:


models = pd.DataFrame({
                'Models': ['Random Forest Classifier'],
                'Score':  [RFCCV]})

models.sort_values(by='Score', ascending=False)


# ### 5.5. ROC CURVE
# Accuracy is measured by the area under the ROC curve. An area of 1 represents a perfect test; an area of .5 represents a worthless test.
# 
# A rough guide for classifying the accuracy of a diagnostic test is the traditional academic point system:
# 
# .90-1 = excellent (A)
# 
# .80-.90 = good (B)
# 
# .70-.80 = fair (C)
# 
# .60-.70 = poor (D)
# 
# .50-.60 = fail (F)

# In[23]:


from sklearn import metrics
fig, ax = plt.subplots(figsize = (6,6))
probs = rfc.predict_proba(X_test)
preds = probs[:,1]
fprrfc, tprrfc, thresholdrfc = metrics.roc_curve(y_test, preds)
roc_aucrfc = metrics.auc(fprrfc, tprrfc)

ax.plot(fprrfc, tprrfc, 'b', label = 'AUC = %0.2f' % roc_aucrfc)
ax.plot([0, 1], [0, 1],'r--')
ax.set_title('Receiver Operating Characteristic Random Forest ',fontsize=20)
ax.set_ylabel('True Positive Rate',fontsize=20)
ax.set_xlabel('False Positive Rate',fontsize=15)
ax.legend(loc = 'lower right', prop={'size': 16})

plt.subplots_adjust(wspace=1)


# # ANALYZING THE RESULTS

# **Explanation about the False(wrong) Values:**
# - False Positive, means the client do NOT SUBSCRIBED to term deposit, but the model thinks he did.
# - False Negative, means the client SUBSCRIBED to term deposit, but the model said he dont.
# 
# **In my opinion:**
# - The first one its most harmful, because we think that we already have that client but we dont and maybe we lost him in other future campaings 
# - The second its not good but its ok, we have that client and in the future we'll discovery that in truth he's already our client
# 
# Obs: - i'll do the math manualy to be more visible and understanding

# In[24]:


print('Cross Validation mean: ', (cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=2, scoring = 'accuracy').mean()))


# In[25]:


print('Cross Validation values: ', cross_val_score(rfc, X_train, y_train, cv=k_fold, n_jobs=2, scoring = 'accuracy'))


# ### WHITOUT BALANCE CLASSES

# **RFC Confusion Matrix**
# 
#                              [6884  212]
#                              [ 277  627]
#     
#     RFC Reports
#                     precision    recall  f1-score   support
#  
#                 0       0.96      0.97      0.97      7096
#                 1       0.75      0.69      0.72       904
#       avg / total       0.94      0.94      0.94      8000

# ### BALANCE CLASSES

# In[26]:


print('RFC Confusion Matrix\n', confusion_matrix(y_test, rfcpred))


# In[27]:


print('RFC Reports\n',classification_report(y_test, rfcpred))


# ###  Ok, now lets go deep into this values
# # RECALL

# In[28]:


print('RFC Confusion Matrix\n', confusion_matrix(y_test, rfcpred))


# ##### Recall - Specificity #####
# TN / (TN + FP) [ MATRIX LINE 1 ]
# 
#  - For all NEGATIVE(0) **REAL** VALUES how much we predict correct ?
# 
#  - other way to understand, our real test set has 6739 + 365 = 7104 clients that didin't subscribe(0), and our model predict 95% correct or 6739 correct and 365 incorrect   

# In[29]:


print('Specificity/ Recall 0 : ', round(6739 /(6739 + 365),2))


# ##### Recall - Sensitivity #####
# TP / (TP + FN) [ MATRIX LINE 2 ]
# 
#  - For all POSITIVE(1) **REAL** VALUES how much we predict correct ?
# 
#  - other way to understand, our real test set has 175 + 1623 = 1798 clients that subscribe(1), and our model predict 90% correct or 1623 correct and 175 incorrect.

# In[30]:


print('Sensitivity/ Recall 1 : ',round(1623 / (1623 + 175),3))
print('Sensitivity/ Recall 1 : ',round(metrics.recall_score(y_test, rfcpred),2))


# # PRECISION

# In[31]:


print('RFC Confusion Matrix\n', confusion_matrix(y_test, rfcpred))


# ##### Precision  #####
# TN / (TN + FN) [ MATRIX COLUMN 1 ]
# 
# - For all NEGATIVE(0) **PREDICTIONS** by our model, how much we predict correct ?
# 
# - other way to understand, our model pointed 6739 + 175 = 6914 clients that didin't subscribe(0), and our model predict 97% correct or 6739 correct and 175 incorrect   

# In[32]:


print('Precision 0 : ',round(6739 / (6739 + 175),2))


# ##### Precision  #####
# TN / (TN + FN) [ MATRIX COLUMN 1 ]
# 
# - For all POSITIVE(1) **PREDICTIONS** by our model, how much we predict correct ?
# 
# - other way to understand, our model pointed 365 + 1623 = 1988 clients that subscribe(1), and our model predict 82% correct or 253 correct and 116 incorrect   

# In[33]:


print('Precision 1 : ',round(1623 / (1623 + 365),2))
print('Precision 1 : ',round(metrics.precision_score(y_test, rfcpred),2))


# # F1-SCORE
# - F1-Score is a "median" of Recall and Precision, consider this when you want a balance between this metrics
# 
# F1 = 2(*Precision(0) * Recall(0)) / (Precision(0) + Recall(0))

# In[34]:


F1_0 = 2*0.97*0.95/(0.97+0.95)
print('F1-Score 0: ',round(F1_0,2))


# In[35]:


F1_1 = 2*0.82*0.9/(0.82+0.9)
print('F1-Score 1: ',round(F1_1,2))


# # AVG/ TOTAL
#  - this consider the weights of sum of REAL VALUES [line 1] [line2]
# 

# In[36]:


AVG_precision =  (0.97*(7104/8902))+ (0.82*(1798/8902))
print('AVT/Total Precision', round(AVG_precision,2))


# In[37]:


AVG_Recall =  (0.95*(7104/8902))+ (0.9*(1798/8902))
print('AVT/Total Recall', round(AVG_Recall,2))


# In[38]:


AVG_f1 =  (0.96*(7104/8902))+ (0.86*(1798/8902))
print('AVT/Total F1-Score', round(AVG_f1,2))

