#!/usr/bin/env python
# coding: utf-8

# A short description of the problem at hand 

# For the problem at hand and the data which we have we will go forward with our analysis in the structured manner as discussed below 
#     1. Section 1 
#            a. Rough analysis and understanding of the data and identify how useful the data is for analysis 
#            b. Try answering the below questions :
#                 i.   What are the most common Procedures and Diagnosis codes
#                 ii.  What are the most common procedures and diagnosis codes performed by the potential fradulent providers
#                 iii. How many providers submit potential fraud claims
#                 iv.  Which states/localities have the highest number of frauds
#                 v.   What is the average cost of potential fraud claims and also what is the cost as a % of whole a. Checking the outliers for such claims
#                 vi.  Is there any average age on which most of the time fradulent procedures or claims are applied on
# 
# 
#     2. Section 2
#             a. Descriptive analysis of the data 
#             b. Missing values
#             c. Outliers 
#             d. Prepare the data
#             
#             
#     3. Section 3 
#             a. Apply models 
# 
# 
# 
#     

# Please note the colors for the following:
# Inpatient -> green
# Outpatient -> cayan
# Combined -> blue 

# # **Section 1**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
# import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

Train=pd.read_csv("../input/healthcare-provider-fraud-detection-analysis/Train-1542865627584.csv")       
Train.head(2)


# Any results you write to the current directory are saved as output.


# In[ ]:


Test = pd.read_csv('../input/healthcare-provider-fraud-detection-analysis/Test-1542969243754.csv');
Test.head(2)


# In[ ]:


Train_Outpatient=pd.read_csv("../input/healthcare-provider-fraud-detection-analysis/Train_Outpatientdata-1542865627584.csv")       
Train_Outpatient.head(2)


# In[ ]:


Test_Outpatient = pd.read_csv('../input/healthcare-provider-fraud-detection-analysis/Test_Outpatientdata-1542969243754.csv');
Test_Outpatient.head(2)


# In[ ]:


Train_Inpatient = pd.read_csv('../input/healthcare-provider-fraud-detection-analysis/Train_Inpatientdata-1542865627584.csv');
Train_Inpatient.head(2)


# In[ ]:


Test_Inpatient=pd.read_csv("../input/healthcare-provider-fraud-detection-analysis/Test_Inpatientdata-1542969243754.csv")       
Test_Inpatient.head(2)


# In[ ]:


Train_Beneficiary=pd.read_csv("../input/healthcare-provider-fraud-detection-analysis/Train_Beneficiarydata-1542865627584.csv")       
Train_Beneficiary.head(2)


# In[ ]:



Test_Beneficiary = pd.read_csv('../input/healthcare-provider-fraud-detection-analysis/Test_Beneficiarydata-1542969243754.csv');
Test_Beneficiary.head(2)


# # Exploratory Data Analysis 
# 
# * Looking for the most common procedure codes which are applied for the fradulent and non fradulent services to see any specific pattern 

# # Inpatient Data 

# In[ ]:


df_procedures1 =  pd.DataFrame(columns = ['Procedures'])
df_procedures1['Procedures'] = pd.concat([Train_Inpatient["ClmProcedureCode_1"], Train_Inpatient["ClmProcedureCode_2"], Train_Inpatient["ClmProcedureCode_3"], Train_Inpatient["ClmProcedureCode_4"], Train_Inpatient["ClmProcedureCode_5"], Train_Inpatient["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures1['Procedures'].head(10)
grouped_procedure_df = df_procedures1['Procedures'].value_counts()

df_diagnosis = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis['Diagnosis'] = pd.concat([Train_Inpatient["ClmDiagnosisCode_1"], Train_Inpatient["ClmDiagnosisCode_2"], Train_Inpatient["ClmDiagnosisCode_3"], Train_Inpatient["ClmDiagnosisCode_4"], Train_Inpatient["ClmDiagnosisCode_5"], Train_Inpatient["ClmDiagnosisCode_6"], Train_Inpatient["ClmDiagnosisCode_7"],  Train_Inpatient["ClmDiagnosisCode_8"], Train_Inpatient["ClmDiagnosisCode_9"], Train_Inpatient["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis['Diagnosis'].head(10)
grouped_diagnosis_df = df_diagnosis['Diagnosis'].value_counts()

grouped_procedure_df1 = grouped_procedure_df.to_frame()
grouped_procedure_df1.columns = ['count']
grouped_procedure_df1['Procedure'] = grouped_procedure_df1.index
grouped_procedure_df1['Percentage'] = (grouped_procedure_df1['count']/sum(grouped_procedure_df1['count']))*100

grouped_diagnosis_df = grouped_diagnosis_df.to_frame()
grouped_diagnosis_df.columns = ['count']
grouped_diagnosis_df['Diagnosis'] = grouped_diagnosis_df.index
grouped_diagnosis_df['Percentage'] = (grouped_diagnosis_df['count']/sum(grouped_diagnosis_df['count']))*100

# taking only top 20 

plot_procedure_df1 = grouped_procedure_df1.head(20)
plot_diagnosis_df1 = grouped_diagnosis_df.head(20)

# Plotting the most commonly used diagnosis and procedures 
from matplotlib import pyplot as plt

plot_procedure_df1['Procedure'] = 'P' + plot_procedure_df1['Procedure'].astype(str)
plot_procedure_df1.sort_values(by=['Percentage'])
plot_procedure_df1.iplot(x ='Procedure', y='Percentage', kind='bar',xTitle='Procedure', color ='green',
                  yTitle='Percentage', title='Procedure Distribution', categoryorder='total descending')

plot_diagnosis_df1['Diagnosis'] = 'D' + plot_diagnosis_df1['Diagnosis'].astype(str)
plot_diagnosis_df1.sort_values(by=['Percentage'])
plot_diagnosis_df1.iplot(x ='Diagnosis', y='Percentage', kind='bar',xTitle='Diagnosis', color ='green',
                  yTitle='Percentage', title='Diagnosis Distribution', categoryorder='total descending')


# We see that for inpatinet the most common procedure used is 4019, 9904, 2724 among others
# 
# We see that for inpatinet the most common **Diagnosis** used is 4019, 2724,25000 among others

# # OUTPATIENT

# In[ ]:


df_procedures2 =  pd.DataFrame(columns = ['Procedures'])
df_procedures2['Procedures'] = pd.concat([Train_Outpatient["ClmProcedureCode_1"], Train_Outpatient["ClmProcedureCode_2"], Train_Outpatient["ClmProcedureCode_3"], Train_Outpatient["ClmProcedureCode_4"], Train_Outpatient["ClmProcedureCode_5"], Train_Outpatient["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures2['Procedures'].head(10)
grouped_procedure_df2 = df_procedures2['Procedures'].value_counts()

df_diagnosis2 = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis2['Diagnosis'] = pd.concat([Train_Outpatient["ClmDiagnosisCode_1"], Train_Outpatient["ClmDiagnosisCode_2"], Train_Outpatient["ClmDiagnosisCode_3"], Train_Outpatient["ClmDiagnosisCode_4"], Train_Outpatient["ClmDiagnosisCode_5"], Train_Outpatient["ClmDiagnosisCode_6"], Train_Outpatient["ClmDiagnosisCode_7"],  Train_Outpatient["ClmDiagnosisCode_8"], Train_Outpatient["ClmDiagnosisCode_9"], Train_Outpatient["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis2['Diagnosis'].head(10)
grouped_diagnosis_df2 = df_diagnosis2['Diagnosis'].value_counts()

grouped_procedure_df_op = grouped_procedure_df2.to_frame()
grouped_procedure_df_op.columns = ['count']
grouped_procedure_df_op['Procedure'] = grouped_procedure_df_op.index
grouped_procedure_df_op['Percentage'] = (grouped_procedure_df_op['count']/sum(grouped_procedure_df_op['count']))*100

grouped_diagnosis_df_op = grouped_diagnosis_df2.to_frame()
grouped_diagnosis_df_op.columns = ['count']
grouped_diagnosis_df_op['Diagnosis'] = grouped_diagnosis_df_op.index
grouped_diagnosis_df_op['Percentage'] = (grouped_diagnosis_df_op['count']/sum(grouped_diagnosis_df_op['count']))*100

# taking only top 20 

plot_procedure_df2 = grouped_procedure_df_op.head(20)
plot_diagnosis_df2 = grouped_diagnosis_df_op.head(20)

# Plotting the most commonly used diagnosis and procedures 
from matplotlib import pyplot as plt


plot_procedure_df2['Procedure'] = 'P' + plot_procedure_df2['Procedure'].astype(str)
plot_procedure_df2.sort_values(by=['Percentage'])
plot_procedure_df2.iplot(x ='Procedure', y='Percentage', kind='bar',xTitle='Procedure', color ='yellow',
                  yTitle='Percentage', title='Procedure Distribution', categoryorder='total descending')

plot_diagnosis_df2['Diagnosis'] = 'D' + plot_diagnosis_df2['Diagnosis'].astype(str)
plot_diagnosis_df2.sort_values(by=['Percentage'])
plot_diagnosis_df2.iplot(x ='Diagnosis', y='Percentage', kind='bar',xTitle='Diagnosis', color ='yellow',
                  yTitle='Percentage', title='Diagnosis Distribution', categoryorder='total descending')


# We see a minor difference between the most used diagnosis and procedure codes between inpatient and outpatients 
# 
# We see that for inpatinet the most common procedure used is 9904, 3722, 4516 among others
# 
# We see that for inpatinet the most common Diagnosis used is 4019, 25000, 2724 among others

# # **Let's see how many providers are fraudulent  **

# In[ ]:


Train.head()
T_fraud = Train['PotentialFraud'].value_counts()
grouped_train_df = T_fraud.to_frame()

grouped_train_df.columns = ['count']
grouped_train_df['Fraud'] = grouped_train_df.index
grouped_train_df['Percentage'] = (grouped_train_df['count']/sum(grouped_train_df['count']))*100
grouped_train_df['Percentage'].iplot( kind='bar',color = "blue", title = 'Distribution')


# # 2. What are the most common procedures and diagnosis codes performed by the potential fradulent providers

# # INPATIENT

# In[ ]:


len(Train_Inpatient)


# In[ ]:


Train_f =  pd.DataFrame(columns = ['PotentialFraud', 'Provider'])
Train_f = Train.loc[(Train['PotentialFraud'] == 'Yes')]
fraud_provider_ip_df = pd.merge(Train_Inpatient, Train_f, how='inner', on='Provider')
len(fraud_provider_ip_df)


# In[ ]:


(len(fraud_provider_ip_df)/len(Train_Inpatient)) * 100


# # Inference 
# 
# So we see there are **23402** admitted(inpatients) cases that the potential fradulent providers have interacted with at one point or the other during their services at the hospital. This is around **58%** of the cases which we have in our inpatient data. 
# 
# ***This means from our inpatient dataset for training we can have fradulent activities on more than half of them - 58% are potential fradulent encounters***

# # OUTPATIENT

# In[ ]:


len(Train_Outpatient)


# In[ ]:


fraud_provider_op_df = pd.merge(Train_Outpatient, Train_f, how='inner', on='Provider')
len(fraud_provider_op_df)


# In[ ]:


(len(fraud_provider_op_df)/len(Train_Outpatient))*100


# # Inference 
# 
# So we see there are **189394** outpatient cases that the potential fradulent providers have interacted with at one point or the other during their services at the hospital. This is around **37%** of the cases which we have in our inpatient data. 
# 
# ***This means from our outpatient dataset for training we can have fradulent activities on around 38% of encounters***

# # Now let us check which were the most used procedure codes and diagnosis codes used by the potential fradulent providers 

# # INPATIENT
# Potential fradulent encounters analysis of procedures and diagnosis codes used

# In[ ]:


df_procedures2 =  pd.DataFrame(columns = ['Procedures'])
df_procedures2['Procedures'] = pd.concat([fraud_provider_ip_df["ClmProcedureCode_1"], fraud_provider_ip_df["ClmProcedureCode_2"], fraud_provider_ip_df["ClmProcedureCode_3"], fraud_provider_ip_df["ClmProcedureCode_4"], fraud_provider_ip_df["ClmProcedureCode_5"], fraud_provider_ip_df["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures2['Procedures'].head(10)
grouped_F_procedure_df = df_procedures2['Procedures'].value_counts()

df_diagnosis2 = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis2['Diagnosis'] = pd.concat([fraud_provider_ip_df["ClmDiagnosisCode_1"], fraud_provider_ip_df["ClmDiagnosisCode_2"], fraud_provider_ip_df["ClmDiagnosisCode_3"], fraud_provider_ip_df["ClmDiagnosisCode_4"], fraud_provider_ip_df["ClmDiagnosisCode_5"], fraud_provider_ip_df["ClmDiagnosisCode_6"], fraud_provider_ip_df["ClmDiagnosisCode_7"],  fraud_provider_ip_df["ClmDiagnosisCode_8"], fraud_provider_ip_df["ClmDiagnosisCode_9"], fraud_provider_ip_df["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis2['Diagnosis'].head(10)
grouped_F_diagnosis_df = df_diagnosis2['Diagnosis'].value_counts()

grouped_F_procedure_df2 = grouped_F_procedure_df.to_frame()
grouped_F_procedure_df2.columns = ['count']
grouped_F_procedure_df2['Procedure'] = grouped_F_procedure_df2.index
grouped_F_procedure_df2['Percentage'] = (grouped_F_procedure_df2['count']/sum(grouped_F_procedure_df2['count']))*100

grouped_F_diagnosis_df2 = grouped_F_diagnosis_df.to_frame()
grouped_F_diagnosis_df2.columns = ['count']
grouped_F_diagnosis_df2['Diagnosis'] = grouped_F_diagnosis_df2.index
grouped_F_diagnosis_df2['Percentage'] = (grouped_F_diagnosis_df2['count']/sum(grouped_F_diagnosis_df2['count']))*100

plot_F_procedure_df1 = grouped_F_procedure_df2.head(20)

plot_F_diagnosis_df1 = grouped_F_diagnosis_df2.head(20)

plot_F_procedure_df1.plot(x ='Procedure', y='Percentage', kind = 'bar', color ='g')
plot_F_diagnosis_df1.plot(x ='Diagnosis', y='Percentage', kind = 'bar', color ='g')


# Here too we see a similar pattern with a little variation in the type of procedures used for fradulent activities 

# # OUTPATIENT 
# Potential fradulent encounters analysis of procedures and diagnosis codes used

# In[ ]:


df_procedures_op2 =  pd.DataFrame(columns = ['Procedures'])
df_procedures_op2['Procedures'] = pd.concat([fraud_provider_op_df["ClmProcedureCode_1"], fraud_provider_op_df["ClmProcedureCode_2"], fraud_provider_op_df["ClmProcedureCode_3"], fraud_provider_op_df["ClmProcedureCode_4"], fraud_provider_op_df["ClmProcedureCode_5"], fraud_provider_op_df["ClmProcedureCode_6"]], axis=0, sort=True).dropna()
df_procedures_op2['Procedures'].head(10)
grouped_F_procedure_op_df = df_procedures_op2['Procedures'].value_counts()

df_diagnosis_op2 = pd.DataFrame(columns = ['Diagnosis'])
df_diagnosis_op2['Diagnosis'] = pd.concat([fraud_provider_op_df["ClmDiagnosisCode_1"], fraud_provider_op_df["ClmDiagnosisCode_2"], fraud_provider_op_df["ClmDiagnosisCode_3"], fraud_provider_op_df["ClmDiagnosisCode_4"], fraud_provider_op_df["ClmDiagnosisCode_5"], fraud_provider_op_df["ClmDiagnosisCode_6"], fraud_provider_op_df["ClmDiagnosisCode_7"],  fraud_provider_op_df["ClmDiagnosisCode_8"], fraud_provider_op_df["ClmDiagnosisCode_9"], fraud_provider_op_df["ClmDiagnosisCode_10"]], axis=0, sort=True).dropna()
df_diagnosis_op2['Diagnosis'].head(10)
grouped_F_diagnosis_op_df = df_diagnosis2['Diagnosis'].value_counts()

grouped_F_procedure_opdf2 = grouped_F_procedure_op_df.to_frame()
grouped_F_procedure_opdf2.columns = ['count']
grouped_F_procedure_opdf2['Procedure'] = grouped_F_procedure_opdf2.index
grouped_F_procedure_opdf2['Percentage'] = (grouped_F_procedure_opdf2['count']/sum(grouped_F_procedure_opdf2['count']))*100

grouped_F_diagnosis_opdf2 = grouped_F_diagnosis_op_df.to_frame()
grouped_F_diagnosis_opdf2.columns = ['count']
grouped_F_diagnosis_opdf2['Diagnosis'] = grouped_F_diagnosis_opdf2.index
grouped_F_diagnosis_opdf2['Percentage'] = (grouped_F_diagnosis_opdf2['count']/sum(grouped_F_diagnosis_opdf2['count']))*100

plot_F_procedure_opdf1 = grouped_F_procedure_opdf2.head(20)

plot_F_diagnosis_opdf1 = grouped_F_diagnosis_opdf2.head(20)

plot_F_procedure_opdf1.plot(x ='Procedure', y='Percentage', kind = 'bar', color ='c')
plot_F_diagnosis_opdf1.plot(x ='Diagnosis', y='Percentage', kind = 'bar', color ='c')


# A little variation from the inpatient data if we knew the actual procedure codes we could identify why this was the case. 

# # 4.Which states/localities have the highest number of potential frauds 
# Data contains both inpatient and outpatient encounters

# In[ ]:


Train_Beneficiary.head(2)


# In[ ]:


fraud_beneficiary_ip_op_df = pd.merge(Train_Beneficiary, fraud_provider_ip_df, how='inner', on='BeneID')
fraud_beneficiary_ip_op_df = pd.merge(Train_Beneficiary, fraud_provider_op_df, how='inner', on='BeneID')
Train_F_Beneficiary_grouped = fraud_beneficiary_ip_op_df['State'].value_counts()
Train_F_Beneficiary_grouped1 = Train_F_Beneficiary_grouped.to_frame()
Train_F_Beneficiary_grouped1['Count'] =  Train_F_Beneficiary_grouped1['State']
Train_F_Beneficiary_grouped1['STATE'] = Train_F_Beneficiary_grouped1.index
Train_F_Beneficiary_grouped1 = Train_F_Beneficiary_grouped1.drop(['State'], axis = 1)
Train_F_Beneficiary_grouped1 = Train_F_Beneficiary_grouped1.head(20)
Train_F_Beneficiary_grouped1.plot(x ='STATE', y='Count', kind = 'bar')


# # Average Age for the data set and as a comparison for the probable fradulent activites applied on what age range

# In[ ]:


import seaborn as sns

fraud_beneficiary_ip_op_df['DOB'] =  pd.to_datetime(fraud_beneficiary_ip_op_df['DOB'], format='%Y-%m-%d')  
now = pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') # Assuming this is 2009 data as the last recorded death is for 2009
fraud_beneficiary_ip_op_df['DOB'] = fraud_beneficiary_ip_op_df['DOB'].where(fraud_beneficiary_ip_op_df['DOB'] < now, fraud_beneficiary_ip_op_df['DOB'] -  np.timedelta64(100, 'Y'))   # 2
fraud_beneficiary_ip_op_df['age'] = (now - fraud_beneficiary_ip_op_df['DOB']).astype('<m8[Y]')    # 3
ax = fraud_beneficiary_ip_op_df['age'].plot.hist(bins=20, alpha=0.5, figsize=(8, 6), edgecolor='k')


# This seems logical as most of the patients are of an age >65 

# Inpatient data as a whole not just the fradulent activities

# In[ ]:


Train_Beneficiary['DOB'] =  pd.to_datetime(Train_Beneficiary['DOB'], format='%Y-%m-%d')  
now = pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') # Assuming this is 2009 data as the last recorded death is for 2009
Train_Beneficiary['DOB'] = Train_Beneficiary['DOB'].where(Train_Beneficiary['DOB'] < now, Train_Beneficiary['DOB'] -  np.timedelta64(100, 'Y'))   # 2
Train_Beneficiary['age'] = (now - Train_Beneficiary['DOB']).astype('<m8[Y]')    # 3
ax = Train_Beneficiary['age'].plot.hist(bins=20, alpha=0.5, figsize=(8, 6), edgecolor='k')


# Here too we see a similar pattern

# # What is the average cost of potential fraud claims and also what is the cost as a % of whole a. Checking the outliers for such claims

# # INPATIENT

# In[ ]:


ax = Train_Inpatient['InscClaimAmtReimbursed'].plot.hist(bins=20, alpha=0.5, figsize=(8, 6), facecolor='g', edgecolor='k')


# In[ ]:


Train_Inpatient_1 = pd.merge(Train_Inpatient, Train, how='inner', on='Provider')
g = sns.FacetGrid(Train_Inpatient_1, col='PotentialFraud', height=8)
g.map(plt.hist, 'InscClaimAmtReimbursed', bins=20, color = 'g')


# We see that it is a significantly large amount which might be fradulent.

# In[ ]:


Train_Inpatient_1 = Train_Inpatient_1.loc[(Train_Inpatient_1['PotentialFraud'] == 'Yes')]
Total = Train_Inpatient_1['InscClaimAmtReimbursed'].sum()
print(Total)


# 241288510 - around 240 Million dollars worth of claim might have some fradulent activity. Even if we assume that it has just 10% fradulent activity the amount will be quite huge

# # OUTPATIENT

# In[ ]:


ax = Train_Outpatient['InscClaimAmtReimbursed'].plot.hist(bins=100,range=[0, 5000], alpha=0.5, figsize=(8, 6), facecolor='c', edgecolor='k')


# It makes sense most of the outpatient claims are of a smaller amount 

# In[ ]:


Train_Outpatient_1 = pd.merge(Train_Outpatient, Train, how='inner', on='Provider')
g = sns.FacetGrid(Train_Outpatient_1, col='PotentialFraud', height=8)
g.map(plt.hist, 'InscClaimAmtReimbursed', bins=20, range=[0, 5000], color ='c')


# # Section 2

# Checking for missing values in the data set 

# In[ ]:


Train_Beneficiary.head()


# In[ ]:


Train_Beneficiary.isna().sum()


# In[ ]:


Test_Beneficiary.isna().sum()


# Only the date of death is empty - makes sense for the people who are alive

# Adding the columns - Age

# In[ ]:


Train_Beneficiary['DOB'] = pd.to_datetime(Train_Beneficiary['DOB'] , format = '%Y-%m-%d')
Train_Beneficiary['DOD'] = pd.to_datetime(Train_Beneficiary['DOD'],format = '%Y-%m-%d',errors='ignore')
Train_Beneficiary['Age'] = round(((Train_Beneficiary['DOD'] - Train_Beneficiary['DOB']).dt.days)/365)

Test_Beneficiary['DOB'] = pd.to_datetime(Test_Beneficiary['DOB'] , format = '%Y-%m-%d')
Test_Beneficiary['DOD'] = pd.to_datetime(Test_Beneficiary['DOD'],format = '%Y-%m-%d',errors='ignore')
Test_Beneficiary['Age'] = round(((Test_Beneficiary['DOD'] - Test_Beneficiary['DOB']).dt.days)/365)

## As we see that last DOD value is 2009-12-01 ,which means Beneficiary Details data is of year 2009.
## so we will calculate age of other benficiaries for year 2009.

Train_Beneficiary.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - Test_Beneficiary['DOB']).dt.days)/365),
                                 inplace=True)


Test_Beneficiary.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - Test_Beneficiary['DOB']).dt.days)/365),
                                 inplace=True)


Test_Beneficiary.head(2)


# Let us create the data joining beneficiaries, Inpatient and out patient data and then we will see if any data conversion is required or not

# In[ ]:


# Joining inpatinet and outpatient data with a column defining if it is inpatient or outpatient 


# In[ ]:


## Creating the master DF for test
Train_Inpatient['EncounterType'] = 0
Train_Outpatient['EncounterType'] = 1
frames = [Train_Inpatient, Train_Outpatient]
TrainInAndOut = pd.concat(frames)
TrainInAndOutBenf = pd.merge(TrainInAndOut, Train_Beneficiary, how='inner', on='BeneID')
Master_df = pd.merge(TrainInAndOutBenf, Train, how='inner', on='Provider')

##Creating the master DF for Test 
Test_Inpatient['EncounterType'] = 0
Test_Outpatient['EncounterType'] = 1
frames = [Test_Inpatient, Test_Outpatient]
TestInAndOut = pd.concat(frames)
TestInAndOutBenf = pd.merge(TestInAndOut, Test_Beneficiary, how='inner', on='BeneID')
MasterTest_df = pd.merge(TestInAndOutBenf, Test, how='inner', on='Provider')


# In[ ]:


Master_df['DOB'] = pd.to_datetime(Master_df['DOB'] , format = '%Y-%m-%d')
Master_df['DOD'] = pd.to_datetime(Master_df['DOD'],format = '%Y-%m-%d',errors='ignore')
Master_df['Age'] = round(((Master_df['DOD'] - Master_df['DOB']).dt.days)/365)

MasterTest_df['DOB'] = pd.to_datetime(MasterTest_df['DOB'] , format = '%Y-%m-%d')
MasterTest_df['DOD'] = pd.to_datetime(MasterTest_df['DOD'],format = '%Y-%m-%d',errors='ignore')
MasterTest_df['Age'] = round(((MasterTest_df['DOD'] - MasterTest_df['DOB']).dt.days)/365)

## As we see that last DOD value is 2009-12-01 ,which means Beneficiary Details data is of year 2009.
## so we will calculate age of other benficiaries for year 2009.

Master_df.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - Master_df['DOB']).dt.days)/365),
                                 inplace=True)


MasterTest_df.Age.fillna(round(((pd.to_datetime('2009-12-01' , format = '%Y-%m-%d') - MasterTest_df['DOB']).dt.days)/365),
                                 inplace=True)


MasterTest_df.head(2)
Master_df = Master_df.drop(['age'], axis = 1) 


# In[ ]:


## removing the column DOD and DOB also creating a new column IsDead as we already have the age we do not need date of death and date of birth 

Master_df.loc[Master_df['DOD'].isnull(), 'IsDead'] = '0'
Master_df.loc[(Master_df['DOD'].notnull()), 'IsDead'] = '1'
Master_df = Master_df.drop(['DOD'], axis = 1)
Master_df = Master_df.drop(['DOB'], axis = 1)

## Same activity on the test data 
MasterTest_df.loc[MasterTest_df['DOD'].isnull(), 'IsDead'] = '0'
MasterTest_df.loc[(MasterTest_df['DOD'].notnull()), 'IsDead'] = '1'
MasterTest_df = MasterTest_df.drop(['DOD'], axis = 1)
MasterTest_df = MasterTest_df.drop(['DOB'], axis = 1)
MasterTest_df.head()


# Calculating the number of days the patient was admitted to the dospital and removing admission and discharge date, For outpatients as they do not get admitted will put number of days admitted = 0 

# In[ ]:


Master_df['AdmissionDt'] = pd.to_datetime(Master_df['AdmissionDt'] , format = '%Y-%m-%d')
Master_df['DischargeDt'] = pd.to_datetime(Master_df['DischargeDt'],format = '%Y-%m-%d')
Master_df['DaysAdmitted'] = ((Master_df['DischargeDt'] - Master_df['AdmissionDt']).dt.days)+1
Master_df.loc[Master_df['EncounterType'] == 1, 'DaysAdmitted'] = '0'
Master_df[['EncounterType','DaysAdmitted','DischargeDt','AdmissionDt']].head()
Master_df = Master_df.drop(['DischargeDt'], axis = 1)
Master_df = Master_df.drop(['AdmissionDt'], axis = 1)

## Performing the same operations on test data 
MasterTest_df['AdmissionDt'] = pd.to_datetime(MasterTest_df['AdmissionDt'] , format = '%Y-%m-%d')
MasterTest_df['DischargeDt'] = pd.to_datetime(MasterTest_df['DischargeDt'],format = '%Y-%m-%d')
MasterTest_df['DaysAdmitted'] = ((MasterTest_df['DischargeDt'] - MasterTest_df['AdmissionDt']).dt.days)+1
MasterTest_df.loc[MasterTest_df['EncounterType'] == 1, 'DaysAdmitted'] = '0'
MasterTest_df[['EncounterType','DaysAdmitted','DischargeDt','AdmissionDt', 'DeductibleAmtPaid']].head()
MasterTest_df = MasterTest_df.drop(['DischargeDt'], axis = 1)
MasterTest_df = MasterTest_df.drop(['AdmissionDt'], axis = 1)


# In[ ]:


MasterTest_df.loc[MasterTest_df['DeductibleAmtPaid'].isnull(), 'DeductibleAmtPaid'] = '0'
Master_df.loc[Master_df['DeductibleAmtPaid'].isnull(), 'DeductibleAmtPaid'] = '0'


# In[ ]:


#Master_df.isna().sum()
Master_df.isna().sum()


# For procedures, diagnosis codes and Physician let us try one hot encoding and then follwed by PCA 
# 
# 
# **DIAGNOSIS** 
# 
# Please note: 
# 1. Number of diagnosis codes and procedure codes are huge and doing a one hot encoding on the same we will explode the data set - so we are taking the top 80%(cumilatively) used items for one hot encoding and then performing a PCA on them all together.
# (example - performing one hot encoding for all the diagnosis related fields first and then performing PCA on all of them together)
# 
# We can do this because many times the diagnosis codes might be present on any of the columns and it really does not matter at which position the diagnosis code was used at as that might depend on factors such as if the patient is visiting a specialist or a general physician, or if a base diagnosis code is applied or a subbordinate one which in the later case cannot be the diagnosis code 1. 
# 
# All we care for here is if we can find a pattern in the usability of the codes all together and the same logic can be applied for procedure codes as well.

# In[ ]:


Master_df.shape


# Finding the top 80% of the diagnosis codes and making all others 0 from the following fields and then perform one hot encoding over the same 
# * ClmAdmitDiagnosisCode              
# * ClmDiagnosisCode_1                  
# * ClmDiagnosisCode_10                
# * ClmDiagnosisCode_2                 
# * ClmDiagnosisCode_3                 
# * ClmDiagnosisCode_4                 
# * ClmDiagnosisCode_5                 
# * ClmDiagnosisCode_6                 
# * ClmDiagnosisCode_7                 
# * ClmDiagnosisCode_8                 
# * ClmDiagnosisCode_9
# * DiagnosisGroupCode                 
# 

# **ClmAdmitDiagnosisCode**

# In[ ]:


admit_diagnosis = Master_df['ClmAdmitDiagnosisCode'].value_counts()
admit_diagnosis_df = admit_diagnosis.to_frame()
admit_diagnosis_df ['Percentage_ClmAdmitDiagnosis'] = (admit_diagnosis_df['ClmAdmitDiagnosisCode']/admit_diagnosis_df['ClmAdmitDiagnosisCode'].sum())*100
admit_diagnosis_df ['Percentage_ClmAdmitDiagnosis'] = admit_diagnosis_df ['Percentage_ClmAdmitDiagnosis'].cumsum()
admit_diagnosis_df.loc[admit_diagnosis_df['Percentage_ClmAdmitDiagnosis'] > 80, 'Percentage_ClmAdmitDiagnosis'] = 0
admit_diagnosis_df.drop(['ClmAdmitDiagnosisCode'], axis = 1) 
admit_diagnosis_df['ClmAdmitDiagnosisCode'] = admit_diagnosis_df.index
Master_df = pd.merge(Master_df, admit_diagnosis_df, how='inner', on='ClmAdmitDiagnosisCode')
Master_df.loc[Master_df['Percentage_ClmAdmitDiagnosis'] == 0, 'ClmAdmitDiagnosisCode'] = 0
Master_df.tail(5)


# **ClmDiagnosisCode_1**

# In[ ]:


diagnosis1 = Master_df['ClmDiagnosisCode_1'].value_counts()
diagnosis1_df = diagnosis1.to_frame()
diagnosis1_df ['Percentage_ClmDiagnosisCode_1'] = (diagnosis1_df['ClmDiagnosisCode_1']/diagnosis1_df['ClmDiagnosisCode_1'].sum())*100
diagnosis1_df ['Percentage_ClmDiagnosisCode_1'] = diagnosis1_df['Percentage_ClmDiagnosisCode_1'].cumsum()
diagnosis1_df.loc[diagnosis1_df['Percentage_ClmDiagnosisCode_1'] > 60, 'Percentage_ClmDiagnosisCode_1'] = 0
# for checking the number of items we will end up with 
# subset_df = diagnosis1_df[diagnosis1_df['Percentage_ClmDiagnosisCode_1'] != 0]
# subset_df.count()
#diagnosis1_df.head()
diagnosis1_df.drop(['ClmDiagnosisCode_1'], axis = 1) 
diagnosis1_df['ClmDiagnosisCode_1'] = diagnosis1_df.index
Master_df = pd.merge(Master_df, diagnosis1_df, how='inner', on='ClmDiagnosisCode_1')
Master_df.loc[Master_df['Percentage_ClmDiagnosisCode_1'] == 0, 'ClmDiagnosisCode_1'] = 0
Master_df.tail(5)


# **ClmDiagnosisCode_10**

# In[ ]:


diagnosis10 = Master_df['ClmDiagnosisCode_10'].value_counts()
diagnosis10_df = diagnosis10.to_frame()
diagnosis10_df ['Percentage_ClmDiagnosisCode_10'] = (diagnosis10_df['ClmDiagnosisCode_10']/diagnosis10_df['ClmDiagnosisCode_10'].sum())*100
diagnosis10_df ['Percentage_ClmDiagnosisCode_10'] = diagnosis10_df['Percentage_ClmDiagnosisCode_10'].cumsum()
diagnosis10_df.loc[diagnosis10_df['Percentage_ClmDiagnosisCode_10'] > 60, 'Percentage_ClmDiagnosisCode_10'] = 0
# for checking the number of items we will end up with 
# subset_df = diagnosis10_df[diagnosis10_df['Percentage_ClmDiagnosisCode_10'] != 0]
# subset_df.count()
#diagnosis1_df.head()
diagnosis10_df.drop(['ClmDiagnosisCode_10'], axis = 1) 
diagnosis10_df['ClmDiagnosisCode_10'] = diagnosis10_df.index
Master_df = pd.merge(Master_df, diagnosis10_df, how='inner', on='ClmDiagnosisCode_10')
Master_df.loc[Master_df['Percentage_ClmDiagnosisCode_10'] == 0, 'ClmDiagnosisCode_10'] = 0
Master_df.tail(5)


# **ClmDiagnosisCode_2**

# In[ ]:


diagnosis2 = Master_df['ClmDiagnosisCode_2'].value_counts()
diagnosis2_df = diagnosis2.to_frame()
diagnosis2_df ['Percentage_ClmDiagnosisCode_2'] = (diagnosis2_df['ClmDiagnosisCode_2']/diagnosis2_df['ClmDiagnosisCode_2'].sum())*100
diagnosis2_df ['Percentage_ClmDiagnosisCode_2'] = diagnosis2_df['Percentage_ClmDiagnosisCode_2'].cumsum()
diagnosis2_df.loc[diagnosis2_df['Percentage_ClmDiagnosisCode_2'] > 80, 'Percentage_ClmDiagnosisCode_2'] = 0
#for checking the number of items we will end up with 
# subset_df = diagnosis2_df[diagnosis2_df['Percentage_ClmDiagnosisCode_2'] != 0]
# subset_df.count()
#diagnosis1_df.head()
diagnosis2_df.drop(['ClmDiagnosisCode_2'], axis = 1) 
diagnosis2_df['ClmDiagnosisCode_2'] = diagnosis2_df.index
Master_df = pd.merge(Master_df, diagnosis2_df, how='inner', on='ClmDiagnosisCode_2')
Master_df.loc[Master_df['Percentage_ClmDiagnosisCode_2'] == 0, 'ClmDiagnosisCode_2'] = 0
Master_df.tail(5)


# **ClmDiagnosisCode_3**

# In[ ]:


diagnosis3 = Master_df['ClmDiagnosisCode_3'].value_counts()
diagnosis3_df = diagnosis3.to_frame()
diagnosis3_df ['Percentage_ClmDiagnosisCode_3'] = (diagnosis3_df['ClmDiagnosisCode_3']/diagnosis3_df['ClmDiagnosisCode_3'].sum())*100
diagnosis3_df ['Percentage_ClmDiagnosisCode_3'] = diagnosis3_df['Percentage_ClmDiagnosisCode_3'].cumsum()
diagnosis3_df.loc[diagnosis3_df['Percentage_ClmDiagnosisCode_3'] > 80, 'Percentage_ClmDiagnosisCode_3'] = 0
#for checking the number of items we will end up with 
# subset_df = diagnosis3_df[diagnosis3_df['Percentage_ClmDiagnosisCode_3'] != 0]
# subset_df.count()
#diagnosis1_df.head()
diagnosis3_df.drop(['ClmDiagnosisCode_3'], axis = 1) 
diagnosis3_df['ClmDiagnosisCode_3'] = diagnosis3_df.index
Master_df = pd.merge(Master_df, diagnosis3_df, how='inner', on='ClmDiagnosisCode_3')
Master_df.loc[Master_df['Percentage_ClmDiagnosisCode_3'] == 0, 'ClmDiagnosisCode_3'] = 0
Master_df.tail(5)


# **ClmDiagnosisCode_4**

# In[ ]:


diagnosis4 = Master_df['ClmDiagnosisCode_4'].value_counts()
diagnosis4_df = diagnosis4.to_frame()
diagnosis4_df ['Percentage_ClmDiagnosisCode_4'] = (diagnosis4_df['ClmDiagnosisCode_4']/diagnosis4_df['ClmDiagnosisCode_4'].sum())*100
diagnosis4_df ['Percentage_ClmDiagnosisCode_4'] = diagnosis4_df['Percentage_ClmDiagnosisCode_4'].cumsum()
diagnosis4_df.loc[diagnosis4_df['Percentage_ClmDiagnosisCode_4'] > 80, 'Percentage_ClmDiagnosisCode_4'] = 0
#for checking the number of items we will end up with 
# subset_df = diagnosis4_df[diagnosis4_df['Percentage_ClmDiagnosisCode_4'] != 0]
# subset_df.count()
#diagnosis1_df.head()
diagnosis4_df.drop(['ClmDiagnosisCode_4'], axis = 1) 
diagnosis4_df['ClmDiagnosisCode_4'] = diagnosis4_df.index
Master_df = pd.merge(Master_df, diagnosis4_df, how='inner', on='ClmDiagnosisCode_4')
Master_df.loc[Master_df['Percentage_ClmDiagnosisCode_4'] == 0, 'ClmDiagnosisCode_4'] = 0
Master_df.tail(5)


# **ClmDiagnosisCode_5**

# In[ ]:


diagnosis5 = Master_df['ClmDiagnosisCode_5'].value_counts()
diagnosis5_df = diagnosis5.to_frame()
diagnosis5_df ['Percentage_ClmDiagnosisCode_5'] = (diagnosis5_df['ClmDiagnosisCode_5']/diagnosis5_df['ClmDiagnosisCode_5'].sum())*100
diagnosis5_df ['Percentage_ClmDiagnosisCode_5'] = diagnosis5_df['Percentage_ClmDiagnosisCode_5'].cumsum()
diagnosis5_df.loc[diagnosis5_df['Percentage_ClmDiagnosisCode_5'] > 80, 'Percentage_ClmDiagnosisCode_5'] = 0
#for checking the number of items we will end up with 
# subset_df = diagnosis5_df[diagnosis5_df['Percentage_ClmDiagnosisCode_5'] != 0]
# subset_df.count()
#diagnosis1_df.head()
diagnosis5_df.drop(['ClmDiagnosisCode_5'], axis = 1) 
diagnosis5_df['ClmDiagnosisCode_5'] = diagnosis5_df.index
Master_df = pd.merge(Master_df, diagnosis5_df, how='inner', on='ClmDiagnosisCode_5')
Master_df.loc[Master_df['Percentage_ClmDiagnosisCode_5'] == 0, 'ClmDiagnosisCode_5'] = 0
Master_df.tail(5)


# **ClmDiagnosisCode_6**

# In[ ]:


diagnosis6 = Master_df['ClmDiagnosisCode_6'].value_counts()
diagnosis6_df = diagnosis6.to_frame()
diagnosis6_df ['Percentage_ClmDiagnosisCode_6'] = (diagnosis6_df['ClmDiagnosisCode_6']/diagnosis6_df['ClmDiagnosisCode_6'].sum())*100
diagnosis6_df ['Percentage_ClmDiagnosisCode_6'] = diagnosis6_df['Percentage_ClmDiagnosisCode_6'].cumsum()
diagnosis6_df.loc[diagnosis6_df['Percentage_ClmDiagnosisCode_6'] > 70, 'Percentage_ClmDiagnosisCode_6'] = 0
#for checking the number of items we will end up with 
# subset_df = diagnosis6_df[diagnosis6_df['Percentage_ClmDiagnosisCode_6'] != 0]
# subset_df.count()
#diagnosis1_df.head()
diagnosis6_df.drop(['ClmDiagnosisCode_6'], axis = 1) 
diagnosis6_df['ClmDiagnosisCode_6'] = diagnosis6_df.index
Master_df = pd.merge(Master_df, diagnosis6_df, how='inner', on='ClmDiagnosisCode_6')
Master_df.loc[Master_df['Percentage_ClmDiagnosisCode_6'] == 0, 'ClmDiagnosisCode_6'] = 0
Master_df.tail(5)


# **ClmDiagnosisCode_7**

# In[ ]:


diagnosis7 = Master_df['ClmDiagnosisCode_7'].value_counts()
diagnosis7_df = diagnosis7.to_frame()
diagnosis7_df ['Percentage_ClmDiagnosisCode_7'] = (diagnosis7_df['ClmDiagnosisCode_7']/diagnosis7_df['ClmDiagnosisCode_7'].sum())*100
diagnosis7_df ['Percentage_ClmDiagnosisCode_7'] = diagnosis7_df['Percentage_ClmDiagnosisCode_7'].cumsum()
diagnosis7_df.loc[diagnosis7_df['Percentage_ClmDiagnosisCode_7'] > 70, 'Percentage_ClmDiagnosisCode_7'] = 0
#for checking the number of items we will end up with 
# subset_df = diagnosis7_df[diagnosis7_df['Percentage_ClmDiagnosisCode_7'] != 0]
# subset_df.count()
#diagnosis1_df.head()
diagnosis7_df.drop(['ClmDiagnosisCode_7'], axis = 1) 
diagnosis7_df['ClmDiagnosisCode_7'] = diagnosis7_df.index
Master_df = pd.merge(Master_df, diagnosis7_df, how='inner', on='ClmDiagnosisCode_7')
Master_df.loc[Master_df['Percentage_ClmDiagnosisCode_7'] == 0, 'ClmDiagnosisCode_7'] = 0
Master_df.tail(5)


# **ClmDiagnosisCode_8**

# In[ ]:


diagnosis8 = Master_df['ClmDiagnosisCode_8'].value_counts()
diagnosis8_df = diagnosis8.to_frame()
diagnosis8_df ['Percentage_ClmDiagnosisCode_8'] = (diagnosis8_df['ClmDiagnosisCode_8']/diagnosis8_df['ClmDiagnosisCode_8'].sum())*100
diagnosis8_df ['Percentage_ClmDiagnosisCode_8'] = diagnosis8_df['Percentage_ClmDiagnosisCode_8'].cumsum()
diagnosis8_df.loc[diagnosis8_df['Percentage_ClmDiagnosisCode_8'] > 70, 'Percentage_ClmDiagnosisCode_8'] = 0
#for checking the number of items we will end up with 
# subset_df = diagnosis8_df[diagnosis8_df['Percentage_ClmDiagnosisCode_8'] != 0]
# subset_df.count()
#diagnosis1_df.head()
diagnosis8_df.drop(['ClmDiagnosisCode_8'], axis = 1) 
diagnosis8_df['ClmDiagnosisCode_8'] = diagnosis8_df.index
Master_df = pd.merge(Master_df, diagnosis8_df, how='inner', on='ClmDiagnosisCode_8')
Master_df.loc[Master_df['Percentage_ClmDiagnosisCode_8'] == 0, 'ClmDiagnosisCode_8'] = 0
Master_df.tail(5)


# **ClmDiagnosisCode_9**

# In[ ]:


diagnosis9 = Master_df['ClmDiagnosisCode_9'].value_counts()
diagnosis9_df = diagnosis9.to_frame()
diagnosis9_df ['Percentage_ClmDiagnosisCode_9'] = (diagnosis9_df['ClmDiagnosisCode_9']/diagnosis9_df['ClmDiagnosisCode_9'].sum())*100
diagnosis9_df ['Percentage_ClmDiagnosisCode_9'] = diagnosis9_df['Percentage_ClmDiagnosisCode_9'].cumsum()
diagnosis9_df.loc[diagnosis9_df['Percentage_ClmDiagnosisCode_9'] > 70, 'Percentage_ClmDiagnosisCode_9'] = 0
#for checking the number of items we will end up with 
# subset_df = diagnosis9_df[diagnosis9_df['Percentage_ClmDiagnosisCode_9'] != 0]
# subset_df.count()
#diagnosis1_df.head()
diagnosis9_df.drop(['ClmDiagnosisCode_9'], axis = 1) 
diagnosis9_df['ClmDiagnosisCode_9'] = diagnosis9_df.index
Master_df = pd.merge(Master_df, diagnosis9_df, how='inner', on='ClmDiagnosisCode_9')
Master_df.loc[Master_df['Percentage_ClmDiagnosisCode_9'] == 0, 'ClmDiagnosisCode_9'] = 0
Master_df.tail(5)


# **DiagnosisGroupCode**

# In[ ]:


diagnosisGr = Master_df['DiagnosisGroupCode'].value_counts()
diagnosisGr_df = diagnosisGr.to_frame()
diagnosisGr_df ['Percentage_DiagnosisGroupCode'] = (diagnosisGr_df['DiagnosisGroupCode']/diagnosisGr_df['DiagnosisGroupCode'].sum())*100
diagnosisGr_df ['Percentage_DiagnosisGroupCode'] = diagnosisGr_df['Percentage_DiagnosisGroupCode'].cumsum()
diagnosisGr_df.loc[diagnosisGr_df['Percentage_DiagnosisGroupCode'] > 70, 'Percentage_DiagnosisGroupCode'] = 0
#for checking the number of items we will end up with 
# subset_df = diagnosisGr_df[diagnosisGr_df['Percentage_DiagnosisGroupCode'] != 0]
# subset_df.count()
#diagnosis1_df.head()
diagnosisGr_df.drop(['DiagnosisGroupCode'], axis = 1) 
diagnosisGr_df['DiagnosisGroupCode'] = diagnosisGr_df.index
Master_df = pd.merge(Master_df, diagnosisGr_df, how='inner', on='DiagnosisGroupCode')
Master_df.loc[Master_df['Percentage_DiagnosisGroupCode'] == 0, 'DiagnosisGroupCode'] = 0
Master_df.tail(5)


# One hot encoding for Claim Admit diagnosis 

# In[ ]:


Master_df = pd.get_dummies(Master_df,columns=['ClmAdmitDiagnosisCode', 'ClmDiagnosisCode_1','ClmDiagnosisCode_10', 'ClmDiagnosisCode_2','ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7' , 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'DiagnosisGroupCode'], prefix='DiagnosisCodeA')
Master_df.tail()


# In[ ]:


for_PCA_diag = Master_df.loc[:,Master_df.columns.str.startswith("DiagnosisCode")]
for_PCA_diag.head()


# In[ ]:


# #Master_df[Master_df.columns[pd.Series(Master_df.columns).str.startswith('ClmDiagnosisCode')]]
from sklearn.decomposition import PCA
pca = PCA(n_components = .90) # all the PCA features explaining more than 1% (I verified it by trying and checking the explained_variance_ratio_)
reduced = pca.fit_transform(for_PCA_diag)


# In[ ]:


var = pca.explained_variance_ratio_.cumsum()
plt.plot(var)


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# #Master_df.columns[pd.Series(Master_df.columns).str.startswith('AdmitDiagnosis')].tolist()


# :) Work in progress ... 
