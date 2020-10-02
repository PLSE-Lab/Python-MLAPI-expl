#!/usr/bin/env python
# coding: utf-8

# # Kaggle-dataset-analysis
# ## crime in Boston
# # 2019-02-16 start

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


original=pd.read_csv('../input/crime.csv',engine='python')


# In[ ]:


original['YEAR'].unique()


# In[ ]:


original['OCCURRED_ON_DATE']=pd.to_datetime(original['OCCURRED_ON_DATE'])


# In[ ]:


np.isnan(original.any())


# In[ ]:


original['DISTRICT'].unique()


# ## top5 OFFENSE_CODE_GROUP 

# In[ ]:


offense=pd.pivot_table(original.loc[:,['OFFENSE_CODE_GROUP','YEAR','OFFENSE_CODE']],index='OFFENSE_CODE_GROUP',               columns='YEAR',aggfunc=np.count_nonzero)


# In[ ]:


summary=pd.DataFrame(offense.apply(np.sum,axis=1))
summary=summary.rename(columns={0:'total'})


# In[ ]:


sumsort=summary.sort_values(by='total',ascending=False)
top5=sumsort.iloc[0:5,:]
top5


# In[ ]:


sns.set()
p0=plt.figure(figsize=(15,14))
plt.title(r'2015-2018 top 5 group bar plot')
plt.bar(range(top5.index.shape[0]),top5.loc[:,'total'])
plt.xlabel('OFFENSE_CODE_GROUP')
plt.xticks(range(top5.index.shape[0]),top5.index)
x=np.arange(top5.index.shape[0])
y=np.array(top5['total'])
for i,j in zip(x,y):
    plt.text(i,j,'%d'%j,ha='center')
p0.savefig('./top5_group_bar.png')
plt.show()


# ## Visualization-by DISTRICT

# In[ ]:


disgroup=original.groupby(by='DISTRICT')


# In[ ]:


groupcount=disgroup.count()


# In[ ]:


groupcount.head()


# In[ ]:


number=groupcount.iloc[:,0]
number=pd.DataFrame(number)


# In[ ]:


number.rename(columns={'INCIDENT_NUMBER':'NUMBER'},inplace=True)
number.head()


# In[ ]:


plt.figure(figsize=(15,14))
plt.title(r'2015-2018 boston-crime by district bar plot')
p1=sns.barplot(x=number.index,y='NUMBER',data=number)
x=np.arange(number.index.shape[0])
y=np.array(list(number['NUMBER']))
for i,j in zip(x,y):
    plt.text(i,j+0.05,'%d'%j,ha='center')
else:
    pass
p1fig=p1.get_figure()
p1fig.savefig('./total_by_district_bar.png')
plt.show()


# ## DISTRICT
# 
# B2 district has highest number of crime
# 
# C11 and D4 are higher than other district
# 
# A15 is least
# 
# follow this,from number dataframe take top5 district

# In[ ]:


districtsorted=number.sort_values(by='NUMBER',ascending=False)


# In[ ]:


top5=districtsorted.iloc[0:5,:]


# In[ ]:


plt.figure(figsize=(15,14))
plt.title(r'top5 district crime bar plot')
p2=sns.barplot(x=top5.index,y='NUMBER',data=top5)
x=np.arange(top5.index.shape[0])
y=np.array(list(top5['NUMBER']))
for i,j in zip(x,y):
    plt.text(i,j+0.05,'%d'%j,ha='center')
else:
    pass
p2fig=p2.get_figure()
p2fig.savefig('./top5_district_crime_bar.png')
plt.show()


# ## Visualization-by YEAR

# In[ ]:


yeargroup=original.groupby(by='YEAR')


# In[ ]:


count=yeargroup.count()


# In[ ]:


yearnumber=pd.DataFrame(count.iloc[:,0])


# In[ ]:


yearnumber.rename(columns={'INCIDENT_NUMBER':'NUMBER'},inplace=True)


# In[ ]:


yearnumber


# In[ ]:


plt.figure(figsize=(15,14))
plt.title(r'2015-2018 crime by year bar plot')
p3=sns.barplot(x=yearnumber.index,y='NUMBER',data=yearnumber)
x=np.arange(yearnumber.index.shape[0])
y=np.array(list(yearnumber['NUMBER']))
for i,j in zip(x,y):
    plt.text(i,j,'%d'%j,ha='center')
else:
    pass
p3fig=p3.get_figure()
p3fig.savefig('./total_by_year_bar.png')
plt.show()


# ## by year
# 
# With this bar chart,we can see 2017's crime number is highest
# 
# And 2015 is lowest,this may be caused by people get more depressive by year.
# 
# As news,we can see more and more crime has happened
# 
# This may cause this chart that appears higher trend.

# ## Shooting crime summary and Visualization

# In[ ]:


original['SHOOTING'].unique()


# In[ ]:


original.shape


# In[ ]:


original['SHOOTING']=original['SHOOTING'].fillna('N')


# In[ ]:


original['SHOOTING'].unique()


# In[ ]:


original.head()


# In[ ]:


shootcrime=pd.pivot_table(original.loc[original['SHOOTING']=='Y',['YEAR','DISTRICT','SHOOTING']],                index='YEAR',columns='DISTRICT',aggfunc=np.count_nonzero)


# In[ ]:


sns.set()
p4=shootcrime.plot(title=r'2015-2018 crime has shooting bar plot',figsize=(15,14),kind='barh',stacked=True)

p4fig=p4.get_figure()
p4fig.savefig('./total_shooting_crime_barh.png')
plt.show()


# In[ ]:


districtSum=shootcrime.apply(np.sum)
districtSum=pd.DataFrame(districtSum)


# In[ ]:


districtSum=districtSum.rename(columns={0:r'shooting total'})


# In[ ]:


districtSum=districtSum.sort_values(by=r'shooting total',ascending=False)


# In[ ]:


top5=districtSum.iloc[0:5,:]
top5


# In[ ]:


sns.set()
p5=top5.plot(title=r'shooting crime top5 district',figsize=(15,14),kind='bar')
x=np.arange(top5.index.shape[0])
y=np.array(list(top5[r'shooting total']))
for i,j in zip(x,y):
    plt.text(i,j,'%d'%j,ha='center')
p5fig=p5.get_figure()
p5fig.savefig('./shooting_crime_top5_bar.png')
plt.show()


# ## shooting crime
# 
# As this chart,B2 is the highest
# 
# and B2's crime number is the highest
# 
# We may need to be alert with this distrcit
# 
# B3'crime is fifth,but shooting crime is second
# 
# With two charts,the more crime happened,the more shooting crime happened.

# ## Visualization-by Month

# In[ ]:


byMonth=original.groupby(by='MONTH')


# In[ ]:


Monthcount=byMonth.count()


# In[ ]:


MonthNumber=pd.DataFrame(Monthcount.iloc[:,0])
MonthNumber.head()


# In[ ]:


MonthNumber=MonthNumber.rename(columns={'INCIDENT_NUMBER':'NUMBER'})
sns.set()
p6=MonthNumber.plot(title=r'2015-2018 crime total by month bar plot',figsize=(15,14),kind='barh')
x=np.arange(MonthNumber.index.shape[0])
y=np.array(list(MonthNumber['NUMBER']))
for i,j in zip(x,y):
    plt.text(j,i,'%d'%j,ha='center')
p6fig=p6.get_figure()
p6fig.savefig('./total_by_month_bar.png')
plt.show()


# ## by Month
# 
# With this chart,July~August more crime happened
# 
# December~Febuary is less
# 
# Should draw a boxplot to see

# In[ ]:


Month=pd.pivot_table(original.loc[:,['YEAR','MONTH','INCIDENT_NUMBER']],                index='YEAR',columns='MONTH',aggfunc=np.count_nonzero)
Month


# In[ ]:


Month=Month.fillna(0)


# In[ ]:


Monthlist=(list(Month.iloc[:,0])),(list(Month.iloc[:,1])),(list(Month.iloc[:,2])), (list(Month.iloc[:,3])),(list(Month.iloc[:,4])),(list(Month.iloc[:,5])),(list(Month.iloc[:,6])), (list(Month.iloc[:,7])),(list(Month.iloc[:,8])),(list(Month.iloc[:,9])),(list(Month.iloc[:,10])), (list(Month.iloc[:,11]))
def takesecond(elem):
    x=[]
    for i in elem:
        x.append(i[1])
    else:
        return x
label=takesecond(Month.columns)
sns.set()
p1=plt.figure(figsize=(15,14))
plt.boxplot(Monthlist,labels=label,meanline=True)
plt.title(r'each Month change box plot')
plt.xlabel('Month')
plt.ylabel(r'crime number')
p1.savefig('./by_month_boxplot.png')
plt.show()


# ## boxplot-Month
# 
# 2015 January~May and 2018 November~December values are NaN,which is No record
# 
# 2018 October is less,which is seen as abnormal value
# This may be caused by No completed recording
# 
# July~Auguest has less change
# 
# With month bar chart,they may be crime's higher occurred months
# 
# And July in 2017 has high value,which is seen as abnormal value
# 
# Other Month have gentle change 

# ## Principal component analysis
# 
# find the most relative features
# 
# using PCA model to analysis

# In[ ]:


from sklearn.preprocessing import LabelEncoder
tras=original.iloc[:,:]
tras.loc[:,'OFFENSE_CODE_GROUP']=LabelEncoder().fit_transform(tras.loc[:,'OFFENSE_CODE_GROUP'])
tras.loc[:,'OFFENSE_DESCRIPTION']=LabelEncoder().fit_transform(tras.loc[:,'OFFENSE_DESCRIPTION'])
tras.loc[:,'DISTRICT']=LabelEncoder().fit_transform(tras.loc[:,'DISTRICT'].astype('str'))
tras.loc[:,'SHOOTING']=LabelEncoder().fit_transform(tras.loc[:,'SHOOTING'].astype('str'))
tras.loc[:,'DAY_OF_WEEK']=LabelEncoder().fit_transform(tras.loc[:,'DAY_OF_WEEK'])
tras.loc[:,'UCR_PART']=LabelEncoder().fit_transform(tras.loc[:,'UCR_PART'].astype('str'))
tras.loc[:,'STREET']=LabelEncoder().fit_transform(tras.loc[:,'STREET'].astype('str'))


# In[ ]:


tras.head()


# In[ ]:


tras.loc[:,'REPORTING_AREA']=LabelEncoder().fit_transform(tras.loc[:,'REPORTING_AREA'])


# In[ ]:


data=tras.loc[:,['OFFENSE_CODE', 'OFFENSE_CODE_GROUP','OFFENSE_DESCRIPTION',                  'DISTRICT', 'REPORTING_AREA', 'SHOOTING', 'YEAR','DAY_OF_WEEK', 'HOUR', 'UCR_PART', 'STREET']]
target=tras.loc[:,'MONTH']


# In[ ]:


from sklearn.decomposition import PCA
pcamodel=PCA(n_components=11).fit(data)


# In[ ]:


print(pcamodel.explained_variance_ratio_)
'''
top2
OFFENSE_CODE
OFFENSE_CODE_GROUP
'''


# ## PCA
# 
# With using PCA model,OFFENSE_CODE and OFFENSE_CODE_GROUP are main
# 
# ### 2019-02-20
# 
# Use pearson coeffient to analysis correlation
# 
# to choose features to predict

# ## correlation analysis-pearson coefficient

# In[ ]:


pdata=tras.loc[:,['OFFENSE_CODE', 'OFFENSE_CODE_GROUP','OFFENSE_DESCRIPTION',                  'DISTRICT', 'REPORTING_AREA', 'SHOOTING', 'YEAR','DAY_OF_WEEK', 'HOUR', 'UCR_PART', 'STREET','MONTH']]
pearsonMatrix=pd.DataFrame(round(pdata.corr(method='pearson'),6))
pearsonMatrix.sort_values(by='MONTH',ascending=False)


# ## pearson coefficient matrix
# 
# With using pearson coefficient matrix,
# 
# SHOOTING,DAT_OF_WEEK and HOUR are more relative to predict Month
# 
# try to use three features to predict

# ## predict Month-using GBC
# 
# using GradientBoostingClassifier

# In[ ]:


sample=tras.sample(n=10000)


# In[ ]:


features=sample.loc[:,['SHOOTING','DAY_OF_WEEK','HOUR']]
target=sample.loc[:,'MONTH']


# In[ ]:


from sklearn.model_selection import train_test_split
dataTrain,dataTest, targetTrain,targetTest= train_test_split(features,target,train_size=0.8)


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier as GBC


# In[ ]:


crimeGBC=GBC(max_depth=12)


# In[ ]:


crimeGBC.fit(dataTrain,targetTrain)


# In[ ]:


pred=crimeGBC.predict(dataTrain)


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(targetTrain,pred))


# In[ ]:


predict=crimeGBC.predict(dataTest)


# In[ ]:


print(classification_report(targetTest,predict))


# In[ ]:


from sklearn.metrics import accuracy_score
print(accuracy_score(targetTest,predict))


# ## predict Month-using GBC and PCA

# In[ ]:


pcadata=sample.loc[:,['OFFENSE_CODE', 'OFFENSE_CODE_GROUP','OFFENSE_DESCRIPTION',                  'DISTRICT', 'REPORTING_AREA', 'SHOOTING', 'YEAR','DAY_OF_WEEK', 'HOUR', 'UCR_PART', 'STREET']]
target=sample.loc[:,'MONTH']


# In[ ]:


Pcatras=PCA(n_components=2).fit_transform(pcadata)


# In[ ]:


pcaTrain,pcaTest, ptargetTrain,ptargetTest = train_test_split(Pcatras,target,train_size=0.8)


# In[ ]:


pcaGBC=GBC(max_depth=12).fit(pcaTrain,ptargetTrain)


# In[ ]:


pcapre=pcaGBC.predict(pcaTrain)
print(classification_report(ptargetTrain,pcapre))


# In[ ]:


pcapredict=pcaGBC.predict(pcaTest)
print(classification_report(ptargetTest,pcapredict))
print(accuracy_score(ptargetTest,pcapredict))


# ## predict Month-using SVC and three features

# In[ ]:


from sklearn.svm import SVC
crimeSVC=SVC()


# In[ ]:


crimeSVC.fit(dataTrain,targetTrain)


# In[ ]:


SVCpre=crimeSVC.predict(dataTrain)
print(classification_report(targetTrain,SVCpre))


# In[ ]:


SVCpredict=crimeSVC.predict(dataTest)
print(classification_report(targetTest,SVCpredict))
print(accuracy_score(targetTest,SVCpredict))


# ## predict Month-using SVC and PCA

# In[ ]:


pSVC=SVC().fit(pcaTrain,ptargetTrain)


# In[ ]:


pcaSVCpre=pSVC.predict(pcaTrain)
print(classification_report(ptargetTrain,pcaSVCpre))


# In[ ]:


pcaSVCpredict=pSVC.predict(pcaTest)
print(classification_report(ptargetTest,pcaSVCpredict))
print(accuracy_score(ptargetTest,pcaSVCpredict))


# ## predict Month-using RandomForestClassifier and PCA

# In[ ]:


from sklearn.ensemble import RandomForestClassifier as RFC
crimeRFC=RFC(max_depth=12)


# In[ ]:


crimeRFC


# In[ ]:


crimeRFC.fit(pcaTrain,ptargetTrain)


# In[ ]:


RFCpre=crimeRFC.predict(pcaTrain)
print(classification_report(ptargetTrain,RFCpre))


# In[ ]:


RFCpredict=crimeRFC.predict(pcaTest)
print(classification_report(ptargetTest,RFCpredict))
print(accuracy_score(ptargetTest,RFCpredict))


# With using PCA and three features,
# 
# use PCA data seems to be more effective than choose three features,
# 
# and use SVC model get highest accuracy score
# 
# 

# ## predict-offense_code_group 
# 
# ### choose features with pearson matrix

# In[ ]:


pearsonMatrix.sort_values(by='OFFENSE_CODE_GROUP',ascending=False)


# In[ ]:


offenseSample=tras.sample(n=10000)
odata=offenseSample.loc[:,['UCR_PART','YEAR','REPORTING_AREA']]
otarget=offenseSample.loc[:,'OFFENSE_CODE_GROUP']


# In[ ]:


odata.head()


# In[ ]:


odata['YEAR']=LabelEncoder().fit_transform(odata['YEAR'])


# In[ ]:


otarget.unique()


# ## predict offense_code_group-with three features and SVC

# In[ ]:


odataTrain,odataTest, otargetTrain,otargetTest = train_test_split(odata,otarget,train_size=0.8)


# In[ ]:


offenseSVC=SVC()


# In[ ]:


offenseSVC


# In[ ]:


offenseSVC.fit(odataTrain,otargetTrain)


# In[ ]:


offensepred=offenseSVC.predict(odataTrain)


# In[ ]:


print(classification_report(otargetTrain,offensepred))
print(accuracy_score(otargetTrain,offensepred))


# In[ ]:


offensepredict=offenseSVC.predict(odataTest)


# In[ ]:


print(classification_report(otargetTest,offensepredict))
print(accuracy_score(otargetTest,offensepredict))


# ## predict offense_code_group-PCA and SVC

# In[ ]:


opdata=offenseSample.loc[:,['OFFENSE_CODE', 'OFFENSE_CODE_GROUP','OFFENSE_DESCRIPTION',                  'DISTRICT', 'REPORTING_AREA', 'SHOOTING', 'YEAR','DAY_OF_WEEK', 'HOUR', 'UCR_PART', 'STREET','MONTH']]
optarget=offenseSample.loc[:,'OFFENSE_CODE_GROUP']


# In[ ]:


opcamodel=PCA(n_components=12).fit(opdata)


# In[ ]:


opcamodel.explained_variance_ratio_


# In[ ]:


opcadata=PCA(n_components=2).fit_transform(opdata)


# In[ ]:


opcadata.shape


# In[ ]:


opcaDataTrain,opcaDataTest, opcaTargetTrain,opcaTargetTest = train_test_split(opcadata,optarget,train_size=0.8)


# In[ ]:


pcaoffenseSVC=SVC().fit(opcaDataTrain,opcaTargetTrain)


# In[ ]:


pcapre=pcaoffenseSVC.predict(opcaDataTrain)
print(classification_report(opcaTargetTrain,pcapre))
print(accuracy_score(opcaTargetTrain,pcapre))


# In[ ]:


pcapredict=pcaoffenseSVC.predict(opcaDataTest)
print(classification_report(opcaTargetTest,pcapredict))
print(accuracy_score(opcaTargetTest,pcapredict))


# ## predict offense_code_group-using PCA and GBC

# In[ ]:


opcaGBC=GBC(max_depth=66).fit(opcaDataTrain,opcaTargetTrain)


# In[ ]:


opcaGBCpre=opcaGBC.predict(opcaDataTrain)
print(classification_report(opcaTargetTrain,opcaGBCpre))
print(accuracy_score(opcaTargetTrain,opcaGBCpre))


# In[ ]:


opcaGBCpredict=opcaGBC.predict(opcaDataTest)
print(classification_report(opcaTargetTest,opcaGBCpredict))
print(accuracy_score(opcaTargetTest,opcaGBCpredict))


# ## predict offense_code_group
# 
# With using PCA and GradientBoostingClassifier,
# 
# accuracy score is 89%,and some class's f1_score reach 1.00
# 
# So,best way may be using PCA and GradientBoostingClassifier

# In[ ]:




