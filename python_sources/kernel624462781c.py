#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data=pd.read_csv("../input/churn-dataset/Train File.csv")


# In[ ]:


data.head()


# In[ ]:


plt.hist(data['SeniorCitizen'])


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


data_1=data.dropna(subset=['TotalCharges'])


# In[ ]:


data_1.info()


# In[ ]:


data_1.reset_index(drop=True,inplace=True)


# In[ ]:


data_1.head()


# In[ ]:


data_1['OnlineSecurity']


# In[ ]:


Par=[]#Partner
DP=[]#DeviceProtection
CO=[]#Contract
Ch=[]#Churn
O=[]#Online Security
TP=[]
ST=[]
PS=[]
for i in data_1['OnlineSecurity'].values:
    if i=="Yes":
        O.append(1)
    if i=="No":
        O.append(0)
    if i=="No internet service":
        O.append(-1)
for i in data_1['PhoneService'].values:
    if i=="Yes":
        PS.append(1)
    else:
        PS.append(0)

for i in data_1['Churn'].values:
    if i=="Yes":
        Ch.append(1)
    else:
        Ch.append(0)
        
for i in data_1['Partner'].values:
    if i=="Yes":
        Par.append(1)
    else:
        Par.append(0)
for i in data_1['DeviceProtection'].values:
    if i=="Yes":
        DP.append(1)
    if i=="No":
        DP.append(0)
    if i=="No internet service":
        DP.append(-1)
for i in data_1['Contract'].values:
    if i=="Month-to-month":
        CO.append(1)
    if i=="One year":
        CO.append(0)
    if i=="Two year":
        CO.append(-1)


# In[ ]:


np.corrcoef(CO,Ch)


# In[ ]:


np.corrcoef(Par,Ch)


# In[ ]:


np.corrcoef(DP,Ch)


# In[ ]:


sns.boxplot(data_1['tenure'])


# In[ ]:


sns.boxplot(data_1['MonthlyCharges'])


# In[ ]:


sns.boxplot(data_1['TotalCharges'])


# In[ ]:


data_1.corr()


# In[ ]:


sns.heatmap(data_1.corr())


# In[ ]:


G=[]
PL=[]
for i in data_1['PaperlessBilling'].values:
    if i=="Yes":
        PL.append(1)
    else:
        PL.append(0)

for i in data_1['gender'].values:
    if i=="Female":
        G.append(1)
    else:
        G.append(0)


# In[ ]:


np.corrcoef(Ch,G)


# In[ ]:


np.corrcoef(O,G)


# In[ ]:


sns.heatmap(np.corrcoef(O,Ch))


# In[ ]:


sns.heatmap(np.corrcoef(O,DP))


# In[ ]:


np.corrcoef(O,DP)


# In[ ]:


np.corrcoef(O,TP)


# In[ ]:


np.corrcoef(O,ST)


# In[ ]:


data_1.head()


# In[ ]:


X=data_1[['gender','tenure','PhoneService','OnlineSecurity','PaperlessBilling','TotalCharges','Partner',"Contract"]]


# In[ ]:


Y=data_1['Churn']


# In[ ]:


X.shape,Y.shape


# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


Scaler=MinMaxScaler()


# In[ ]:


Scaler.fit(X.TotalCharges.values.reshape(-1,1))


# In[ ]:


X.TotalCharges.min(),X.TotalCharges.max(),Scaler.data_max_


# In[ ]:


X['TotalCharges']=Scaler.transform(X.TotalCharges.values.reshape(-1,1))


# In[ ]:


Scaler.fit(X.tenure.values.reshape(-1,1))
X['tenure']=Scaler.transform(X.tenure.values.reshape(-1,1))


# In[ ]:


X['PhoneService']=PS
X['gender']=G
X['OnlineSecurity']=O
X['PaperlessBilling']=PL


# In[ ]:


X['Partner']=Par
X['Contract']=CO


# In[ ]:


X.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg=LogisticRegression()


# In[ ]:


logreg.fit(X,Y)


# In[ ]:


logreg.intercept_


# In[ ]:


logreg.coef_


# In[ ]:


test=pd.read_csv('../input/churn-dataset/Test File2.csv')


# In[ ]:


test.head()


# In[ ]:


test.tenure.max(),test.tenure.min()


# In[ ]:


#	gender	tenure	PhoneService	OnlineSecurity	PaperlessBilling	TotalCharges
gen=[]
Pl=[]
PhoneS=[]
OS=[]

for i in test['PaperlessBilling'].values:
    if i=="Yes":
        Pl.append(1)
    else:
        Pl.append(0)

for i in test['gender'].values:
    if i=="Female":
        gen.append(1)
    else:
        gen.append(0)

for i in test['OnlineSecurity'].values:
    if i=="Yes":
        OS.append(1)
    if i=="No":
        OS.append(0)
    if i=="No internet service":
        OS.append(-1)

for i in test['PhoneService'].values:
    if i=="Yes":
        PhoneS.append(1)
    else:
        PhoneS.append(0)
        


# In[ ]:


X_test=test[['gender','tenure','PhoneService','OnlineSecurity','PaperlessBilling','TotalCharges','Partner',"Contract"]]


# In[ ]:


X_test.head()


# In[ ]:


Pr=[]
Co=[]
for i in X_test['Partner'].values:
    if i=="Yes":
        Pr.append(1)
    else:
        Pr.append(0)
for i in X_test['Contract'].values:
    if i=="Month-to-month":
        Co.append(1)
    if i=="One year":
        Co.append(0)
    if i=="Two year":
        Co.append(-1)


# In[ ]:


X_test['gender']=gen
X_test['PhoneService']=PhoneS
X_test['OnlineSecurity']=OS
X_test['PaperlessBilling']=Pl
X_test['Partner']=Pr
X_test['Contract']=Co


# In[ ]:


X_test.head()
#X_test.info()


# In[ ]:


test[test.TotalCharges.isna()]


# In[ ]:





# In[ ]:


Scaler.fit(X_test.TotalCharges.values.reshape(-1,1))


# In[ ]:


X_test['TotalCharges']=Scaler.transform(X_test.TotalCharges.values.reshape(-1,1))


# In[ ]:


X_test.head()


# In[ ]:


Scaler.fit(X_test.tenure.values.reshape(-1,1))


# In[ ]:


X_test['tenure']=Scaler.transform(X_test.tenure.values.reshape(-1,1))


# In[ ]:


X_test.head()


# In[ ]:


Y_pred=logreg.predict(X_test)


# In[ ]:


Y_pred


# In[ ]:


CustId=test['customerID']
frame={'customerID':CustId,'Churn':Y_pred}
Final=pd.DataFrame(frame)


# In[ ]:


Final.to_csv('AkashSubmission.csv',index=False)


# In[ ]:


F=pd.read_csv('AkashSubmission.csv')


# In[ ]:


type(CustId)


# In[ ]:


F.head()


# In[ ]:


F.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


data_test=pd.read_csv('G:/Sem5/ML/Soft Computing/Project/Decode2.0/Test File.csv')


# In[ ]:


data_test.info()


# In[ ]:


N=data_test[data_test['TotalCharges'].isnull()]


# In[ ]:


N


# In[ ]:


cut2=N['customerID']
ch=pd.Series(['No','Yes','No','Yes'])


# In[ ]:





# In[ ]:





# In[ ]:


import pandas as pd
Test File2 = pd.read_csv("../input/Test File2.csv")
Train File = pd.read_csv("../input/Train File.csv")

