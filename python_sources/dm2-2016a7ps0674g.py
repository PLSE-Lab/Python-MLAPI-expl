#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
df=pd.read_csv("../input/dmassignment2/train.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.replace({'?':None},inplace=True)


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);


# In[ ]:


df['Enrolled'].unique()


# In[ ]:


df.drop(['Weaks'],axis=1,inplace=True)


# In[ ]:


df['Schooling'].unique()


# In[ ]:


#df.drop(['PREV','Fill'],axis=1,inplace=True)


# In[ ]:


cols = ['COB FATHER','COB MOTHER','COB SELF','MIC','MOC','Worker Class','MSA','REG','MOVE']
for column in cols:
    mode = df[column].mode()[0]
    df[column] = df[column].fillna(mode)


# In[ ]:


df['COB FATHER']


# In[ ]:


# df.drop(['Gain','Loss','Stock'],axis=1,inplace=True)


# In[ ]:


#df.drop(['MIC','MOC'],axis=1,inplace=True)


# In[ ]:


df.drop(['Detailed'],axis=1,inplace=True)


# In[ ]:


#df.drop(['COB FATHER','COB MOTHER','COB SELF'],axis=1,inplace=True)


# In[ ]:


#df.drop(['MSA','REG','MOVE'],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df['Married_Life'].unique()


# In[ ]:


Worker_Class=pd.get_dummies(df['Worker Class'],drop_first=True)


# In[ ]:


Worker_Class.head()


# In[ ]:


Married_Life=pd.get_dummies(df['Married_Life'])


# In[ ]:


Married_Life.head()


# In[ ]:


Cast=pd.get_dummies(df['Cast'])


# In[ ]:


Cast.head()


# In[ ]:


Hispanic=pd.get_dummies(df['Hispanic'],drop_first=True)


# In[ ]:


Hispanic.head()


# In[ ]:


Sex=pd.get_dummies(df['Sex'])


# In[ ]:


Sex.head()


# In[ ]:


Tax_Status=pd.get_dummies(df['Tax Status'])


# In[ ]:


Tax_Status.head()


# In[ ]:


Live=pd.get_dummies(df['Live'],drop_first=True)


# In[ ]:


Live.head()


# In[ ]:


Teen=pd.get_dummies(df['Teen'],drop_first=True)


# In[ ]:


Teen.head()


# In[ ]:


Citizen=pd.get_dummies(df['Citizen'])


# In[ ]:


Citizen.head()


# In[ ]:


Prev=pd.get_dummies(df['PREV'],drop_first=True)


# In[ ]:


Fill=pd.get_dummies(df['Fill'],drop_first=True)


# In[ ]:


COB_FATHER=pd.get_dummies(df['COB FATHER'])
COB_MOTHER=pd.get_dummies(df['COB MOTHER'])
COB_SELF=pd.get_dummies(df['COB SELF'])
MIC=pd.get_dummies(df['MIC'])
MOC=pd.get_dummies(df['MOC'])


# In[ ]:


MSA=pd.get_dummies(df['MSA'])
REG=pd.get_dummies(df['REG'])
MOVE=pd.get_dummies(df['MOVE'])
Enrolled=pd.get_dummies(df['Enrolled'])
MLU=pd.get_dummies(df['MLU'])
Reason=pd.get_dummies(df['Reason'])
Area=pd.get_dummies(df['Area'])
State=pd.get_dummies(df['State'])
Full_Part=pd.get_dummies(df['Full/Part'])
Summary=pd.get_dummies(df['Summary'])
Schooling=pd.get_dummies(df['Schooling'])


# In[ ]:


df.drop(['Worker Class','Married_Life','Hispanic','Sex','Tax Status','Live','Teen','Citizen','PREV','Fill','COB FATHER',
        'COB MOTHER','COB SELF','MIC','MOC','MSA','REG','MOVE','Enrolled','MLU','Reason','Area','State','Full/Part',
         'Summary','Schooling'],axis=1,inplace=True)


# In[ ]:


df=pd.concat([df,Worker_Class,Married_Life,Hispanic,Sex,Tax_Status,Live,Teen,Citizen,Prev,Fill,COB_FATHER,COB_MOTHER,COB_SELF
             ,MIC,MOC,MSA,REG,MOVE,Enrolled,MLU,Reason,Area,State,Full_Part,
         Summary,Schooling]
             ,axis=1)


# In[ ]:


#Changes
df.drop(['Cast'],axis=1,inplace=True)
df=pd.concat([df,Cast],axis=1)


# In[ ]:


Y=df['Class']
X=df.drop(['Class'],axis=1)


# In[ ]:


df.info()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[ ]:


df


# In[ ]:


from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_train)
X_train = pd.DataFrame(np_scaled)
np_scaled_val = min_max_scaler.transform(X_val)
X_val = pd.DataFrame(np_scaled_val)
X_train.head()


# # Naive Bayes
# 

# In[ ]:


np.random.seed(42)


# In[ ]:


from sklearn.naive_bayes import GaussianNB as NB


# In[ ]:


nb = NB()
nb.fit(X_train,y_train)
nb.score(X_val,y_val)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_NB = nb.predict(X_val)
print(confusion_matrix(y_val, y_pred_NB))


# In[ ]:


print(classification_report(y_val, y_pred_NB))


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


roc_auc_score(y_val,y_pred_NB)


# # Logistic Regression
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lg = LogisticRegression(solver = 'liblinear', C = 8, multi_class = 'ovr', random_state = 42)
lg.fit(X_train,y_train)
lg.score(X_val,y_val)
roc_auc_score(y_val,lg.predict(X_val))


# In[ ]:


y_pred_LR = lg.predict(X_val)
print(confusion_matrix(y_val, y_pred_LR))


# In[ ]:


print(classification_report(y_val, y_pred_LR))


# In[ ]:


roc_auc_score(y_val,y_pred_LR)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


score_train_RF = []
score_test_RF = []

for i in range(1,18,1):
    rf = RandomForestClassifier(n_estimators=i, random_state = 42)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_val,y_val)
    score_test_RF.append(sc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')


# In[ ]:


rf = RandomForestClassifier(n_estimators=11, random_state = 42)
rf.fit(X_train, y_train)
rf.score(X_val,y_val)


# In[ ]:


y_pred_RF = rf.predict(X_val)


# In[ ]:


roc_auc_score(y_val,y_pred_RF)


# In[ ]:


print(classification_report(y_val, y_pred_RF))


# In[ ]:


dff=pd.read_csv('../input/test-1/test_1.csv')


# In[ ]:


ids=dff['ID']


# In[ ]:


dff.replace({'?':None},inplace=True)


# In[ ]:


for column in dff[['ID','Age','OC','Timely Income','Stock','NOP','Own/Self','WorkingPeriod','Weight']]:
    dff[column]=dff[column].astype(float) 


# In[ ]:


for column in dff[['COB FATHER','COB MOTHER','COB SELF','MIC','MOC','Worker Class','MSA','REG','MOVE']]:
    mode = dff[column].mode()
    dff[column] = dff[column].fillna(mode)


# In[ ]:


dff.drop(['Weaks','Detailed'],axis=1,inplace=True)
Worker_Class=pd.get_dummies(dff['Worker Class'],drop_first=True)
Married_Life=pd.get_dummies(dff['Married_Life'])
Hispanic=pd.get_dummies(dff['Hispanic'],drop_first=True)
Sex=pd.get_dummies(dff['Sex'])
Tax_Status=pd.get_dummies(dff['Tax Status'])
Live=pd.get_dummies(dff['Live'],drop_first=True)
Teen=pd.get_dummies(dff['Teen'],drop_first=True)
Citizen=pd.get_dummies(dff['Citizen'])
Cast=pd.get_dummies(dff['Cast'])
Prev=pd.get_dummies(dff['PREV'],drop_first=True)
Fill=pd.get_dummies(dff['Fill'],drop_first=True)
COB_FATHER=pd.get_dummies(dff['COB FATHER'])
COB_MOTHER=pd.get_dummies(dff['COB MOTHER'])
COB_SELF=pd.get_dummies(dff['COB SELF'])
MIC=pd.get_dummies(dff['MIC'])
MOC=pd.get_dummies(dff['MOC'])
MSA=pd.get_dummies(dff['MSA'])
REG=pd.get_dummies(dff['REG'])
MOVE=pd.get_dummies(dff['MOVE'])
Enrolled=pd.get_dummies(dff['Enrolled'])
MLU=pd.get_dummies(dff['MLU'])
Reason=pd.get_dummies(dff['Reason'])
Area=pd.get_dummies(dff['Area'])
State=pd.get_dummies(dff['State'])
Full_Part=pd.get_dummies(dff['Full/Part'])
Summary=pd.get_dummies(dff['Summary'])
Schooling=pd.get_dummies(dff['Schooling'])
dff.drop(['Worker Class','Married_Life','Hispanic','Sex','Tax Status','Live','Teen','Citizen','PREV','Fill','COB FATHER',
        'COB MOTHER','COB SELF','MIC','MOC','MSA','REG','MOVE','Enrolled','MLU','Reason','Area','State','Full/Part',
         'Summary','Schooling'],axis=1,inplace=True)
dff=pd.concat([dff,Worker_Class,Married_Life,Hispanic,Sex,Tax_Status,Live,Teen,Citizen,Prev,Fill,COB_FATHER,COB_MOTHER,COB_SELF
             ,MIC,MOC,MSA,REG,MOVE,Enrolled,MLU,Reason,Area,State,Full_Part,
         Summary,Schooling]
             ,axis=1)


# In[ ]:


dff.drop(['Cast'],axis=1,inplace=True)
dff=pd.concat([dff,Cast],axis=1)


# In[ ]:


Y=df['Class']
X=df.drop(['Class'],axis=1)


# In[ ]:


pred=[]


# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled)
np_scaled_val = min_max_scaler.transform(dff)
dff = pd.DataFrame(np_scaled_val)
X.head()


# In[ ]:


Nb = NB()
Nb.fit(X,Y)
pred=Nb.predict(dff)


# In[ ]:


lg.fit(X,Y)
pred=lg.predict(dff)


# In[ ]:


res=pd.DataFrame(pred)
final=pd.concat([ids,res],axis=1).reindex() 
final=final.rename(columns={0:'Class'})


# In[ ]:


final.head()


# In[ ]:


final.to_csv('Final.csv',index=False)
from IPython.display import HTML 
import pandas as pd 
import numpy as np 
import base64 
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):      
    csv = df.to_csv(index=False)     
    b64 = base64.b64encode(csv.encode())     
    payload = b64.decode()     
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'     
    html = html.format(payload=payload,title=title,filename=filename)     
    return HTML(html) 
    create_download_link(Final.csv) 
 
 

