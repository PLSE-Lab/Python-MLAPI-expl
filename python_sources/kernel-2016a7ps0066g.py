#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data_orig = pd.read_csv("../input/train.csv", sep=',')
data_test = pd.read_csv("../input/test.csv", sep=',')
data = data_orig


# In[ ]:


data.info()


# In[ ]:


for i in range (1,9):
    wcs='WC'+str(i)
    data.loc[data['Worker Class']==wcs,'Worker Class']=i
    data_test.loc[data_test['Worker Class']==wcs,'Worker Class']=i
    
data.loc[data['Worker Class']=='?','Worker Class']=0
data_test.loc[data_test['Worker Class']=='?','Worker Class']=0


# In[ ]:


for i in range (1,18):
    wcs='Edu'+str(i)
    data.loc[data['Schooling']==wcs,'Schooling']=i
    data_test.loc[data_test['Schooling']==wcs,'Schooling']=i
    
data.groupby(['Schooling']).count()


# In[ ]:


lis=[1,3,4,7,9,10,13,14,16,17]
for i in lis:
    data.loc[data['Schooling']==i,'Schooling']=1
    data_test.loc[data_test['Schooling']==i,'Schooling']=1
    
data.groupby(['Schooling']).count()


# In[ ]:


data.loc[data['Schooling']==5,'Schooling']=3
data.loc[data['Schooling']==6,'Schooling']=4
data.loc[data['Schooling']==8,'Schooling']=5
data.loc[data['Schooling']==11,'Schooling']=6
data.loc[data['Schooling']==12,'Schooling']=7
data.loc[data['Schooling']==15,'Schooling']=8

data_test.loc[data_test['Schooling']==5,'Schooling']=3
data_test.loc[data_test['Schooling']==6,'Schooling']=4
data_test.loc[data_test['Schooling']==8,'Schooling']=5
data_test.loc[data_test['Schooling']==11,'Schooling']=6
data_test.loc[data_test['Schooling']==12,'Schooling']=7
data_test.loc[data_test['Schooling']==15,'Schooling']=8


# In[ ]:


data2=data
data2.loc[data2['Timely Income']>0,'Timely Income']=1
data_test.loc[data_test['Timely Income']>0,'Timely Income']=1


# In[ ]:


data2.loc[data2.Enrolled=='?','Enrolled']=0
data2.loc[data2.Enrolled=='Uni1','Enrolled']=1
data2.loc[data2.Enrolled=='Uni2','Enrolled']=2

data_test.loc[data_test.Enrolled=='?','Enrolled']=0
data_test.loc[data_test.Enrolled=='Uni1','Enrolled']=1
data_test.loc[data_test.Enrolled=='Uni2','Enrolled']=2


# In[ ]:


for i in range(1,8):
    ss='MS'+str(i)
    data2.loc[data.Married_Life==ss,'Married_Life']=i
    data_test.loc[data_test.Married_Life==ss,'Married_Life']=i


# In[ ]:


data2.loc[(data2.MIC=='?') | (data2.MIC=='MIC_G') | (data2.MIC=='MIC_O') | (data2.MIC=='MIC_P') | (data2.MIC=='MIC_U'),'MIC']=1
data_test.loc[(data_test.MIC=='?') | (data_test.MIC=='MIC_G') | (data_test.MIC=='MIC_O') | (data_test.MIC=='MIC_P') | (data_test.MIC=='MIC_U'),'MIC']=1


# In[ ]:


data2.loc[ data2.MIC!=1 ,'MIC']=2
data_test.loc[ data_test.MIC!=1 ,'MIC']=2


# In[ ]:


data2.loc[(data2.MOC=='?') | (data2.MOC=='MOC_D') | (data2.MOC=='MOC_G') | (data2.MOC=='MOC_I') | (data2.MOC=='MOC_L'),'MOC']=1
data_test.loc[(data_test.MOC=='?') | (data_test.MOC=='MOC_D') | (data_test.MOC=='MOC_G') | (data_test.MOC=='MOC_I') | (data_test.MOC=='MOC_L'),'MOC']=1

#data2.groupby(['MOC','Class']).count()


# In[ ]:


data2.loc[ data2.MOC!=1 ,'MOC']=2
data_test.loc[ data_test.MOC!=1 ,'MOC']=2

#data2.groupby(['MOC','Class']).count()


# In[ ]:


data2.loc[data2.Cast=='TypeA','Cast']=0
data2.loc[data2.Cast=='TypeB','Cast']=1
data2.loc[data2.Cast=='TypeC','Cast']=2
data2.loc[data2.Cast=='TypeD','Cast']=3
data2.loc[data2.Cast=='TypeE','Cast']=4

data_test.loc[data_test.Cast=='TypeA','Cast']=0
data_test.loc[data_test.Cast=='TypeB','Cast']=1
data_test.loc[data_test.Cast=='TypeC','Cast']=2
data_test.loc[data_test.Cast=='TypeD','Cast']=3
data_test.loc[data_test.Cast=='TypeE','Cast']=4

#data2.groupby(['Cast','Class']).count()


# In[ ]:


data2.loc[data2.Hispanic=='HA','Hispanic']=1
data2.loc[data2.Hispanic!=1,'Hispanic']=2

data_test.loc[data_test.Hispanic=='HA','Hispanic']=1
data_test.loc[data_test.Hispanic!=1,'Hispanic']=2

#data2.groupby(['Hispanic','Class']).count()


# In[ ]:


data2.loc[data2.Sex=='M','Sex']=0
data2.loc[data2.Sex=='F','Sex']=1

data_test.loc[data_test.Sex=='M','Sex']=0
data_test.loc[data_test.Sex=='F','Sex']=1


# In[ ]:


data2.loc[data2.MLU=='?','MLU']=0
data2.loc[data2.MLU=='NO','MLU']=1
data2.loc[data2.MLU=='YES','MLU']=2

data_test.loc[data_test.MLU=='?','MLU']=0
data_test.loc[data_test.MLU=='NO','MLU']=1
data_test.loc[data_test.MLU=='YES','MLU']=2


# In[ ]:


for i in range(1,6):
    jj='JL'+str(i)
    data2.loc[data2.Reason==jj,'Reason']=i-1
    data_test.loc[data_test.Reason==jj,'Reason']=i-1
    
data2.loc[data2.Reason=='?','Reason']=5
data_test.loc[data_test.Reason=='?','Reason']=5


# In[ ]:


data2.loc[data2['Full/Part']=='FB','Full/Part']=1
data2.loc[data2['Full/Part']=='FC','Full/Part']=2
data2.loc[data2['Full/Part']=='FF','Full/Part']=3
data2.loc[data2['Full/Part']=='FG','Full/Part']=4
data2.loc[(data2['Full/Part']!=1) & (data2['Full/Part']!=2) & (data2['Full/Part']!=3) & (data2['Full/Part']!=4) ,'Full/Part']=5

data_test.loc[data_test['Full/Part']=='FB','Full/Part']=1
data_test.loc[data_test['Full/Part']=='FC','Full/Part']=2
data_test.loc[data_test['Full/Part']=='FF','Full/Part']=3
data_test.loc[data_test['Full/Part']=='FG','Full/Part']=4
data_test.loc[(data_test['Full/Part']!=1) & (data_test['Full/Part']!=2) & (data_test['Full/Part']!=3) & (data_test['Full/Part']!=4) ,'Full/Part']=5


#data2.groupby(['Full/Part','Class']).count()


# In[ ]:


bins = [-1,1,7000, 20000, 100005]
labels = [1,2,3,4]
data2['Gain'] = pd.cut(data2['Gain'], bins=bins, labels=labels)
data_test['Gain'] = pd.cut(data_test['Gain'], bins=bins, labels=labels)


# In[ ]:


data2.groupby(['Gain','Class']).count()


# In[ ]:


for i in range (1,7):
    ss='J'+str(i)
    data2.loc[data2['Tax Status']==ss,'Tax Status']=i-1
    data_test.loc[data_test['Tax Status']==ss,'Tax Status']=i-1
    
data2.groupby(['Tax Status','Class']).count()


# In[ ]:


data2=data2.drop(['State'],axis=1)
data_test=data_test.drop(['State'],axis=1)


# In[ ]:


data2.loc[data2.Area=='OUT','Area']=0
data2.loc[data2.Area=='NE','Area']=1
data2.loc[data2.Area=='MW','Area']=2
data2.loc[data2.Area=='W','Area']=3
data2.loc[data2.Area=='S','Area']=4
data2.loc[data2.Area=='?','Area']=5

data_test.loc[data_test.Area=='OUT','Area']=0
data_test.loc[data_test.Area=='NE','Area']=1
data_test.loc[data_test.Area=='MW','Area']=2
data_test.loc[data_test.Area=='W','Area']=3
data_test.loc[data_test.Area=='S','Area']=4
data_test.loc[data_test.Area=='?','Area']=5

data2.groupby(['Area','Class']).count()


# In[ ]:


data2=data2.drop(['Detailed'],axis=1)
data_test=data_test.drop(['Detailed'],axis=1)


# In[ ]:


for i in range(1,9):
    ss='sum'+str(i)
    data2.loc[data2.Summary==ss,'Summary']=i
    data_test.loc[data_test.Summary==ss,'Summary']=i

data2.groupby(['Summary','Class']).count()


# In[ ]:


data2.loc[data2.MSA=='StatusQ','MSA']=0
data2.loc[data2.MSA=='StatusN','MSA']=1
data2.loc[data2.MSA=='StatusO','MSA']=2
data2.loc[data2.MSA=='StatusJ','MSA']=3
data2.loc[data2.MSA=='StatusH','MSA']=4
data2.loc[data2.MSA=='StatusM','MSA']=5
data2.loc[data2.MSA=='StatusL','MSA']=6
data2.loc[data2.MSA=='StatusA','MSA']=7
data2.loc[data2.MSA=='?','MSA']=8

data_test.loc[data_test.MSA=='StatusQ','MSA']=0
data_test.loc[data_test.MSA=='StatusN','MSA']=1
data_test.loc[data_test.MSA=='StatusO','MSA']=2
data_test.loc[data_test.MSA=='StatusJ','MSA']=3
data_test.loc[data_test.MSA=='StatusH','MSA']=4
data_test.loc[data_test.MSA=='StatusM','MSA']=5
data_test.loc[data_test.MSA=='StatusL','MSA']=6
data_test.loc[data_test.MSA=='StatusA','MSA']=7
data_test.loc[data_test.MSA=='?','MSA']=8


# In[ ]:


data2=data2.drop(['REG'],axis=1)
data_test=data_test.drop(['REG'],axis=1)


# In[ ]:


data2=data2.drop(['MOVE'],axis=1)
data_test=data_test.drop(['MOVE'],axis=1)


# In[ ]:


data2.loc[data2.Live=='?','Live']=0
data2.loc[data2.Live=='NO','Live']=1
data2.loc[data2.Live=='YES','Live']=2

data_test.loc[data_test.Live=='?','Live']=0
data_test.loc[data_test.Live=='NO','Live']=1
data_test.loc[data_test.Live=='YES','Live']=2

#data2.groupby(['Live','Class']).count()


# In[ ]:


data2.loc[data2.PREV=='YES','PREV']=0
data2.loc[data2.PREV=='NO','PREV']=1
data2.loc[data2.PREV=='?','PREV']=2

data_test.loc[data_test.PREV=='YES','PREV']=0
data_test.loc[data_test.PREV=='NO','PREV']=1
data_test.loc[data_test.PREV=='?','PREV']=2


# In[ ]:


data2.loc[data2.Teen=='?','Teen']=0
data2.loc[data2.Teen=='B','Teen']=1
data2.loc[data2.Teen=='F','Teen']=2
data2.loc[data2.Teen=='M','Teen']=3
data2.loc[data2.Teen=='N','Teen']=4

data_test.loc[data_test.Teen=='?','Teen']=0
data_test.loc[data_test.Teen=='B','Teen']=1
data_test.loc[data_test.Teen=='F','Teen']=2
data_test.loc[data_test.Teen=='M','Teen']=3
data_test.loc[data_test.Teen=='N','Teen']=4

#data2.groupby(['Teen','Class']).count()


# In[ ]:


data2=data2.drop(['COB FATHER','COB MOTHER','COB SELF'],axis=1)
data_test=data_test.drop(['COB FATHER','COB MOTHER','COB SELF'],axis=1)


# In[ ]:


for i in range(1,6):
    cc='Case'+str(i)
    data2.loc[data2.Citizen==cc,'Citizen']=i
    data_test.loc[data_test.Citizen==cc,'Citizen']=i
data2.groupby(['Citizen','Class']).count() 


# In[ ]:


data2.loc[data2.Fill=='YES','Fill']=0
data2.loc[data2.Fill=='NO','Fill']=1
data2.loc[data2.Fill=='?','Fill']=2

data_test.loc[data_test.Fill=='YES','Fill']=0
data_test.loc[data_test.Fill=='NO','Fill']=1
data_test.loc[data_test.Fill=='?','Fill']=2


# In[ ]:


data2=data2.drop(['ID'],axis=1)
data_test2=data_test.drop(['ID'],axis=1)


# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(20, 16))
corr = data2.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);


# In[ ]:


data_test2['Age']=data_test2['Age'].astype(np.int64)
data_test2['IC']=data_test2['IC'].astype(np.int64)
data_test2['OC']=data_test2['OC'].astype(np.int64)
data_test2['Gain']=data_test2['Gain'].astype(np.int64)

data2['Age']=data2['Age'].astype(np.int64)
data2['IC']=data2['IC'].astype(np.int64)
data2['OC']=data2['OC'].astype(np.int64)
data2['Gain']=data2['Gain'].astype(np.int64)


# In[ ]:


from sklearn import preprocessing
#Performing Min_Max Normalization
col=data2.columns
col2=data_test2.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data2[col])
data2[col] = pd.DataFrame(np_scaled)

np_scaled = min_max_scaler.fit_transform(data_test2[col2])
data_test2[col2] = pd.DataFrame(np_scaled)


# In[ ]:


col=['Gain','Loss','Stock','Weight','Weaks','Age','IC','OC','Timely Income','NOP']
std_scaler= preprocessing.StandardScaler()
np_scaled = std_scaler.fit_transform(data2[col])
data2[col] = pd.DataFrame(np_scaled)

np_scaled = std_scaler.fit_transform(data_test2[col])
data_test2[col] = pd.DataFrame(np_scaled)


# In[ ]:


y=data2['Class']
X=data2.drop(['Class'],axis=1)
X.head()


# In[ ]:


#Oversampling more than under
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
ros = RandomOverSampler(random_state=42)
rus = RandomUnderSampler(random_state=42)
rus.fit(X, y)
X_resampled, y_resampled = rus.fit_resample(X,y)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.20, random_state=42)


# In[ ]:


from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_train)
X_train = pd.DataFrame(np_scaled)
np_scaled_val = min_max_scaler.transform(X_val)
X_val = pd.DataFrame(np_scaled_val)
#X_train.head()


# In[ ]:


#Validation Score for NB
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.metrics import roc_auc_score

np.random.seed(42)
nb = NB()
nb.fit(X_train,y_train)
y_pred_NB = nb.predict(X_val)
print(roc_auc_score(y_val, y_pred_NB))


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(classification_report(y_val, y_pred_NB))


# In[ ]:


X=data_test2
from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled_val = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled_val)
X.head()


# In[ ]:


y_pred_NB = nb.predict(X)
data_test['Class1']=y_pred_NB


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

train_acc = []
test_acc = []
for i in range(1,15):
    dTree = DecisionTreeClassifier(max_depth=i)
    dTree.fit(X_train,y_train)
    acc_train = dTree.score(X_train,y_train)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,y_val)
    test_acc.append(acc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Score", "Validation Score"])
plt.title('Score vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Score')


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

train_acc = []
test_acc = []
for i in range(2,50):
    dTree = DecisionTreeClassifier(max_depth = 7, min_samples_split=i, random_state = 42)
    dTree.fit(X_train,y_train)
    acc_train = dTree.score(X_train,y_train)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,y_val)
    test_acc.append(acc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(2,50),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(2,50),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Score", "Validation Score"])
plt.title('Score vs min_samples_split')
plt.xlabel('Max Depth')
plt.ylabel('Score')


# In[ ]:


#Decision Tree Cross - Validation Score
from tqdm import tqdm
fpred=[]
ma=0
bi=2
bj=11
for i in tqdm(range (2,45)):
    for j in range(10,50):
        dTree = DecisionTreeClassifier(max_depth=i, random_state = j,min_samples_split=37)
        dTree.fit(X_train,y_train)
        y_pred_DT = dTree.predict(X_val)
        if (roc_auc_score(y_val, y_pred_DT)>ma):
            ma=roc_auc_score(y_val, y_pred_DT)
            fpred=y_pred_DT
            bi=i
            bj=j
print(ma,bi,bj)


# In[ ]:


print(confusion_matrix(y_val, fpred))


# In[ ]:


dTree = DecisionTreeClassifier(max_depth=bi, random_state = bj,min_samples_split=37)
dTree.fit(X_train,y_train)
y_pred_DT = dTree.predict(X)
data_test['Class2']=y_pred_DT


# In[ ]:


#Cross - Validation for Random Forest
from sklearn.ensemble import RandomForestClassifier
ma=0
fpred=[]
bi=0
bj=0
for i in tqdm(range (1,30) ):
    for j in range(30):
        rf = RandomForestClassifier(n_estimators=i, random_state = j)
        rf.fit(X_train, y_train)
        y_pred_RF = rf.predict(X_val)
        if (ma<roc_auc_score(y_val, y_pred_RF)):
            ma=roc_auc_score(y_val, y_pred_RF)
            fpred=y_pred_RF
            bi=i
            bj=j
print(ma,bi,bj)


# In[ ]:


print(confusion_matrix(y_val, fpred))


# In[ ]:


rf = RandomForestClassifier(n_estimators=bi, random_state = bj)
rf.fit(X_train, y_train)
y_pred_RF = rf.predict(X)
data_test['Class3']=y_pred_RF


# In[ ]:


data_sub=data_test[['ID','Class3']]
#data_sub.info()


# In[ ]:


#Cross-Validation for Logistic Regression
from sklearn.linear_model import LogisticRegression

lg = LogisticRegression(max_iter=100000,solver = 'lbfgs', C = 8, multi_class = 'multinomial', random_state = 42, class_weight={0:0.1,1:0.9})
lg.fit(X_train,y_train)
y_pred_LR = lg.predict(X_val)
print(roc_auc_score(y_val, y_pred_LR))


# In[ ]:


data_test['Class4']=lg.predict(X)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

train_acc = []
test_acc = []
for i in tqdm(range(1,20)):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    acc_train = knn.score(X_train,y_train)
    train_acc.append(acc_train)
    acc_test = knn.score(X_val,y_val)
    test_acc.append(acc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,20),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,20),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Score", "Test Score"])
plt.title('Score vs K neighbors')
plt.xlabel('K neighbors')
plt.ylabel('Score')


# In[ ]:


#Cross Validation kNN
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)
y_pred_knn=knn.predict(X_val)
print(roc_auc_score(y_val,y_pred_knn))


# In[ ]:


data_test['Class5']=knn.predict(X)


# In[ ]:


#Selecting best n_estimators on the basis of cross validation score
from sklearn.ensemble import AdaBoostClassifier
ma=0
fpred=[]
bi=0
bj=0
train_acc = []
test_acc = []
for i in tqdm(range (1,60) ):
    ad = AdaBoostClassifier(n_estimators=i, random_state = 42)
    ad.fit(X_train, y_train)
    acc_train = ad.score(X_train,y_train)
    train_acc.append(acc_train)
    acc_test = ad.score(X_val,y_val)
    test_acc.append(acc_test)
    y_pred_AB = ad.predict(X_val)
    if (ma<roc_auc_score(y_val, y_pred_AB)):
        ma=roc_auc_score(y_val, y_pred_AB)
        fpred=y_pred_AB
        bi=i
        bj=j
print(ma,bi)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,60),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,60),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Score", "Validation Score"])
plt.title('Score vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Score')


# In[ ]:


#Cross - Validation Adaboost
ma=0
bj=0
fpred=[]
for j in range (50):
    ad = AdaBoostClassifier(n_estimators=bi, random_state = j)
    ad.fit(X_train, y_train)
    y_pred_AB = ad.predict(X_val)
    if (ma<roc_auc_score(y_val, y_pred_AB)):
        ma=roc_auc_score(y_val, y_pred_AB)
        fpred=y_pred_AB
        bj=j
print(ma,bj)


# In[ ]:


ad = AdaBoostClassifier(n_estimators=44, random_state = 0)
ad.fit(X_train, y_train)
fpred = ad.predict(X)


# In[ ]:


data_test['Class6']=fpred


# In[ ]:


#Binning/Assigning weights to different results
lis=[]
for i in range (1,7):
    vv='Class'+str(i)
    lis.append(vv)
lis+=['Class2','Class3','Class3','Class3','Class6']
#print(lis)


# In[ ]:


pred = data_test[lis].mean(axis=1)


# In[ ]:


#Selecting threshold
data_test['Class']=pred
data_test.loc[data_test.Class<=0.55,'Class']=0
data_test.loc[data_test.Class>0.55,'Class']=1


# In[ ]:


data_sub=data_test[['ID','Class']]
data_sub.info()


# In[ ]:


data_sub['Class']=data_sub['Class'].astype(np.int64)


# In[ ]:


data_sub.groupby(['Class']).count()


# In[ ]:


data_sub.to_csv('final_sub.csv', encoding='utf-8', index=False)


# In[ ]:


from IPython.display import HTML import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"): csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode())
payload = b64.decode()
html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
html = html.format(payload=payload,title=title,filename=filename)
return HTML(html) create_download_link(data_sub)

