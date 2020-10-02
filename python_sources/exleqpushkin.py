#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
df = pd.read_csv("../input/mnmnmnmn/EXL_EQ_2020_Train_datasets.csv")
df2 = df.copy()


# In[ ]:


df.drop('var30',axis=1,inplace=True)
df.drop('cust_id',axis=1,inplace=True)
df2.drop('var30',axis=1,inplace=True)
df2.drop('cust_id',axis=1,inplace=True)


# In[ ]:


def iqr(v):
    return np.subtract(*np.percentile(df[v],[75,25]))
def iqr2(v):
    return np.subtract(*np.percentile(df2[v],[75,25]))


# In[ ]:


def appiqr(v):
    return df.loc[(df[v]<np.percentile(df[v],75)+2*iqr(v))&(df[v]>np.percentile(df[v],25)-2*iqr(v))]
def appiqr2(v):
    return df2.loc[(df2[v]<np.percentile(df2[v],75)+2*iqr2(v))&(df2[v]>np.percentile(df2[v],25)-2*iqr2(v))]


# # categorical final: = var3, var6,var16, var32, var33, var34, var35,var36, var37 var 39, var40

# > continuous = 1,2,4,57,8,9,10,11,12,13,14,15,16,21,22,23,24,25,2627,28,29,31

# In[ ]:


r=[]
l =[1,2,4,5,7,8,9,10,11,12,13,14,15,16,21,22,23,24,25,26,27,28,29,31]
for i in l:
    v = 'var'+str(i)
    if(appiqr(v).shape[0]>0):
        print(v,appiqr(v).shape[0])


# In[ ]:


df['var3'] = df['var3'].astype('category')
df['var6'] = df['var6'].astype('category')
df['var16'] = df['var16'].astype('category')
df['var32'] = df['var32'].astype('category')
df['var33'] = df['var33'].astype('category')
df['var34'] = df['var34'].astype('category')
df['var35'] = df['var35'].astype('category')
df['var40'] = df['var40'].astype('category')
df['self_service_platform'] = df['self_service_platform'].astype('category')


# In[ ]:


df2['var3'] = df2['var3'].astype('category')
df2['var6'] = df2['var6'].astype('category')
df2['var16'] = df2['var16'].astype('category')
df2['var32'] = df2['var32'].astype('category')
df2['var33'] = df2['var33'].astype('category')
df2['var34'] = df2['var34'].astype('category')
df2['var35'] = df2['var35'].astype('category')
df2['var40'] = df2['var40'].astype('category')
df2['self_service_platform'] = df2['self_service_platform'].astype('category')


# # var36, var37 and var39 are left

# In[ ]:


# print(df.dtypes)


# # dropping var38

# In[ ]:





# In[ ]:


df.drop('var38',axis=1, inplace=True)
df2.drop('var38',axis=1, inplace=True)


# In[ ]:


df.head(1)


# In[ ]:





# In[ ]:


#var24,var36,var37,
#var39: prediction
#var37: prediction
#var24: considering continuous


# In[ ]:





# In[ ]:





# # filling var 24

# In[ ]:


appiqr2('var24').shape[0]


# In[ ]:


df['var24'] = df['var24'].fillna(df['var24'].mean())
df2['var24'] = df2['var24'].fillna(df2['var24'].mean())


# # filling 39

# In[ ]:


df['var39']=df['var39'].fillna('Single Housing')
df2['var39']=df2['var39'].fillna('Single Housing')


# In[ ]:


df['var39'] = df['var39'].astype('category')
df2['var39'] = df2['var39'].astype('category')


# # converting categorical to columns

# In[ ]:


df = pd.get_dummies(df,prefix=['var3'],columns=['var3'])
df = pd.get_dummies(df,prefix=['var6'],columns=['var6'])
df = pd.get_dummies(df,prefix=['var16'],columns=['var16'])
df = pd.get_dummies(df,prefix=['var32'],columns=['var32'])
df = pd.get_dummies(df,prefix=['var33'],columns=['var33'])
df = pd.get_dummies(df,prefix=['var34'],columns=['var34'])
df = pd.get_dummies(df,prefix=['var35'],columns=['var35'])
df = pd.get_dummies(df,prefix=['var39'],columns=['var39'])
df = pd.get_dummies(df,prefix=['var40'],columns=['var40'])
df = pd.get_dummies(df,prefix=['self_service_platform'],columns=['self_service_platform'])

# df2 = pd.get_dummies(df2,prefix=['var3'],columns=['var3'])
# df2 = pd.get_dummies(df2,prefix=['var6'],columns=['var6'])
# df2 = pd.get_dummies(df2,prefix=['var16'],columns=['var16'])
# df2 = pd.get_dummies(df2,prefix=['var32'],columns=['var32'])
# df2 = pd.get_dummies(df2,prefix=['var33'],columns=['var33'])
# df2 = pd.get_dummies(df2,prefix=['var34'],columns=['var34'])
# df2 = pd.get_dummies(df2,prefix=['var35'],columns=['var35'])
# df2 = pd.get_dummies(df2,prefix=['var39'],columns=['var39'])
# df2 = pd.get_dummies(df2,prefix=['var40'],columns=['var40'])
# df2 = pd.get_dummies(df2,prefix=['self_service_platform'],columns=['self_service_platform'])


# In[ ]:


df.columns


# # Filling 37 by making a model 

# In[ ]:


import xgboost as xgb


# In[ ]:


# df['var36']


# In[ ]:


df.columns


# In[ ]:


df1 = df.dropna()
#df1 = pd.get_dummies(df1,prefix=['var36'],columns=['var36'])


# In[ ]:


df1.columns


# In[ ]:


df1 = df1.drop('var36',axis=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
df1['var37'] = LabelEncoder().fit_transform(df1['var37'])


# In[ ]:


# df.columns


# In[ ]:


# j = 0
# for i in df1.dtypes:
#     print(df1.dtypes.index[j],i)
#     j+=1


# In[ ]:


# for i in df1.dtypes:
#     print(i)


# In[ ]:


df1.columns


# In[ ]:


X = df1[['var1', 'var2', 'var4', 'var5', 'var7', 'var8', 'var9',
       'var10', 'var11', 'var12', 'var13', 'var14', 'var15', 'var21', 'var22',
       'var23', 'var24', 'var25', 'var26', 'var27', 'var28', 'var29', 'var31',
       'var3_0', 'var3_1', 'var3_2', 'var3_3', 'var3_4', 'var6_0',
       'var6_1', 'var6_2', 'var6_3', 'var6_4', 'var6_5', 'var6_6', 'var6_7',
       'var6_8', 'var6_9', 'var16_0', 'var16_1', 'var32_0', 'var32_1',
       'var33_entertainment_channel1', 'var33_entertainment_channel2',
       'var33_movie_channel1', 'var33_news_channel1', 'var33_news_channel2',
       'var33_other', 'var34_Active', 'var34_Cancelled', 'var34_Inactive',
       'var34_Never', 'var34_Pending', 'var35_CreditCard',
       'var35_Electronic Transfer', 'var35_Standard', 'var39_Commercial',
       'var39_Exclude mapping', 'var39_Multi Housing', 'var39_Other',
       'var39_Single Housing', 'var40_N', 'var40_Y',
       'self_service_platform_Desktop', 'self_service_platform_Mobile App',
       'self_service_platform_Mobile Web', 'self_service_platform_STB']]
y = df1['var37']


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)


# In[ ]:


D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)


# In[ ]:


param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 3} 

steps = 20  # The number of training iterations


# In[ ]:


model = xgb.train(param, D_train, steps)


# In[ ]:


import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score

preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(Y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(Y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(Y_test, best_preds)))


# In[ ]:


df.loc[df['var37'].isna()]


# In[ ]:


ct = df.loc[df['var37'].isna()][X.columns]
D_test = xgb.DMatrix(ct, label=Y_test)
preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])


# In[ ]:


# df.var37.iloc[]
ind = np.array(df.loc[df['var37'].isna()].index)


# In[ ]:



df.var37.iloc[ind]


# In[ ]:


k = []
for i in range(len(best_preds)):
    if(i==0):
        k.append('N')
    else:
        k.append('Y')


# In[ ]:


# df.var37.iloc[df.loc[df['var37'].isna().index] 
df.var37.iloc[df.loc[df['var37'].isna()].index] = np.array(k)


# In[ ]:


df['var37'].value_counts()


# In[ ]:


df = pd.get_dummies(df,prefix=['var37'],columns=['var37'])


# In[ ]:


# for i in range(len(df.dtypes)):
#     print(df.dtypes.index[i],df.dtypes[i])


# 

# In[ ]:


df['var36']=df['var36'].fillna('video/internet')


# In[ ]:


def isvideo(arr):
    if "video" in arr:
        return 1
    else:
        return 0
def isinternet(arr):
    if "internet" in arr:
        return 1
    else:
        return 0
def isvoice(arr):
    if "voice" in arr:
        return 1
    else:
        return 0
def ishomesecurity(arr):
    if 'homesecurity' in arr:
        return 1
    else:
        return 0


# In[ ]:


df['var36'] = df['var36'].str.lower()


# In[ ]:


df['video'] = df['var36'].apply(lambda x: isvideo(str(x).split('/')))
df['internet'] = df['var36'].apply(lambda x: isinternet(str(x).split('/')))
df['voice'] = df['var36'].apply(lambda x: isvoice(str(x).split('/')))
df['homesecurity'] = df['var36'].apply(lambda x: ishomesecurity(str(x).split('/')))


# In[ ]:


df = df.drop('var36',axis=1)


# In[ ]:


# for i in range(len(df.dtypes)):
#     print(df.dtypes.index[i],df.dtypes[i])


# In[ ]:


# for i in df.isnull().sum():
#     if(i>0):
#         break
#         print(1)


# In[ ]:


X = df[['var1', 'var2', 'var4', 'var5', 'var7', 'var8', 'var9',
       'var10', 'var11', 'var12', 'var13', 'var14', 'var15', 'var21', 'var22',
       'var23', 'var24', 'var25', 'var26', 'var27', 'var28', 'var29', 'var31',
       'var3_0', 'var3_1', 'var3_2', 'var3_3', 'var3_4', 'var6_0', 'var6_1',
       'var6_2', 'var6_3', 'var6_4', 'var6_5', 'var6_6', 'var6_7', 'var6_8',
       'var6_9', 'var16_0', 'var16_1', 'var32_0', 'var32_1',
       'var33_entertainment_channel1', 'var33_entertainment_channel2',
       'var33_movie_channel1', 'var33_news_channel1', 'var33_news_channel2',
       'var33_other', 'var34_Active', 'var34_Cancelled', 'var34_Inactive',
       'var34_Never', 'var34_Pending', 'var35_CreditCard',
       'var35_Electronic Transfer', 'var35_Standard', 'var39_Commercial',
       'var39_Exclude mapping', 'var39_Multi Housing', 'var39_Other',
       'var39_Single Housing', 'var40_N', 'var40_Y',
       'var37_N', 'var37_Y', 'video', 'internet', 'voice', 'homesecurity']]


# In[ ]:


df2.columns


# In[ ]:


g = df2.copy()


# In[ ]:


# from sklearn.preprocessing import MultiLabelBinarizer
 y = g['self_service_platform']
# mlb = MultiLabelBinarizer()
# #y = mlb.fit_transform(y)


# In[ ]:


def mlb(s):
    if s=='Desktop':
        return 0
    if s=='Mobile App':
        return 1
    if s=='Mobile Web':
        return 2
    if s=='STB':
        return 3


# In[ ]:


y = y.apply(lambda x: mlb(str(x)))


# In[ ]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, random_state=42, stratify=y)


# In[ ]:


import xgboost as xgb
dtrain = xgb.DMatrix(data=X_train1, label=y_train1)
dtest = xgb.DMatrix(data=X_test1)


# In[ ]:


from sklearn.metrics import classification_report
params = {
    'max_depth': 7,
    'objective': 'multi:softmax',  # error evaluation for multiclass training
    'num_class': 4,
    'n_gpus': 0
}
bst = xgb.train(params, dtrain)
pred = bst.predict(dtest)
print(classification_report(y_test1, pred))


# In[ ]:


p = pd.read_csv('../input/mamamama/EXL_EQ_2020_Test_Datasets.csv')


# In[ ]:


p.drop(['cust_id','var38','var30'],axis=1,inplace =True)


# In[ ]:


p['var3'] = p['var3'].astype('category')
p['var6'] = p['var6'].astype('category')
p['var16'] = p['var16'].astype('category')
p['var32'] = p['var32'].astype('category')
p['var33'] = p['var33'].astype('category')
p['var34'] = p['var34'].astype('category')
p['var35'] = p['var35'].astype('category')
p['var37'] = p['var37'].astype('category')
p['var39'] = p['var39'].astype('category')
p['var40'] = p['var40'].astype('category')


# In[ ]:


p = pd.get_dummies(p,prefix=['var3'],columns=['var3'])
p = pd.get_dummies(p,prefix=['var6'],columns=['var6'])
p = pd.get_dummies(p,prefix=['var16'],columns=['var16'])
p = pd.get_dummies(p,prefix=['var32'],columns=['var32'])
p = pd.get_dummies(p,prefix=['var33'],columns=['var33'])
p = pd.get_dummies(p,prefix=['var34'],columns=['var34'])
p = pd.get_dummies(p,prefix=['var35'],columns=['var35'])
p = pd.get_dummies(p,prefix=['var39'],columns=['var39'])
p = pd.get_dummies(p,prefix=['var40'],columns=['var40'])
p = pd.get_dummies(p,prefix=['var37'],columns=['var37'])


# In[ ]:


p['video'] = p['var36'].apply(lambda x: isvideo(str(x).split('/')))
p['internet'] = p['var36'].apply(lambda x: isinternet(str(x).split('/')))
p['voice'] = p['var36'].apply(lambda x: isvoice(str(x).split('/')))
p['homesecurity'] = p['var36'].apply(lambda x: ishomesecurity(str(x).split('/')))


# In[ ]:


p = p.drop('var36',axis=1)


# In[ ]:


p.columns


# In[ ]:


dtest1 = xgb.DMatrix(data=p)
pred = bst.predict(dtest1)


# In[ ]:


pred = pd.DataFrame(pred)
pred[0] = pred[0]+1
pred.to_excel('pushkin.xlsx')


# In[ ]:


pred[0].value_counts()


# In[ ]:


df


# In[ ]:


#sum(df.isnull().sum())


# # Data Analysis

# # Correlation between continuous columns

# In[ ]:


cont_l = [1,2,4,5,7,8,9,10,11,12,13,14,15,16,21,22,23,24,25,26,27,28,29,31]


# In[ ]:


contl = []
for i in cont_l:
    contl.append('var'+str(i))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt 
plt.figure(figsize=(20,10))
sns.heatmap(df2[contl].corr(),annot=True)


# This gives us the following conclusions
# 1. Account tenure depends on the wealth conditions of the family and tenure for which video product is subscribed.
# 2. Internet connect status = 1 is related to the number of ticket raised by customer and number of on demand videos watched
# 3. Number of videos watched is correlated to video tickets raised
# 4. Number of emails opened depends on the number of times internet connect status is 1.
# 5. Number of tickets raised is related to the number of times the email is opened
# 6. count of emails opened is corelated to the bumber of nigital devices owned by the customer
# 7. More the Median home value in the area of housingunit, more the total duration of news channel watched 
# 8. More the Median home value in the area of housing unit, more the tenure for which video product is subscribed
# 9. More the Median home value in the area of housing unit more tickets raised
# 10. More the Median home value in the area of housing unit more the % trades delinquent by 90 or more days
# 11. More the likelihood to use credit card, more the tenure for which video product is subscribed.
# 12. more No. of digital video recorders, more the no. of tickets raised related to video product.
# 13. more the tenure for which video is purchased, more the % of trades delinquent.
# 14. more the likelihood of using a credit card, more the % trades delinquent
# 15. More the family income, more the likelihood of using a credit card.

# 1. Family income is an important point. Tenure, total duration of news channel, number of tickets,%trades delinquent, digital equipments owned by the customer factors depend on the income.
# 2. 

# In[ ]:


df2


# In[ ]:


plt.figure(figsize=(20,10))
d = sns.distplot(df2['var1']/365)
plt.xticks(np.arange(min(df2['var1']/365), max(df2['var1']/365)+1, 3))


# In[ ]:


g = ['var3', 'var6','var16', 'var32', 'var33', 'var34', 'var35','var36', 'var37','var39', 'var40']


# In[ ]:


import pandas as pd
v = pd.read_csv("../input/variables101/EXL_EQ_2020_Train_datasets.csv")
dictv = {}
for i in v.index:
    dictv[v['Feature'][i]]=v['Description'][i]


# # Variation of continuous variables with categorical

# In[ ]:


g.append('self_service_platform')


# In[ ]:


def box(i):
    for j in g:
        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(15,6))
        plt.xticks(rotation=90)
        ax = sns.boxplot(x=j, y=i,data=df2)
        plt.title(i+'/'+j)
        plt.xlabel(dictv[j])
        plt.ylabel(dictv[i])


# In[ ]:


box('var1')


# 1. More the number of wireless equipment, more the tenure
# 2. Higher the social position, higher the tenure and the increase is approximately exponential.
# 3. Customer subscribed for spanish video have much higher tenure than others.
# 4. Tenure is more if there is an active offer thus mostly number of inactive offers are from new members.
# 5. t

# In[ ]:


box('var2')


# 

# In[ ]:


box('var4')


# 

# In[ ]:


box('var5')


# 

# In[ ]:


box('var7')


# 

# In[ ]:


box('var8')


# 

# In[ ]:


box('var9')


# 

# In[ ]:


box('var10')


# 

# In[ ]:


box('var11')


# 

# In[ ]:


box('var12')


# 

# In[ ]:


box('var13')


# 

# In[ ]:


box('var14')


# 

# In[ ]:


box('var15')


# 

# In[ ]:


contl = ['var1',
 'var2',
 'var4',
 'var5',
 'var7',
 'var8',
 'var9',
 'var10',
 'var11',
 'var12',
 'var13',
 'var14',
 'var15',
 'var21',
 'var22',
 'var23',
 'var24',
 'var25',
 'var26',
 'var27',
 'var28',
 'var29',
 'var31']


# In[ ]:


box('var21')


# 

# In[ ]:


box('var22')


# 

# In[ ]:


box('var23')


# 

# In[ ]:


box('var24')


# 

# In[ ]:


box('var25')


# 

# In[ ]:


box('var26')


# 

# In[ ]:


box('var27')


# 

# In[ ]:


box('var28')


# 

# In[ ]:


box('var29')


# 

# In[ ]:


box('var31')


# 

# In[ ]:


v


# In[ ]:




