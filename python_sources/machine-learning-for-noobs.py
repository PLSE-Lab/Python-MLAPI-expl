#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[ ]:


# Load .csv to variables
breeds = pd.read_csv('../input/breed_labels.csv')
colors = pd.read_csv('../input/color_labels.csv')

train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')

train['dstype']='train'
test['dstype']='test'
all=pd.concat([train,test])
all=all.reset_index(drop=True)
train.drop('Description',axis=1,inplace=True)
test.drop('Description',axis=1,inplace=True)
all.drop('Description',axis=1,inplace=True)


# In[ ]:


# Fetch train dataset's infomation
all.info()


# As we can see, we have a dataset with 14993 data points, over a thousand name & description are missing but those should be trivial :D 

# In[ ]:


train.head()


# # Adoption Speed

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
data = train['AdoptionSpeed'].value_counts()
data.plot('barh')
for i,v in enumerate(data.values):
    plt.gca().text(v+50,i-0.1,str(v),color='teal',fontweight='bold')


# # Cat or dog

# In[ ]:


# Change type 1 to dog, 2 to cat
all['Type']=all['Type'].apply(lambda x:'Dog' if x==1 else 'Cat')
all[all.Name=="Brisco"]


# In[ ]:


plt.figure(figsize=(5,3))
sns.countplot(x='dstype',data=all,hue='Type')
plt.title('Amount of cats and dogs in test set and train set')


# # Age

# In[ ]:


# See the age of dogs & cats
train.Age.value_counts().head(10)


#  Mostly pets are young, there are pet with the age equal to multiples of 12. Maybe the owners didn't really bother with ages
# 

# In[ ]:


plt.figure(figsize=(10, 6))
sns.violinplot(x="AdoptionSpeed", y="Age", hue="Type", data=train)
plt.title('AdoptionSpeed by Type and age')


# The more young the pets are, the more likely they are getting adopted

# # Name

# Maybe pets with no names are less likely to get adopted because they haven't received loves from the owner (by the appearance maybe). Lets find out if it's true

# In[ ]:



# Set all cells in column 'Name' to 1 if there is 'no name' or blank (null)
for i,value in enumerate(all.Name):
    if(str(value).lower().find('no name')==0 ):
        all.at[i,'Name']=1
for i in all[all['Name'].isnull()].index:
    all.at[i,'Name']=1
# Set the remaining cells to 0
for i,value in enumerate(all.Name):
    if(str(value)!= '1' ):
        all.at[i,'Name']=0
# Rename Name to No-Name
all.rename(columns={'Name':'NoName'},inplace=True)


# In[ ]:


ax = sns.barplot(x="AdoptionSpeed", y="AdoptionSpeed", data=all[all.NoName==1], estimator=lambda x: len(x) / len(all[all.NoName==1]) * 100)
ax.set(ylabel="Percent")
ax.set_title('Adoption Speed for no names')


# In[ ]:


ax = sns.barplot(x="AdoptionSpeed", y="AdoptionSpeed", data=all[all.NoName==0], estimator=lambda x: len(x) / len(all[all.NoName==0]) * 100)
ax.set(ylabel="Percent")
ax.set_title('Adoption Speed for named pets')


# Looks like I was right! Not-named pets are less likely to get adopted compared to named pets by 5% :D

# # Quantity

# In[ ]:


all.groupby('Quantity').agg(['count','mean'])['AdoptionSpeed']


# The greater amount of pets in a group (an advertisement) the less likely of them to get adopted. Maybe they come in a group and are not treated as valuable as a single one

# # Breed

# In[ ]:


breeds[breeds.BreedName=='Mixed Breed']


# Sometimes they write in Breed2 to indicate Mixed Breed, sometimes they just straight up write BreedID=307 for indicating Mixed Breed

# In[ ]:


train['Pure']=0
train.loc[train['Breed2']==0,'Pure']=1
train.loc[train['Breed1']==307,'Pure']=0
test['Pure']=0
test.loc[test['Breed2']==0,'Pure']=1
test.loc[test['Breed1']==307,'Pure']=0
print('-Train:')
print('There are',len(train[train.Pure==1]), 'Pure Breed',len(train[train.Pure==1])/len(train)*100,"%")
print('There are',len(train[train.Pure==0]), 'Mixed Breed',len(train[train.Pure==0])/len(train)*100,"%")
print('-Test:')
print('There are',len(test[test.Pure==1]), 'Pure Breed',len(test[test.Pure==1])/len(test)*100,"%")
print('There are',len(test[test.Pure==0]), 'Mixed Breed',len(test[test.Pure==0])/len(test)*100,"%")
all['Pure']=0
all.loc[all['Breed2']==0,'Pure']=1
all.loc[all['Breed1']==307,'Pure']=0
all


# In[ ]:


train[train.Pure==1]['AdoptionSpeed'].mean()


# In[ ]:


train[train.Pure==0]['AdoptionSpeed'].mean()


# Pure Breed are likely to get adopted quicker than mixed breed

# # Gender

# In[ ]:


sns.factorplot('Type', col='Gender', data=all, kind='count', hue='dstype');
plt.subplots_adjust(top=0.8)
plt.suptitle('Count of cats and dogs in train and test set by gender');


# There are more Female in the data

# In[ ]:


sns.countplot(x='AdoptionSpeed',data=all,hue='Gender')


# In[ ]:


#One Hot Encoder
from sklearn.preprocessing import LabelBinarizer

LaBi = LabelBinarizer()

Breed1_lb=LaBi.fit_transform(all.Breed1)
Breed2_lb=LaBi.fit_transform(all.Breed2)
Type_lb=LaBi.fit_transform(all.Type)
Gender_lb=LaBi.fit_transform(all.Gender)
Vaccinated_lb=LaBi.fit_transform(all.Vaccinated)
Dewormed_lb = LaBi.fit_transform(all.Dewormed)
FurLength_lb = LaBi.fit_transform(all.FurLength)
Sterilized_lb = LaBi.fit_transform(all.Sterilized)
Health_lb = LaBi.fit_transform(all.Health)
Color1_lb = LaBi.fit_transform(all.Color1)
Color2_lb = LaBi.fit_transform(all.Color2)
Color3_lb = LaBi.fit_transform(all.Color3)
allLB=np.append(Breed1_lb,Breed2_lb,axis=1)
allLB=np.append(allLB,Type_lb,axis=1)
allLB=np.append(allLB,Gender_lb,axis=1)
allLB=np.append(allLB,Vaccinated_lb,axis=1)
allLB=np.append(allLB,Dewormed_lb,axis=1)
#allLB=np.append(allLB,FurLength_lb,axis=1)
allLB=np.append(allLB,Sterilized_lb,axis=1)
allLB=np.append(allLB,Health_lb,axis=1)
allLB=np.append(allLB,Color1_lb,axis=1)
allLB=np.append(allLB,Color2_lb,axis=1)
#allLB=np.append(allLB,Color3_lb,axis=1)
allLB.shape


# In[ ]:


all_mat=np.append(allLB,all[['Age','NoName','Pure','Quantity']].values.reshape(18941,4),axis=1)
all_mat=np.append(all_mat,all['AdoptionSpeed'].values.reshape(18941,1),axis=1)
train_x=all_mat[:14993,:-1]
train_y=all_mat[:14993,-1]
test_x=all_mat[14993:,:-1]
test_y=all_mat[14993:,-1]
all_mat.shape


# In[ ]:


submission=pd.read_csv('../input/test/sample_submission.csv')
submission.head()


# In[ ]:


from sklearn.model_selection import train_test_split
train_x,cv_x,train_y,cv_y=train_test_split(train_x,train_y,test_size=0.2)


# # Logistic Regression

# In[ ]:


import sklearn
from sklearn.preprocessing import scale 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing


# In[ ]:


train.values[:,:-2]


# In[ ]:


train.columns


# In[ ]:


LogReg = LogisticRegression()
LogReg.fit(train_x, list(train_y))


# In[ ]:


y_pred = LogReg.predict(cv_x)


# In[ ]:


y_pred


# In[ ]:


# Metrics
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(list(cv_y),y_pred)
accuracy


# In[ ]:


#y_test_pred=LogReg.predict(test_x)


# In[ ]:


# Get predicted array into dataframe
#for i,value in enumerate(y_test_pred):
#    submission.set_value(i,'AdoptionSpeed',value)
#submission.tail()


# In[ ]:


# Import into CSV
#submission.to_csv('submission.csv', index=False)


# # XGBoost

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgb_model = XGBClassifier()
xgb_model.fit(train_x, train_y)


# In[ ]:


# Predict
prediction = xgb_model.predict(cv_x)


# In[ ]:


# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(list(cv_y),list(prediction))
accuracy


# In[ ]:


#y_pred = xgb_model.predict(test_x)


# In[ ]:


#for i,value in enumerate(y_pred):
#    submission.set_value(i,'AdoptionSpeed',value)
#submission.to_csv('submission.csv', index=False)
#y_pred[-1]


# In[ ]:


#y_pred


# In[ ]:


#submission.head()


# # Deep Learning

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.layers import Dropout


# In[ ]:


model = Sequential()
model.add(Dense(600, activation='relu',  kernel_initializer='normal',input_dim=train_x.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))
mcp = keras.callbacks.ModelCheckpoint("model.h5", monitor="val_acc",  save_best_only=True, save_weights_only=False)


# In[ ]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


train_y_lb = LaBi.fit_transform(list(train_y))
cv_y_lb = LaBi.fit_transform(list(cv_y))


# In[ ]:


model.fit(train_x, train_y_lb, epochs=10, validation_data=(cv_x,cv_y_lb), callbacks=[mcp], batch_size=16)


# In[ ]:


from keras.models import load_model
best_model = load_model("model.h5")


# In[ ]:


#Evaluate
score = best_model.evaluate(cv_x, cv_y_lb, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:


y_pred=best_model.predict(test_x)
y_pred[0:20]
import operator
# Y_list: from one-hot y_pred to choosing output based index. e.g : if [0 0 1] then 2 or if [0 1 0] then 1
y_list=[]
for i in y_pred:
    index, value = max(enumerate(i), key=operator.itemgetter(1))
    y_list.append(index)
y_pred=best_model.predict(test_x)
y_list


# In[ ]:


for i,value in enumerate(y_list):
        submission.set_value(i,'AdoptionSpeed',value)
    
submission.to_csv('submission.csv', index=False)


# In[ ]:


submission.head()


# In[ ]:


train_x.shape


# In[ ]:


train_y_lb.shape


# In[ ]:




