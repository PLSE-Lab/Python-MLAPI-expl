#!/usr/bin/env python
# coding: utf-8

# ## Importing Important Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report,roc_curve,accuracy_score,auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Conv2D,MaxPooling2D, Flatten
df = pd.read_csv("../input/pima-diabetes-dataset/Diabetes.csv")


# In[ ]:


title_mapping = {'YES':1,'NO':0}
df[' Class variable']=df[' Class variable'].map(title_mapping)


# ## Plotting Zeros Count in Data

# In[ ]:


z=(df == 0).sum(axis=0)
z=pd.DataFrame(z)
z.columns=['Zeros Count']
z.drop(' Class variable',inplace=True)
z.plot(kind='bar',stacked=True, figsize=(10,5),grid=True)


# In[ ]:


col=['n_pregnant','glucose_conc','bp','skin_len','insulin','bmi','pedigree_fun','age','Output']
df.columns=col
df.head()


# In[ ]:


diabetes_true_count = len(df.loc[df['Output'] == True])
diabetes_false_count = len(df.loc[df['Output'] == False])
(diabetes_true_count,diabetes_false_count)


# ## Replacing 0 with NaN to Handle Easily

# In[ ]:


col=['glucose_conc','bp','insulin','bmi','skin_len']
for i in col:
    df[i].replace(0, np.nan, inplace= True)


# In[ ]:


df.isnull().sum()


# ### Function to calculate Median according to the Output

# In[ ]:


def median_target(var):   
    temp = df[df[var].notnull()]
    temp = temp[[var, 'Output']].groupby(['Output'])[[var]].median().reset_index()
    return temp


# In[ ]:


median_target('insulin')


# In[ ]:


median_target('glucose_conc')


# In[ ]:


median_target('skin_len')


# In[ ]:


median_target('bp')


# In[ ]:


median_target('bmi')


# ## Filling the NaN value with Median according to Output

# In[ ]:


df.loc[(df['Output'] == 0 ) & (df['insulin'].isnull()), 'insulin'] = 102.5
df.loc[(df['Output'] == 1 ) & (df['insulin'].isnull()), 'insulin'] = 169.5
df.loc[(df['Output'] == 0 ) & (df['glucose_conc'].isnull()), 'glucose_conc'] = 107
df.loc[(df['Output'] == 1 ) & (df['glucose_conc'].isnull()), 'glucose_conc'] = 140
df.loc[(df['Output'] == 0 ) & (df['skin_len'].isnull()), 'skin_len'] = 27
df.loc[(df['Output'] == 1 ) & (df['skin_len'].isnull()), 'skin_len'] = 32
df.loc[(df['Output'] == 0 ) & (df['bp'].isnull()), 'bp'] = 70
df.loc[(df['Output'] == 1 ) & (df['bp'].isnull()), 'bp'] = 74.5
df.loc[(df['Output'] == 0 ) & (df['bmi'].isnull()), 'bmi'] = 30.1
df.loc[(df['Output'] == 1 ) & (df['bmi'].isnull()), 'bmi'] = 34.3


# In[ ]:


df.isnull().sum()


# ## Box Plot to check for Outliers in the Data

# In[ ]:


plt.style.use('ggplot') # Using ggplot2 style visuals 

f, ax = plt.subplots(figsize=(11, 15))

ax.set_facecolor('#fafafa')
ax.set(xlim=(-.05, 200))
plt.ylabel('Variables')
plt.title("Overview Data Set")
ax = sns.boxplot(data = df, 
  orient = 'h', 
  palette = 'Set2')


# # Outlier Correction with Median

# In[ ]:


sns.boxplot(df.n_pregnant)


# In[ ]:


df['n_pregnant'].value_counts()


# In[ ]:


median_target('n_pregnant')


# In[ ]:


df.loc[(df['Output'] == 0 ) & (df['n_pregnant']>13), 'n_pregnant'] = 2
df.loc[(df['Output'] == 1 ) & (df['n_pregnant']>13), 'n_pregnant'] = 4


# In[ ]:


df['n_pregnant'].value_counts()


# In[ ]:


sns.boxplot(df.bp)


# In[ ]:


median_target('bp')


# In[ ]:


df.loc[(df['Output'] == 0 ) & (df['bp']<40), 'bp'] = 70
df.loc[(df['Output'] == 1 ) & (df['bp']<40), 'bp'] = 74.5


# In[ ]:


df.loc[(df['Output'] == 0 ) & (df['bp']>103), 'bp'] = 70
df.loc[(df['Output'] == 1 ) & (df['bp']>103), 'bp'] = 74.5


# In[ ]:


sns.boxplot(df.bp)


# In[ ]:


sns.boxplot(df.skin_len)


# In[ ]:


median_target('skin_len')


# In[ ]:


df.loc[(df['Output'] == 0 ) & (df['skin_len']>38), 'skin_len'] = 27
df.loc[(df['Output'] == 1 ) & (df['skin_len']>38), 'skin_len'] = 32


# In[ ]:


df.loc[(df['Output'] == 0 ) & (df['skin_len']<20), 'skin_len'] = 27
df.loc[(df['Output'] == 1 ) & (df['skin_len']<20), 'skin_len'] = 32


# In[ ]:


sns.boxplot(df.bmi)


# In[ ]:


median_target('bmi')


# In[ ]:


df.loc[(df['Output'] == 0 ) & (df['bmi']>48), 'bmi'] = 30.1
df.loc[(df['Output'] == 1 ) & (df['bmi']>48), 'bmi'] = 34.3


# In[ ]:


sns.boxplot(df.pedigree_fun)


# In[ ]:


median_target('pedigree_fun')


# In[ ]:


df.loc[(df['Output'] == 0 ) & (df['pedigree_fun']>1), 'pedigree_fun'] = 0.336
df.loc[(df['Output'] == 1 ) & (df['pedigree_fun']>1), 'pedigree_fun'] = 0.449


# In[ ]:


sns.boxplot(df.age)


# In[ ]:


median_target('age')


# In[ ]:


df.loc[(df['Output'] == 0 ) & (df['age']>61), 'age'] = 27
df.loc[(df['Output'] == 1 ) & (df['age']>61), 'age'] = 36


# ## Splitting the Data

# In[ ]:


X = df.drop(['Output'], 1)
y = df['Output']


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# ## Scaling the Data

# In[ ]:


std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)


# ## SVM With RBF Kernel

# In[ ]:


model=SVC(kernel='rbf')
model.fit(x_train,y_train)


# In[ ]:


y_pred=model.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


fpr,tpr,_=roc_curve(y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
print('AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ## SVM With Linear Kernel

# In[ ]:


model=SVC(kernel='linear')
model.fit(x_train,y_train)


# In[ ]:


y_pred=model.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


fpr,tpr,_=roc_curve(y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
print('ROC AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ## Logistic Regression

# In[ ]:



regressor=LogisticRegression()


# In[ ]:


regressor.fit(x_train,y_train)


# In[ ]:


y_pred=regressor.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


fpr,tpr,_=roc_curve(y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
print('ROC AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ## KNN

# In[ ]:


clf = KNeighborsClassifier(n_neighbors=3) 
clf.fit(x_train,y_train)  
print(clf.score(x_test,y_test))


# In[ ]:


y_pred=clf.predict(x_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


fpr,tpr,_=roc_curve(y_test,y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
print('ROC AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ## Random Forest

# In[ ]:


classifier=RandomForestClassifier()
classifier.fit(x_train,y_train)


# In[ ]:


Y_pred=classifier.predict(x_test)
confusion_matrix(y_test,Y_pred)


# In[ ]:


accuracy_score(y_test,Y_pred)


# In[ ]:


print(classification_report(y_test,Y_pred))


# In[ ]:


fpr,tpr,_=roc_curve(y_test,Y_pred)
#calculate AUC
roc_auc=auc(fpr,tpr)
print('ROC AUC: %0.2f' % roc_auc)
#plot of ROC curve for a specified class
plt.figure()
plt.plot(fpr,tpr,label='ROC curve(area= %2.f)' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# ## Artificial Neural Networks

# In[ ]:


model = Sequential()
model.add(Dense(32,input_shape=(x_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(64,input_shape=(x_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(64,input_shape=(x_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(128,input_shape=(x_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(128,input_shape=(x_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(256,input_shape=(x_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(256,input_shape=(x_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))


# In[ ]:


print(model.summary())
model.compile(loss='sparse_categorical_crossentropy',
              optimizer="sgd",metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=10, epochs=50, verbose=1, validation_data=(x_test, y_test))
loss, accuracy = model.evaluate(x_test,y_test, verbose=0)
print("Loss : "+str(loss))
print("Accuracy :"+str(accuracy*100.0))

