#!/usr/bin/env python
# coding: utf-8

# This is the second version of my public kernel(Intrusion Detection System). ANN is also trained & tested on the dataset in this version.
# I would really appreciate your feedback.

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


# In[ ]:


print(os.listdir('../input/kdd-cup-1999-data'))


# In[ ]:


with open("../input/kdd-cup-1999-data/kddcup.names",'r') as f:
    print(f.read())


# In[ ]:


cols="""duration,
protocol_type,
service,
flag,
src_bytes,
dst_bytes,
land,
wrong_fragment,
urgent,
hot,
num_failed_logins,
logged_in,
num_compromised,
root_shell,
su_attempted,
num_root,
num_file_creations,
num_shells,
num_access_files,
num_outbound_cmds,
is_host_login,
is_guest_login,
count,
srv_count,
serror_rate,
srv_serror_rate,
rerror_rate,
srv_rerror_rate,
same_srv_rate,
diff_srv_rate,
srv_diff_host_rate,
dst_host_count,
dst_host_srv_count,
dst_host_same_srv_rate,
dst_host_diff_srv_rate,
dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate,
dst_host_serror_rate,
dst_host_srv_serror_rate,
dst_host_rerror_rate,
dst_host_srv_rerror_rate"""

columns=[]
for c in cols.split(','):
    if(c.strip()):
       columns.append(c.strip())

columns.append('target')
#print(columns)
print(len(columns))


# In[ ]:


with open("../input/kdd-cup-1999-data/training_attack_types",'r') as f:
    print(f.read())


# In[ ]:


attacks_types = {
    'normal': 'normal',
'back': 'dos',
'buffer_overflow': 'u2r',
'ftp_write': 'r2l',
'guess_passwd': 'r2l',
'imap': 'r2l',
'ipsweep': 'probe',
'land': 'dos',
'loadmodule': 'u2r',
'multihop': 'r2l',
'neptune': 'dos',
'nmap': 'probe',
'perl': 'u2r',
'phf': 'r2l',
'pod': 'dos',
'portsweep': 'probe',
'rootkit': 'u2r',
'satan': 'probe',
'smurf': 'dos',
'spy': 'r2l',
'teardrop': 'dos',
'warezclient': 'r2l',
'warezmaster': 'r2l',
}


# READING DATASET

# In[ ]:


path = "../input/kdd-cup-1999-data/kddcup.data_10_percent.gz"
df = pd.read_csv(path,names=columns)

#Adding Attack Type column
df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]])

df.head()


# In[ ]:


df.shape


# In[ ]:


df['target'].value_counts()


# In[ ]:


df['Attack Type'].value_counts()


# In[ ]:


df.dtypes


# DATA PREPROCESSING

# In[ ]:


df.isnull().sum()


# In[ ]:


#Finding categorical features
num_cols = df._get_numeric_data().columns

cate_cols = list(set(df.columns)-set(num_cols))
cate_cols.remove('target')
cate_cols.remove('Attack Type')

cate_cols


# CATEGORICAL FEATURES DISTRIBUTION

# In[ ]:


#Visualization
def bar_graph(feature):
    df[feature].value_counts().plot(kind="bar")


# In[ ]:


bar_graph('protocol_type')


# Protocol type: We notice that ICMP is the most present in the used data, then TCP and almost 20000 packets of UDP type

# In[ ]:


plt.figure(figsize=(15,3))
bar_graph('service')


# In[ ]:


bar_graph('flag')


# In[ ]:


bar_graph('logged_in')


# logged_in (1 if successfully logged in; 0 otherwise): We notice that just 70000 packets are successfully logged in.

# TARGET FEATURE DISTRIBUTION

# In[ ]:


bar_graph('target')


# Attack Type(The attack types grouped by attack, it's what we will predict)

# In[ ]:


bar_graph('Attack Type')


# In[ ]:


df.columns


# DATA CORRELATION

# In[ ]:


df = df.dropna('columns')# drop columns with NaN

df = df[[col for col in df if df[col].nunique() > 1]]# keep columns where there are more than 1 unique values

corr = df.corr()

plt.figure(figsize=(15,12))

sns.heatmap(corr)

plt.show()


# In[ ]:


df['num_root'].corr(df['num_compromised'])


# In[ ]:


df['srv_serror_rate'].corr(df['serror_rate'])


# In[ ]:


df['srv_count'].corr(df['count'])


# In[ ]:


df['srv_rerror_rate'].corr(df['rerror_rate'])


# In[ ]:


df['dst_host_same_srv_rate'].corr(df['dst_host_srv_count'])


# In[ ]:


df['dst_host_srv_serror_rate'].corr(df['dst_host_serror_rate'])


# In[ ]:


df['dst_host_srv_rerror_rate'].corr(df['dst_host_rerror_rate'])


# In[ ]:


df['dst_host_same_srv_rate'].corr(df['same_srv_rate'])


# In[ ]:


df['dst_host_srv_count'].corr(df['same_srv_rate'])


# In[ ]:


df['dst_host_same_src_port_rate'].corr(df['srv_count'])


# In[ ]:


df['dst_host_serror_rate'].corr(df['serror_rate'])


# In[ ]:


df['dst_host_serror_rate'].corr(df['srv_serror_rate'])


# In[ ]:


df['dst_host_srv_serror_rate'].corr(df['serror_rate'])


# In[ ]:


df['dst_host_srv_serror_rate'].corr(df['srv_serror_rate'])


# In[ ]:


df['dst_host_rerror_rate'].corr(df['rerror_rate'])


# In[ ]:


df['dst_host_rerror_rate'].corr(df['srv_rerror_rate'])


# In[ ]:


df['dst_host_srv_rerror_rate'].corr(df['rerror_rate'])


# In[ ]:


df['dst_host_srv_rerror_rate'].corr(df['srv_rerror_rate'])


# In[ ]:


#This variable is highly correlated with num_compromised and should be ignored for analysis.
#(Correlation = 0.9938277978738366)
df.drop('num_root',axis = 1,inplace = True)

#This variable is highly correlated with serror_rate and should be ignored for analysis.
#(Correlation = 0.9983615072725952)
df.drop('srv_serror_rate',axis = 1,inplace = True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9947309539817937)
df.drop('srv_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
#(Correlation = 0.9993041091850098)
df.drop('dst_host_srv_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9869947924956001)
df.drop('dst_host_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
#(Correlation = 0.9821663427308375)
df.drop('dst_host_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9851995540751249)
df.drop('dst_host_srv_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
#(Correlation = 0.9865705438845669)
df.drop('dst_host_same_srv_rate',axis = 1, inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df_std = df.std()
df_std = df_std.sort_values(ascending = True)
df_std


# FEATURE MAPPING

# In[ ]:


df['protocol_type'].value_counts()


# In[ ]:


#protocol_type feature mapping
pmap = {'icmp':0,'tcp':1,'udp':2}
df['protocol_type'] = df['protocol_type'].map(pmap)


# In[ ]:


df['flag'].value_counts()


# In[ ]:


#flag feature mapping
fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
df['flag'] = df['flag'].map(fmap)


# In[ ]:


df.head()


# In[ ]:


df.drop('service',axis = 1,inplace= True)


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# MODELLING

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


# In[ ]:


df = df.drop(['target',], axis=1)
print(df.shape)

# Target variable and train set
Y = df[['Attack Type']]
X = df.drop(['Attack Type',], axis=1)

sc = MinMaxScaler()
X = sc.fit_transform(X)

# Split test and train data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)


# GAUSSIAN NAIVE BAYES

# In[ ]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB


# In[ ]:


model1 = GaussianNB()


# In[ ]:


start_time = time.time()
model1.fit(X_train, Y_train.values.ravel())
end_time = time.time()


# In[ ]:


print("Training time: ",end_time-start_time)


# In[ ]:


start_time = time.time()
Y_test_pred1 = model1.predict(X_test)
end_time = time.time()


# In[ ]:


print("Testing time: ",end_time-start_time)


# In[ ]:


print("Train score is:", model1.score(X_train, Y_train))
print("Test score is:",model1.score(X_test,Y_test))


# DECISION TREE

# In[ ]:


#Decision Tree 
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


model2 = DecisionTreeClassifier(criterion="entropy", max_depth = 4)


# In[ ]:


start_time = time.time()
model2.fit(X_train, Y_train.values.ravel())
end_time = time.time()


# In[ ]:


print("Training time: ",end_time-start_time)


# In[ ]:


start_time = time.time()
Y_test_pred2 = model2.predict(X_test)
end_time = time.time()


# In[ ]:


print("Testing time: ",end_time-start_time)


# In[ ]:


print("Train score is:", model2.score(X_train, Y_train))
print("Test score is:",model2.score(X_test,Y_test))


# RANDOM FOREST

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model3 = RandomForestClassifier(n_estimators=30)


# In[ ]:


start_time = time.time()
model3.fit(X_train, Y_train.values.ravel())
end_time = time.time()


# In[ ]:


print("Training time: ",end_time-start_time)


# In[ ]:


start_time = time.time()
Y_test_pred3 = model3.predict(X_test)
end_time = time.time()


# In[ ]:


print("Testing time: ",end_time-start_time)


# In[ ]:


print("Train score is:", model3.score(X_train, Y_train))
print("Test score is:",model3.score(X_test,Y_test))


# SUPPORT VECTOR MACHINE

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model4 = SVC(gamma = 'scale')


# In[ ]:


start_time = time.time()
model4.fit(X_train, Y_train.values.ravel())
end_time = time.time()


# In[ ]:


print("Training time: ",end_time-start_time)


# In[ ]:


start_time = time.time()
Y_test_pred4 = model4.predict(X_test)
end_time = time.time()


# In[ ]:


print("Testing time: ",end_time-start_time)


# In[ ]:


print("Train score is:", model4.score(X_train, Y_train))
print("Test score is:", model4.score(X_test,Y_test))


# LOGISTIC REGRESSION

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model5 = LogisticRegression(max_iter=1200000)


# In[ ]:


start_time = time.time()
model5.fit(X_train, Y_train.values.ravel())
end_time = time.time()


# In[ ]:


print("Training time: ",end_time-start_time)


# In[ ]:


start_time = time.time()
Y_test_pred5 = model5.predict(X_test)
end_time = time.time()


# In[ ]:


print("Testing time: ",end_time-start_time)


# In[ ]:


print("Train score is:", model5.score(X_train, Y_train))
print("Test score is:",model5.score(X_test,Y_test))


# GRADIENT BOOSTING CLASSIFIER

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


model6 = GradientBoostingClassifier(random_state=0)


# In[ ]:


start_time = time.time()
model6.fit(X_train, Y_train.values.ravel())
end_time = time.time()


# In[ ]:


print("Training time: ",end_time-start_time)


# In[ ]:


start_time = time.time()
Y_test_pred6 = model6.predict(X_test)
end_time = time.time()


# In[ ]:


print("Testing time: ",end_time-start_time)


# In[ ]:


print("Train score is:", model6.score(X_train, Y_train))
print("Test score is:", model6.score(X_test,Y_test))


# Artificial Neural Network

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


# In[ ]:


def fun():
    model = Sequential()
    
    #here 30 is output dimension
    model.add(Dense(30,input_dim =30,activation = 'relu',kernel_initializer='random_uniform'))
    
    #in next layer we do not specify the input_dim as the model is sequential so output of previous layer is input to next layer
    model.add(Dense(1,activation='sigmoid',kernel_initializer='random_uniform'))
    
    #5 classes-normal,dos,probe,r2l,u2r
    model.add(Dense(5,activation='softmax'))
    
    #loss is categorical_crossentropy which specifies that we have multiple classes
    
    model.compile(loss ='categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    
    return model


# In[ ]:


#Since,the dataset is very big and we cannot fit complete data at once so we use batch size.
#This divides our data into batches each of size equal to batch_size.
#Now only this number of samples will be loaded into memory and processed. 
#Once we are done with one batch it is flushed from memory and the next batch will be processed.
model7 = KerasClassifier(build_fn=fun,epochs=100,batch_size=64)


# In[ ]:


start = time.time()
model7.fit(X_train, Y_train.values.ravel())
end = time.time()


# In[ ]:


print('Training time')
print((end-start))


# In[ ]:


start_time = time.time()
Y_test_pred7 = model7.predict(X_test)
end_time = time.time()


# In[ ]:


print("Testing time: ",end_time-start_time)


# In[ ]:


start_time = time.time()
Y_train_pred7 = model7.predict(X_train)
end_time = time.time()


# In[ ]:


accuracy_score(Y_train,Y_train_pred7)


# In[ ]:


accuracy_score(Y_test,Y_test_pred7)


# TRAINING ACCURACY

# In[ ]:


names = ['NB','DT','RF','SVM','LR','GB','ANN']
values = [87.951,99.058,99.997,99.875,99.352,99.793,99.914]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.ylim(80,102)
plt.bar(names,values)


# In[ ]:


f.savefig('training_accuracy_figure.png',bbox_inches='tight')


# TESTING ACCURACY

# In[ ]:


names = ['NB','DT','RF','SVM','LR','GB','ANN']
values = [87.903,99.052,99.969,99.879,99.352,99.771,99.886]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.ylim(80,102)
plt.bar(names,values)


# In[ ]:


f.savefig('test_accuracy_figure.png',bbox_inches='tight')


# TRAINING TIME

# In[ ]:


names = ['NB','DT','RF','SVM','LR','GB','ANN']
values = [1.04721,1.50483,11.45332,126.96016,56.67286,446.69099,1211.54094]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.bar(names,values)


# In[ ]:


f.savefig('train_time_figure.png',bbox_inches='tight')


# TESTING TIME

# In[ ]:


names = ['NB','DT','RF','SVM','LR','GB','ANN']
values = [0.79089,0.10471,0.60961,32.72654,0.02198,1.41416,1.72521]
f = plt.figure(figsize=(15,3),num=10)
plt.subplot(131)
plt.bar(names,values)


# In[ ]:


f.savefig('test_time_figure.png',bbox_inches='tight')


# In[ ]:




