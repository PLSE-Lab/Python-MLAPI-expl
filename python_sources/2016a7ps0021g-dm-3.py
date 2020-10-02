#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


data_orig = pd.read_csv("../input/opcode_frequency_benign.csv", sep=',')
data2_orig = pd.read_csv("../input/opcode_frequency_malware.csv", sep=',')
data3_orig = pd.read_csv("../input/Test_data.csv", sep=',')
data = data_orig
datat=data2_orig
datay=data3_orig


# In[ ]:


data['Class']=0


# In[ ]:


fn=datay['FileName']
fn


# In[ ]:


datat['Class']=1


# In[ ]:


datat=datat.drop(['19','20','21','22','24','25','26','27','28','29','30'],1)


# In[ ]:


datay=datay.drop(['19','20','21','22','24','25','26','27','28','29','30'],1)


# In[ ]:


data=data.drop(['19','20','21','22','24','25','26','27','28','29','30'],1)


# In[ ]:


dataf=pd.concat([data,datat],ignore_index=True)


# In[ ]:


dataf.shape
dataf.columns = dataf.columns.astype(str)
datay.columns = datay.columns.astype(str)


# In[ ]:


dataf.columns.map(type)
dataf.shape


# In[ ]:


dataf=dataf.dropna(axis='columns')
datay=datay.dropna(axis='columns')


# In[ ]:


y=dataf['Class']
dataf.shape
#datay=datay.drop(datay[datay.columns["1809"]],axis='columns')


# In[ ]:


dataf=dataf.drop(['Class','FileName'],1)
datay=datay.drop(['FileName'],1)
datay.shape


# In[ ]:


X=dataf
dataf.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)


# # Normalisation (Train)

# In[ ]:


from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_train)
X_train = pd.DataFrame(np_scaled)
np_scaled_val = min_max_scaler.transform(X_val)
X_val = pd.DataFrame(np_scaled_val)
X_train.head()


# # Normalisation (Test)

# In[ ]:


from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled)
np_scaled_val = min_max_scaler.transform(datay)
datay = pd.DataFrame(np_scaled_val)
datay.head()


# In[ ]:


from sklearn.metrics import roc_auc_score


# # Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB as NB
#NB?


# In[ ]:


nb = NB()
nb.fit(X_train,y_train)
y_pred2=nb.predict(X_val)
roc_auc_score(y_val, y_pred2)
#nb.score(X_val,y_val)


# In[ ]:





# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_NB = nb.predict(X_val)
print(confusion_matrix(y_val, y_pred_NB))


# In[ ]:


print(classification_report(y_val, y_pred_NB))


# 

# In[ ]:





# In[ ]:





# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
#LogisticRegression?


# In[ ]:


lg = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)
lg.fit(X_train,y_train)
y_pred2=lg.predict(X_val)
roc_auc_score(y_val, y_pred2)
#sc=make_scorer(roc_auc_score)
#result=cross_validate(lg,X_val,y_val,cv=5,scoring=(sc),return_train_score=True)
#lg.score(X_val,y_val)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_LR = lg.predict(X_val)
print(confusion_matrix(y_val, y_pred_LR))


# In[ ]:


print(classification_report(y_val, y_pred_LR))


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
#DecisionTreeClassifier?


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
plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
plt.title('Accuracy vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')


# In[ ]:


from sklearn.tree import DecisionTreeClassifier

train_acc = []
test_acc = []
for i in range(2,30):
    dTree = DecisionTreeClassifier(max_depth = 6, min_samples_split=i, random_state = 42)
    dTree.fit(X_train,y_train)
    acc_train = dTree.score(X_train,y_train)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,y_val)
    test_acc.append(acc_test)


# In[ ]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(2,30),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(2,30),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
plt.title('Accuracy vs min_samples_split')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')


# In[ ]:


dTree = DecisionTreeClassifier(max_depth=17, random_state = 42)
dTree.fit(X_train,y_train)
dTree.score(X_val,y_val)


# In[ ]:


y_pred_DT = dTree.predict(X_val)
#y_pred2=lg.predict(X_val)
roc_auc_score(y_val, y_pred_DT)
#print(confusion_matrix(y_val, y_pred_DT))


# In[ ]:


print(classification_report(y_val, y_pred_DT))


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


rf = RandomForestClassifier(n_estimators=400, random_state = 42)
rf.fit(X_train, y_train)
#rf.score(X_val,y_val)
y_pred2=rf.predict(X_val)
roc_auc_score(y_val, y_pred2)


# In[ ]:


param_grid = { 
    'n_estimators': [400,500,600],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [25,30,35],
    'criterion' :['gini', 'entropy']
}


# In[ ]:


#from sklearn.model_selection import GridSearchCV
#CV = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5, verbose=20,n_jobs=-1)
#CV.fit(X,y)


# In[ ]:


#CV.best_params_


# In[ ]:



rfc=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 400, max_depth=25, criterion='entropy')


# In[ ]:


rfc.fit(X,y)


# In[ ]:


importantfea= zip(rfc.feature_importances_,X.columns)


# In[ ]:


#X.columns.shape


# In[ ]:


importantfea=sorted(importantfea,reverse=True)


# In[ ]:


importantfea
xx=X.columns
yy=rfc.feature_importances_


# In[ ]:


importantfea


# In[ ]:


imf=[xx for yy,xx in importantfea]


# In[ ]:





# In[ ]:


rfc.fit(X,y)


# # AdaBoost

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
clf = AdaBoostClassifier(n_estimators=400,learning_rate=1,random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
#accuracy_score(y_val, y_pred)
#y_pred2=lg.predict(X_val)
roc_auc_score(y_val, y_pred)


# # XGBoost

# In[ ]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


xg_reg = xgb.XGBRegressor(colsample_bytree = 0.5, learning_rate = 0.6,
                max_depth = 40, alpha = 10, n_estimators = 15,objective = "binary:logistic")


# In[ ]:


xg_reg.fit(X_train,y_train)
y_pred = xg_reg.predict(X_val)
roc_auc_score(y_val, y_pred)
#xg_reg.score(y_val,y_pred)
#accuracy_score(y_val, y_pred)


# # Neural Network

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[ ]:


def build_model():
    model = Sequential()
    model.add(Dense(256, input_dim=1797, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


# In[ ]:


nn_model = build_model()


# In[ ]:


y_train_oh = pd.get_dummies(y_train)
y_train_oh = np.asarray(y_train_oh)


# In[ ]:


from keras.utils.np_utils import to_categorical
y_binary = to_categorical(y_train_oh)


# In[ ]:


#nn_model.fit(X_train,y_binary,validation_split = 0.1,batch_size=10,verbose=1)


# In[ ]:





# In[ ]:





# In[ ]:


predy = rfc.predict(datay)


# In[ ]:


predy


# In[ ]:


dframe = pd.DataFrame(predy)


# In[ ]:


dff=pd.DataFrame(fn)


# In[ ]:


final_data=dff.join(dframe,how='left')


# In[ ]:


final_data= final_data.rename(columns={0: "Class"})


# In[ ]:


final_data.to_csv('final_submission_dhruv.csv', index = False)


# In[ ]:


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

create_download_link(final_data)


# In[ ]:




