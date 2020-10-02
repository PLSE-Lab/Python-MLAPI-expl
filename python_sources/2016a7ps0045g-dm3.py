#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn.preprocessing as sk
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# In[42]:


df_b = pd.read_csv("../input/opcode_frequency_benign.csv")
df_m = pd.read_csv("../input/opcode_frequency_malware.csv")


# In[43]:


df_b['Class'] = 0
df_m['Class'] = 1


# In[ ]:





# In[44]:


df_train_original2 = df_b.append(df_m)
df_train_original = df_b.append(df_m , ignore_index = True)


# In[45]:


df_train_original.head()


# # Pre Processing

# In[46]:


df_train = df_train_original.copy()


# In[47]:


df_train.duplicated(keep = 'first').any()


# In[48]:


df_train.isnull().values.any()


# In[49]:


df_train.drop('FileName',axis = 1,inplace = True)


# # Feature Scaling

# In[50]:


y = df_train['Class']
X = df_train.drop('Class',axis = 1)
X.head()


# In[51]:


# Test train split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)


# In[52]:


from sklearn import preprocessing
#Performing Min_Max Normalization
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_train)
X_train = pd.DataFrame(np_scaled)
np_scaled_val = min_max_scaler.transform(X_val)
X_val = pd.DataFrame(np_scaled_val)
X_train.head()


# # Naive Bayes

# In[53]:


np.random.seed(42)


# In[54]:


from sklearn.naive_bayes import GaussianNB as NB
nb = NB()
nb.fit(X_train,y_train)
nb.score(X_val,y_val)


# In[55]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_NB = nb.predict(X_val)
print(confusion_matrix(y_val, y_pred_NB))


# In[56]:


print(classification_report(y_val, y_pred_NB))


# In[57]:


from sklearn.metrics import roc_auc_score

roc_auc_score(y_val, y_pred_NB)


# # Logistic Regression

# In[58]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(solver = 'liblinear', C = 1, multi_class = 'ovr', random_state = 42)
lg.fit(X_train,y_train)
lg.score(X_val,y_val)


# In[59]:


lg = LogisticRegression(solver = 'lbfgs', C = 8, multi_class = 'multinomial', random_state = 42)
lg.fit(X_train,y_train)
lg.score(X_val,y_val)


# In[60]:


y_pred_LR = lg.predict(X_val)
print(confusion_matrix(y_val, y_pred_LR))


# In[61]:


print(classification_report(y_val, y_pred_LR))


# In[62]:


from sklearn.metrics import roc_auc_score

roc_auc_score(y_val, y_pred_LR)


# # Decision Tree

# In[63]:


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


# In[64]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,15),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,15),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
plt.title('Accuracy vs Max Depth')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')


# In[65]:


from sklearn.tree import DecisionTreeClassifier

train_acc = []
test_acc = []
for i in range(2,30):
    dTree = DecisionTreeClassifier(max_depth = 10, min_samples_split=i, random_state = 42)
    dTree.fit(X_train,y_train)
    acc_train = dTree.score(X_train,y_train)
    train_acc.append(acc_train)
    acc_test = dTree.score(X_val,y_val)
    test_acc.append(acc_test)


# In[66]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(2,30),train_acc,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(2,30),test_acc,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score, test_score],["Train Accuracy", "Validation Accuracy"])
plt.title('Accuracy vs min_samples_split')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')


# In[67]:


dTree = DecisionTreeClassifier(max_depth=9, random_state = 42)
dTree.fit(X_train,y_train)
dTree.score(X_val,y_val)


# In[68]:


y_pred_DT = dTree.predict(X_val)
print(confusion_matrix(y_val, y_pred_DT))


# In[69]:


from sklearn.metrics import roc_auc_score

roc_auc_score(y_val,y_pred_DT)


# # Random Forest

# In[70]:


from sklearn.ensemble import RandomForestClassifier

score_train_RF = []
score_test_RF = []

for i in range(1,18,1):
    rf = RandomForestClassifier(n_estimators=i, random_state = 42)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_val,y_val)
    score_test_RF.append(sc_test)


# In[71]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,18,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,18,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')


# In[72]:


rf = RandomForestClassifier(n_estimators=13, random_state = 42)
rf.fit(X_train, y_train)
rf.score(X_val,y_val)


# In[73]:


y_pred_RF = rf.predict(X_val)
confusion_matrix(y_val, y_pred_RF)


# In[74]:


print(classification_report(y_val, y_pred_RF))


# In[75]:


param_grid = {
'n_estimators': [650,700,800,900],
'max_features': ['auto', 'sqrt', 'log2'],
'max_depth' : [16,17,18],
'criterion' :['gini', 'entropy']
}


# In[76]:


#import xgboost as xgb


# In[77]:


#xgc = xgb.XGBClassifier()
#xgc.fit(X_train, y_train)
#xgc.score(X_val,y_val)


# In[78]:


#y_pred_X = xgc.predict(X_val)
#roc_auc_score(y_val, y_pred_X)


# # Grid Search

# In[79]:


from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5, verbose=20,n_jobs=-1)
CV_rfc.fit(X,y)


# In[80]:


CV_rfc.best_params_


# In[ ]:





# # Bagging

# In[81]:


from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
model.fit(X_train, y_train)
model.score(X_val,y_val)


# In[82]:


y_pred_bagging = model.predict(X_val)
accuracy_score(y_val,y_pred_bagging)


# # Testing on the model

# In[83]:


df_test_original = pd.read_csv("../input/Test_data.csv")
df_test = df_test_original.copy()
df_test.head()


# In[84]:


df_testing = df_test.drop(df_test.columns[1809],axis  = 1)
df_testing = df_testing.drop('FileName',axis = 1)
df_testing.head()


# In[85]:


min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X1 = pd.DataFrame(np_scaled)
X1.head()


# In[ ]:





# In[86]:


min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df_testing)
df_minmax = pd.DataFrame(np_scaled)
df_minmax.head()


# In[87]:


rf.fit(X,y)
predictions = rf.predict(df_minmax)


# In[88]:


submit = pd.read_csv("../input/Test_data.csv" , sep = ",")
colID = submit[['FileName']]


# In[89]:


res = colID.assign(Class = predictions)


# In[90]:


res['FileName'] = res['FileName'].astype(int)


# In[91]:


res.to_csv("Submission3.csv",index = False)


# In[92]:


res.info()


# In[93]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "Submissionffg.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(res)


# In[ ]:




