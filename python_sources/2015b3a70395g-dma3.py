#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Reading Benign Data
benign_data = pd.read_csv("../input/opcode_frequency_benign.csv", sep=',')
benign_data.info()


# In[3]:


benign_data.isnull().values.any()


# In[4]:


benign_data.duplicated().sum()


# In[5]:


#Reading Malware Data
malware_data = pd.read_csv("../input/opcode_frequency_malware.csv", sep=',')
malware_data.info()


# In[6]:


malware_data.isnull().values.any()


# In[7]:


malware_data.duplicated().sum()


# In[8]:


benign_data['class'] = 0
malware_data['class'] = 1


# In[9]:


data = benign_data.append(malware_data)


# In[10]:


data.head()


# In[11]:


orig_data = data.copy()
X = data
X = X.drop(columns=['class','FileName'])
y = data['class']


# In[12]:


from sklearn import preprocessing

arr = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
arr_scaled = min_max_scaler.fit_transform(arr)
X_scaled = pd.DataFrame(arr_scaled)
X_scaled.head()


# In[13]:


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_scaled)
pca_data = pd.DataFrame(data = principalComponents, columns = ['PC 1', 'PC 2'])


# In[14]:


mat=pca_data.values
plt.figure(figsize=(20,15)) 
plt.scatter(mat[:,0],mat[:,1],c=y,cmap='rainbow')
# plt.xlim(-10, 40)
# plt.ylim(-10, 10)


# In[15]:


from sklearn.cluster import KMeans


# In[16]:


sse = {}
new_df = X
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(new_df)
    #new_df["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.figure(figsize=(15,10))
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.savefig('optimalK_KMEANS.png')
plt.show()


# In[17]:


X.head()


# In[18]:


kmeans = KMeans(n_clusters = 2,random_state=42).fit(data)
data['KMeans'] = kmeans.labels_


# In[19]:


X0 = data[data['KMeans']==0]
X1 = data[data['KMeans']==1]


# In[20]:


File0 = X0['FileName']
y0 = X0['class']
X0 = X0.drop(columns=['FileName','class','KMeans'])
File1 = X1['FileName']
y1 = X1['class']
X1 = X1.drop(columns=['FileName','class','KMeans'])


# In[21]:


# X = X.drop(columns=['KMeans'])


# ### Applying Classification Models on dataset

# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.20, random_state=42)


# In[23]:


import seaborn as sns
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[24]:


def score(classification, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1_sc = f1_score(y_true, y_pred, average = 'binary', pos_label = 1)
    print(" ")
    print('Accuracy: {0}\nPrecision: {1}\nRecall: {2}\nF1 Score: {3}'.format(accuracy, precision, recall, f1_sc))
    print(" ")
    print(" ")
    return accuracy, precision, recall, f1_sc


# In[25]:


def run_model(name, model, X, y, parameters = None, cros_val_splits = 5,plot=False):
    model_instance = model
    prediction = cross_val_predict(model_instance, X, y.values.ravel(), cv = cros_val_splits)
    cm = metrics.confusion_matrix(y,prediction)
    accuracy, precision, recall, f1_score = score(name, y, prediction)
    cm = cm.astype('float')/ cm.sum()
    
    if plot:
        #plot confusion matrix
        plt.figure(figsize=(9,9))
        sns.heatmap(cm, annot = True, fmt='.3f', linewidths=.5, square=True)
        plt.ylabel('Actual Label',size = 15)
        plt.xlabel('Predicted Label', size = 15)
        all_sample_title = 'Accuracy Score: {}'.format(accuracy)
        plt.title(all_sample_title,size = 15)
        plt.show()
    


# In[26]:


models = [DecisionTreeClassifier(), RandomForestClassifier(), AdaBoostClassifier(), 
          GradientBoostingClassifier(), GaussianNB(), SVC()]
names = ['Decision Tree', 'Random Forest', 'AdaBoost', 'Gradient Boosting', 'Naive Bayes', 'SVM']


# In[27]:


for i in range(len(models)):
    print(names[i])
    model = models[i]
    run_model('name', model, X_scaled, y,plot=False)


# In[28]:


score_train_RF = []
score_test_RF = []

for i in range(1,30,1):
    rf = RandomForestClassifier(max_depth=i,n_estimators=23, random_state = 42)
    rf.fit(X_train,y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_val,y_val)
    score_test_RF.append(sc_test)


# In[29]:


plt.figure(figsize=(10,6))
train_score,=plt.plot(range(1,30,1),score_train_RF,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(1,30,1),score_test_RF,color='red',linestyle='dashed',  marker='o',
         markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.axvline(x=25, linewidth=2)
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')


# In[30]:


rf = RandomForestClassifier(n_estimators=23, max_depth=25,random_state = 42)
rf.fit(X_train,y_train)
rf.score(X_val,y_val)


# In[31]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_RF = rf.predict(X_val)
confusion_matrix(y_val, y_pred_RF)


# In[32]:


from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_val,y_pred_RF))


# In[33]:


X_test = pd.read_csv('../input/Test_data.csv')


# In[34]:


X_test.head()


# In[35]:


ID = X_test['FileName']


# In[36]:


X_test = X_test.drop(columns=['Unnamed: 1809'])


# In[37]:


kmeans = KMeans(n_clusters = 2,random_state=42).fit(X_test)
X_test['KMeans'] = kmeans.labels_


# In[38]:


X_test0 = X_test[X_test['KMeans']==0]
X_test1 = X_test[X_test['KMeans']==1]


# In[39]:


ID0 = X_test0['FileName']
ID1 = X_test1['FileName']


# In[40]:


X_test0 = X_test0.drop(columns=['FileName','KMeans'])
X_test1 = X_test1.drop(columns=['FileName','KMeans'])


# In[41]:


rf = RandomForestClassifier(n_estimators=23,max_depth=25,random_state = 42)
rf.fit(X,y)


# In[42]:


preds0 = rf.predict(X_test0)
preds1 = rf.predict(X_test1)


# In[43]:


df0 = pd.DataFrame(columns=['FileName', 'class'])
df0['FileName']=ID0
df0['class']=preds0
df0.head()


# In[44]:


df1 = pd.DataFrame(columns=['FileName', 'class'])
df1['FileName']=ID1
df1['class']=preds1
df1.head()


# In[45]:


df = df0.append(df1,ignore_index=True)
df = df.sort_values(by='FileName')
df.head()


# In[46]:


# df.to_csv('final_sub2.csv',index=False)


# In[47]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html='<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df)


# In[ ]:




