#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd


# In[9]:


train_data1 = pd.read_csv('../input/opcode_frequency_benign.csv')
train_data2 = pd.read_csv('../input/opcode_frequency_malware.csv')


# In[10]:


train_data1['Class']=0
train_data2['Class']=1


# In[11]:


train_data1.head()
#train_data2.head()


# In[12]:


train_data=train_data1.append(train_data2)
train_data.head()


# In[13]:


train_data = train_data.drop(['FileName'], 1)


# In[14]:


train_data.head()


# In[15]:


train_data=train_data.sample(frac=1).reset_index(drop=True)
train_data.head()


# In[16]:


label=train_data['Class']


# In[17]:


#label


# In[18]:


train_data = train_data.drop(['Class'], 1)


# In[19]:


train_data.head()


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

Xstd = ss.fit_transform(train_data.values)
Xstd


# In[22]:


from sklearn.decomposition import PCA
pca = PCA(n_components=1000)
pc = pca.fit_transform(Xstd)


# In[23]:


pc


# In[24]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
pc,label = ros.fit_resample(pc,label)
len(label)


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(pc, label,test_size=0.08)


# In[26]:


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth=50).fit(X_train,y_train)
accuracy_dtree = dtree_model.score(X_test,y_test)
accuracy_dtree


# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[28]:


rf = RandomForestClassifier(n_estimators=90, random_state = 52)
rf.fit(X_train, y_train)
rf.score(X_test,y_test)


# In[29]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 4).fit(X_train,y_train)
accuracy_knn = knn.score(X_test,y_test)
accuracy_knn


# In[30]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train,y_train)
accuracy_gnb = gnb.score(X_test,y_test)
accuracy_gnb


# In[31]:


from sklearn.svm import SVC
svm_model = SVC(kernel = 'linear',C = 1).fit(X_train, y_train)
svm_predict = svm_model.predict(X_test)

accuracy = svm_model.score(X_test,y_test)
accuracy


# In[33]:


test_data = pd.read_csv('../input/Test_data.csv')


# In[34]:


test_data.head()
IDs=test_data['FileName']
test_data=test_data.drop(['FileName'],axis=1)
test_data.head()


# In[35]:


test_data=test_data.drop(['Unnamed: 1809'],axis=1)


# In[36]:


test_data.head()


# In[37]:


Xtstd = ss.transform(test_data.values)


# In[38]:


Xtstd=pca.transform(Xtstd)


# In[39]:


opDtree= rf.predict(Xtstd)
opDtreeList=opDtree.tolist()


# In[40]:


res1 = pd.DataFrame(opDtreeList)
final = pd.concat([IDs, res1], axis=1).reindex()
final = final.rename(columns={0: "Class"})
final['Class'] = final.Class.astype(int)


# In[41]:


final.to_csv('submission.csv', index = False,  float_format='%.f')


# In[42]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
 csv = df.to_csv(index=False)
 b64 = base64.b64encode(csv.encode())
 payload = b64.decode()
 html = '<a download="{filename}"href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
 html = html.format(payload=payload,title=title,filename=filename)
 return HTML(html)
create_download_link(final)

