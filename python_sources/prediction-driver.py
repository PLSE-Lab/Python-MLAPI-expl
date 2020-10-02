#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


dataset=pd.read_csv("../input/train.csv")


# In[ ]:


dataset.convert_objects(convert_numeric=True)


# In[ ]:


print (dataset.head())


# In[ ]:


cols=list(dataset)
lenAttr=len(cols)
print (lenAttr)
print (cols)


# In[ ]:


cols.insert(lenAttr-1,cols.pop(cols.index('target')))


# In[ ]:


print (cols)


# In[ ]:


cat=[]
binary=[]
con_ord=[]
for i in range(1,lenAttr-1):
    if str(cols[i][-3:len(cols[i])])=="bin":
        binary.append(cols[i])
    elif str(cols[i][-3:len(cols[i])])=="cat":
        cat.append(cols[i])
    else:
        con_ord.append(cols[i])
print (cat)
print (binary)
print (con_ord)


# In[ ]:


cols=['id']+cat+binary+con_ord+['target']
print (cols)


# In[ ]:





# In[ ]:


dataset=dataset.loc[:,cols]
print (dataset.head())


# In[ ]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 58].values


# In[ ]:


print (np.count_nonzero(dataset.iloc[:, 1:14].values==-1))
print (np.sum(dataset.iloc[:, 15:31].values==-1))
print (np.sum(dataset.iloc[:, 32:57].values==-1))


# In[ ]:


n_cat=len(cat)
n_bin=len(binary)
n_con_ord=len(con_ord)
print (n_cat,n_bin,n_con_ord)


# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:





# In[ ]:


imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:n_cat])
X[:, 1:n_cat] = imputer.transform(X[:, 1:n_cat])


# In[ ]:


print (np.count_nonzero(X[:, 1:14]==-1))


# In[ ]:


imputer = Imputer(missing_values = -1, strategy = 'median', axis = 0)
imputer = imputer.fit(X[:, 32:57])
X[:, 32:57] = imputer.transform(X[:, 32:57])


# In[ ]:


print (np.count_nonzero(X[:, 32:57]==-1))


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:,1:14] = labelencoder_X.fit_transform(X[:, 1:14])
onehotencoder = OneHotEncoder(categorical_features = [i for i in range(1,15)])
X = onehotencoder.fit_transform(X).toarray()


# In[ ]:


temp=X


# In[ ]:


X=temp


# In[ ]:


print (X.shape)


# In[ ]:


X=X[:,1:219]
print (X.shape)


# In[ ]:


print (X[0:10,])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=X
X_train[:,192:] = sc_X.fit_transform(X_train[:,192:])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=150,max_depth=20,random_state=0)
clf.fit(X_train,y)


# In[ ]:





# In[ ]:


from sklearn.externals import joblib
joblib.dump(clf,"model.pkl")


# In[ ]:


dataset=pd.read_csv("../input/test.csv")


# In[ ]:


dataset.convert_objects(convert_numeric=True)


# In[ ]:





# In[ ]:


print (dataset.shape)


# In[ ]:


cols=list(dataset)
lenAttr=len(cols)
print (lenAttr)
print (cols)


# In[ ]:


cat=[]
binary=[]
con_ord=[]
for i in range(1,lenAttr):
    if str(cols[i][-3:len(cols[i])])=="bin":
        binary.append(cols[i])
    elif str(cols[i][-3:len(cols[i])])=="cat":
        cat.append(cols[i])
    else:
        con_ord.append(cols[i])
print (cat)
print (binary)
print (con_ord)


# In[ ]:


cols=['id']+cat+binary+con_ord
print (cols)


# In[ ]:


dataset=dataset.loc[:,cols]
print (dataset.head())


# In[ ]:


X = dataset.iloc[:, :].values


# In[ ]:


print (X.shape)


# In[ ]:


dataID=list(dataset['id'])
print (dataID)


# In[ ]:


print (np.count_nonzero(dataset.iloc[:, 1:14].values==-1))
print (np.sum(dataset.iloc[:, 15:31].values==-1))
print (np.sum(dataset.iloc[:, 32:57].values==-1))


# In[ ]:


n_cat=len(cat)
n_bin=len(binary)
n_con_ord=len(con_ord)
print (n_cat,n_bin,n_con_ord)


# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:


imputer = Imputer(missing_values = -1, strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:n_cat])
X[:, 1:n_cat] = imputer.transform(X[:, 1:n_cat])


# In[ ]:


print (np.count_nonzero(X[:, 1:14]==-1))


# In[ ]:


imputer = Imputer(missing_values = -1, strategy = 'median', axis = 0)
imputer = imputer.fit(X[:, 32:57])
X[:, 32:57] = imputer.transform(X[:, 32:57])


# In[ ]:


print (np.count_nonzero(X[:, 32:57]==-1))


# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:,1:14] = labelencoder_X.fit_transform(X[:, 1:14])
onehotencoder = OneHotEncoder(categorical_features = [i for i in range(1,15)])
X = onehotencoder.fit_transform(X).toarray()


# In[ ]:


temp=X


# In[ ]:


X=temp


# In[ ]:


print (X.shape)


# In[ ]:


print (X)


# In[ ]:





# In[ ]:





# In[ ]:


X=X[:,1:219]
print (X.shape)


# In[ ]:


print (X[0:10,])


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=X
X_train[:,192:] = sc_X.fit_transform(X_train[:,192:])


# In[ ]:


y_prob=(clf.predict_proba(X_train[:,:]))


# In[ ]:


print (len(y_prob))
print (y_prob[0:50])


# In[ ]:


print (y_prob[0][1])
y_pred=clf.predict(X_train)
print (X_train.shape)


# In[ ]:


import csv
with open("output.csv","w+") as f:
    writer=csv.writer(f,delimiter=",")
    writer.writerow(['id','target'])
    for i in range(0,len(y_pred)):
        writer.writerow([int(dataID[i]),float(y_prob[i][1])])
f.close()
                         

