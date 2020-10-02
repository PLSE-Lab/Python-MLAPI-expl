#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


train_data = pd.read_csv('input/train.csv')


# In[ ]:


train_data.info()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


corr=train_data.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#x = data.drop('AveragePrice', axis=1)


# In[ ]:


train_data = train_data.drop(['ID','Worker Class','IC','OC','Enrolled','MIC','MOC','Cast','Hispanic','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV','Teen','COB FATHER','COB MOTHER','COB SELF','Fill','Hispanic','Detailed'], 1)


# In[ ]:


train_data.info()


# In[ ]:


label=train_data['Class']


# In[ ]:


label


# In[ ]:


X=train_data.drop(['Class'],axis=1)
X = pd.get_dummies(X, columns=['Schooling','Married_Life','Sex','Full/Part','Tax Status','Summary','Citizen'])


# In[ ]:


X.info()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

Xstd = ss.fit_transform(X.values)
Xstd


# In[ ]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
Xstd,label = ros.fit_resample(Xstd,label)
len(label)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Xstd, label,test_size=0.3)


# In[ ]:


'''from sklearn.svm import SVC
svm_model = SVC(kernel = 'linear',C = 1).fit(X_train, y_train)
svm_predict = svm_model.predict(X_test)

accuracy = svm_model.score(X_test,y_test)'''


# In[ ]:


#accuracy


# In[ ]:


'''from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 4).fit(X_train,y_train)
accuracy_knn = knn.score(X_test,y_test)
accuracy_knn'''


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth=9).fit(X_train,y_train)
accuracy_dtree = dtree_model.score(X_test,y_test)
accuracy_dtree


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train,y_train)
accuracy_gnb = gnb.score(X_test,y_test)
accuracy_gnb


# In[ ]:


test_data = pd.read_csv('input/test_1.csv')
test_data.info()


# In[ ]:


IDs=test_data['ID']


# In[ ]:


test_data = test_data.drop(['ID','Worker Class','IC','OC','Enrolled','MIC','MOC','Cast','Hispanic','MLU','Reason','Area','State','MSA','REG','MOVE','Live','PREV','Teen','COB FATHER','COB MOTHER','COB SELF','Fill','Hispanic','Detailed'], 1)
test_data.info()


# In[ ]:


Xt = pd.get_dummies(test_data, columns=['Schooling','Married_Life','Sex','Full/Part','Tax Status','Summary','Citizen'])


# In[ ]:


Xtstd = ss.transform(Xt.values)
Xtstd


# In[ ]:


opDtree= dtree_model.predict(Xtstd)
opDtreeList=opDtree.tolist()


# In[ ]:


res1 = pd.DataFrame(opDtreeList)
final = pd.concat([IDs, res1], axis=1).reindex()
final = final.rename(columns={0: "Class"})
final['Class'] = final.Class.astype(int)


# In[ ]:


final.to_csv('submission.csv', index = False,  float_format='%.f')


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
 csv = df.to_csv(index=False)
 b64 = base64.b64encode(csv.encode())
 payload = b64.decode()
 html = '<a download="{filename}" href="data:text/csv;base64,{payload}"
target="_blank">{title}</a>'
 html = html.format(payload=payload,title=title,filename=filename)
 return HTML(html)
create_download_link(final)

