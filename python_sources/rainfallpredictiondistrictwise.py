#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df =pd.read_csv('../input/rainfall-in-india/district wise rainfall normal.csv')
df.head()


# #**Let us analyse the District Rainfall **

# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.columns


# In[7]:


df['STATE_UT_NAME'].value_counts()


# In[8]:


df_up =df[df.STATE_UT_NAME == 'UTTAR PRADESH']
df_up.head(4)


# In[9]:


df_up.plot()


# In[10]:


sort_up = df_up.sort_values('ANNUAL')
sort_up.head()


# #Ten District with less rainfall anually 

# In[11]:


sort_up.head(10).plot(x = 'DISTRICT',y= 'ANNUAL' )


# **Here I am doing only for the country level not by state level any more**

# In[12]:


df.head()


# In[13]:


df_so_ana = df.sort_values('ANNUAL')
df_so_ana.head()


# In[14]:


df_so_ana.head().plot(kind = 'bar')


# In[15]:


df_so_ana.tail().plot(kind = 'bar')


# In[16]:


df_so_ana.head().plot(x = 'DISTRICT', y='ANNUAL',kind = 'bar')


# In[17]:


df_so_ana.tail().plot(x = 'DISTRICT', y='ANNUAL',kind = 'bar')


# **Highest rainfall satates**

# In[18]:


df_so_ana.head().plot(x = 'STATE_UT_NAME', y='ANNUAL',kind = 'bar')


# In[19]:


sort_val_sat = df_so_ana.groupby('STATE_UT_NAME').ANNUAL.mean().sort_values()


# **the state with less rain fall**

# In[20]:


sort_val_sat.plot(kind= 'bar')


# In[21]:


sort_val_sat.head(10).plot(kind= 'bar')


# In[22]:


sort_val_sat.tail(10).plot(kind= 'bar')


# In[23]:


sns.FacetGrid(sort_val_sat)


# In[24]:


df_so_ana.head()


# In[25]:


df_so_ana.columns = ['STATE_UT_NAME','DISTRICT','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL','Jan_Feb','Mar_May','Jun_Sep','Oct_Dec']


# In[26]:


df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values()


# In[27]:


df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().plot()
df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().plot(kind = 'bar')


# In[28]:


df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().head(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().head(10).plot(kind = 'bar')


# In[29]:


df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().tail(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Jan_Feb.mean().sort_values().tail(10).plot(kind = 'bar')


# In[30]:


df_so_ana.columns


# In[31]:


df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values()
df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().plot()
df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().plot(kind = 'bar')


# In[32]:


df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().head(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().head(10).plot(kind = 'bar')


# In[33]:


df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().tail(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Mar_May.mean().sort_values().tail(10).plot(kind = 'bar')


# In[34]:


df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values()
df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().plot()
df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().plot(kind = 'bar')


# In[35]:


df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().tail(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().tail(10).plot(kind = 'bar')


# In[36]:


df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().head(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Jun_Sep.mean().sort_values().head(10).plot(kind = 'bar')


# In[37]:


df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values()
df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().plot()
df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().plot(kind = 'bar')


# In[38]:


df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().head(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().head(10).plot(kind = 'bar')


# In[39]:


df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().tail(10).plot()
df_so_ana.groupby('STATE_UT_NAME').Oct_Dec.mean().sort_values().tail(10).plot(kind = 'bar')


# In[40]:


data =pd.read_csv('../input/rainfall-in-india/rainfall in india 1901-2015.csv')


# In[41]:


data.head()


# In[42]:


data['SUBDIVISION'].value_counts()


# **data analysis wrt subdivision**

# In[43]:


data.plot()


# In[44]:


data.describe()


# In[45]:


data.info()


# **we will select he columns that are required for the analysis**

# In[46]:


df1 = data[['SUBDIVISION','YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']].copy()
df1.head()


# In[47]:


df1['YEAR'].value_counts(sort =False).head()


# In[48]:


df1.describe()


# In[49]:


corr = df1.corr()
sns.heatmap(corr)


# In[50]:


sns.clustermap(corr)


# In[51]:


df1[:100].plot(x ='YEAR', y= 'JUN')
#df1[:100].plot(x='YEAR', y='ANNUAL')
df1[:100].plot(x='YEAR', y='JUL')


# In[52]:


df1[:100].JUN.plot()
df1[:100].ANNUAL.plot()
df1[:100].JUL.plot()


# In[53]:


df1[:100].JUN.plot()
#df[:100].ANNUAL.plot()
df1[:100].JUL.plot()


# In[54]:


df1[:100].JUN.plot()
#df[:100].ANNUAL.plot()
df1[:100].JUL.plot()
df1[:100].AUG.plot()


# In[55]:


df[:100].JUN.plot()
#df[:100].ANNUAL.plot()
df[:100].JUL.plot()
df[:100].AUG.plot()


# In[56]:


corr = df1.corr()
print(corr)


# In[57]:


df2 = df1.sort_values(['YEAR','ANNUAL'])
df2.head()


# In[58]:


df2[:100].JUN.plot()

df2[:100].JUL.plot()
df2[:100].AUG.plot()


# In[59]:


df2


# In[60]:


df3 = df2.sort_values('YEAR')
annual_array = []
for element in range(0,4116,116):
    annual_array.append(df2.loc[element,"ANNUAL"])
    print(element,df2.loc[element,"YEAR"])


# In[61]:


df2.head(10)


# In[62]:


df2['SUBDIVISION'].value_counts()


# In[63]:


#Changing department to unique numbers
df2.SUBDIVISION.unique()
dictionary = {}
for c, value in enumerate(df2.SUBDIVISION.unique(), 1):
    #print(c, value)
    dictionary[value] = c
print(dictionary)
df2["SUBDIVISION"] = df2.SUBDIVISION.map(dictionary)
df2.head()


# In[64]:


df2.columns


# In[65]:


sns.heatmap(df2.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[66]:


df2.dropna( inplace = True) 


# In[67]:


df2.isnull().sum().sum()


# In[68]:


df2.replace([np.inf, -np.inf], np.nan, inplace = True)


# In[69]:


df2.isnull().sum().sum()


# In[70]:


df3.head()


# In[71]:


df3 = df2


# In[72]:


df3.head()


# Principle component analysis

# In[73]:


sns.heatmap(df3.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[74]:


from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing 
from sklearn.pipeline import Pipeline
df2.head(0)
features = []
for element in df2.head(0):
    features.append(element)
features.remove('ANNUAL')
features.remove('JAN')
features.remove('FEB')
features.remove('MAR')
features.remove('APR')
features.remove('OCT')
features.remove('NOV')
features.remove('DEC')
features.remove('AUG')
features.remove('SEP')
print(features)


# In[75]:


features2 = ['AUG']


# In[76]:


# Separating out the features
x  = df2.loc[:, features].values
# Separating out the target
y = df2.loc[:,features2].values


# In[77]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[78]:


from sklearn.decomposition import PCA 
pca = PCA(n_components=4)
pca.fit(x)


# In[79]:


print(pca.explained_variance_ratio_)


# In[80]:


print(pca.singular_values_)


# In[81]:


pca.score(x,y)


# In[82]:


pca.score_samples(x)


# SVM

# In[83]:


from sklearn import svm
from sklearn.datasets import make_classification
clf = svm.SVC()
X_train, y_train = make_classification()
X_test, y_test = make_classification()
clf.fit(X_train, y_train)


# In[84]:


print(clf.intercept_)


# In[85]:


clf.predict(X_test)


# In[86]:


clf.score(X_test, y_test, sample_weight=None)


# In[87]:


y_score = clf.decision_function(X_test)


# In[88]:


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)
print(average_precision)


# In[89]:


from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


# **Random Forest **

# In[90]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)


# **Naive Bayes**

# In[91]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)


# In[92]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
clf.predict(X_test)
clf.score(X_test, y_test)


# *now we took MAY, JUNE, JULY months as our training set and predicted for the august month
# now we will take MAY and JUNE as ou training set and predict for the JULY*

# In[93]:


features = []
for element in df3.head(0):
    features.append(element)
features.remove('ANNUAL')
features.remove('JAN')
features.remove('FEB')
features.remove('MAR')
features.remove('APR')
features.remove('OCT')
features.remove('NOV')
features.remove('DEC')
features.remove('AUG')
features.remove('SEP')
features.remove('JUL')
print(features)


# In[94]:


features2 = ['JUL']


# In[95]:


# Separating out the features
p  = df2.loc[:, features].values
# Separating out the target
q = df2.loc[:,features2].values


# In[96]:


from sklearn.model_selection import train_test_split
P_train, P_test, q_train, q_test = train_test_split(p, q, test_size=0.33, random_state=42)


# In[97]:


from sklearn import svm
from sklearn.datasets import make_classification
clf = svm.SVC()
P_train, q_train = make_classification()
P_test, q_test = make_classification()
clf.fit(P_train, q_train)
clf.predict(P_test)
clf.score(P_test, q_test)


# In[98]:


q_score = clf.decision_function(P_test)


# In[99]:


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(q_test, q_score)
print(average_precision)


# In[100]:


from sklearn.metrics import precision_recall_curve
precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


# In[101]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(P_train, q_train)
clf.predict(P_test)
clf.score(P_test, q_test)


# In[102]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(P_train, q_train)
clf.predict(P_test)
clf.score(P_test, q_test)


# In[103]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(P_train, q_train)
clf.predict(P_test)
clf.score(P_test, q_test)


# **Now we will take Only april as train set and may as test set**

# In[104]:


df4 = df3
df4.head()


# In[105]:


features = []
for element in df3.head(0):
    features.append(element)
features.remove('ANNUAL')
features.remove('JAN')
features.remove('FEB')
features.remove('MAR')
features.remove('APR')
features.remove('OCT')
features.remove('NOV')
features.remove('DEC')
features.remove('AUG')
features.remove('SEP')
features.remove('JUL')
features.remove('JUN')
print(features)


# In[106]:


features2 = ['JUL']


# In[107]:


# Separating out the features
r  = df2.loc[:, features].values
# Separating out the target
s = df2.loc[:,features2].values


# In[108]:


from sklearn.model_selection import train_test_split
R_train, R_test, s_train, s_test = train_test_split(r, s, test_size=0.33, random_state=42)


# In[109]:


from sklearn import svm
from sklearn.datasets import make_classification
clf = svm.SVC()
R_train, s_train = make_classification()
R_test, s_test = make_classification()
clf.fit(R_train, s_train)
clf.predict(R_test)
clf.score(R_test, s_test)


# In[110]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(R_train, s_train)
clf.predict(R_test)
clf.score(R_test, s_test)


# In[111]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(R_train, s_train)
clf.predict(R_test)
clf.score(R_test, s_test)


# In[117]:


ax=plt.figure(figsize=(30,20))
ax=sns.countplot(x="SUBDIVISION",palette="inferno",data=data)
ax.set_xlabel("Classes")
ax.set_ylabel("Count")
ax.set_title("Subdivision count")


# In[ ]:




