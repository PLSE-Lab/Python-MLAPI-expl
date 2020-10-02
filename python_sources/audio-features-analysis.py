#!/usr/bin/env python
# coding: utf-8

# # Spotify Features

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


class_names=['dinner','party','sleep','workout']
dfs = []
for i in range(len(class_names)):
    dfs.append(pd.read_csv('../input/'+class_names[i]+'_track.csv'))
    dfs[i]['class'] = i + 1
    dfs[i]['class_names'] = class_names[i]


# In[ ]:


df = pd.concat(dfs)


# In[ ]:


df.describe()


# In[ ]:


df.dropna().describe()


# In[ ]:


df = df.dropna()
df.head()


# In[ ]:


df.columns


# In[ ]:


numerical_features = ['acousticness', 'danceability',
       'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness',
       'loudness', 'mode', 'speechiness', 'tempo', 'time_signature', 'valence']


# In[ ]:


df.shape


# In[ ]:


X = df[numerical_features]
y = df['class']
X.shape


# In[ ]:


mi = mutual_info_classif(X, y)
print(mi)


# Mutual information between the target class and each feature is estimated. Features with high mutual information are "loudness", "acousticness", "energy", and "instrumentalness". The pair plot shows that for instance sleep music is mostly instrumental.

# In[ ]:


sns.pairplot(df,vars= ['loudness','acousticness','energy','instrumentalness'],hue='class_names')


# In[ ]:


sc = StandardScaler()
Xsc = sc.fit_transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Xsc, y, test_size=0.2, random_state=42)


# ## Random Forest Classifier

# In[ ]:


rfc = RandomForestClassifier(1000)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


y_pred = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("accuracy: %.1f%%" % (np.mean(y_test==y_pred)*100))


# classification works quite well.

# # Signal Processing Features

# In[ ]:


class_names=['dinner','party','sleep','workout']
dfs = []
for i in range(len(class_names)):
    dfs.append(pd.read_csv('../input/'+class_names[i]+'_audio.csv'))
    dfs[i]['class'] = i + 1
    dfs[i]['class_names'] = class_names[i]


# In[ ]:


df = pd.concat(dfs)


# In[ ]:


df.describe()


# In[ ]:


df.head()


# In[ ]:


X = df[['mfcc','scem','scom','srom','sbwm','tempo','rmse']]
y = df['class']


# In[ ]:


mi = mutual_info_classif(X, y)
print(mi)


# Mutual information between the target class and each feature is estimated. All features (except "tempo") have high mutual information.

# In[ ]:


sns.pairplot(df,vars= ['mfcc','scem','scom','srom','sbwm','rmse'],hue='class_names')


# The plot shows that the class "sleep" can be separated very good.

# In[ ]:


print(X.corr())


# The features "scem" and "srom" are nearly linear dependent. (pearson correlation is 99.2%)

# In[ ]:


Xsc = StandardScaler().fit_transform(X)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Xsc, y, test_size=0.3, random_state=42)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


y_pred = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("accuracy: %.1f %%"%(np.mean(y_test==y_pred)*100))

