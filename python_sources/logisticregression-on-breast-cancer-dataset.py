#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df =pd.read_table('../input/breast-cancer-wisconsin.data.txt', delimiter=',', names=('id number','clump_thickness','cell_size_uniformity','cell_chape_uniformity','marginal_adhesion','epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitoses','class'))


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df["bare_nuclei"] = df["bare_nuclei"][df["bare_nuclei"]!='?']
#removed_those_rows_for_which_bare_nuclei_=_'?'


# In[ ]:


sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis') #visualizing missing_data


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis') #no_missing_data


# In[ ]:


df.info()


# In[ ]:


df["bare_nuclei"] = df["bare_nuclei"].astype("int64")


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


pd.get_dummies(df["class"]).head()


# In[ ]:


df["class"] = pd.get_dummies(df["class"],drop_first=True) #"class" column had 2 values - 2 for benign, 4 for malignant


# In[ ]:





# In[ ]:





# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df[df.columns[1:-1]], df["class"], test_size=0.33, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logreg = LogisticRegression(C=100)


# In[ ]:


logreg.fit(X_train,y_train)


# In[ ]:


pred = logreg.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


cm=confusion_matrix(y_test,pred)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(y_test,pred))


# In[ ]:


logreg.score(X_train,y_train)


# In[ ]:


logreg.score(X_test,y_test)


# In[ ]:


cm


# In[ ]:


df_cm = pd.DataFrame(cm,index = ['AN','AP'],columns=['PN','PP'])


# In[ ]:


sns.heatmap(df_cm,cbar=True,cmap='viridis') #AN-actually negative AP-actually positive PN- predicted neagative PP- predicted positive


# In[ ]:




