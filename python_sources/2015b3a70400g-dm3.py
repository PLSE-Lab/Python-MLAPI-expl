#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


mal=pd.read_csv('../input/opcode_frequency_malware.csv')
ben=pd.read_csv('../input/opcode_frequency_benign.csv')


# In[ ]:


mal.head()


# In[ ]:


ben.head()


# In[ ]:


ben["Class"]=0


# In[ ]:


mal["Class"]=1


# In[ ]:


mal.head()


# In[ ]:


df=ben.append(mal, ignore_index=True)


# In[ ]:


len(df)==len(ben)+len(mal)


# In[ ]:


len(mal)


# In[ ]:


df.head()


# In[ ]:


df['Class'].value_counts()


# In[ ]:


df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# In[ ]:


df.head()


# In[ ]:


df=df.drop(columns=["FileName"])


# In[ ]:


X=df.drop(columns=['Class'])
y=df['Class']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1, random_state=42)


# In[ ]:


len(X_train)/(len(X_train) + len(X_test))


# ##Run and Test

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


abc= AdaBoostClassifier(n_estimators=550, random_state=42)


# In[ ]:


abc.fit(X_train, y_train)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(abc.predict(X_test),y_test)


# In[ ]:


accuracy_score(abc.predict(X_train),y_train)


# In[ ]:


df_test=pd.read_csv('../input/Test_data.csv')
df_test.head()


# In[ ]:


FileName=df_test["FileName"]


# In[ ]:


df_test=df_test.drop(columns=["FileName", "Unnamed: 1809"])


# In[ ]:


df_test.head()


# In[ ]:


ys = abc.predict(df_test)


# In[ ]:


len(FileName)


# In[ ]:


dicty= {"FileName" : list(FileName), "Class" : list(ys)}


# In[ ]:


sub=pd.DataFrame.from_dict(dicty)


# In[ ]:


sub=sub[['FileName', 'Class']]


# In[ ]:


sub.tail()


# In[ ]:


sub.to_csv("subs1.csv", index=False)


# In[ ]:


#cp subs1.csv "gdrive/My Drive/DM3/Subs" 


# In[ ]:


sub["Class"].value_counts()


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

create_download_link(sub)


# In[ ]:




