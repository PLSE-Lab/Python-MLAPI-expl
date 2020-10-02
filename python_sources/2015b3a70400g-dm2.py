#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[ ]:





# In[ ]:


pd.set_option('display.max_columns', 500)
df = pd.read_csv("../input/train.csv")
df.head()


# In[ ]:


to_keep = ['ID', 'Age', 'IC', 'OC', 'Schooling', 'Timely Income',
       'Married_Life', 'Cast', 'Hispanic', 'Sex',
       'Full/Part', 'Gain', 'Loss', 'Stock', 'Tax Status',
       'Summary', 'Weight',
       'NOP',
       'Citizen', 'Own/Self', 'Vet_Benefits', 'Weaks', 'WorkingPeriod',
       'Class']


# In[ ]:


for i in df.columns:
  print(str(i) + " " + str(len(df[i].unique())))


# In[ ]:


df2 = df[to_keep]


# In[ ]:


df2.head()


# In[ ]:


df2 = df2.replace({"?": np.nan})


# In[ ]:


df2 = df2.fillna(df.mode().iloc[0])


# In[ ]:


df2.isnull().values.any()


# In[ ]:


df2.head()


# In[ ]:


y=df2["Class"]
x=df2.drop(columns=["ID", "Class"])


# In[ ]:


x_new = pd.get_dummies(x, columns=['Schooling', 'Married_Life', 'Cast',
        'Hispanic', 'Sex', 'Full/Part', 'Tax Status',
        'Summary', 'NOP', 'Citizen', 'Own/Self', 'Vet_Benefits',])


# In[ ]:


cols = x_new.columns


# In[ ]:


x_new.head()


# In[ ]:


abc = AdaBoostClassifier(random_state=42)
abc = abc.fit(x_new,y)


# In[ ]:


f_importances = abc.feature_importances_


# In[ ]:


print(f_importances)


# In[ ]:


imp_index = np.argsort(f_importances)


# In[ ]:


print(imp_index)


# In[ ]:


print([cols[i] for i in imp_index])


# In[ ]:


sel_features = cols[imp_index[-30:]]
print(str(sel_features))
X_train = x_new[[cols for cols in sel_features]]


# In[ ]:


X_train.head()


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)


# In[ ]:


ros = RandomOverSampler(random_state=42)
X_r, y_r = ros.fit_resample(X_train, y_train)


# In[ ]:





# In[ ]:


abc2 = AdaBoostClassifier(n_estimators=500, random_state = 42)
abc2.fit(X_r,y_r)


# In[ ]:


y_pred_val = abc2.predict(X_val)


# In[ ]:


roc_auc_score(y_val,y_pred_val)


# In[ ]:


#pd.set_option('display.max_columns', 500)
df_test = pd.read_csv("../input/test.csv")
df_test.head()


# In[ ]:


to_keep2 = ['ID', 'Age', 'IC', 'OC', 'Schooling', 'Timely Income',
       'Married_Life', 'Cast', 'Hispanic', 'Sex',
       'Full/Part', 'Gain', 'Loss', 'Stock', 'Tax Status',
       'Summary', 'Weight',
       'NOP',
       'Citizen', 'Own/Self', 'Vet_Benefits', 'Weaks', 'WorkingPeriod']


# In[ ]:


dft2 = df_test[to_keep2]


# In[ ]:


dft2.head()


# In[ ]:


dft2 = dft2.replace({"?": np.nan})


# In[ ]:


dft2 = dft2.fillna(dft2.mode().iloc[0])


# In[ ]:


dft2.isnull().values.any()


# In[ ]:


dft2.head()


# In[ ]:


ids=dft2["ID"]
x_test=dft2.drop(columns=["ID"])


# In[ ]:


x_test = pd.get_dummies(x_test, columns=['Schooling', 'Married_Life', 'Cast',
        'Hispanic', 'Sex', 'Full/Part', 'Tax Status',
        'Summary', 'NOP', 'Citizen', 'Own/Self', 'Vet_Benefits',])


# In[ ]:


x_test.head()


# In[ ]:


x_test = x_test[sel_features]


# In[ ]:


x_test.head()


# In[ ]:


len(x_test.columns)


# In[ ]:


y_test = abc2.predict(x_test)


# In[ ]:


len(y_test)


# In[ ]:


dicty= {"ID" : list(ids), "Class" : list(y_test)}


# In[ ]:


sub=pd.DataFrame.from_dict(dicty)


# In[ ]:


sub=sub[['ID', 'Class']]


# In[ ]:


sub.head()


# In[ ]:


sub.to_csv("last_submission.csv", index=False)


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

