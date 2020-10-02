#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[25]:


data_orig = pd.read_csv("../input/train.csv", sep=',')
data = data_orig
data_test_orig = pd.read_csv("../input/test.csv", sep=',')
data_test = data_test_orig.copy()


# In[26]:


data.replace('?',np.nan,inplace=True)
data_test.replace('?',np.nan,inplace=True)


# In[27]:


col_drop = ['Worker Class','Enrolled','MIC','MOC','MLU','Reason','Area','State','MSA','REG',
           'MOVE','Live','PREV','Teen','Fill','ID']


# In[28]:


data.drop(col_drop,axis=1,inplace=True)


# In[29]:


data_test.drop(col_drop,axis=1,inplace=True)


# In[30]:


data_test['Hispanic'].replace(np.nan,'HA',inplace=True)
data['Hispanic'].replace(np.nan,'HA',inplace=True)


# In[31]:


columns_to_drop = ['COB FATHER','COB MOTHER','COB SELF','Detailed']


# In[32]:


data.drop(columns_to_drop,axis=1, inplace=True)
data_test.drop(columns_to_drop,axis=1, inplace=True)


# In[33]:


drop_again = ['WorkingPeriod','Weight']


# In[34]:


data.drop(drop_again,axis=1, inplace=True)
data_test.drop(drop_again,axis=1, inplace=True)


# In[35]:


categorical_columns = ['Married_Life','Cast','Hispanic','Sex','Full/Part','Tax Status','Summary',
                       'NOP','Citizen','Vet_Benefits','Own/Self','Schooling']


# In[36]:


data1 = pd.get_dummies(data, columns=categorical_columns)
data_test_dummies = pd.get_dummies(data_test, columns=categorical_columns)


# In[37]:


y=data1['Class']
X=data1.drop(['Class'],axis=1)
X.head()


# In[38]:


data_test_dummies.head()


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=45)


# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score, roc_auc_score,make_scorer
from sklearn.ensemble import AdaBoostClassifier


# In[41]:


y_val.value_counts()


# In[ ]:


rf = RandomForestClassifier(random_state = 42, n_estimators=50, max_depth=16, min_samples_split=468, criterion='gini', 
                            max_features=14, bootstrap=True, class_weight='balanced_subsample')

model = AdaBoostClassifier(base_estimator=rf,n_estimators=5, random_state=45)

model.fit(X_train, y_train)
y_pred_RF = model.predict(X_val)


print(roc_auc_score(y_val,y_pred_RF))
print(confusion_matrix(y_val,y_pred_RF))


# In[ ]:


model.fit(X, y)

preds = model.predict(data_test_dummies)
preds = pd.DataFrame(preds)

final = pd.concat([data_test_orig["ID"], preds], axis=1).reindex()
final = final.rename(columns={0: "Class"})
final['Class'].value_counts()


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
create_download_link(final) 
 
 


# In[ ]:




