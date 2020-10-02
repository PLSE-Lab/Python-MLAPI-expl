#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


df=pd.read_csv("../input/train.csv",sep=",")


# In[ ]:


df.head()


# In[ ]:


df.columns = [c.replace(' ','_') for c in df.columns]


# In[ ]:


df.info()


# In[ ]:


#list = df[(df['MLU'] == '?')].index.tolist()


# In[ ]:


df['Enrolled'].unique()


# In[ ]:


df = df.drop_duplicates()
df.info()


# In[ ]:


df['Sex'].unique()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);
#mask=np.zeros_like(corr, dtype=np.bool)


# In[ ]:


y_data = df['Class']
x_data = df.drop(['ID','NOP','Weaks','Vet_Benefits','Class'], 1)
x_data.head()


# In[ ]:


df2 = df.drop(df[df.MOC == '?'].index)
repl = df2["MOC"].mode()
repl


# In[ ]:


#x_data['Enrolled'].replace({    '?': 'Uni1' },inplace=True)
#x_data['MLU'].replace({    '?': 'NO' },inplace=True)
#x_data['Live'].replace({    '?': 'YES' },inplace=True)
#x_data['PREV'].replace({    '?': 'NO' },inplace=True)
#x_data['Fill'].replace({    '?': 'NO' },inplace=True)
x_data['Worker_Class'].replace({    '?': 'WC2' },inplace=True)
x_data['MIC'].replace({    '?': 'MIC_N' },inplace=True)
x_data['MOC'].replace({    '?': 'MOC_E' },inplace=True)
x_data['Hispanic'].replace({    '?': 'HA' },inplace=True)
#x_data['Reason'].replace({    '?': 'JL2' },inplace=True)
#x_data['Area'].replace({    '?': 'S' },inplace=True)
x_data['MSA'].replace({    '?': 'StatusA' },inplace=True)
x_data['REG'].replace({    '?': 'StatusA' },inplace=True)
x_data['MOVE'].replace({    '?': 'StatusA' },inplace=True)
x_data['Teen'].replace({    '?': 'B' },inplace=True)
x_data.head()


# In[ ]:


x_data = x_data.drop(['Enrolled','MLU','Reason','Area','PREV','Fill','Live'],1)
x_data = x_data.drop(['State','COB_FATHER','COB_MOTHER','COB_SELF'],1)
x_data = x_data.drop(['Detailed'],1)


# In[ ]:


x_data = pd.get_dummies(x_data,columns = ['Schooling','Sex','Cast','Citizen','Married_Life','Full/Part','Tax_Status','Summary',
                                          'MSA','REG','MOVE','Worker_Class','Hispanic','MIC','MOC','Teen'])
x_data.head()


# In[ ]:


x_data.info()


# In[ ]:


np.random.seed(42)


# In[ ]:


from sklearn.naive_bayes import GaussianNB as NB
#NB?


# In[ ]:


nb = NB()
#nb.fit(X_train,y_train)
#nb.score(X_val,y_val)
nb.fit(x_data,y_data)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

#y_pred_NB = nb.predict(X_val)
#print(confusion_matrix(y_val, y_pred_NB))

y_pred_NB = nb.predict(x_data)
print(confusion_matrix(y_data, y_pred_NB))


# In[ ]:


print(classification_report(y_data, y_pred_NB))


# In[ ]:


df1 = pd.read_csv("../input/test.csv",sep=",")


# In[ ]:


df1.info()


# In[ ]:


df1['Fill'].unique()


# In[ ]:


df1.head()


# In[ ]:


df1.columns = [c.replace(' ','_') for c in df1.columns]
df1.info()


# In[ ]:


#df1['Live'].replace({    '?': 'YES' },inplace=True)
#x_data['PREV'].replace({    '?': 'NO' },inplace=True)
#x_data['Fill'].replace({    '?': 'NO' },inplace=True)
df1['Worker_Class'].replace({    '?': 'WC2' },inplace=True)
df1['Hispanic'].replace({    '?': 'HA' },inplace=True)
df1['MIC'].replace({    '?': 'MIC_N' },inplace=True)
df1['MOC'].replace({    '?': 'MOC_E' },inplace=True)
#x_data['Reason'].replace({    '?': 'JL2' },inplace=True)
#x_data['Area'].replace({    '?': 'S' },inplace=True)
df1['MSA'].replace({    '?': 'StatusA' },inplace=True)
df1['REG'].replace({    '?': 'StatusA' },inplace=True)
df1['MOVE'].replace({    '?': 'StatusA' },inplace=True)
df1['Teen'].replace({    '?': 'B' },inplace=True)
df1.head()


# In[ ]:


df_t = df1.drop(['ID','Weaks','NOP','Vet_Benefits',
                  'State','COB_FATHER','COB_MOTHER','COB_SELF','Detailed',
                  'Enrolled','MLU','Reason','Area','PREV','Fill','Live'],1)
df_t.head()


# In[ ]:


df_t = pd.get_dummies(df_t,columns = ['Schooling','Sex','Cast','Citizen','Married_Life','Full/Part','Tax_Status','Summary',
                                      'Worker_Class','MIC','MOC','Hispanic','MSA','REG','MOVE','Teen'])
df_t.head()


# In[ ]:


df_t.info()


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

y_pred_NB = nb.predict(df_t)
y_pred_NB
#print(confusion_matrix(y_val, y_pred_NB))


# In[ ]:


resfinal = pd.DataFrame(y_pred_NB)
final = pd.concat([df1["ID"], resfinal], axis=1).reindex()
final = final.rename(columns={0: "Class"})
final.tail()


# In[ ]:


final['Class']=final.Class.astype(int) 
final.head()


# In[ ]:


final.info()


# In[ ]:


final.to_csv('submission_NB.csv', index = False)


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
 
 

