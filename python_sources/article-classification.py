#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))  
        df=pd.read_json(os.path.join(dirname,filename),lines=True)

# Any results you write to the current directory are saved as output.


# In[ ]:


df.category.unique()


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder 
from sklearn.utils import shuffle,resample
from sklearn.feature_extraction.text import TfidfVectorizer 
import os 


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))  
        df=pd.read_json(os.path.join(dirname,filename),lines=True)

b1=df['category']=='HEALTHY LIVING'
b2=df['category']=='SPORTS'
b3=df['category']=="SCIENCE"
b=np.logical_or(b1,b2)
df_filter=df[np.logical_or(b,b3)]

df_filter.category.value_counts().plot(kind="bar")

encoder=LabelEncoder()
df_filter["label"]=encoder.fit_transform(df_filter.category)
df_filter=shuffle(df_filter)
df_resample=df_filter[df_filter.category=="HEALTHY LIVING"] 
df_resample_science=df_filter[df_filter.category=="SCIENCE"].sample(len(df_resample),replace=True)
df_resample_sport=df_filter[df_filter.category=="SPORTS"].sample(len(df_resample),replace=True)
df_resample=pd.concat([df_resample,df_resample_science])
df_resample=pd.concat([df_resample,df_resample_sport])
df_resample=shuffle(df_resample)


tfidfvect=TfidfVectorizer(stop_words="english",max_features=18000)
data=tfidfvect.fit_transform(df_resample.short_description)


# ### Creating A French DataBase fot the 3 categories
# 
# For this Task we will be using the translation api <strong>YenDex</strong>. We did just as a demo in case we want to build a French dataset from a set of articles written in English.
# 

# In[ ]:


pip install yandex-translater


# In[ ]:


from yandex.Translater import Translater  

list_fr=[]
 
def translate_yendex_en_fr(s): 
    tr=Translater() 
    tr.set_key("API-KEY")
    tr.set_from_lang("en")
    tr.set_to_lang("fr")
    try:
        tr.set_text(s)
        f=tr.translate()
        print(f)
        return f
    except Exception as e:
        return None
        
#df_filter["short_description_fr"]=df_filter["short_description"].apply(lambda x:translate_yendex_en_fr(x))
for i in range(100):
    list_fr.append(translate_yendex_en_fr(df_filter["short_description"].iloc[i]))


# ## Creating ML Models which Classifies articles into 3 distinct categories
# 
# In the Code below we will test the performance of a set of Machine Learning algorithms predefined in the SKlearn python library. 
# <ul>
#     <li>Multinomial Naive Bayes</li>
#     <li>Random Forest Classifier</li>
#     <li>Gradient Boosting Classifier</li>
#     <li>Multi Layers Perceptron Classifier</li>
#     <li>Extreme Gradient Boosting Classifier (XGBoost)</li>
# </ul>
# 
# The metrics used to compare the performance of each algorithm are:
# <ul>
#     <li><strong>Accuracy: </strong> <em> (TP+TN)/(TN+TP+FP+FN)</em></li>
#     <li><strong>Precision: </strong> <em> (TP)/(TP+FN)</em></li>
#     <li><strong>Recall: </strong> <em> (TP)/(TP+FP)</em></li>
#     <li><strong>F1-Score: </strong> which is the harmonic mean between Precision and Recall<em>2*(Recall*Precision)/(Recall+Precision)</em></li>
# </ul>
# All those metrics can be gathered all together using the classification_report from the SKlearn library. We will also use a confusion matrix to visualize all this.

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.pipeline import Pipeline

clfs={"Multinomial Naivs Bayes":MultinomialNB(),"Random Forest Classifier":RandomForestClassifier(),
      "GBM Classifier":GradientBoostingClassifier(),"MLP Classifier":MLPClassifier(),"XGB Classifier":XGBClassifier()}
x_train,x_test,y_train,y_test=train_test_split(data,df_resample.label,stratify=df_resample.label,test_size=0.2)


# In[ ]:


for k,clf in clfs.items(): 
    print(k)
    clfs[k]=clfs[k].fit(x_train,y_train)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns 

y_preds=[]
classification_reports=[]
for clf in clfs.values():
    y_pred=clf.predict(x_test)
    y_preds.append(y_pred)
    classification_reports.append(classification_report(y_test,y_pred))


# In[ ]:


for clf_report,k in zip(classification_reports,clfs.keys()): 
    print(f"classifcation report for {k}")
    print(clf_report)


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
#d={"accuracy":[],"recall":[],"precision":[],"f1_score":[],"roc_auc":[]}
confusions=[]
for y_pred,k in zip(y_preds,clfs.keys()): 
    sns.heatmap(confusion_matrix(y_pred,y_test))
    plt.title(f"Metrics Evaluation of{k}",color="white")
    plt.show()


# From the above results, we can conclude that RandomForestClassifier and Multi-Layer Perceptron performed best. With more advanced hardware infrastructure we would have applied a Grid Search to perform a Hyper-Parameter tuning which may result to better performance.
