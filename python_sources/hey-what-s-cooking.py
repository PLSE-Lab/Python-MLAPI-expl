#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# To create plots
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

# To create nicer plots
import seaborn as sns

from ipywidgets import interact,interactive, fixed, interact_manual
from ipywidgets.widgets import Select


# To create interactive plots
import plotly
import plotly.offline as pyo
import plotly.graph_objs as go

# Set notebook mode to work in offline
pyo.init_notebook_mode(connected=True)
import plotly.graph_objs as go


import os
print(os.listdir("../input"))


import re, string, unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


Cuisine_count=train.groupby(['cuisine'])['id'].count().sort_values(ascending=False)


# In[ ]:


def barplot(table_name,title_name,x_name,y_name):
            n = table_name.shape[0]
            colormap = get_cmap('viridis')
            colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1/(n-1))]
            data = go.Bar(x = table_name.index,
              y = table_name,
              marker = dict(color = colors)
             )
            layout = go.Layout(title = title_name,
                   xaxis = dict(title = x_name),
                   yaxis = dict(title = y_name))
            fig = go.Figure(data=[data], layout=layout)
            pyo.iplot(fig)


# In[ ]:


barplot(Cuisine_count,'Number of Cuisine Recipe','Cuisine','Number of Recipe')


# In[ ]:


def text_clean(text):
    text=re.sub(r'([^a-zA-Z\s]+?)','',text)
    text=re.sub(' ','',text)
    text=re.sub('P{P}+','',text)
    return text


# In[ ]:


train['Data'] = 'Train'
test['Data'] = 'Test'
both_df = pd.concat([train, test], axis=0,sort=False).reset_index(drop=True)
both_df['Ing']=" "


# In[ ]:


both_df.tail()


# In[ ]:


ingredi=[]
for i,item in both_df.iterrows():
    ingredient=[]
    for ingre in item.ingredients:
        ingred=text_clean(ingre)
        if ingred not in ingredient:
            ingredient.append(ingred)
    ingredi.append(ingredient)
both_df['Ing']=ingredi


# In[ ]:


cuisine=[]
ingred=[]
id_=[]

for i,row in train.iterrows():
    cusine=row.cuisine
    id=row.id
    for ingredient in row.ingredients:
        cuisine.append(cusine)
        ingred.append(ingredient)
        id_.append(id)


# In[ ]:


data=pd.DataFrame({'id':id_,'target':cuisine,'ingredient':ingred})


# In[ ]:


data.groupby(['id'])['ingredient'].count().hist(bins=50)


# In[ ]:


data.groupby(['ingredient'])['target'].count().sort_values(ascending=False)[:15]


# In[ ]:


data[data['ingredient']=='hot pepperoni']                          


# In[ ]:


data[data['ingredient'].str.contains('pepperoni')].groupby(['ingredient','target']).count()


# In[ ]:


unique=[]

def unique_words(text):
    v=text.split(' ',)
    for i in v:
        if i not in unique:
            unique.append(i)
        
for i,item in data.iterrows():
    unique_words(item.ingredient)


# In[ ]:


data.head()


# In[ ]:


@interact(Cuisine=Cuisine_count.index)

def plot(Cuisine):
    res=data[data['target']==Cuisine].groupby('ingredient')['id'].count().sort_values(ascending=False)[:10]
    tile_name='Top Ingredients for'+' '+str.upper(Cuisine) +' '+ 'Cuisine'
    barplot(res,tile_name,'Ingredient','Count') 


# In[ ]:


ingred_c=data.groupby('ingredient')['target'].unique()


# In[ ]:


Ingred_Cu=pd.DataFrame({'ingred':ingred_c.index,'Cuisine':ingred_c.values})


# In[ ]:


@interact(Ingredient=Ingred_Cu['ingred'])

def Ing(Ingredient):
    print('The ' + str.upper(Ingredient)+ ' is added in the following Ingredients:')
    for item in Ingred_Cu[Ingred_Cu['ingred']==Ingredient].Cuisine:
        for p in item:
            print(p)


# In[ ]:


both_df['Ing'] = both_df['Ing'].map(";".join)


# In[ ]:


both_df.head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X = cv.fit_transform(both_df['Ing'])


# In[ ]:


X.shape


# In[ ]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(both_df[both_df['Data']=='Train']['cuisine'])


# In[ ]:


enc.classes_


# In[ ]:


X1=X[0:39774,]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2)


# In[ ]:


print(list(cv.vocabulary_.keys())[:100])


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logistic = LogisticRegression()
logistic.fit(X_train,y_train)


# In[ ]:


logistic.score(X_test, y_test)


# In[ ]:


from sklearn.metrics import confusion_matrix

plt.figure(figsize=(10, 10))

cm = confusion_matrix(y_test, logistic.predict(X_test))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.imshow(cm_normalized, interpolation='nearest')
plt.title("confusion matrix")
plt.colorbar(shrink=0.3)
cuisines = both_df[both_df['Data']=='Train']['cuisine'].value_counts().index
tick_marks = np.arange(len(cuisines))
plt.xticks(tick_marks, cuisines, rotation=90)
plt.yticks(tick_marks, cuisines)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[ ]:


name={'0':'brazilian', '1':'british', '2':'cajun_creole', '3':'chinese', '4':'filipino',
       '5':'french', '6':'greek', '7':'indian', '8':'irish', '9':'italian', '10':'jamaican',
       '11':'japanese', '12':'korean', '13':'mexican', '14':'moroccan', '15':'russian',
       '16':'southern_us', '17':'spanish', '18':'thai', '19':'vietnamese'}


# In[ ]:


from sklearn.metrics import classification_report
y_pred = logistic.predict(X_test)
print(classification_report(y_test, y_pred, target_names=cuisines))


# In[ ]:


X2=X[39774:,]


# In[ ]:


pred=logistic.predict(X2)


# In[ ]:


test_id=both_df[both_df['Data']=='Test']['id']


# In[ ]:


sub=pd.DataFrame({'id':test_id,'cuisine':pred})


# In[46]:


sub['cuisine']=sub['cuisine'].astype(str).replace(name)


# In[47]:


sub.to_csv('sample_submission.csv',index=False)


# In[ ]:




