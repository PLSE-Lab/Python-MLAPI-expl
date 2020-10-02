#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Reading the train and test dataset ****

# In[ ]:


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')


# **Looking at the description and info of train and test dataset**

# In[ ]:


train.head()


# In[ ]:


train.info()


# In[ ]:


train.describe()


# In[ ]:


test.head()


# In[ ]:


test.describe()


# In[ ]:


test.info()


# **Importing Required ML models from sklearn for prediction purposes**

# In[ ]:



from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# **Converting Target as pandas Series **

# In[ ]:


target=train['Survived']


# > **FEATURE ENGINEERING ****
# 

# 1-Generating new features -->Family and Alone

# In[ ]:


def aloner(d):
    if d>0:
        return 1
    else:
        return 0
train['Family_Size']=train['SibSp']+train['Parch']
train['Alone']=train['Family_Size'].transform(lambda f:aloner(f))
test['Family_Size']=test['SibSp']+test['Parch']
test['Alone']=test['Family_Size'].transform(lambda f:aloner(f))


# 2-Generating 'Honorific' feature from name feature

# In[ ]:


import re
train['Honorific']=train['Name'].transform(lambda f:re.findall('(Mr|Mrs|Master|Miss)',f))
test['Honorific']=test['Name'].transform(lambda f:re.findall('(Mr|Mrs|Master|Miss)',f))


# **VISUALIZATION BY TABLEAU**

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1535710855713' style='position: relative'><noscript><a href='#'><img alt='various features alongside Survived ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic_610&#47;Story1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Titanic_610&#47;Story1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic_610&#47;Story1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1535710855713');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1016px';vizElement.style.height='991px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1535718122614' style='position: relative'><noscript><a href='#'><img alt='Story 2 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic2_32&#47;Story2&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Titanic2_32&#47;Story2' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ti&#47;Titanic2_32&#47;Story2&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1535718122614');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='1016px';vizElement.style.height='991px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# Now looking towards correlation between various features of the dataset 

# In[ ]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1536140804752' style='position: relative'><noscript><a href='#'><img alt='Sheet 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Co&#47;CorrelationInTitanic&#47;Sheet1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='CorrelationInTitanic&#47;Sheet1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Co&#47;CorrelationInTitanic&#47;Sheet1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1536140804752');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>")


# Asseen in the above correlation table.No feature is highly correlated with survival rate
# with highest pearson correlation found is -0.338 with PClass

# **Converting categorical data to numeric form**

# In[ ]:


from sklearn.preprocessing import LabelEncoder as le
from sklearn.preprocessing import OneHotEncoder as ohe
for c in train.columns:
    if train[c].dtype=='object' :
        z1=le().fit_transform(train[c].astype(str))
        train[c]=ohe(sparse=False).fit_transform(z1.reshape(len(z1),1))
        z1=le().fit_transform(test[c].astype(str))
        test[c]=ohe(sparse=False).fit_transform(z1.reshape(len(z1),1))
z1=le().fit_transform(train['Age'].astype(str))
train['Age']=ohe(sparse=False).fit_transform(z1.reshape(len(z1),1))
z1=le().fit_transform(test['Age'].astype(str))
test['Age']=ohe(sparse=False).fit_transform(z1.reshape(len(z1),1))
z1=le().fit_transform(train['Fare'].astype(str))
train['Fare']=ohe(sparse=False).fit_transform(z1.reshape(len(z1),1))
z1=le().fit_transform(test['Fare'].astype(str))
test['Fare']=ohe(sparse=False).fit_transform(z1.reshape(len(z1),1))


# **Dropping Some unnecessary Features**
# cabin has more than 70% of the data missing

# In[ ]:


target_1=train['Survived']
train=train.drop(['PassengerId','Survived','Name','Ticket','Cabin'],axis=1)
test=test[train.columns]


# **Filling NA values***

# In[ ]:


from scipy.stats import mode
train=train.apply(lambda f:f.fillna(f.mode()[0]))
test=test.apply(lambda f:f.fillna(f.mode()[0]))


# In[ ]:


import tensorflow as tf


# In[ ]:


for x in train.columns:
    train[x]=train[x].astype(np.float32)


# In[ ]:


from  sklearn.model_selection import train_test_split as tts
xtrain,ztrain,xtest,ztest=tts(train,target_1,train_size=0.8)


# In[ ]:


import tensorflow as tf
tf.reset_default_graph()
inputs=tf.placeholder(tf.float32,shape=(None,10),name='inputs')
target=tf.placeholder(tf.float32,shape=(None,1),name='target')
layer_1=tf.layers.dense(inputs,5,activation='relu')
layer_2=tf.layers.dense(layer_1,1,activation='sigmoid')
cross_entropy=tf.nn.sigmoid_cross_entropy_with_logits(labels=target,logits=layer_2)
cost=tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)
result=tf.equal(tf.round(layer_2),target)
accuracy=tf.reduce_mean(tf.cast(result,tf.float32))

        
    


# In[ ]:


def batcher():
    i1=0
    i2=0
    batch_size=32
    while i1<len(xtrain):
        yield xtrain.iloc[i1:i1+batch_size,:],xtest[i1:i1+batch_size]
        i1+=32
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(500):
        z=batcher()
        for x,y in z:
            result=sess.run([cost,optimizer,accuracy],feed_dict={'inputs:0':np.array(x).reshape(-1,10),'target:0':np.array(y).reshape(-1,1)})
        result=sess.run([cost,accuracy],feed_dict={'inputs:0':np.array(ztrain).reshape(-1,10),'target:0':np.array(ztest).reshape(-1,1)})
        print("Epoch "+str(e)+" accuracy"+str(result[1])+" loss="+str(result[0]))
    f=sess.run([layer_2],feed_dict={'inputs:0':np.array(test).reshape(-1,10)})
    
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




