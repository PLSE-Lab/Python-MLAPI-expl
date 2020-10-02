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

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cufflinks as cf
import sklearn
from sklearn import svm,preprocessing
import seaborn as sns
import plotly.graph_objs as go
import plotly
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import os


# In[ ]:


import pandas as pd
data_more = pd.read_csv("../input/diamond-price-prediction/data-more.csv")
df= pd.read_csv("../input/diamond-price-prediction/diamonds.csv",index_col=0)


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


sns.FacetGrid(df,hue='cut',height=6).map(sns.distplot,'price').add_legend()
plt.plot()


# #### Preprocessing Data

# ##### 2.1 Categorical data to numerical Data

# In[ ]:


cut_dict={'Fair':1,'Good':2,'Very Good':3,'Premium':4,'Ideal':5}
clarity_dict={'I1':1,'SI2':2,'SI1':3,'VS2':4,'VS1':5,'VVS2':6,'VVS1':7,'IF':8}
color_dict={'D':7,'E':6,'F':5,'G':4,'H':3,'I':2,'J':1}


# In[ ]:


df['cut']=df['cut'].map(cut_dict)
df['clarity']=df['clarity'].map(clarity_dict)
df['color']=df['color'].map(color_dict)


# #### 2.2 Dropping additional index attribute

# In[ ]:


df.head()


# In[ ]:


df.isnull().any()


# In[ ]:


df.isnull().sum()


# In[ ]:


df=sklearn.utils.shuffle(df,random_state=42)
X=df.drop(['price'],axis=1).values
X=preprocessing.scale(X)
y=df['price'].values
y=preprocessing.scale(y)


# In[ ]:


test_size=200
X_train=X[:-test_size]
y_train=y[:-test_size]
X_test=X[-test_size:]
y_test=y[-test_size:]


# ### Data Modeling

# In[ ]:





# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
score=[]
for k in range(1,20):
    clf=KNeighborsRegressor(n_neighbors=k,weights='distance',p=1)
    clf.fit(X_train,y_train)
    score.append(clf.score(X_test,y_test))


# In[ ]:


trace0=go.Scatter(
y=score,
x=np.arange(1,len(score)+1),
mode='lines+markers',
marker=dict(
    color='rgb(150,10,10)'
    )
)
layout=go.Layout(
     title='',
     xaxis=dict(title='K value',
                tickmode='linear'
               ),
    yaxis=dict(
        title='Score'
    ))
fig=go.Figure(data=[trace0],layout=layout)
iplot(fig,filename='basic-line')


# In[ ]:




