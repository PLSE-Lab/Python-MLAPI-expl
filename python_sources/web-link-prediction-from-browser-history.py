#!/usr/bin/env python
# coding: utf-8

# Here, We will predict the frecency value from browser history and suggest the favourite links to the user. According to frecency value we can give prediction to user. Here is the demo: [Prediction System](http://myresearch.epizy.com/) . You can read this [paper](https://arxiv.org/abs/1902.08496 ) to understand the link prediction part. Category wise recommendation will be shown on next kernel of this dataset. For website or URL classification you can take help from this [Kernel](https://www.kaggle.com/shawon10/url-classification-by-naive-bayes) . You can also read our another [paper](https://lnkd.in/g3gNg3P) for Website/URL classification.
# 
# ![](https://media.giphy.com/media/7zAIkF2ZsdjTJi7htC/giphy.gif)

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns


# In[ ]:


names=['URL','First_Visit','Last_Visit','Click_Count','Frecency']
nn=['URL','First_Visit','Last_Visit','Click_Count','Frecency']
df=pd.read_csv('../input/TrainingHistory.csv',names=names, na_filter=False)
dt=pd.read_csv('../input/TestingHistory.csv',names=nn, na_filter=False)
df.head(100)


# In[ ]:


sns.set(font_scale=1.5)
g=sns.pairplot(data=df,x_vars=['First_Visit','Last_Visit','Click_Count'],y_vars='Frecency', kind='reg',size=5)


# In[ ]:


y=np.array(df[names[4]]) #train
yt=np.array(dt[nn[4]]) #test


# In[ ]:


X=np.array(df[['First_Visit','Last_Visit','Click_Count']]) #train
Xt=np.array(dt[['First_Visit','Last_Visit','Click_Count']]) #test


# In[ ]:


clf=LinearRegression().fit(X,y) #training by linear regression
mse = mean_squared_error(yt, clf.predict(Xt)) #mean square error
print("MSE: %.4f" % mse)
rmse=np.sqrt(mse)
print("RMSE: %.4f" % rmse) #root mean square error


# In[ ]:


y_pred=clf.predict(Xt) #prediction
print(y_pred)


# In[ ]:


import matplotlib.patches as mpatches
Xp=dt['Click_Count']
plt.scatter(Xp, yt)
plt.plot(Xp, y_pred, color='red', label='Predicted Relationship')
x_actual = Xp
y_actual = yt
plt.plot(x_actual, y_actual, color='green', label='Actual Relationship')
plt.xlabel('Features')
plt.ylabel('Frecency')
plt.legend()
plt.rcParams["figure.figsize"] = [6,5]
plt.show()


# In[ ]:


print(clf.score(Xt,yt))


# In[ ]:


Xf=np.array(dt[['URL','First_Visit','Last_Visit','Click_Count']])


# In[ ]:


yf=clf.predict(Xt) #prediction
yf=yf.reshape(2305,1)
Xd=np.append(Xf,yf,axis=1) #appending prediction to test dataset
print(Xd[:,1])


# In[ ]:


print(yf[0:10]) #prediction


# In[ ]:


submission = pd.DataFrame(Xd) #creating dataframe and csv file
submission.to_csv('result.csv', index=False)


# In[ ]:


submission = pd.read_csv('result.csv')
submission.head() #all columns with prediction

