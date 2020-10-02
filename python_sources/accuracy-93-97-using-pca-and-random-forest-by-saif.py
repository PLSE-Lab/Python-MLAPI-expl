#!/usr/bin/env python
# coding: utf-8

# In[ ]:


ls '../input'


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[ ]:




import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

data1=pd.read_csv('../input/glass/glass.csv')


data1= pd.DataFrame(data=data1)
total_miss = data1.isnull().sum()
perc_miss = total_miss/data1.isnull().count()*100

missing_data = pd.DataFrame({'Total missing':total_miss,
                            '% missing':perc_miss})

missing_data.sort_values(by='Total missing',
                         ascending=False).head()


# In[ ]:


print(data1.shape)

data1.head()


# In[ ]:


data1['Type'].unique()


# In[ ]:


#Create an empty list called traget
target = []
for i in data1['Type']:
    if i >= 1 and i <= 4:
        target.append('1')
    elif i >= 5 and i <= 7:
        target.append('2')
data1['Target'] = target


# In[ ]:


data1['Target'].value_counts()


# In[ ]:


data1.drop(['Type'],axis=1,inplace =True)
data1.shape


# In[ ]:


y=data1['Target']
x=data1.drop('Target',axis=1)


# In[ ]:


def fit_random_forest_classifier(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
    clf=RandomForestClassifier()
    clf.fit(x_train,y_train)
    y_preds=clf.predict(x_test)
    mat=confusion_matrix(y_test,y_preds)
    print(mat)
    print(sns.heatmap(mat,annot=True,cmap='bwr',linewidths=.5))
    acc=accuracy_score(y_test,y_preds)
    print(acc)
    return acc
fit_random_forest_classifier(x,y)


# In[ ]:


def do_pca(n_components,data):
    X=StandardScaler().fit_transform(data)
    pca=PCA(n_components)
    x_pca=pca.fit_transform(X)
    return pca,x_pca


# In[ ]:


pca,x_pca=do_pca(2,x) 


# In[ ]:


x_pca.shape


# In[ ]:


x.shape


# In[ ]:


fit_random_forest_classifier(x_pca,y) 


# In[ ]:


accs=[]
for num_features in range(3,5):
    pca,x_pca=do_pca(num_features,x)
    accs.append(fit_random_forest_classifier(x_pca,y))
num_features=list(range(3,5))    


# In[ ]:


accs=[]
for num_features in range(2,100):
    pca,x_pca=do_pca(num_features,x)
    acc=fit_random_forest_classifier(x_pca,y)
    
    accs.append(acc)
    if acc>.95:
        break


# In[ ]:


features = data1.columns[:-1].tolist()
for feat in features:
    skew = data1[feat].skew()
    sns.distplot(data1[feat], kde= False, label='Skew = %.3f' %(skew), bins=30)
    plt.legend(loc='best')
    plt.show()


# In[ ]:


plt.figure(figsize=(8,8))
sns.pairplot(data1[features],palette='coolwarm')
plt.show()


# In[ ]:


corr = data1[features].corr()
plt.figure(figsize=(16,16))
sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},
           xticklabels= features, yticklabels= features, alpha = 0.7,   cmap= 'coolwarm')
plt.show()


# In[ ]:


#count of the target variable
sns.countplot(x='Target', data=data1)


# In[ ]:


sns.boxplot('Target', 'RI', data =data1)


# In[ ]:


sns.boxplot('Target', 'Mg', data =data1)


# In[ ]:


sns.boxplot('Target','Al', data =data1)


# In[ ]:


sns.boxplot('Target','Si', data =data1)


# In[ ]:





# In[ ]:


sns.boxplot('Target','K', data =data1)


# In[ ]:


sns.boxplot('Target','Ca', data =data1)


# In[ ]:





# In[ ]:


sns.boxplot('Target','Ba', data =data1)


# In[ ]:


sns.boxplot('Target','Fe', data =data1)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

data1.hist(figsize = (20, 20))
plt.show()


# In[ ]:





# In[ ]:




