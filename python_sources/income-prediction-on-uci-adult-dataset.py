#!/usr/bin/env python
# coding: utf-8

# # Consensus

# ### Importing libraries

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# ### Loading *csv* file in *dataframe*

# In[8]:


df = pd.read_csv("../input/adult.csv",1,",")
data = [df]
print (df.head())


# ### Convert *salary* to integer

# In[10]:


salary_map={' <=50K':1,' >50K':0}
df['salary']=df['salary'].map(salary_map).astype(int)
    
print (df.head(10))


# ### convert *sex* into *integer*

# In[11]:


df['sex'] = df['sex'].map({' Male':1,' Female':0}).astype(int)

print (df.head(10))
print (("-"*40))
print (df.info())


# ### Find correlation between columns

# In[12]:


def plot_correlation(df, size=15):
    corr= df.corr()
    fig, ax =plt.subplots(figsize=(size,size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)),corr.columns)
    plt.yticks(range(len(corr.columns)),corr.columns)
    plt.show()


# In[6]:


plot_correlation(df)


# ### Categorise in US and Non-US candidates

# In[13]:


print (df[['country','salary']].groupby(['country']).mean())


# ### Drop empty value marked as '?'

# In[14]:


print (df.shape)
df['country'] = df['country'].replace(' ?',np.nan)
df['workclass'] = df['workclass'].replace(' ?',np.nan)
df['occupation'] = df['occupation'].replace(' ?',np.nan)

df.dropna(how='any',inplace=True)

print (df.shape)
print (df.head(10))


# In[17]:



for dataset in data:
    dataset.loc[dataset['country'] != ' United-States', 'country'] = 'Non-US'
    dataset.loc[dataset['country'] == ' United-States', 'country'] = 'US'


# In[18]:


df.head(10)


# ### Convert *country* in *integer*

# In[19]:


df['country'] = df['country'].map({'US':1,'Non-US':0}).astype(int)


# In[20]:


df.head(10)


# ### Data visualisation using histogram

# In[21]:


x= df['hours-per-week']
plt.hist(x,bins=None,density=True,normed=None,histtype='bar')
plt.show()


# In[23]:


df[['relationship','salary']].groupby(['relationship']).mean()


# In[25]:


df[['marital-status','salary']].groupby(['marital-status']).mean()


# ### Categorise marital-status into single and couple

# In[26]:



df['marital-status'] = df['marital-status'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
df['marital-status'] = df['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')

df.head(10)


# In[27]:


df[['marital-status','salary']].groupby(['marital-status']).mean()


# In[28]:


df[['marital-status','relationship','salary']].groupby(['marital-status','relationship']).mean()


# In[29]:


df[['marital-status','relationship','salary']].groupby(['relationship','marital-status']).mean()


# In[30]:



df['marital-status'] = df['marital-status'].map({'Couple':0,'Single':1})
   
df.head(10)


# In[31]:


rel_map = {' Unmarried':0,' Wife':1,' Husband':2,' Not-in-family':3,' Own-child':4,' Other-relative':5}

df['relationship'] = df['relationship'].map(rel_map)
    
df.head(10)


# ### Analyse *race*

# In[32]:


df[['race','salary']].groupby('race').mean()


# In[33]:


race_map={' White':0,' Amer-Indian-Eskimo':1,' Asian-Pac-Islander':2,' Black':3,' Other':4}


df['race']= df['race'].map(race_map)
    
df.head(10)


# In[34]:


df[['occupation','salary']].groupby(['occupation']).mean()


# In[35]:


df[['workclass','salary']].groupby(['workclass']).mean()


# In[36]:


def f(x):
    if x['workclass'] == ' Federal-gov' or x['workclass']== ' Local-gov' or x['workclass']==' State-gov': return 'govt'
    elif x['workclass'] == ' Private':return 'private'
    elif x['workclass'] == ' Self-emp-inc' or x['workclass'] == ' Self-emp-not-inc': return 'self_employed'
    else: return 'without_pay'
    
    
df['employment_type']=df.apply(f, axis=1)

df.head(10)


# In[38]:


df[['employment_type','salary']].groupby(['employment_type']).mean()


# In[37]:


employment_map = {'govt':0,'private':1,'self_employed':2,'without_pay':3}

df['employment_type'] = df['employment_type'].map(employment_map)
df.head(10)


# In[39]:


df[['education','salary']].groupby(['education']).mean()


# In[40]:


df.drop(labels=['workclass','education','occupation'],axis=1,inplace=True)
df.head(10)


# In[41]:


x= df['education-num']
plt.hist(x,bins=None,density=True,normed=None,histtype='bar')
plt.show()


# In[42]:


x=df['capital-gain']
plt.hist(x,bins=None,normed=None)
plt.show()


# In[44]:


df.loc[(df['capital-gain'] > 0),'capital-gain'] = 1
df.loc[(df['capital-gain'] == 0 ,'capital-gain')]= 0


# In[45]:


df.head(25)


# In[46]:


x=df['capital-loss']
plt.hist(x,bins=None)
plt.show()


# In[47]:


df.loc[(df['capital-loss'] > 0),'capital-loss'] = 1
df.loc[(df['capital-loss'] == 0 ,'capital-loss')]= 0

df.head(10)


# In[48]:


df['age'].count()


# ## Applying model for learning

# ### Divide data in training, validation and test dataset

# #### 50% training data, 20% validation data, 30% test data

# In[49]:



from sklearn.model_selection import train_test_split

X= df.drop(['salary'],axis=1)
y=df['salary']

split_size=0.3

#Creation of Train and Test dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split_size,random_state=22)

#Creation of Train and validation dataset
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=5)


# In[51]:


print ("Train dataset: {0}{1}".format(X_train.shape, y_train.shape))
print ("Validation dataset: {0}{1}".format(X_val.shape, y_val.shape))
print ("Test dataset: {0}{1}".format(X_test.shape, y_test.shape))


# ### Let's select few algorithm used for classification

# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# In[54]:


models = []
names = ['LR','Random Forest','Neural Network','GaussianNB','DecisionTreeClassifier','SVM',]

models.append((LogisticRegression()))
models.append((RandomForestClassifier(n_estimators=100)))
models.append((MLPClassifier()))
models.append((GaussianNB()))
models.append((DecisionTreeClassifier()))
models.append((SVC()))


# In[55]:


print (models)


# In[56]:


from sklearn import model_selection
from sklearn.metrics import accuracy_score


# In[58]:



kfold = model_selection.KFold(n_splits=5,random_state=7)

for i in range(0,len(models)):    
    cv_result = model_selection.cross_val_score(models[i],X_train,y_train,cv=kfold,scoring='accuracy')
    score=models[i].fit(X_train,y_train)
    prediction = models[i].predict(X_val)
    acc_score = accuracy_score(y_val,prediction)     
    print ('-'*40)
    print ('{0}: {1}'.format(names[i],acc_score))
    


# ##### Let's proceed further with Random Forest algorithm as it showed good accuracy

# #### Let's predict our test data and see prediction results

# In[59]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[60]:


randomForest = RandomForestClassifier(n_estimators=100)
randomForest.fit(X_train,y_train)
prediction = randomForest.predict(X_test)


# In[61]:


print ('-'*40)
print ('Accuracy score:')
print (accuracy_score(y_test,prediction))
print ('-'*40)
print ('Confusion Matrix:')
print (confusion_matrix(y_test,prediction))
print ('-'*40)
print ('Classification Matrix:')
print (classification_report(y_test,prediction))


# In[ ]:




