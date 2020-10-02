#!/usr/bin/env python
# coding: utf-8

# ### Loading libraries

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier')


# #### Reading test and train data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport os\nprint(os.listdir("../input"))\n# Any results you write to the current directory are saved as output.\n\ndf_train = pd.read_csv(\'../input/train.csv\')\ndf_test = pd.read_csv(\'../input/test.csv\')')


# #### INPUT size

# In[ ]:


df_train.shape


# In[ ]:


df_test.shape


# In[ ]:


df_train.head(2)


# In[ ]:


df_test.head(2)


# ### Checking for missing  values

# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# In[ ]:





# ### Date split

# ##### Train data

# In[ ]:



df_train.date  = pd.to_datetime(df_train.date, format='%Y-%m-%d')


# In[ ]:


df_train['year'] = df_train.date.dt.year
df_train['month']=df_train.date.dt.month
df_train['day']=df_train.date.dt.day


# In[ ]:


df_train.head(2)


# ##### Test data

# In[ ]:



df_test.date  = pd.to_datetime(df_test.date, format='%Y-%m-%d')
df_test['year'] = df_test.date.dt.year
df_test['month']=df_test.date.dt.month
df_test['day']=df_test.date.dt.day


# In[ ]:


df_test.head(2)


# ### Categorical conversion

# In[ ]:


df_train['year']=df_train['year'].astype('category')
df_train['month']=df_train['month'].astype('category')
df_train['day']=df_train['day'].astype('category')
df_train['store']=df_train['store'].astype('category')
df_train['item']=df_train['item'].astype('category')
df_train['sales']=df_train['sales'].astype('category')


# In[ ]:


y=pd.DataFrame()
y['sales']=df_train['sales']


# #### Dropping columns

# In[ ]:


df_train=df_train.drop(columns='date',axis=1)


# In[ ]:


df_train=df_train.drop(columns='sales',axis=1)


# ### Checking datatypes

# In[ ]:


df_train.dtypes


# #### Test data 

# In[ ]:


df_test['year']=df_test['year'].astype('category')
df_test['month']=df_test['month'].astype('category')
df_test['day']=df_test['day'].astype('category')
df_test['store']=df_test['store'].astype('category')
df_test['item']=df_test['item'].astype('category')
df_test=df_test.drop(columns='date',axis=1)


# #### Droping columns

# In[ ]:


df_test=df_test.drop(columns='id',axis=1)


# #### Checking datatypes 

# In[ ]:


df_test.dtypes


# ### Modelling with Random forest classifier

# In[ ]:


clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[ ]:


x=df_train.iloc[:,0:5]  


# In[ ]:


clf=clf.fit(x,y)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'output=clf.predict(df_test)\nresult=pd.DataFrame(output)\nresult')


# ### Submission

# In[ ]:


test=pd.read_csv('../input/test.csv',usecols=['id'])
fin=pd.DataFrame(test)
fin['sales']=result
fin.to_csv('Sales.csv',index=False)
 

