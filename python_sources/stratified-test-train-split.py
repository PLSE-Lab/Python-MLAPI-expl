#!/usr/bin/env python
# coding: utf-8

# In the ML processing it is important that the data have close distribution in the train and test set. Let's graphically explore what does standard sklearn train_test_split does with the distribution of the data.

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# We will try this on the famous Titanic dataframe

# In[3]:


df = pd.read_csv('../input/train.csv')
df['Surname'] = df['Name'].apply(lambda l: l.split(',')[0])
y = df['Survived']
X = df.fillna(0)


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# Let's print some data characteristics and compare train (on the left) and test (on the right) set

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(15,14))
plt.subplots_adjust(hspace = 0.5)
plt.subplot(4, 2, 1)
sns.barplot(x = "Pclass",y = "Survived",data=X_train,linewidth=2)

plt.subplot(4, 2, 2)
sns.barplot(x = "Pclass",y = "Survived",data=X_test,linewidth=2)

plt.subplot(4, 2, 3)
sns.barplot(x = "Sex",y = "Survived",data=X_train,linewidth=2)

plt.subplot(4, 2, 4)
sns.barplot(x = "Sex",y = "Survived",data=X_test,linewidth=2)


plt.subplot(4, 2, 5)
sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=X_train)

plt.subplot(4, 2, 6)
sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=X_test)

plt.subplot(4,2,7)
sns.distplot(X_train['Age'],label='train');
sns.distplot(X_test['Age'],label='test');
plt.legend()

plt.subplot(4,2,8)
sns.distplot(X_train['Pclass'],label='train',bins=3);
sns.distplot(X_test['Pclass'],label='test',bins=3);
plt.title('Comparison of Pclasses')
plt.legend()


fig.show()
plt.show()


# Just to make sure, let's review the distribution of survivors based on their gender

# In[6]:


pd.DataFrame({'train':X_test[['Survived','Sex']].groupby(['Survived','Sex']).size(),'test':X_train[['Survived','Sex']].groupby(['Survived','Sex']).size()})


# Let's evaluate if the distributions are the same for few values using Kolmogorov-Smirnov test. If the p value is smaller than our statistical significance we can reject the null hypothesis that the distributions are the same. However all our tests, should high values of p which suggest that the distributions are the same (considering the values are distributed in the normal distribution)

# In[7]:


from scipy.stats import ks_2samp
print(ks_2samp(X_train['Pclass'],X_test['Pclass']))
print(ks_2samp(X_train['Sex'],X_test['Sex']))
print(ks_2samp(X_train['Age'],X_test['Age']))
print(ks_2samp(X_train['Fare'],X_test['Fare']))
print(ks_2samp(X_train['Surname'],X_test['Surname']))


# When we printed several distribution of the key fields, everything looks well distributed

# The stratify tries to split the major variables into representative groups, however if you look on a feature with big number of values, you can see that the split is not ideal. For example some surnames appear only the train set while some are more significant in the test set (despite high p value above)

# In[8]:


top_values = 15
a = X_train[X_train['Surname'].isin(df['Surname'].value_counts()[:top_values].index)]['Surname'].value_counts()
b = X_test[X_test['Surname'].isin(df['Surname'].value_counts()[:top_values].index)]['Surname'].value_counts()
surname_count = pd.DataFrame({'train':a,'test':b}).fillna(0)
surname_count.plot.bar(rot=45)
plt.show()


# Let's try to do the split specifying stratify on the 'Surname' column

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=X['Surname'])


# Initially it fails, because plenty of surnames appear just once in the data frame. So let's try to update the dataframe to contain only surnames which appear at least twice

# In[10]:


df2 = df[df['Surname'].isin(df['Surname'].value_counts()[df['Surname'].value_counts()>2].index)]
y2 = df2['Survived']
X2 = df2.fillna(0)


# and stratify specifically based on 'Surname' column

# In[11]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.33, random_state=42, stratify=X2['Surname'])


# In[12]:


top_values = 15
a = X2_train[X2_train['Surname'].isin(df['Surname'].value_counts()[:top_values].index)]['Surname'].value_counts()
b = X2_test[X2_test['Surname'].isin(df['Surname'].value_counts()[:top_values].index)]['Surname'].value_counts()
surname_count2 = pd.DataFrame({'train':a,'test':b}).fillna(0)


# In[13]:


fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(15,2))

ax1 = plt.subplot(121)
ax2= plt.subplot(122)

surname_count.plot.bar(rot=45,ax=ax1)
surname_count2.plot.bar(rot=45,ax=ax2)

plt.show()


# In[14]:


print(ks_2samp(X_train['Surname'],X_test['Surname']))
print(ks_2samp(X2_train['Surname'],X2_test['Surname']))


# But how does the other values look like if we specify the stratications

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(15,14))
plt.subplots_adjust(hspace = 0.5)
plt.subplot(4, 2, 1)
sns.barplot(x = "Pclass",y = "Survived",data=X2_train,linewidth=2)

plt.subplot(4, 2, 2)
sns.barplot(x = "Pclass",y = "Survived",data=X2_test,linewidth=2)

plt.subplot(4, 2, 3)
sns.barplot(x = "Sex",y = "Survived",data=X2_train,linewidth=2)

plt.subplot(4, 2, 4)
sns.barplot(x = "Sex",y = "Survived",data=X2_test,linewidth=2)


plt.subplot(4, 2, 5)
sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=X2_train)

plt.subplot(4, 2, 6)
sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=X2_test)

plt.subplot(4,2,7)
sns.distplot(X_train['Age'],label='train');
sns.distplot(X_test['Age'],label='test');
plt.legend()

plt.subplot(4,2,8)
sns.distplot(X_train['Pclass'],label='train',bins=3);
sns.distplot(X_test['Pclass'],label='test',bins=3);
plt.title('Comparison of Pclasses')
plt.legend()


fig.show()
plt.show()


# In[16]:


print(ks_2samp(X2_train['Pclass'],X2_test['Pclass']))
print(ks_2samp(X2_train['Sex'],X2_test['Sex']))
print(ks_2samp(X2_train['Age'],X2_test['Age']))
print(ks_2samp(X2_train['Fare'],X2_test['Fare']))
print(ks_2samp(X2_train['Surname'],X2_test['Surname']))


# Nevertheless we will never get the ideal distribution in relation to all the variables. Look at the distribution of survivors in the 1st class based on fare on the above chart. 
