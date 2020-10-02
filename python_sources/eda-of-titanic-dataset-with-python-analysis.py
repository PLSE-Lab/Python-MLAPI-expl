#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis with Titanic dataset
# 
# 
# 

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[13]:


plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


train_data=pd.read_csv('../input/titanicdataset-traincsv/train.csv')
test_data=pd.read_csv('../input/testrag/test.csv')


# In[17]:


train_data.shape


# In[18]:


train_data.head(10)


# In[19]:


test_data.head()


# In[20]:


train_data.isnull().sum()


# In[21]:


sb.countplot('Survived',data=train_data)
plt.show()


# From the above graph it is clear that not many persons survived.
# Out of 891 persons in training dataset only 350, 38.4% of total training dataset survived. We will get more insight of data by exploring more.

# Here we'll explore features

# In[22]:


train_data.groupby(['Sex', 'Survived'])['Survived'].count()


# It is clear that 233 female survived out of 344. And out of 577 male 109 survived. The survival ratio of female is much greater than that of male. It can be seen clearly in following graph

# In[23]:


train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
sb.countplot('Sex',hue='Survived',data=train_data,)
plt.show()


# 'Sex' is very interesting feature. Isn't it? Let's explore more features

# In[24]:


sb.countplot('Pclass', hue='Survived', data=train_data)
plt.title('Pclass: Sruvived vs Dead')
plt.show()


# Wow.... That looks amazing. It is usually said that Money can't buy Everything, But it is clearly seen that pasangers of Class 1 are given high priority while Rescue. There are greater number of passangers in Class 3 than Class 1 and Class 2 but very few, almost 25% in Class 3 survived. In Class 2, survivail and non-survival rate is 49% and 51% approx.
# While in Class 1 almost 68% people survived. So money and status matters here.
# 
# Let's dive in again into data to check more interesting observations.

# In[25]:


pd.crosstab([train_data.Sex,train_data.Survived],train_data.Pclass,margins=True).style.background_gradient(cmap='summer_r')


# In[26]:


sb.factorplot('Pclass', 'Survived', hue='Sex', data=train_data)
plt.show()


# I use FactorPlot and CrossTab here because with these plots categorical variables can easily be visualized. Looking at FactorPlot and CrossTab, it is clear that women survival rate in Class 1 is about 95-96%, as only 3 out of 94 women died. So, it is now more clear that irrespective of Class, women are given first priority during Rescue. Because survival rate for men in even Class 1 is also very low.
# From this conclusion, PClass is also a important feature.

# In[27]:


print('Oldest person Survived was of:',train_data['Age'].max())
print('Youngest person Survived was of:',train_data['Age'].min())
print('Average person Survived was of:',train_data['Age'].mean())


# In[28]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sb.violinplot('Pclass','Age',hue='Survived',data=train_data,split=True,ax=ax[0])
ax[0].set_title('PClass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sb.violinplot("Sex","Age", hue="Survived", data=train_data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# From above violen plots, following observations are clear,
# 1) The no of children is increasing from Class 1 to 3, the number of children in Class 3 is greater than other two. 
# 2) Survival rate of children, for age 10 and below is good irrespective of Class
# 3) Survival rate between age 20-30 is well and is quite better for women.
# 
# Now, in Age feature we have 177 null values filled with NaN. We have to deal with it. But we can't enter mean of age in every NaN column, because our average/mean is 29 and we cannot put 29 for a child or some olde man. So we have to discover something better. 
# Let's do something more interesting with dataset by exploring more.

# What is, if I look at 'Name' feature, It looks interesting. Let's check it....

# In[29]:


train_data['Initial']=0
for i in train_data:
    train_data['Initial']=train_data.Name.str.extract('([A-Za-z]+)\.') #extracting Name initials


# In[30]:


pd.crosstab(train_data.Initial,train_data.Sex).T.style.background_gradient(cmap='summer_r')


# There are many names which are not relevant like Mr, Mrs etc. So I will replace them with some relevant names,

# In[31]:


train_data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',
                               'Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss',
                                'Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[32]:


train_data.groupby('Initial')['Age'].mean()


# In[33]:


train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Mr'),'Age']=33
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Mrs'),'Age']=36
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Master'),'Age']=5
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Miss'),'Age']=22
train_data.loc[(train_data.Age.isnull()) & (train_data.Initial=='Other'),'Age']=46


# In[34]:


train_data.Age.isnull().any()


# In[35]:


f,ax=plt.subplots(1,2,figsize=(20,20))
train_data[train_data['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived = 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train_data[train_data['Survived']==1].Age.plot.hist(ax=ax[1],bins=20,edgecolor='black',color='green')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
ax[1].set_title('Survived = 1')
plt.show()


# From the above plots, I found the following observations
# 

# (1) First priority during Rescue is given to children and women, as the persons<5 are save by large numbers
# (2) The oldest saved passanger is of 80
# (3) The most deaths were between 30-40

# In[36]:


sb.factorplot('Pclass','Survived',col='Initial',data=train_data)
plt.show()


# From the above FactorPlots it is Clearly seen that women and children were saved irrespective of PClass

# Let's explore some more

# # Feature: SibSip

# SibSip feature indicates that whether a person is alone or with his family. Siblings=brother,sister, etc
# and Spouse= husband,wife

# In[37]:


pd.crosstab([train_data.SibSp],train_data.Survived).style.background_gradient('summer_r')


# In[38]:


f,ax=plt.subplots(1,2,figsize=(20,8))
sb.barplot('SibSp','Survived', data=train_data,ax=ax[0])
ax[0].set_title('SipSp vs Survived in BarPlot')
sb.factorplot('SibSp','Survived', data=train_data,ax=ax[1])
ax[1].set_title('SibSp vs Survived in FactorPlot')
plt.close(2)
plt.show()


# In[39]:


pd.crosstab(train_data.SibSp,train_data.Pclass).style.background_gradient('summer_r')


# There are many interesting facts with this feature. Barplot and FactorPlot shows that if a passanger is alone in ship with no siblings, survival rate is 34.5%. The graph decreases as no of siblings increase. This is interesting because, If I have a family onboard, I will save them instead of saving myself. But there's something wrong, the survival rate for families with 5-8 members is 0%. Is this because of PClass?
# Yes this is PClass,  The crosstab shows that Person with SibSp>3 were all in Pclass3. It is imminent that all the large families in Pclass3(>3) died.

# That are some interesting facts we have observed with Titanic dataset.

# In[ ]:




