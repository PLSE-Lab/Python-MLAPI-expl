#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn import preprocessing
#from sklearn import cross_validation, metrics
from sklearn.preprocessing import Normalizer
#from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from pylab import plot, show, subplot, specgram, imshow, savefig


import matplotlib.pyplot as plote

get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')


# In[2]:


train = pd.read_csv("../input/train_u6lujuX_CVtuZ9i.csv") 
test = pd.read_csv("../input/test_Y3wMUE5_7gLdaTN.csv") 


# In[3]:


test.tail()


# In[4]:


train.head()


# In[5]:


train.describe()


# In[6]:


# To check how many columns have missing values - this can be repeated to see the progress made
def show_missing():
    missing = train.columns[train.isnull().any()].tolist()
    return missing


# In[7]:


#from this we can find the total missing data in each columns

train[show_missing()].isnull().sum()


# In[8]:


print (train['Property_Area'].value_counts())
print (train['Education'].value_counts())
print (train['Gender'].value_counts())
print (train['Dependents'].value_counts())
print (train['Married'].value_counts())
print (train['Self_Employed'].value_counts())
print (train['Loan_Status'].value_counts())


# In[9]:


#filling data with approperiate measure of central tendency
train['Gender'] = train['Gender'].fillna( train['Gender'].dropna().mode().values[0] )
train['Married'] = train['Married'].fillna( train['Married'].dropna().mode().values[0] )
train['Dependents'] = train['Dependents'].fillna( train['Dependents'].dropna().mode().values[0] )
train['Self_Employed'] = train['Self_Employed'].fillna( train['Self_Employed'].dropna().mode().values[0] )
train['LoanAmount'] = train['LoanAmount'].fillna( train['LoanAmount'].dropna().mean() )
train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna( train['Loan_Amount_Term'].dropna().mode().values[0] )
train['Credit_History'] = train['Credit_History'].fillna( train['Credit_History'].dropna().mode().values[0] )
train['Dependents'] = train['Dependents'].str.rstrip('+')


# In[10]:


train[show_missing()].isnull().sum()


# In[11]:


train.isnull().sum()


# In[12]:


train.head()


# In[13]:


train.tail()


# In[14]:


import matplotlib.pyplot as plt
plt.hist(train.ApplicantIncome,bins=10)
plt.show()


# In[15]:


train['ApplicantIncome'].hist()


# In[16]:


train['LoanAmount'].hist()


# In[17]:


train['Loan_Amount_Term'].hist()


# In[18]:


ax = train.groupby('Gender').ApplicantIncome.mean().plot(kind='bar')
ax.set_xlabel("Gender")
ax.set_ylabel("mean ApplicantIncom")


# In[19]:


ax = train.groupby('Education').ApplicantIncome.mean().plot(kind='bar')
ax.set_xlabel("Education(1=Graduate)")
ax.set_ylabel("mean ApplicantIncom")


# In[20]:


ax = train.groupby('Married').ApplicantIncome.mean().plot(kind='bar')
ax.set_xlabel("Married(1=yes)")
ax.set_ylabel("mean ApplicantIncom")


# In[21]:


temp3 = pd.crosstab(train['Credit_History'], train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[22]:


temp3 = pd.crosstab(train['Dependents'], train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[23]:


temp3 = pd.crosstab(train['Gender'], train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[24]:


temp3 = pd.crosstab(train['Education'], train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[25]:


temp3 = pd.crosstab(train['Property_Area'], train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[ ]:





# In[26]:


print (train['Property_Area'].value_counts())
print (train['Education'].value_counts())
print (train['Gender'].value_counts())
print (train['Dependents'].value_counts())
print (train['Married'].value_counts())
print (train['Self_Employed'].value_counts())
print (train['Loan_Status'].value_counts())


# In[27]:


#Converting categorical data to continuous data using 0 1 hot encoding
train['Gender'] = train['Gender'].map({'Female':0,'Male':1}).astype(np.int)
train['Married'] = train['Married'].map({'No':0, 'Yes':1}).astype(np.int)
train['Education'] = train['Education'].map({'Not Graduate':0, 'Graduate':1}).astype(np.int)
train['Self_Employed'] = train['Self_Employed'].map({'No':0, 'Yes':1}).astype(np.int)
train['Loan_Status'] = train['Loan_Status'].map({'N':0, 'Y':1}).astype(np.int)
#train['Property_Area'] = train['Property_Area'].map({'Urban':0, 'Rural':1,}).astype(np.int)

train['Dependents'] = train['Dependents'].astype(np.int)


# In[28]:


train.tail()


# In[29]:


sns.barplot(x="Property_Area", y="Loan_Status", hue="Education", data=train);


# In[30]:


sns.barplot(x="Education", y="Loan_Status", hue="Education", data=train);


# In[31]:


sns.barplot(x="Gender", y="Loan_Status", hue="Education", data=train);


# In[32]:


sns.factorplot(x='Education',y='Loan_Status',data=train)


# In[33]:


sns.factorplot(x='Property_Area',y='Loan_Status',data=train)


# In[34]:


train['ApplicantIncome'].hist()


# In[35]:


train['Property_Area'] = LabelEncoder().fit_transform(train['Property_Area'])


# In[36]:


train.tail()


# In[37]:


print (train['Property_Area'].value_counts())


# In[38]:


corr=train.corr()["Loan_Status"]
corr[np.argsort(corr, axis=0)[::-1]]


# In[39]:


import seaborn as sns
corrmat = train.corr()
plt.figure(figsize = (10,7))


corr_features = corrmat.index[abs(corrmat["Loan_Status"])>-0.1]
g = sns.heatmap(train[corr_features].corr(),annot=True,cmap="RdYlGn")


# In[40]:


targetfet = train.Loan_Status

features = train.drop(['Loan_Status','Loan_ID'], axis = 1)
targetfet.shape
#features.shape


# In[ ]:





# In[41]:


####Prediction model########
#Train-Test split
from sklearn.model_selection import train_test_split
data_train, data_test, label_train, label_test = train_test_split(features, targetfet, test_size = 0.2, random_state = 42)
label_train.shape


# In[42]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
logis.fit(data_train, label_train)
logis_score_train = logis.score(data_train, label_train)
print("Training score: ",logis_score_train)
logis_score_test = logis.score(data_test, label_test)
print("Testing score: ",logis_score_test)


# In[43]:


#decision tree
from sklearn.ensemble import RandomForestClassifier
dt = RandomForestClassifier()
dt.fit(data_train, label_train)
dt_score_train = dt.score(data_train, label_train)
print("Training score: ",dt_score_train)
dt_score_test = dt.score(data_test, label_test)
print("Testing score: ",dt_score_test)


# In[44]:


#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(data_train, label_train)
rfc_score_train = rfc.score(data_train, label_train)
print("Training score: ",rfc_score_train)
rfc_score_test = rfc.score(data_test, label_test)
print("Testing score: ",rfc_score_test)


# In[45]:


#Model comparison
models = pd.DataFrame({
        'Model'          : ['Logistic Regression',  'Decision Tree', 'Random Forest'],
        'Training_Score' : [logis_score_train,  dt_score_train, rfc_score_train],
        'Testing_Score'  : [logis_score_test, dt_score_test, rfc_score_test]
    })
models.sort_values(by='Testing_Score', ascending=False)


# In[ ]:





# ## Same thing after doing label encoding

# In[46]:


clf = LogisticRegression()
clf
#Fiting into model
clf.fit(data_train, label_train)
#Prediction using test data
label_pred = clf.predict(data_test)
#classification accuracy
from sklearn import metrics
print(metrics.accuracy_score(label_test, label_pred))


# In[47]:


train = train.apply(LabelEncoder().fit_transform)


# In[48]:


train.head()


# In[49]:


targetfet1 = train.Loan_Status

features1 = train.drop(['Loan_Status'], axis = 1)


# In[50]:


####Prediction model########
#Train-Test split
from sklearn.model_selection import train_test_split
data_train, data_test, label_train, label_test = train_test_split(features1, targetfet1, test_size = 0.2, random_state = 42)


# In[51]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
logis.fit(data_train, label_train)
logis_score_train = logis.score(data_train, label_train)
print("Training score: ",logis_score_train)
logis_score_test = logis.score(data_test, label_test)
print("Testing score: ",logis_score_test)


# In[ ]:




