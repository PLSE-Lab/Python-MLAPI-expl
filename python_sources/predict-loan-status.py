#!/usr/bin/env python
# coding: utf-8

# In[176]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn import preprocessing
from sklearn import cross_validation, metrics
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[177]:


data_train = pd.read_csv("../input/train_u6lujuX_CVtuZ9i.csv") 

data_test = pd.read_csv("../input/test_Y3wMUE5_7gLdaTN.csv") 


# In[178]:


data_train.shape


# In[179]:


data_test.shape


# In[180]:


data_train.head(5)


# In[181]:


data_train.describe()


# In[182]:


data_test.head()


# In[183]:


data_test.describe()


# In[184]:


data_train.isnull().sum()


# In[185]:


test.isnull().sum()


# In[186]:


train = pd.read_csv("../input/train_u6lujuX_CVtuZ9i.csv") 
test = pd.read_csv("../input/test_Y3wMUE5_7gLdaTN.csv") 
targets = train.Loan_Status

train.drop('Loan_Status', 1, inplace=True)
combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop(['index', 'Loan_ID'], inplace=True, axis=1)


# In[187]:


combined.head(5)


# In[188]:


combined.shape


# In[189]:


combined.describe()


# In[190]:


# To check how many columns have missing values - this can be repeated to see the progress made
def show_missing():
    missing = combined.columns[combined.isnull().any()].tolist()
    return missing


# In[191]:


#from this we can find the total missing data in each columns

combined[show_missing()].isnull().sum()


# In[192]:


print (combined['Property_Area'].value_counts())
print (combined['Education'].value_counts())
print (combined['Gender'].value_counts())
print (combined['Dependents'].value_counts())
print (combined['Married'].value_counts())
print (combined['Self_Employed'].value_counts())
print (combined['Credit_History'].value_counts())


# In[193]:


#filling data with approperiate measure of central tendency
combined['Gender'].fillna('Male', inplace=True)
combined['Married'].fillna('Yes', inplace=True)

combined['Self_Employed'].fillna('Yes', inplace=True)

combined['Credit_History'].fillna(1, inplace=True)

combined['LoanAmount'].fillna(combined['LoanAmount'].median(), inplace=True)
#combined['Loan_Amount_Term'].fillna(combined['Loan_Amount_Term'].mean(), inplace=True)


# In[194]:


combined.isnull().sum()


# ## Visualization

# In[195]:


combined['ApplicantIncome'].hist()


# In[196]:


combined['LoanAmount'].hist()


# In[197]:


combined['Loan_Amount_Term'].hist()


# In[198]:


ax = combined.groupby('Gender').ApplicantIncome.mean().plot(kind='bar')
ax.set_xlabel("Gender")
ax.set_ylabel("mean ApplicantIncom")


# In[199]:


ax = combined.groupby('Education').ApplicantIncome.mean().plot(kind='bar')
ax.set_xlabel("Education(1=Graduate)")
ax.set_ylabel("mean ApplicantIncom")


# In[200]:


ax = combined.groupby('Married').ApplicantIncome.mean().plot(kind='bar')
ax.set_xlabel("Married(1=yes)")
ax.set_ylabel("mean ApplicantIncom")


# In[201]:



temp = pd.crosstab(data_train['Credit_History'], data_train['Loan_Status'])
temp.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[202]:


temp3 = pd.crosstab(data_train['Dependents'], data_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[203]:


temp3 = pd.crosstab(data_train['Gender'], data_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[204]:


temp3 = pd.crosstab(data_train['Education'], data_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[205]:


temp3 = pd.crosstab(data_train['Property_Area'], data_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# # feature engineering

# ## one hot encoding

# In[206]:


combined['Gender'] = combined['Gender'].map({'Male':1,'Female':0})
combined['Married'] = combined['Married'].map({'Yes':1,'No':0})
combined['Education'] = combined['Education'].map({'Graduate':1,'Not Graduate':0})
combined['Self_Employed'] = combined['Self_Employed'].map({'Yes':1,'No':0})


# In[207]:


combined['Singleton'] = combined['Dependents'].map(lambda d: 1 if d=='1' else 0)
combined['Small_Family'] = combined['Dependents'].map(lambda d: 1 if d=='2' else 0)
combined['Large_Family'] = combined['Dependents'].map(lambda d: 1 if d=='3+' else 0)
combined.drop(['Dependents'], axis=1, inplace=True)


# In[208]:


combined['Total_Income'] = combined['ApplicantIncome'] + combined['CoapplicantIncome']
combined.drop(['ApplicantIncome','CoapplicantIncome'], axis=1, inplace=True)


# In[209]:


combined['Income_Ratio'] = combined['Total_Income'] / combined['LoanAmount']


# In[210]:


combined['Loan_Amount_Term'].value_counts()


# In[211]:


approved_term = data_train[data_train['Loan_Status']=='Y']['Loan_Amount_Term'].value_counts()
unapproved_term = data_train[data_train['Loan_Status']=='N']['Loan_Amount_Term'].value_counts()
df = pd.DataFrame([approved_term,unapproved_term])
df.index = ['Approved','Unapproved']
df.plot(kind='bar', stacked=True, figsize=(15,8))


# In[212]:


combined['Very_Short_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t<=60 else 0)
combined['Short_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>60 and t<180 else 0)
combined['Long_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>=180 and t<=300  else 0)
combined['Very_Long_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>300 else 0)
combined.drop('Loan_Amount_Term', axis=1, inplace=True)


# In[213]:


combined['Credit_History_Bad'] = combined['Credit_History'].map(lambda c: 1 if c==0 else 0)
combined['Credit_History_Good'] = combined['Credit_History'].map(lambda c: 1 if c==1 else 0)
combined['Credit_History_Unknown'] = combined['Credit_History'].map(lambda c: 1 if c==2 else 0)
combined.drop('Credit_History', axis=1, inplace=True)


# In[214]:


property_dummies = pd.get_dummies(combined['Property_Area'], prefix='Property')
combined = pd.concat([combined, property_dummies], axis=1)
combined.drop('Property_Area', axis=1, inplace=True)


# In[215]:


combined[60:70]


# ## feature scaling

# In[216]:


def feature_scaling(dataframe):
    dataframe -= dataframe.min()
    dataframe /= dataframe.max()
    return dataframe


# In[217]:


combined['LoanAmount'] = feature_scaling(combined['LoanAmount'])
combined['Total_Income'] = feature_scaling(combined['Total_Income'])
combined['Income_Ratio'] = feature_scaling(combined['Income_Ratio'])


# In[218]:


combined.head()


# ## prediction model

# In[219]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# In[220]:


#function for computing score
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


# In[221]:


#recovering train test &target
global combined, data_train
targets = data_train['Loan_Status'].map({'Y':1,'N':0})
train = combined.head(614)
test = combined.iloc[614:]


# # Random forest algorithm

# ## feature imortance

# In[222]:


clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)


# In[223]:


features = pd.DataFrame()
features['Feature'] = train.columns
features['Importance'] = clf.feature_importances_
features.sort_values(by=['Importance'], ascending=False, inplace=True)
features.set_index('Feature', inplace=True)


# In[224]:


features.plot(kind='bar', figsize=(20, 10))


# In[225]:


model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
train_reduced.shape


# In[226]:


test_reduced = model.transform(test)
test_reduced.shape


# In[227]:


parameters  = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}

model = RandomForestClassifier(**parameters)
model.fit(train, targets)


# In[228]:


compute_score(model, train, targets, scoring='accuracy')


# In[237]:


#saving output as output.csv
output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('../input/test_Y3wMUE5_7gLdaTN.csv')
df_output['Loan_ID'] = aux['Loan_ID']
df_output['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output)
df_output[['Loan_ID','Loan_Status']].to_csv('output.csv',index=False)


# # other algorithms

# In[230]:


####Prediction model########
#Train-Test split
from sklearn.model_selection import train_test_split
datatrain, datatest, labeltrain, labeltest = train_test_split(train, targets, test_size = 0.2, random_state = 42)
labeltrain.shape


# In[231]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
logis.fit(datatrain, labeltrain)
logis_score_train = logis.score(datatrain, labeltrain)
print("Training score: ",logis_score_train)
logis_score_test = logis.score(datatest, labeltest)
print("Testing score: ",logis_score_test)


# In[232]:


#saving output as output.csv of decision tree
output2 = logis.predict(test).astype(int)
df_output2 = pd.DataFrame()
aux = pd.read_csv('../input/test_Y3wMUE5_7gLdaTN.csv')
df_output2['Loan_ID'] = aux['Loan_ID']
df_output2['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output2)
df_output2[['Loan_ID','Loan_Status']].to_csv('output2.csv',index=False)


# In[ ]:





# In[233]:


#decision tree
from sklearn.ensemble import RandomForestClassifier
dt = RandomForestClassifier()
dt.fit(datatrain, labeltrain)
dt_score_train = dt.score(datatrain, labeltrain)
print("Training score: ",dt_score_train)
dt_score_test = dt.score(datatest, labeltest)
print("Testing score: ",dt_score_test)


# In[234]:


#saving output as output.csv of decision tree
#output2 = dt.predict(test).astype(int)
#df_output2 = pd.DataFrame()
#aux = pd.read_csv('../input/test_Y3wMUE5_7gLdaTN.csv')
#df_output2['Loan_ID'] = aux['Loan_ID']
#df_output2['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output2)
#df_output2[['Loan_ID','Loan_Status']].to_csv('output2.csv',index=False)


# In[ ]:





# In[235]:


#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(datatrain, labeltrain)
rfc_score_train = rfc.score(datatrain, labeltrain)
print("Training score: ",rfc_score_train)
rfc_score_test = rfc.score(datatest, labeltest)
print("Testing score: ",rfc_score_test)


# In[236]:


#Model comparison
models = pd.DataFrame({
        'Model'          : ['Logistic Regression',  'Decision Tree', 'Random Forest'],
        'Training_Score' : [logis_score_train,  dt_score_train, rfc_score_train],
        'Testing_Score'  : [logis_score_test, dt_score_test, rfc_score_test]
    })
models.sort_values(by='Testing_Score', ascending=False)


# In[ ]:





# In[ ]:





# In[ ]:




