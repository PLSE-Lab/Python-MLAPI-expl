#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import codecs
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[3]:


data=pd.read_csv('../input/Interview.csv')


# In[4]:


data.head()


# In[5]:


#columns in our data
data.columns


# In[6]:


print(data.shape)
data.info()


# In[7]:


# Deleting unnamed rows
del data['Unnamed: 23']
del data['Unnamed: 24']
del data['Unnamed: 25']
del data['Unnamed: 26']
del data['Unnamed: 27']


# In[8]:


data.head()


# In[ ]:


data['Client name'].value_counts()


# In[9]:


data['Observed Attendance'].value_counts()


# In[10]:


data.loc[data['Observed Attendance']=='NO','Observed Attendance']='No'
data['Observed Attendance'].value_counts()


# In[11]:


data.loc[data['Observed Attendance']=='No ','Observed Attendance']='No'
data.loc[data['Observed Attendance']=='no ','Observed Attendance']='No'
data.loc[data['Observed Attendance']=='no','Observed Attendance']='No'
data['Observed Attendance'].value_counts()


# In[12]:


data.loc[np.logical_or(data['Observed Attendance']=='yes', data['Observed Attendance']=='yes '),'Observed Attendance']='Yes'
data['Observed Attendance'].value_counts()


# In[13]:


data.head()


# In[14]:


data.loc[np.logical_or(data['Location']=='CHENNAI', data['Location']=='chennai '),'Location']='Chennai'
data.loc[data['Location']=='chennai','Location']='Chennai'
data.loc[data['Location']=='Gurgaonr','Location']='Gurgaon'
data.loc[np.logical_or(data['Location']=='Gurgaon', data['Location']=='Delhi'),'Location']='Others'
data.loc[np.logical_or(data['Location']=='- Cochin- ', data['Location']=='Noida'),'Location']='Others'
data.loc[data['Location']=='Hyderabad','Location']='Chennai'
#print dict(data['Location'].value_counts())
plt.figure(figsize=(15,6))
sns.countplot(data=data,x='Location',hue='Observed Attendance')


# In[15]:


data['Candidate Current Location'].value_counts()


# In[16]:


data.loc[np.logical_or(data['Candidate Current Location']=='CHENNAI', data['Candidate Current Location']=='chennai '),'Candidate Current Location']='Chennai'
data.loc[data['Candidate Current Location']=='chennai','Candidate Current Location']='Chennai'
data.loc[data['Candidate Current Location']=='Gurgaonr','Candidate Current Location']='Gurgaon'
data.loc[np.logical_or(data['Candidate Current Location']=='Gurgaon', data['Candidate Current Location']=='Delhi'),'Candidate Current Location']='Others'
data.loc[np.logical_or(data['Candidate Current Location']=='- Cochin- ', data['Candidate Current Location']=='Noida'),'Candidate Current Location']='Others'
data.loc[data['Candidate Current Location']=='Hyderabad','Candidate Current Location']='Chennai'
#print dict(data['Location'].value_counts())
plt.figure(figsize=(15,6))
sns.countplot(data=data,x='Candidate Current Location',hue='Observed Attendance')


# In[17]:


print(dict(data['Candidate Current Location'].value_counts()))


# In[18]:


data.info()


# In[19]:


data['Industry'].value_counts()


# In[20]:


data.loc[data['Industry']!='BFSI','Industry']='Others'
#data.loc[data['Industry']=='IT Products and Services','Industry']='Others'
#data.loc[data['Industry']=='IT Services','Industry']='Others'
#data.loc[data['Industry']=='Electronics','Industry']='Others'
#data.loc[data['Industry']=='Telecom','Industry']='Others'
#data.loc[data['Industry']=='IT','Industry']='Others'


# In[21]:


data['Industry'].value_counts()


# In[22]:


data['Position to be closed'].value_counts()


# In[23]:


data.loc[data['Position to be closed']!='Routine','Position to be closed']='Others'
#data=data[data['Position to be closed']!='AML']
#data=data[data['Position to be closed']!='Production- Sterile']


# In[24]:


data['Position to be closed'].value_counts()


# In[25]:


data['Client name'].value_counts()


# In[26]:


data.loc[data['Client name']!='Standard Chartered Bank','Client name']='Others'
#data.loc[data['Client name']=='ANZ','Client name']='Others'
#data.loc[data['Client name']=='Standard Chartered Bank Chennai','Client name']='Others'
#data.loc[data['Client name']=='Astrazeneca','Client name']='Others'
#data.loc[data['Client name']=='Barclays','Client name']='Others'
#data.loc[data['Client name']=='Woori Bank','Client name']='Others'


# In[27]:


data['Client name'].value_counts()


# In[28]:


data['Client name'].unique()


# In[29]:


data =data[data['Client name']!='\xef\xbb\xbf\xef\xbb\xbf']


# In[30]:


data['Client name'].unique()


# In[31]:


del data['Name(Cand ID)']


# In[32]:


del data['Candidate Native location']
data.columns


# In[33]:


del data['Hope there will be no unscheduled meetings']


# In[34]:


del data['Can I have an alternative number/ desk number. I assure you that I will not trouble you too much']


# In[35]:


data['Candidate Job Location'].value_counts()


# In[36]:


data.loc[data['Candidate Job Location']=='Gurgaon','Candidate Job Location']='Others'
data.loc[data['Candidate Job Location']=='Visakapatinam','Candidate Job Location']='Others'
data.loc[data['Candidate Job Location']=='Noida','Candidate Job Location']='Others'
data.loc[data['Candidate Job Location']=='- Cochin- ','Candidate Job Location']='Others'
data.loc[data['Candidate Job Location']=='Hosur','Candidate Job Location']='Others'


# In[37]:


data['Candidate Job Location'].value_counts()


# In[38]:


data.loc[data['Expected Attendance']=='No ','Expected Attendance']='No'
data.loc[data['Expected Attendance']=='NO','Expected Attendance']='No'
data.loc[data['Expected Attendance']=='Uncertain','Expected Attendance']='No'
data.loc[data['Expected Attendance']=='yes','Expected Attendance']='Yes'

data =data[data['Expected Attendance']!='10.30 Am']
data =data[data['Expected Attendance']!='11:00 AM']
data['Expected Attendance'].value_counts()
sns.countplot(data=data,x='Observed Attendance',hue='Expected Attendance')


# In[39]:


data.columns


# In[40]:


data['Date of Interview'].value_counts()
del data['Date of Interview']


# In[41]:


data['Marital Status'].value_counts()


# In[42]:


data['Have you obtained the necessary permission to start at the required time'].value_counts()
#data['Can I Call you three hours before the interview and follow up on your attendance for the interview'].value_counts()
#data['Have you taken a printout of your updated resume. Have you read the JD and understood the same'].value_counts()
#data['Are you clear with the venue details and the landmark.'].value_counts()
#data['Has the call letter been shared'].value_counts()
data.loc[data['Have you obtained the necessary permission to start at the required time']=='Na','Have you obtained the necessary permission to start at the required time']='No'

data.loc[data['Have you obtained the necessary permission to start at the required time']=='Not yet','Have you obtained the necessary permission to start at the required time']='No'

data.loc[data['Have you obtained the necessary permission to start at the required time']=='NO','Have you obtained the necessary permission to start at the required time']='No'
data.loc[data['Have you obtained the necessary permission to start at the required time']=='yes','Have you obtained the necessary permission to start at the required time']='Yes'
data =data[data['Have you obtained the necessary permission to start at the required time']!='Yet to confirm']
data['Have you obtained the necessary permission to start at the required time'].value_counts()


# In[43]:


data['Can I Call you three hours before the interview and follow up on your attendance for the interview'].value_counts()
data.loc[data['Can I Call you three hours before the interview and follow up on your attendance for the interview']=='Na','Can I Call you three hours before the interview and follow up on your attendance for the interview']='No'

data.loc[data['Can I Call you three hours before the interview and follow up on your attendance for the interview']=='No Dont','Can I Call you three hours before the interview and follow up on your attendance for the interview']='No'

data.loc[data['Can I Call you three hours before the interview and follow up on your attendance for the interview']=='yes','Can I Call you three hours before the interview and follow up on your attendance for the interview']='Yes'


# In[44]:


data['Can I Call you three hours before the interview and follow up on your attendance for the interview'].value_counts()


# In[45]:


data['Have you taken a printout of your updated resume. Have you read the JD and understood the same'].value_counts()


# In[46]:


data.loc[data['Have you taken a printout of your updated resume. Have you read the JD and understood the same']=='Na','Have you taken a printout of your updated resume. Have you read the JD and understood the same']='No'
data.loc[data['Have you taken a printout of your updated resume. Have you read the JD and understood the same']=='Not Yet','Have you taken a printout of your updated resume. Have you read the JD and understood the same']='No'
data.loc[data['Have you taken a printout of your updated resume. Have you read the JD and understood the same']=='Not yet','Have you taken a printout of your updated resume. Have you read the JD and understood the same']='No'
data.loc[data['Have you taken a printout of your updated resume. Have you read the JD and understood the same']=='na','Have you taken a printout of your updated resume. Have you read the JD and understood the same']='No'
data.loc[data['Have you taken a printout of your updated resume. Have you read the JD and understood the same']=='yes','Have you taken a printout of your updated resume. Have you read the JD and understood the same']='Yes'
data.loc[data['Have you taken a printout of your updated resume. Have you read the JD and understood the same']=='No- will take it soon','Have you taken a printout of your updated resume. Have you read the JD and understood the same']='No'
data['Have you taken a printout of your updated resume. Have you read the JD and understood the same'].value_counts()


# In[47]:


data['Are you clear with the venue details and the landmark.'].value_counts()


# In[48]:


data.loc[data['Are you clear with the venue details and the landmark.']=='Na','Are you clear with the venue details and the landmark.']='No'
data.loc[data['Are you clear with the venue details and the landmark.']=='na','Are you clear with the venue details and the landmark.']='No'
data.loc[data['Are you clear with the venue details and the landmark.']=='yes','Are you clear with the venue details and the landmark.']='Yes'
data.loc[data['Are you clear with the venue details and the landmark.']=='No- I need to check','Are you clear with the venue details and the landmark.']='No'


# In[49]:


data['Are you clear with the venue details and the landmark.'].value_counts()


# In[50]:


data['Has the call letter been shared'].value_counts()


# In[51]:


data.loc[data['Has the call letter been shared']=='Na','Has the call letter been shared']='No'
data.loc[data['Has the call letter been shared']=='Not yet','Has the call letter been shared']='No'
data.loc[data['Has the call letter been shared']=='Not Sure','Has the call letter been shared']='No'
data.loc[data['Has the call letter been shared']=='Need To Check','Has the call letter been shared']='No'
data.loc[data['Has the call letter been shared']=='Yet to Check','Has the call letter been shared']='No'
data.loc[data['Has the call letter been shared']=='Havent Checked','Has the call letter been shared']='No'
data.loc[data['Has the call letter been shared']=='no','Has the call letter been shared']='No'
data.loc[data['Has the call letter been shared']=='na','Has the call letter been shared']='No'
data.loc[data['Has the call letter been shared']=='Not sure','Has the call letter been shared']='No'
data.loc[data['Has the call letter been shared']=='yes','Has the call letter been shared']='Yes'
data['Has the call letter been shared'].value_counts()


# In[52]:


data.head(20)


# In[54]:


for i in data.columns:
    print(data[i].value_counts())


# In[55]:


del data['Nature of Skillset']


# In[57]:


for i in data.columns:
    print(data[i].value_counts())


# In[58]:


data.loc[data['Interview Venue']=='Gurgaon','Interview Venue']='Others'
data.loc[data['Interview Venue']=='Hyderabad','Interview Venue']='Others'
data.loc[data['Interview Venue']=='Noida','Interview Venue']='Others'
data.loc[data['Interview Venue']=='- Cochin- ','Interview Venue']='Others'
data.loc[data['Interview Venue']=='Hosur','Interview Venue']='Others'


# In[60]:


for i in data.columns:
    print(data[i].value_counts())


# In[63]:


print(dict(data['Interview Type'].value_counts()))


# In[64]:


data.loc[data['Interview Type']=='Scheduled Walkin','Interview Type']='Scheduled Walk In'
data.loc[data['Interview Type']=='Sceduled walkin','Interview Type']='Scheduled Walk In'
data.loc[data['Interview Type']=='Scheduled ','Interview Type']='Scheduled Walk In'
data.loc[data['Interview Type']=='Walkin ','Interview Type']='Walkin'
data['Interview Type'].value_counts()


# In[66]:


for i in data.columns:
    print(data[i].value_counts())


# In[ ]:





# In[68]:


for i in data.columns:
    print(data[i].value_counts())


# In[ ]:





# In[69]:


data['Marital Status'].unique()


# In[70]:


data=data.dropna()


# In[71]:


data['Marital Status'].unique()


# In[72]:


data=data.apply(LabelEncoder().fit_transform)


# In[73]:


data.head()


# In[74]:


from sklearn.linear_model import LogisticRegression


# In[75]:


Y_train=data['Observed Attendance']


# In[76]:


del data['Observed Attendance']


# In[77]:


X_train = data


# In[78]:


data.head()


# In[79]:


from sklearn.model_selection import train_test_split


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)


# In[81]:


log=LogisticRegression()
log.fit(X_train,y_train)
#score = LogisticRegression().score(X_test, y_test)
#print(score)


# In[82]:


score = log.score(X_test, y_test)
print(score)


# In[86]:


def show_performance_model(model, X_train, y_train, X_test, y_test):
    # check classification scores of logistic regression
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train=model.predict(X_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    [fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
    print('Train/Test split results:')
    print(model.__class__.__name__+" accuracy on train set is %2.3f" % accuracy_score(y_train, y_pred_train))
    print(model.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
    print(model.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
    print(model.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))

    idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95

    plt.figure()
    plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')
    plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (recall)', fontsize=14)
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")


    print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
          "and a specificity of %.3f" % (1-fpr[idx]) + 
          ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
    return


# In[87]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
from sklearn.model_selection import  cross_val_score


# In[88]:


show_performance_model(log, X_train, y_train, X_test, y_test )


# In[90]:


from sklearn.ensemble import RandomForestClassifier
randomf = RandomForestClassifier(n_jobs=2, random_state=0)
randomf.fit(X_train,y_train)
show_performance_model(randomf, X_train, y_train, X_test, y_test )


# In[91]:


from sklearn.ensemble import GradientBoostingClassifier


# In[92]:


gb= GradientBoostingClassifier()
gb.fit(X_train,y_train)
show_performance_model(gb, X_train, y_train, X_test, y_test )


# In[ ]:




