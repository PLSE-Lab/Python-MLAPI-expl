#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize=(16,8))
sns.heatmap(data.corr(),annot=True,linewidths=1,linecolor='w')
plt.xlabel('Columns')
plt.ylabel('Columns')
plt.title('Heatmap')
plt.savefig('Heatmap.png')


# In[ ]:


training_data=data[['PassengerId','Pclass','Fare']]


# In[ ]:


output_data=data['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split as tts
X,x_test,Y,y_test= tts(training_data,output_data,test_size=0.3,random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[ ]:


model.fit(X,Y)


# In[ ]:


predictions=model.predict(x_test)


# In[ ]:


predictions[:5]


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


score = accuracy_score(predictions,y_test)
score


# In[ ]:


test_data=pd.read_csv('../input/test.csv')
test_data.head()


# In[ ]:


test_data=test_data[['PassengerId','Pclass','Fare']]


# In[ ]:


test_data.head()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_predictions=model.predict(test_data)


# In[ ]:


submission=pd.DataFrame({
    'PassengerId':test_data['PassengerId'],
    'Survived':test_predictions
})


# In[ ]:


submission.head()


# In[ ]:


submission=submission.set_index('PassengerId')


# In[ ]:


submission.to_csv('Prediction1.csv')


# In[ ]:


data['Sex']=data['Sex'].apply(lambda x:1 if x=='male' else 0)
data.head()


# In[ ]:


td=data[['PassengerId','Pclass','Fare','Sex']]
od=data['Survived']


# In[ ]:


td.head()


# In[ ]:


X,x_test,Y,y_test= tts(td,od,test_size=0.3,random_state=42)


# In[ ]:


model=LogisticRegression()


# In[ ]:


model.fit(X,Y)


# In[ ]:


predictions=model.predict(x_test)


# In[ ]:


predictions[:5]


# In[ ]:


score = accuracy_score(predictions,y_test)
score


# In[ ]:


test_data=pd.read_csv('../input/test.csv')
test_data.head()


# In[ ]:


test_data['Sex']=test_data['Sex'].apply(lambda x:1 if x=='male' else 0)
test_data.head()


# In[ ]:


test_data=test_data[['PassengerId','Pclass','Fare','Sex']]


# In[ ]:


test_data.head()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_predictions=model.predict(test_data)


# In[ ]:


submission=pd.DataFrame({
    'PassengerId':test_data['PassengerId'],
    'Survived':test_predictions
})


# In[ ]:


submission.head()


# In[ ]:


submission=submission.set_index('PassengerId')


# In[ ]:


submission.to_csv('Prediction2.csv')

