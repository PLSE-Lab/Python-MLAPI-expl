#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


data=pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.set()


# In[ ]:


font={
    'size':18
}


# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(data.corr(),annot=True,linewidths=1,linecolor='w')
plt.xlabel('columns', fontdict=font)
plt.ylabel('columns', fontdict=font)
plt.xticks(rotation=0)
plt.title('Correlations', fontdict=font)
plt.savefig('1.png')


# In[ ]:


data.info()


# In[ ]:


data.Sex[data.Sex == 'male'] = 1
data.Sex[data.Sex == 'female'] =0


# In[ ]:


data.head()


# In[ ]:


numerical_data=data[['PassengerId','Pclass','Fare','Age','Sex','Parch','SibSp']].copy()


# In[ ]:


numerical_data.isnull().sum()


# In[ ]:


numerical_data['Age']=numerical_data['Age'].fillna(22)


# In[ ]:


survival_data=data['Survived'].copy()


# In[ ]:


from sklearn.model_selection import train_test_split 


# In[ ]:


X,x_test,Y,y_test=train_test_split(numerical_data,survival_data,test_size=0.3,random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model=LogisticRegression()


# In[ ]:


model.fit(X,Y)


# In[ ]:


predictions=model.predict(x_test)


# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[ ]:


confusion= confusion_matrix(predictions,y_test)


# In[ ]:


confusion


# In[ ]:


score= accuracy_score(predictions,y_test)


# In[ ]:


score


# ### importing the test dataset

# In[ ]:


test_data=pd.read_csv('../input/test.csv')


# In[ ]:


test_data.head()


# In[ ]:


test_data=test_data[['PassengerId','Pclass','Fare','Age','Sex','SibSp','Parch']].copy()


# In[ ]:


test_data.Sex[test_data.Sex == 'male'] = 1
test_data.Sex[test_data.Sex == 'female'] =0


# In[ ]:


test_data.head()


# In[ ]:


test_data=test_data[['PassengerId','Pclass','Fare','Age','Sex','SibSp','Parch']].copy()


# In[ ]:


test_data['Fare']=test_data['Fare'].fillna(10)


# In[ ]:


test_data['Age']=test_data['Age'].fillna(22)


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_predictions=model.predict(test_data)


# In[ ]:


test_predictions[:3]


# In[ ]:


test_predictions_df={
    'PassengerId':test_data['PassengerId'],
    'Survived': test_predictions
}


# In[ ]:


test_predictions_df=pd.DataFrame(test_predictions_df)


# In[ ]:


test_predictions_df.head()


# In[ ]:


test_predictions_df=test_predictions_df.set_index('PassengerId')


# In[ ]:


test_predictions_df.to_csv('Submission.csv')

