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


test=data['Sex'].value_counts()


# In[ ]:


test=pd.Series(test)


# In[ ]:


sns.set(rc={'figure.figsize':(15, 7)})
test.plot(kind='bar')
plt.savefig('male.png')


# In[ ]:


data1=data.groupby('Sex')['Survived'].value_counts()


# In[ ]:


data1


# In[ ]:


survived_data = pd.DataFrame()


# In[ ]:


female_survival=data1[0][1]
male_survival=data1[1][1]
female_unsurvival=data1[0][0]
male_unsurvival=data1[1][0]


# In[ ]:


survived_data['Sex'] = ['male', 'female']
survived_data['Survived'] = [male_survival, female_survival]
survived_data['unSurvived'] = [male_unsurvival, female_unsurvival]


# In[ ]:


survived_data


# In[ ]:


survived_data=survived_data.set_index('Sex')


# In[ ]:


sns.set(rc={'figure.figsize':(15, 7)})
survived_data.plot(kind='bar',stacked=True)
plt.savefig('comparision.png')
plt.xticks(rotation=0)


# In[ ]:


survived_data


# In[ ]:


test


# In[ ]:


survived_data['total']=[test[1],test[0]]


# In[ ]:


survived_data


# In[ ]:


new_data = survived_data.apply(lambda x: round(100 * x/survived_data['total']))


# In[ ]:


new_data


# In[ ]:


new_data.drop('total',axis=1,inplace=True)


# In[ ]:


new_data


# In[ ]:


sns.set(rc={'figure.figsize':(12,6)})
new_data.plot(kind='bar',stacked=True)


# USE OF CROSSTAB

# In[ ]:


ag=pd.crosstab(data['Sex'], data['Survived'])


# In[ ]:


sns.set(rc={'figure.figsize':(15, 7)})
ag.plot(kind='bar',stacked=True)
plt.savefig('comparision1.png')
plt.xticks(rotation=0)


# In[ ]:


numerical_data.isnull().sum()


# In[ ]:


numerical_data=data[['PassengerId','Pclass','Fare','Age','Sex','Parch','SibSp']].copy()


# In[ ]:


numerical_data['Age']=numerical_data['Age'].fillna(22)


# In[ ]:


survival_data=data['Survived'].copy()


# In[ ]:


from sklearn.model_selection import train_test_split 


# In[ ]:


X,x_test,Y,y_test=train_test_split(numerical_data,survival_data,test_size=0.3,random_state=5)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


model=LogisticRegression()


# In[ ]:


model.fit(X,Y)


# In[ ]:


coefficients=model.coef_


# In[ ]:


coeff_dict={
    'PassengerId':coefficients[0][0],
    'Pclass':coefficients[0][1],
    'Fare' :coefficients[0][2],
    'Age' :coefficients[0][3],
    'Sex' :coefficients[0][4],
    'Parch' :coefficients[0][5],
    'SibSp' :coefficients[0][6]
}


# In[ ]:


coeff_dict=pd.Series(coeff_dict)


# In[ ]:


plt.figure(figsize=(10,5))
coeff_dict.plot(kind='bar')
plt.xticks(rotation=0)


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


# In[ ]:




