#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv('../input/advertising.csv')


# In[ ]:


data.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.set()


# # How 'age' of the person, affects the user advertisement click.

# In[ ]:


plt.hist(x=data['Age'],label= 'Total Users')
plt.hist(x=data['Age'][data['Clicked on Ad']==1],label= 'Clicked on Add')
plt.xlabel('Age')
plt.title('AGE vs ADD CLICKS')
plt.legend()
plt.savefig('AGE vs ADD CLICKS.png')


# # How 'gender' of the person, affects the user advertisement click.

# In[ ]:


gender_data=pd.crosstab(data['Male'],data['Clicked on Ad'])


# In[ ]:


gender_data


# In[ ]:


gender_data.rename({0:'Female',1:'Male'},axis=0,inplace=True)


# In[ ]:


gender_data


# In[ ]:


gender_data.plot(kind='bar')
plt.xlabel('Gender')
plt.title('GENDER vs ADD CLICKS')
plt.xticks(rotation=0)
plt.savefig('GENDER vs ADD CLICKS.png')


# # How 'country' of the person, affects the user advertisement click.

# In[ ]:


country_data=pd.crosstab(data['Country'],data['Clicked on Ad'])


# In[ ]:


country_data['total']=country_data.sum(axis=1)


# In[ ]:


country_data.head()


# In[ ]:


country_data=country_data.apply(lambda x:round(100*x/country_data['total']))


# In[ ]:


country_data.drop('total',axis=1,inplace=True)


# In[ ]:


country_data.head()


# In[ ]:


country_data= country_data.sort_values(1,ascending=False)


# In[ ]:


country_data.head(10)


# In[ ]:


country_data=country_data[:10].index.tolist()


# In[ ]:


country_data


# # How 'daily internet usage' of the person, affects the user advertisement click.

# In[ ]:


plt.scatter(x=data['Daily Internet Usage'], y=data['Clicked on Ad'])
plt.title('DAILY USAGE vs ADD CLICKS')
plt.legend()
plt.savefig('DAILY USAGE vs ADD CLICKS.png')


# # Logistic Regression

# In[ ]:


training_data=data[['Age','Male','Daily Internet Usage']]
output_data=data['Clicked on Ad']


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


test_data=pd.read_csv('../input/Test.csv')
test_data.head()


# In[ ]:


test_data=test_data[['Age','Male','Daily Internet Usage']]
test_data.head()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_predictions=model.predict(test_data)


# In[ ]:


submission=pd.DataFrame({
    'Clicked on Ad':test_predictions
})


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('Prediction.csv')

