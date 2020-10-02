#!/usr/bin/env python
# coding: utf-8

# beginner trying the deep learning
# used mean to fill less count of null values 
# used only 7 input variables removed the cabin column from the data set since 600+/840 missing values are present 
# improved the accuracy with normalizing the fare value by 12%
# decreased the accuracy with normalizing the age parameter by 1.5% (nothing can be concluded)
# -- things to try :
#     1.use feature tools to find the importance of features present(but will that be helpfull in the case of very small dataset  

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow
import featuretools as ft


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
# fix random seed for reproducibility
np.random.seed(7)


# In[ ]:


train_data=pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv')


# In[ ]:


#train_data.describe()
train_data.isnull().sum()
train_data.describe()


# In[ ]:


train_data['Age']=train_data['Age'].fillna(train_data['Age'].mean())
train_data=train_data.drop(['Cabin'],axis=1)


# In[ ]:


train_data['Embarked']=train_data['Embarked'].fillna(method='ffill')


# In[ ]:


# female is 0 male 1
train_data['Sex']=train_data['Sex'].apply(lambda x:1 if x=='male' else 0 )
train_data['Embarked']=train_data['Embarked'].apply(lambda x:1 if x=='S'else 2 if x=='C' else 3)
train_data['Fare']=train_data['Fare'].apply(lambda x: x/513)
#train_data['Age']=train_data['Age'].apply(lambda x: x/79.6)


# In[ ]:


X=train_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
Y=train_data[['Survived']]


# In[ ]:


# create model
model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model.fit(X, Y, epochs=75, batch_size=100)


# In[ ]:


# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


test_data['Age']=test_data['Age'].fillna(test_data['Age'].mean())
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].mean())
test_data['Fare']=test_data['Fare'].apply(lambda x: x/513)
#test_data['Age']=test_data['Age'].apply(lambda x: x/79.6)
test_data=test_data.drop(['Cabin'],axis=1)
test_data['Embarked']=test_data['Embarked'].fillna(method='ffill')
test_data['Sex']=test_data['Sex'].apply(lambda x:1 if x=='male' else 0 )
test_data['Embarked']=test_data['Embarked'].apply(lambda x:1 if x=='S'else 2 if x=='C' else 3)


# In[ ]:


test_data.isnull().sum()


# In[ ]:


X=test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]


# In[ ]:


predict=model.predict(X)


# In[ ]:


final_out=pd.DataFrame(columns=['PassengerId','Survived'])


# In[ ]:


final_out['PassengerId']=test_data['PassengerId']


# In[ ]:


final_out['Survived']=list(predict)


# In[ ]:


final_out['Survived']=final_out['Survived'].fillna(method='ffill')


# In[ ]:


final_out['Survived']=final_out['Survived'].apply(lambda x:int(round(x[0])))


# In[ ]:


final_out.reset_index()
final_out.to_csv('./final_predictions.csv',index=False)


# In[ ]:


final_out['Survived'].value_counts()

