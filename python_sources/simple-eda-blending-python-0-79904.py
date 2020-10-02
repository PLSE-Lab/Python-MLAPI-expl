#!/usr/bin/env python
# coding: utf-8

# ## Advanced Ensemble Techniques Blending
# this Kernel is built in base to this documents: https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/ 
# If I help you please upvote

# In[ ]:


import numpy as np 
import pandas as pd 

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os
print(os.listdir("../input"))
np.random.seed(0)


# This Kernel es a basic example of implementation about Stacking, this technique is very import for understand techniques more avanzed 

# Upload dataset

# In[ ]:


train = pd.read_csv("../input/train.csv",index_col='PassengerId')
test = pd.read_csv("../input/test.csv",index_col='PassengerId')#


# i count data missing and count dataset

# In[ ]:


train.shape,test.shape


# Show my Dataset

# In[ ]:


train[(train['Age'].isna())].head(12)


# In[ ]:


train[(train['SibSp']==0) & (train['Parch']==0) &(train['Embarked']=='Q')&(train['Sex']=='male')].head(5)


# In[ ]:


train.head(5)


# ## Engineer Features
# We work in the features Dataset, this example is about Stacking, and it's not important to deep in the feature

# In[ ]:


def replaceGen(sex):
    gen =0
    if sex=='male':
        gen=0
    elif sex=='female':
        gen=1
    return gen
    


# In[ ]:


train['Sex'] = train['Sex'].apply(replaceGen)
test['Sex'] = test['Sex'].apply(replaceGen)


# In[ ]:


train['Age'].hist(figsize=(10, 4));


# In[ ]:


train['Age'].fillna(train['Age'].mean(), inplace=True)
test['Age'].fillna(test['Age'].mean(), inplace=True)


# In[ ]:


test[test['Fare'].isna()]


# In[ ]:


Age_mean = train[(train['Pclass']==3) & (train['Embarked']=='S') & (train['Age']>55) & (train['Sex']==0)]['Fare'].mean()


# In[ ]:


test['Fare'].fillna(Age_mean, inplace=True)


# In[ ]:


X =train.drop(['Survived','Name','Ticket','Cabin','Embarked'],axis=1)
y =pd.DataFrame(train['Survived'])
test_f =test.drop(['Name','Ticket','Cabin','Embarked'],axis=1)


# In[ ]:


X.shape,y.shape


# Split the Dataset in Train and test and data validation
# 
# ![Explained](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/05/image-7-300x226.png)

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)


# We test the models and its accuracy

# In[ ]:


model1 = DecisionTreeClassifier()
model1.fit(x_train, y_train)
val_pred1=pd.DataFrame(model1.predict(x_val))
test_pred1=pd.DataFrame(model1.predict(x_test))

model2 = KNeighborsClassifier()
model2.fit(x_train,y_train)
val_pred2=pd.DataFrame(model2.predict(x_val))
test_pred2=pd.DataFrame(model2.predict(x_test))

model3 = RandomForestClassifier()
model3.fit(x_train,y_train)
val_pred3=pd.DataFrame(model3.predict(x_val))
test_pred3=pd.DataFrame(model3.predict(x_test))

model4 = GradientBoostingClassifier()
model4.fit(x_train,y_train)
val_pred4=pd.DataFrame(model4.predict(x_val))
test_pred4=pd.DataFrame(model4.predict(x_test))


# In[ ]:


x_val.head()


# In[ ]:


df_val=pd.concat([x_val.reset_index(drop=True), val_pred1.reset_index(drop=True),val_pred2.reset_index(drop=True),val_pred3.reset_index(drop=True),val_pred4.reset_index(drop=True)],axis=1)
df_test=pd.concat([x_test.reset_index(drop=True), test_pred1.reset_index(drop=True),test_pred2.reset_index(drop=True),test_pred3.reset_index(drop=True),test_pred4.reset_index(drop=True)],axis=1)


# In[ ]:


df_val.head()


# Show the accuracy in the model of second level

# In[ ]:


model = LogisticRegression()
model.fit(df_val,y_val)
model.score(df_test,y_test)


# 
# 
# 
# 
# Run model with all Dataset

# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


model1 = DecisionTreeClassifier()
model1.fit(x_train, y_train)
val_pred1=pd.DataFrame(model1.predict(x_val))
test_pred1=pd.DataFrame(model1.predict(test_f))

model2 = KNeighborsClassifier()
model2.fit(x_train,y_train)
val_pred2=pd.DataFrame(model2.predict(x_val))
test_pred2=pd.DataFrame(model2.predict(test_f))

model3 = RandomForestClassifier()
model3.fit(x_train,y_train)
val_pred3=pd.DataFrame(model3.predict(x_val))
test_pred3=pd.DataFrame(model3.predict(test_f))

model4 = GradientBoostingClassifier()
model4.fit(x_train,y_train)
val_pred4=pd.DataFrame(model4.predict(x_val))
test_pred4=pd.DataFrame(model4.predict(test_f))


# In[ ]:


x_val.head()


# In[ ]:


df_val=pd.concat([x_val.reset_index(drop=True), val_pred1.reset_index(drop=True),val_pred2.reset_index(drop=True),val_pred3.reset_index(drop=True),val_pred4.reset_index(drop=True)],axis=1)
df_test=pd.concat([test_f.reset_index(drop=True), test_pred1.reset_index(drop=True),test_pred2.reset_index(drop=True),test_pred3.reset_index(drop=True),test_pred4.reset_index(drop=True)],axis=1)


# In[ ]:


df_val.head()


# In[ ]:


model = LogisticRegression()
model.fit(df_val,y_val)
y_target = model.predict(df_test)


# In[ ]:


test_salida = pd.DataFrame( { 'PassengerId': test_f.index , 'Survived': y_target } )


# In[ ]:


#Show Output
test_salida.head(20)


# In[ ]:


test_salida.to_csv( 'titanic_pred.csv' , index = False )


# In[ ]:




