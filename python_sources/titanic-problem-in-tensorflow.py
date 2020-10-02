#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# In[ ]:


train.shape,test.shape


# In[ ]:


train.head()


# In[ ]:


train.describe(include='all')


# Get the count of Null values in each column.

# In[ ]:


train.isnull().sum(), test.isnull().sum()
#full_data=[train,test], full_data[1].shape,full_data[0].shape


# In[ ]:


train.columns


# To get the unique records from the column

# In[ ]:


train['Sex'].unique() #pd.unique(train['Sex']) , np.unique(train['Sex'])


# In[ ]:


np.unique(train['Sex'])


# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[ ]:


label=LabelEncoder()
train['Sex_code']=label.fit_transform(train['Sex'])


# In[ ]:


train['Sex_code'].unique()


#  Difference between count and size, count() applies the function to each column, returning the number of not null records within each.

# In[ ]:


train.groupby('Pclass').size()


# In[ ]:


train.groupby('Pclass').count()


# In[ ]:


print(train[['Pclass','Survived']].groupby(['Pclass'],as_index=True).size())


# In[ ]:


print(train[train['Survived']==1].groupby(['Pclass'],as_index=False).size())


# In[ ]:


print(train[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean())


# In[ ]:


print(train[['Sex','Survived']].groupby(['Sex'],as_index=False).count())


# In[ ]:


print(train[train['Survived']==1].groupby(['Sex'],as_index=False).size())


# In[ ]:


print(train[['SibSp','Survived']].groupby(['SibSp'],as_index=False).count())


# In[ ]:


print(train[train['Survived']==1].groupby(['SibSp'],as_index=False).size())


# In[ ]:


print(train[['Embarked','Survived']].groupby(['Embarked'],as_index=False).size())


# In[ ]:


print(train[train['Survived']==1].groupby(['Embarked'],as_index=False).size())


# In[ ]:


sns.factorplot('Pclass','Survived', data=train,size=4,aspect=3)


# In[ ]:


g=sns.FacetGrid(train,col='Survived')
g.map(plt.hist,'Age',bins=20,color='m')


# In[ ]:


g=sns.FacetGrid(train,col='Survived')
g.map(plt.scatter,'Fare','Age',edgecolor='w')


# Alpha is the brightness 0.1 darkest and 0.9 brightest.
# size size of the graph 1 is smaller 10 is bigger. bins-number of division along x axis

# In[ ]:


grid=sns.FacetGrid(train,col='Survived',row='Pclass',size=3,aspect=1.2)
grid.map(plt.hist,'Age',alpha=.7,bins=20).add_legend()


# In[ ]:


g1=sns.FacetGrid(train,row='Embarked')
g1.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep').add_legend()


# In[ ]:


g2=sns.FacetGrid(train,row='Embarked',col='Survived')
g2.map(sns.barplot,'Sex','Fare').add_legend()


# In[ ]:


train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q':2} ).astype(int)


# In[ ]:


test['Embarked'] = test['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q':2} ).astype(int)


# In[ ]:


grid=sns.FacetGrid(train,col='Pclass',row='Sex',size=3,aspect=1.2)
grid.map(plt.hist,'Age',alpha=.7,bins=20).add_legend()


# In[ ]:


avg_age=train['Age'].mean()
std_age=train['Age'].std()
age_null_count=train['Age'].isnull().sum()
age_null_random_list = np.random.randint(avg_age - std_age, avg_age + std_age, size=age_null_count)
train['Age'][np.isnan(train['Age'])] = age_null_random_list
train['Age']=train['Age'].astype(int)
train['cat_age']=pd.cut(train['Age'],5)


# In[ ]:


avg_age,std_age,age_null_count
#age_null_random_list


# In[ ]:


avg_age=test['Age'].mean()
std_age=test['Age'].std()
age_null_count=test['Age'].isnull().sum()
age_null_random_list = np.random.randint(avg_age - std_age, avg_age + std_age, size=age_null_count)
test['Age'][np.isnan(test['Age'])] = age_null_random_list
test['Age']=test['Age'].astype(int)
test['cat_age']=pd.cut(test['Age'],5)


# In[ ]:


test.loc[ test['Age'] <= 16, 'Age'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <= 32), 'Age'] = 1
test.loc[(test['Age'] > 32) & (test['Age'] <= 48), 'Age'] = 2
test.loc[(test['Age'] > 48) & (test['Age'] <= 64), 'Age'] = 3
test.loc[ test['Age'] > 64, 'Age'] = 4


# In[ ]:


test['Sex'].unique()


# In[ ]:


train.groupby('Pclass').count()


# In[ ]:


test.groupby('Pclass').count()


# In[ ]:


train


# In[ ]:


train.loc[ train['Age'] <= 16, 'Age'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <= 32), 'Age'] = 1
train.loc[(train['Age'] > 32) & (train['Age'] <= 48), 'Age'] = 2
train.loc[(train['Age'] > 48) & (train['Age'] <= 64), 'Age'] = 3
train.loc[ train['Age'] > 64, 'Age'] = 4


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


rf=RandomForestClassifier()


# In[ ]:


y1=train['Survived']#.ravel()
x2=train.drop(['Survived','Name','Ticket','Cabin','cat_age'],axis=1)
#x1=x2.values


# In[ ]:


y1.shape,x2.shape,x2.columns


# In[ ]:


#def feature_importances(self,x1,y1):
#       print(self.clf.fit(x1,y1).feature_importances_)


# In[ ]:


def feature_selection(x2,y1):
    clf = ExtraTreesClassifier(n_estimators=10)
    #clf = clf.fit(X_train.iloc[0:780269,:], np.ravel(y_train.iloc[0:780269,:]))
    clf = clf.fit(x2,np.ravel(y1))
    feat_importances = pd.Series(clf.feature_importances_, index=x2.columns)
    dict={'A':clf.feature_importances_,'B':x2.columns}
    df_new=pd.DataFrame(dict)
    #df_new.to_csv('../working/fet.csv')
    print(df_new)


# In[ ]:


feature_selection(x2,y1)


# In[ ]:


#rf.fit(x1,y1)
#rf.feature_importances(x1,y1)
#print (zip(map(lambda x2: round(x2, 4), rf.feature_importances_), x2.columns) )             


# In[ ]:


#X_train = train.drop("Survived",axis=1)
#Y_train = train["Survived"]
#X_test  = test.drop("PassengerId",axis=1).copy()


# In[ ]:


train.shape,test.shape


# In[ ]:


X_train = train.drop(['Ticket','Cabin','cat_age','Name','PassengerId','Survived','Sex_code'],axis=1)
Y_train = train['Survived']
X_test  = test.drop(['Ticket','Cabin','Name','PassengerId','cat_age'],axis=1)


# In[ ]:


X_train.shape,X_test.shape,Y_train.shape,test.shape,train.shape


# In[ ]:


X_train.columns,X_test.columns


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.20,random_state=42)


# In[ ]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape,X_test.shape


# In[ ]:


import tensorflow as tf
from keras.utils import to_categorical 


# In[ ]:


classes=2
y_train=to_categorical(y_train,num_classes = classes)
y_test=to_categorical(y_test,num_classes = classes)


# In[ ]:


y_train[0]


# In[ ]:


epochs=2
batch_size=32
display_progress=40
wt_init=tf.contrib.layers.xavier_initializer()


# In[ ]:


n_input=7
n_dense_1=32
n_dense_2=32
n_classes=2


# In[ ]:


x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_classes])


# In[ ]:


def dense(x,W,b):
    z=tf.add(tf.matmul(x,W),b)
    a=tf.nn.relu(z)
    return a


# In[ ]:


def network(x,weights,biases):
    dense_1=dense(x,weights['W1'],biases['b1'])
    dense_2=dense(dense_1,weights['W2'],biases['b2'])
    output_layer_z=tf.add(tf.matmul(dense_2,weights['W_out']),biases['b_out'])
    return output_layer_z


# In[ ]:


bias_dict={
    'b1':tf.Variable(tf.zeros([n_dense_1])),
    'b2':tf.Variable(tf.zeros([n_dense_2])),
    'b_out':tf.Variable(tf.zeros([n_classes]))
}

weight_dict={
    'W1':tf.get_variable('W1',[n_input,n_dense_1],initializer=wt_init),
    'W2':tf.get_variable('W2',[n_dense_1,n_dense_2],initializer=wt_init),
    'W_out':tf.get_variable('W_out',[n_dense_2,n_classes],initializer=wt_init)
}


# In[ ]:


predictions=network(x,weights=weight_dict ,biases=bias_dict)


# In[ ]:


print(predictions.shape),print(y.shape)


# In[ ]:


cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions ,labels=y))
optimizer=tf.train.AdamOptimizer().minimize(cost)


# In[ ]:


correct_prediction=tf.equal(tf.argmax(predictions,1),tf.argmax(y,1))
accuracy_pct=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))*100


# In[ ]:


init=tf.global_variables_initializer()


# In[ ]:


predict=tf.argmax(predictions,1)


# In[ ]:


with tf.Session() as session:
    session.run(init)
    
    print("Training for", epochs, "epochs.")
    
    # loop over epochs: 
    for epoch in range(epochs):
        
        avg_cost = 0.0 # track cost to monitor performance during training
        avg_accuracy_pct = 0.0
        
        # loop over all batches of the epoch:
        n_batches = int(x_train.shape[0] / batch_size)
        #batchnumber=0
        for i in range(n_batches):
            
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            #batchnumber= batchnumber+1
            batch_start_idx = (i * batch_size) % (x_train.shape[0] - batch_size)
            batch_end_idx = batch_start_idx + batch_size
            batch_X = x_train[batch_start_idx:batch_end_idx]
            batch_Y = y_train[batch_start_idx:batch_end_idx]
            
            # feed batch data to run optimization and fetching cost and accuracy: 
            _, batch_cost, batch_acc, Predict = session.run([optimizer, cost, accuracy_pct, predictions], 
                                                   feed_dict={x: batch_X, y: batch_Y})
            
            # accumulate mean loss and accuracy over epoch: 
            avg_cost += batch_cost / n_batches
            avg_accuracy_pct += batch_acc / n_batches
            
        # output logs at end of each epoch of training:
        print("Epoch ", '%03d' % (epoch+1), 
              ": cost = ", '{:.3f}'.format(avg_cost), 
              ", accuracy = ", '{:.2f}'.format(avg_accuracy_pct), "%", 
              sep='')
    
    print("Training Complete. Testing Model.\n")
    
    test_cost = cost.eval({x: x_test, y: y_test})
    test_accuracy_pct = accuracy_pct.eval({x: x_test, y: y_test})
    
    print("Test Cost:", '{:.3f}'.format(test_cost))
    print("Test Accuracy: ", '{:.2f}'.format(test_accuracy_pct), "%", sep='')
    
    predicted_lables = predict.eval({x: X_test})
    print(len(predicted_lables))
    #predicted_lables = np.zeros(X_test.shape[0])
    #for i in range(0,X_test.shape[0]//batch_size):
        #predicted_lables[i*batch_size : (i+1)*batch_size] = predict.eval({x: X_test[i*batch_size : (i+1)*batch_size], 
                                       


# In[ ]:


X_test.shape[0]


# In[ ]:


predicted_lables[0]


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predicted_lables
    })
submission.to_csv('titanic.csv', index=False)


# In[ ]:




