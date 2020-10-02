# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def normalize(x): return (x - np.min(x)) / (np.max(x) - np.min(x))

train = pd.read_csv("../input/train.csv", header = 0, dtype = {'Age': np.float64})
test = pd.read_csv("../input/test.csv", header = 0, dtype = {'Age': np.float64})

dataset =pd.concat([train,test])

len_train=(len(train))
len_test=(len(test))


dataset['Pclass'] = normalize(dataset.Pclass)

dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset['FamilySize'] = dataset['FamilySize'].astype('float64')
dataset['FamilySize'] = normalize(dataset.FamilySize)

dataset['Embarked'] = dataset['Embarked'].fillna('NA')

dataset['Age'] = dataset['Age'].fillna(train['Age'].median())
dataset['Age'] = dataset['Age'].astype('float64')
dataset['Age'] = normalize(dataset.Age)

dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
dataset['Fare'] = dataset['Fare'].astype('float64')
dataset['Fare'] = normalize(dataset.Fare)

dataset['TicketNumber'] = dataset.Ticket.str.extract('(\d+)', expand=True)
dataset['TicketNumber'] = dataset['TicketNumber'].fillna(0)
dataset['TicketNumber'] = pd.to_numeric(dataset['TicketNumber'])	
dataset['TicketNumber'] = normalize(dataset.TicketNumber)

dataset['NumCabins'] = dataset['Cabin'].str.split().str.len()
dataset['NumCabins'] = dataset['NumCabins'].fillna(0)
dataset['NumCabins'] = normalize(dataset.NumCabins)

dataset['CabinLevel'] = dataset['Cabin'].astype(str).str[0]

dataset['CabinNumber'] = dataset.Cabin.str.extract('(\d+)', expand=True)
dataset['CabinNumber'] = dataset['CabinNumber'].fillna(0)
dataset['CabinNumber'] = pd.to_numeric(dataset['CabinNumber'])
dataset['CabinNumber'] = normalize(dataset.CabinNumber)

dataset['Sex'].replace(['male','female'],[1,0],inplace=True)

dummies1=pd.get_dummies(dataset['Embarked'],'Embarked')
dummies2=pd.get_dummies(dataset['CabinLevel'],'CabinLevel')

dataset=pd.concat([dataset,dummies1],1)
dataset =pd.concat([dataset,dummies2],1)

dataset = dataset.drop(columns=['CabinLevel','Embarked','Cabin','Ticket','Name','PassengerId','SibSp','Parch'])

train = dataset.iloc[:len_train]
test = dataset.iloc[len_train:]

train_y = train['Survived'].as_matrix()
train_y = train_y.reshape(len_train,1)

train =  train.drop(columns=['Survived'])
test = test.drop(columns=['Survived'])

train_x = train.as_matrix()
test_x = test.as_matrix()

x_data = train_x
y_data = train_y
n_input = x_data.shape[1] 
n_hidden = x_data.shape[1]
n_output = 1
learning_rate = 0.1
epochs = 20000

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], 0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], 0, 1.0))

b1 = tf.Variable(tf.random_uniform([n_output],0,1))
b2 = tf.Variable(tf.random_uniform([n_output],0,1))


L2 = tf.sigmoid(tf.matmul(X, W1)+ b1)
hy = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = tf.reduce_mean(-Y*tf.log(hy) - (1-Y)*tf.log(1-hy)) 
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:
  session.run(init)
    
  for step in range(epochs):
	    
    session.run(optimizer, feed_dict={X: x_data, Y: y_data})
	    
    if step % 1000 == 0:
	        
      print (session.run(cost, feed_dict={X: x_data, Y: y_data}))
	
    answer = tf.equal(tf.floor(hy+.5), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))
    Y_test = tf.floor(hy + 0.5)
    print (session.run([Y_test], feed_dict={X: x_data, Y: y_data}))
    print ("Accuracy: ", accuracy.eval({X: x_data, Y: y_data}))
	
    prediction = session.run([Y_test]  , feed_dict = {X:test_x})
    prediction = np.asarray(prediction)
    prediction = prediction.reshape(len_test,1)
    submission = pd.DataFrame(prediction,columns=['Survived']).astype(int)
    submission['PassengerId']= submission.index + len_train + 1
    submission = submission[['PassengerId','Survived']]
    submission.to_csv("submission.csv",index=False)