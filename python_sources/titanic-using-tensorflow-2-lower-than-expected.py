# this approch is deeper than https://www.kaggle.com/tomorowo/titanic-using-tensorflow and higher from expected point of view, but result is worse than previous approach.
from __future__ import absolute_import, unicode_literals
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import re
import datetime

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
x_train = train_data.drop(['PassengerId','Ticket','Survived'], axis=1)
y_train = pd.DataFrame({'Dead':(train_data['Survived']+1)%2,'Survived':train_data['Survived']})
x_test = test_data.drop(['PassengerId','Ticket'], axis=1)

x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean())
x_test['Age'] = x_test['Age'].fillna(x_test['Age'].mean())

def simplify_ages(df):
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df['Age'], bins, labels=group_names)
    df['Age'] = categories.cat.codes 
    return df

def simplify_cabins(df):
    df['Cabin'] = df['Cabin'].fillna('N')
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])
    df['Cabin'] = pd.Categorical(df['Cabin'])
    df['Cabin'] = df['Cabin'].cat.codes 
    return df

def simplify_fares(df):
    df['Fare'] = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df['Fare'], bins, labels=group_names)
    df['Fare'] = categories.cat.codes 
    return df

def simplify_sex(df):
    df['Sex'] = pd.Categorical(df['Sex'])
    df['Sex'] = df['Sex'].cat.codes 
    return df

def simplify_embarked(df):
    df['Embarked'] = pd.Categorical(df['Embarked'])
    df['Embarked'] = df['Embarked'].cat.codes + 1
    return df

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

def simplify_name(df):
    df['Name'] = df['Name'].apply(get_title)
    df['Name'] = df['Name'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Name'] = df['Name'].replace('Mlle', 'Miss')
    df['Name'] = df['Name'].replace('Ms', 'Miss')
    df['Name'] = df['Name'].replace('Mme', 'Mrs')    
    df['Name'] = pd.Categorical(df['Name'])
    df['Name'] = df['Name'].cat.codes + 1
    return df

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = simplify_sex(df)
    df = simplify_embarked(df)
    df = simplify_name(df)
    return df

transform_features(x_train)
transform_features(x_test)
x_train.head(2)

HD = 20
STDDEV = 0.35

x = tf.placeholder("float", [None, 9])

#W = tf.Variable(tf.zeros([9, HD]))
W = tf.Variable(tf.random_normal([9,HD], stddev=STDDEV)) # this change is important. if use zeros, then return all zero for W and b
b = tf.Variable(tf.zeros([HD]))
x2 = tf.nn.relu(tf.matmul(x, W) + b)

W2 = tf.Variable(tf.zeros([HD, HD]))
b2 = tf.Variable(tf.zeros([HD]))
x3 = tf.nn.softmax(tf.matmul(x2, W2) + b2)

W3 = tf.Variable(tf.zeros([HD, 2]))
b3 = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x3, W3) + b3)

y_ = tf.placeholder("float", [None, 2])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 10
length = len(x_train)

for i in range(1000):
    #print("train loop=" , i)
    #print(i,".",)
    for j in range(int(length/batch_size)):
        batch_xs = x_train[j*batch_size:(j+1)*batch_size-1]
        batch_ys = y_train[j*batch_size:(j+1)*batch_size-1]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i%10 == 0:
        c=0
        t=0
        for index,value in x_train.iterrows():
            batch_xs = [value]
            dy = sess.run(y, feed_dict={x: batch_xs})
            if train_data.loc[index]['Survived'] == int(np.round(dy[0][1])):
                c = c + 1
            t = t +1
        print(datetime.datetime.today())
        print("try = ", i , ": result = ", c/t)
    if i%250 == 249:
        fn = './Titanic_deep_more_' + str(i)  + '_' + str(c/t) + '.csv' 
        f = open(fn,'w')
        f.write('PassengerId,Survived\n')
        for index,value in x_test.iterrows():
            batch_xs = [value]
            dy = sess.run(y, feed_dict={x: batch_xs})
            f.write(str(test_data.loc[index]['PassengerId'])+','+str(int(np.round(dy[0][1])))+'\n') # Survived
        f.close()

print('PassengerId,Survived')
for index,value in x_test.iterrows():
    batch_xs = [value]
    dy = sess.run(y, feed_dict={x: batch_xs})
    #print(np.round(dy))
    print(test_data.loc[index]['PassengerId'], ',' , int(np.round(dy[0][1]))) # Survived