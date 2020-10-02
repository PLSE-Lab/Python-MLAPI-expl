# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.contrib.layers.python.layers import initializers

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



test=pd.read_csv("../input/test.csv")

train=pd.read_csv("../input/train.csv")



# -------------------- Start one hot encodign --------------------

train=train.drop(["Name","Ticket","Cabin","Embarked"],axis=1)

categ=pd.cut(train["Fare"],3,labels=["Low","Med","High"])

train=train.drop("Fare",axis=1)

train=train.join(categ)



dum=pd.get_dummies(data=train["Pclass"],prefix="Pclass")

train=train.drop("Pclass",axis=1)

train=train.join(dum)



dum=pd.get_dummies(data=train["Sex"],prefix="Sex")

train=train.drop("Sex",axis=1)

train=train.join(dum)



categ=pd.cut(train["Age"],bins=[0,2,6,12,18,50,200])

train=train.drop("Age",axis=1)

train=train.join(categ)

dum=pd.get_dummies(data=train["Age"],prefix="Age")

train=train.drop("Age",axis=1)

train=train.join(dum)



#dum=pd.get_dummies(data=train["SibSp"],prefix="SibSp")

train=train.drop("SibSp",axis=1)

#train=train.join(dum)



#dum=pd.get_dummies(data=train["Parch"],prefix="Parch")

train=train.drop("Parch",axis=1)

#train=train.join(dum)



#dum=pd.get_dummies(data=train["Fare"],prefix="Fare")

train=train.drop("Fare",axis=1)

#train=train.join(dum)





test=test.drop(["Name","Ticket","Cabin","Embarked"],axis=1)

categ=pd.cut(test["Fare"],3,labels=["Low","Med","High"])

test=test.drop("Fare",axis=1)

test=test.join(categ)



dum=pd.get_dummies(data=test["Pclass"],prefix="Pclass")

test=test.drop("Pclass",axis=1)

test=test.join(dum)



dum=pd.get_dummies(data=test["Sex"],prefix="Sex")

test=test.drop("Sex",axis=1)

test=test.join(dum)



categ=pd.cut(test["Age"],bins=[0,2,6,12,18,50,200])

test=test.drop("Age",axis=1)

test=test.join(categ)

dum=pd.get_dummies(data=test["Age"],prefix="Age")

test=test.drop("Age",axis=1)

test=test.join(dum)



#dum=pd.get_dummies(data=test["SibSp"],prefix="SibSp")

test=test.drop("SibSp",axis=1)

#test=test.join(dum)



#dum=pd.get_dummies(data=test["Parch"],prefix="Parch")

test=test.drop("Parch",axis=1)

#test=test.join(dum)



#dum=pd.get_dummies(data=test["Fare"],prefix="Fare")

test=test.drop("Fare",axis=1)

#test=test.join(dum)



trainY=train['Survived']

train=train.drop('Survived',axis=1)

trainY=pd.get_dummies(data=trainY,prefix="Survived")

train=train.drop('PassengerId',axis=1)



testID=test["PassengerId"]

test=test.drop('PassengerId',axis=1)

# -------------------- Stop one hot encodign -------------------

learning_rate = 0.1

training_epochs =100

learning_drop_sub=(learning_rate-0.005)/ training_epochs

batch_size = 50

display_step = training_epochs/5

dropout_flag=True



x_shape=train.shape[1]

y_shape=trainY.shape[1]

x = tf.placeholder(tf.float32, [None, x_shape])

y = tf.placeholder(tf.float32, [None, y_shape])



l1=tf.layers.dense( inputs=x,

                    units=8,

                    activation=tf.sigmoid,

                    kernel_initializer=tf.random_uniform_initializer,

                    bias_initializer=tf.random_uniform_initializer,

                    name="First_layer")

l2=tf.layers.dense( inputs=l1,

                    units=5,

                    activation=tf.sigmoid,

                    kernel_initializer=tf.random_uniform_initializer,

                    bias_initializer=tf.random_uniform_initializer,

                    name="Second_layer")

l3=tf.layers.dropout(inputs=l2,rate=0.4,training=dropout_flag)

pred=tf.layers.dense( inputs=l3,

                    units=2,

                    activation=tf.sigmoid,

                    kernel_initializer=tf.random_uniform_initializer,

                    bias_initializer=tf.random_uniform_initializer,

                    name="Output_layer")



output=tf.argmax(pred, axis=1)



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init_op)

    

    total_batch = int(len(train)/batch_size)

    for epoch in range(training_epochs):

        X_batches = np.array_split(train, total_batch)

        Y_batches = np.array_split(trainY, total_batch)

        count=0

        for i in range(total_batch):

            batch_x, batch_y = X_batches[i], Y_batches[i]

            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,

                                                          y: batch_y})

            count=count+c

        avg=count/total_batch

        if epoch % display_step == 0:

            print("Epoch:", '%04d' % (epoch+1), "\navg cost=", "%0.9f" % avg)

        learning_rate=learning_rate-learning_drop_sub;

    print("Optimization Finished!")

    

    dropout_flag=False

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Accuracy:", accuracy.eval({x: train, y: trainY}))

    

    prediction = sess.run([output], feed_dict={x:test})

    testID=testID.tolist()

    StackingSubmission = pd.DataFrame({ 'PassengerId': testID,'Survived': prediction[0]})

    

    StackingSubmission.to_csv("Submission13.csv", index=False)