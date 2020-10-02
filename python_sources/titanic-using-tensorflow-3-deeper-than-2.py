# ToDo
# DONE: random access to minibatch
# DONE: add noize to sample data 
# Feauture Hashing / Data separation, maybe use word2vec ?
# Learning rate decay

# Hi, this is third trial for "Titanic using Tensorflow" 
# 1st https://www.kaggle.com/tomorowo/titanic-using-tensorflow 
# 2nd https://www.kaggle.com/tomorowo/titanic-using-tensorflow-2-lower-than-expected
from __future__ import absolute_import, unicode_literals
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import re
import datetime
import random
import math

global output,output_data,max_score
output      = pd.DataFrame(index=[],columns=['hd','depth','opt','total_trys','batch','noize','try','forward','back','score','is_best'])
output_data = pd.DataFrame() #will be defined
max_score   = 0

global x_train,y_train,x_train_forward,y_train_foward,x_test
train_data = pd.read_csv('../input/train.csv')
test_data  = pd.read_csv('../input/test.csv')
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
x_train_forward = x_train[750:]
y_train_forward = x_train[750:]
x_train = x_train[0:750]
y_train = y_train[0:750]

##
def execute_learning(HD,DEPTH,OPT,TRYS,BATCH_SIZE,NOIZE):
    global output,output_data,max_score
    global x_train,y_train,x_train_forward,y_train_foward,x_test

    STDDEV = 0.35
    LR = 0.001   #Learning Rate
    #NOIZE = 0.05 #Nozing Rate
    print("***** With optimizer = " + OPT)
    x_hl = [0 for i in range(DEPTH+1)]
    W_hl = [0 for i in range(DEPTH+1)]
    b_hl = [0 for i in range(DEPTH+1)]

    x = tf.placeholder("float", [None, 9])
    W = tf.Variable(tf.random_normal([9,HD], stddev=STDDEV)) #Don't use zero filling
    b = tf.Variable(tf.zeros([HD]))
    x_hl[0] = tf.nn.relu(tf.matmul(x, W) + b)

    for lp in range(DEPTH):
        W_hl[lp] = tf.Variable(tf.random_normal([HD,HD], stddev=STDDEV)) #Don't use zero filling
        b_hl[lp] = tf.Variable(tf.zeros([HD]))
        #x_hl[lp+1] = tf.nn.softmax(tf.matmul(x_hl[lp], W_hl[lp] + b_hl[lp]))
        x_hl[lp+1] = tf.nn.relu(tf.matmul(x_hl[lp], W_hl[lp] + b_hl[lp]))

    W_last = tf.Variable(tf.zeros([HD, 2])) #Last W should be zero
    b_last = tf.Variable(tf.zeros([2]))
    y = tf.nn.softmax(tf.matmul(x_hl[DEPTH], W_last) + b_last)

    y_ = tf.placeholder("float", [None, 2])
    #cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y,labels=y_))*100
    #train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    if OPT == "Adam":
        train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
    elif OPT == "Adag":
        train_step = tf.train.AdagradOptimizer(LR).minimize(cross_entropy)
    elif OPT == "Adad":
        train_step = tf.train.AdadeltaOptimizer(LR).minimize(cross_entropy)
    elif OPT == "Ftrl":
        train_step = tf.train.FtrlOptimizer(LR).minimize(cross_entropy)
    elif OPT == "Grad":
        train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    #try_time = 100
    #for i in range(try_time):
    #        batch_xs, batch_ys = mnist.train.next_batch(100)
    length = len(x_train)
    score = 0

    for i in range(TRYS):
        #print("train loop=" , i)
        #print(i,".",)
        for j in range(int(length/BATCH_SIZE)): # I think this loop has no mean.. ? marge to trys x length ?
            #batch_xs = x_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE-1]
            #batch_ys = y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE-1]
            batch_xs = x_train.sample(BATCH_SIZE)
            batch_ys = y_train.iloc[batch_xs.index]

            # insert noize into randam area
            for k in range(int(BATCH_SIZE*NOIZE/100)):
                batch_xs.iloc[k,[random.randint(0,8)]] = -1

            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if (i+1)%10 == 0:
            c=0
            t=0
            print(datetime.datetime.today())
            print("TRY = ", str(i+1) , ":")
            for index,value in x_train.iterrows():
                batch_xs = [value]
                dy = sess.run(y, feed_dict={x: batch_xs})
                if math.isnan(np.round(dy[0][1])):
                    continue
                elif train_data.loc[index]['Survived'] == int(np.round(dy[0][1])):
                    c = c + 1
                t = t +1
            ct_b = c/t
            print("      BACK TEST: result = ", str(round(100*ct_b,3)))
            c=0
            t=0
            for index,value in x_train_forward.iterrows():
                batch_xs = [value]
                dy = sess.run(y, feed_dict={x: batch_xs})
                if math.isnan(np.round(dy[0][1])):
                    continue
                elif train_data.loc[index]['Survived'] == int(np.round(dy[0][1])):
                    c = c + 1
                t = t +1
            ct_f = c/t
            score = ct_f * (1 - abs(1 - ct_f/ct_b))
            print("   FORWARD TEST: result = ", str(round(100*ct_f,3)))
            print("   ::SCORE : " , str(round(100 * score,3))) 
            #print("W_last=" , sess.run(W_last))


            is_best = 0
            if score > max_score:
                is_best = 1
                output['is_best'] = 0
            #output = pd.DataFrame(index=[],columns=['hd','depth','opt','total_trys','batch','noize','try','forward','back','score'])
            sr = pd.Series([HD,DEPTH,OPT,TRYS,BATCH_SIZE,NOIZE,i,ct_f,ct_b,score,is_best],index=output.columns)
            output = output.append(sr,ignore_index=True)

        if score > max_score:
            max_score = score 
            output_data = pd.DataFrame(index=[],columns=['PassengerId','Survived'])

            fn = './result_' + OPT + '_'  + str(HD) + '_' + str(DEPTH) + '_' + str(NOIZE) + '_' + \
                    str(i+1).zfill(len(str(TRYS))) + '_' + str(BATCH_SIZE) + '_' + \
                    str(round(100*ct_f,2)) + '_' + str(round(100*score,2)) + '.csv' 
            print("---> Writing to : " , fn)
            f = open(fn,'w')
            f.write('PassengerId,Survived\n')

            for index,value in x_test.iterrows():
                batch_xs = [value]
                dy = sess.run(y, feed_dict={x: batch_xs})
                sr = pd.Series([str(test_data.loc[index]['PassengerId']),str(int(np.round(dy[0][1])))],index=output_data.columns)

                output_data = output_data.append(sr,ignore_index=True)
                f.write(str(test_data.loc[index]['PassengerId'])+','+str(int(np.round(dy[0][1])))+'\n') # Survived
            f.close()
        #if (i+1)%100 == 0:
            #fn = './titanic_' + OPT + '_'  + str(HD) + '_' + str(DEPTH) +'_' + \
            #        str(i+1).zfill(len(str(TRYS))) + '_' + str(BATCH_SIZE) + '_' + \
            #        str(round(100*ct_f,2)) + '_' + str(round(100*score,2)) + '.csv' 
            #print("---> Writing to : " , fn)
            #f = open(fn,'w')
            #f.write('PassengerId,Survived\n')
            #for index,value in x_test.iterrows():
            #    batch_xs = [value]
            #    dy = sess.run(y, feed_dict={x: batch_xs})
            #    f.write(str(test_data.loc[index]['PassengerId'])+','+str(int(np.round(dy[0][1])))+'\n') # Survived
            #f.close()

HDS    = [5,10] #[5,10,20,50,100]  # 5 is good. demension of Hidden Layers
DEPTHS = [2,5,10] #[2,5,7,10]        # 2 is good. the number of Hidden Layers
OPTS   = ["Adam"] #["Adam","Adag","Adad","Ftrl","Grad"] # Adam is good.
TRY_AND_BATCHS = [[1000,100]] #[[1000,10],[1000,30],[3000,100]] # 3000,100 is good.
NOIZES = [0.05]


for HD in HDS:
    for DEPTH in DEPTHS:
        for OPT in OPTS:
            for TB in TRY_AND_BATCHS:
                for NOIZE in NOIZES:
                    execute_learning(HD,DEPTH,OPT,TRYS=TB[0],BATCH_SIZE=TB[1],NOIZE=NOIZE)

output.to_csv("titanic_output.csv")
output_data.to_csv("titanic_output_data_bestscore.csv",index=False)
#EOF