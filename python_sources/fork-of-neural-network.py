import pandas as pd
import numpy as np
import random as rnd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
combine = [train_data, test_data]
scaler = MinMaxScaler(feature_range=(0, 1))

print(train_data.columns.values)

train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_data, test_data]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Lady',
                        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_data = train_data.drop(['Name', 'PassengerId'], axis=1)
test_data = test_data.drop(['Name'], axis=1)
combine = [train_data, test_data]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
      

train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
     
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

train_data = train_data.drop(['AgeBand'], axis=1)
combine = [train_data, test_data]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_data, test_data]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_data.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

freq_port = train_data.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

x_train = train_data.drop("Survived", axis=1)
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
y_train = train_data["Survived"]
x_test  = test_data.drop("PassengerId", axis=1).copy()
x_test = pd.DataFrame(scaler.fit_transform(x_test), columns=x_test.columns)
y_test = test_data["PassengerId"]

print(x_train.head())

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

n_features = x_train.shape[1]
n_classes = 2

x_placeholder = tf.placeholder(tf.float64,(None,n_features), name='input')
prob = tf.placeholder(tf.float64)

#hidden = tf.layers.dense(x_placeholder,256,name='hidden_1',  activation=tf.nn.tanh,
#                         kernel_initializer=tf.truncated_normal_initializer() )
hidden = tf.layers.dense(x_placeholder,700, activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer() )

#hidden2 = tf.layers.dropout(hidden,rate = prob)
#hidden2 = tf.layers.dense(hidden,32, activation=tf.nn.relu, 
#                         kernel_initializer=None )
#hidden3 = tf.layers.dense(hidden2,600, activation=tf.nn.tanh, 
#                         kernel_initializer=tf.truncated_normal_initializer() )
                        
                         
#hidden4 = tf.layers.dense(hidden3,100, activation=tf.nn.relu, 
#                         kernel_initializer=tf.truncated_normal_initializer() )
                         
logit = tf.layers.dense(hidden,2,name='logit',activation=tf.nn.softmax, 
                         kernel_initializer=None )
                         #kernel_initializer=tf.truncated_normal_initializer() )

y_placeholder = tf.placeholder(tf.int32, (None),name='output')
one_hot_y = tf.one_hot(y_placeholder, n_classes,name='onehot_output')

#learning_rate = 0.0003
rate = 0.35
reg_constant = 0.035
EPOCHS = 1000
BATCH_SIZE = int(len(x_train)/1)

regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels = one_hot_y)
loss = tf.reduce_mean(cross_entropy) + reg_constant * sum(regularization_loss)

global_step = tf.Variable(2, trainable=True)
starter_learning_rate = 0.003
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           400, 2.0, staircase=True)
# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.train.GradientDescentOptimizer(learning_rate)
    .minimize(loss , global_step=global_step)
)

#optimizer = tf.train.AdamOptimizer(learning_rate)
#training_op = optimizer.minimize(loss)

# if predicton is correct
pred = tf.argmax(logit, 1)
correct_prediction = tf.equal(tf.argmax(logit,1), tf.argmax(one_hot_y,1))
accuracy_ops = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(x,y,sess):
    global BATCH_SIZE
    num_examples = len(x)
    total_accuracy = 0
    #sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x[offset:offset+BATCH_SIZE], y[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_ops, 
                          feed_dict={x_placeholder: batch_x, y_placeholder:batch_y, prob:1.0})
        total_accuracy += (accuracy*len(batch_x))
    return total_accuracy/num_examples

epochs = []
list_train_accuracy = []
list_valid_accuracy = []
plt.ion()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
num_examples = len(x_train)
print("Training ..")
print()
for i in range(EPOCHS):
    x_train_sh, y_train_sh = x_train, y_train
    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = x_train_sh[offset:end], y_train_sh[offset:end]
        #sess.run(training_op,feed_dict={x_placeholder:batch_x, y_placeholder:batch_y,prob:1.0})
        sess.run(learning_step,feed_dict={x_placeholder:batch_x, y_placeholder:batch_y,prob:1.0})

    valid_acc = evaluate(x_test, y_test,sess)
    train_acc = evaluate(x_train, y_train,sess)
    list_train_accuracy.append(train_acc)
    list_valid_accuracy.append(valid_acc)
    print('\r',"EPOCH {} ...".format(i+1),
          "Validation Accuracy = {:.3f} ...".format(valid_acc),
          "Training Accuracy = {:.3f} ...".format(train_acc),end='')
    if len(epochs) == 0:
        epochs.append(0)
    else:
        epochs.append(epochs[-1]+1)
plt.plot(epochs,list_train_accuracy,'b-',epochs,list_valid_accuracy,'r-')
plt.show()

predicted_labels = sess.run(pred, feed_dict={x_placeholder: x_test,prob: 1.0})
d = {'PassengerId': test_data['PassengerId'], 'Survived': predicted_labels}
prediction_df = pd.DataFrame(data=d)
prediction_df.to_csv('test_out.csv', index=False)