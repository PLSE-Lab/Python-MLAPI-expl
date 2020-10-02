import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf


filename = '../input/framingham.csv'
df = pd.read_csv(filename)
df.fillna(method='ffill', inplace=True)

# male0 = Female; 1 = Male
# ageAge at exam time.
# education1 = Some High School; 2 = High School or GED; 3 = Some College or Vocational School; 4 = college
# currentSmoker0 = nonsmoker; 1 = smoker
# cigsPerDaynumber of cigarettes smoked per day (estimated average)
# BPMeds0 = Not on Blood Pressure medications; 1 = Is on Blood Pressure medications
# prevalentStroked
# prevalentHyp
# diabetes0 = No; 1 = Yes
# totCholmg/dL
# sysBPmmHg
# diaBPmmHg
# BMIBody Mass Index calculated as: Weight (kg) / Height(meter-squared)
# heartRateBeats/Min (Ventricular)
# glucosemg/dL
# TenYearCHD


N, D = len(df), 2
X = np.ones((N, D+1))
X[:,1] = df.glucose.values
X[:,2] = df.sysBP.values
Y = df.diabetes.values

X, Y = shuffle(X, Y)

for i in range(D):
    X[:,i+1] = (X[:,i+1] - X[:,i+1].mean()) / X[:,i+1].std()

split = int(len(df)*0.60)
x_train, x_test = X[:split], X[split:]
y_train, y_test = Y[:split], Y[split:]

# Sklearn Section
log = LogisticRegression()
log.fit(x_train, y_train)
y_pred = log.predict(x_test)
sklearn_score = accuracy_score(y_test, y_pred)

# Numpy Section

def cross_entropy(t, y):
    return -(t*np.log(y) + (1-t)*np.log(1-y)).sum()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

train_costs = []
test_costs = []
w = np.random.randn(D+1)
epochs = 10000
lr = 0.0001
y_train_pred = sigmoid(x_train.dot(w))
y_test_pred = sigmoid(x_test.dot(w))
for i in range(epochs):
    train_costs.append(cross_entropy(y_train, y_train_pred))
    test_costs.append(cross_entropy(y_test, y_test_pred))
    w += (x_train.T.dot(y_train - y_train_pred) - 0.001*w) * lr
    y_train_pred = sigmoid(x_train.dot(w))
    y_test_pred = sigmoid(x_test.dot(w))

plt.title('Numpy')
plt.plot(train_costs)
plt.plot(test_costs)
plt.legend()
plt.show()

y_pred = sigmoid(x_test.dot(w))
numpy_score = accuracy_score(y_test, np.round(y_test_pred))


# Tensorflow Section

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
x_tf = tf.placeholder(tf.float64, (None, D+1))
y_tf = tf.placeholder(tf.float64, (None, 1))

w = tf.Variable(np.random.randn(D+1, 1))
z = tf.matmul(x_tf, w)

y_pred_tf = tf.nn.sigmoid(z)

lr = 1
cost = tf.losses.sigmoid_cross_entropy(y_tf, y_pred_tf)
train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
epochs = 10000
train_costs = []
test_costs = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        _, train_loss = sess.run([train_op, cost], feed_dict={
            x_tf: x_train, y_tf: y_train
        })
        test_loss = sess.run(cost, feed_dict={
            x_tf: x_test, y_tf: y_test
        })
        train_costs.append(train_loss)
        test_costs.append(test_loss)
    
    y_pred = sess.run(y_pred_tf, feed_dict={
        x_tf: x_test
    })

tensorflow_score = accuracy_score(y_test, np.round(y_pred))

plt.title('Tensorflow')
plt.plot(train_costs)
plt.plot(test_costs)
plt.legend()
plt.show()

print("numpy score: ", numpy_score)
print("sklearn score: ", sklearn_score)
print("tensorflow score: ", tensorflow_score) 
#%%
