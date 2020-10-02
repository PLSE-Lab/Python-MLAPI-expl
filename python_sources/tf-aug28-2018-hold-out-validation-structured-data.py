'''
August28-2018
Author: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for structured tabular data surgical csv file binary classification with deepneural nets tensorflow, pandas and keras
Results:
train loss: 0.0865 - acc: 0.9618 - val_loss: 1.9394 - val_acc: 0.7330
test loss,accuracy 1.8126787839715772 0.7488612835204873

train acc is high, but test and val acc are low...overfitting....reduce model complexity(decrease number of units and layers) or get more training data.

confusion matrix: [[3056  581]
 [ 573  620]]
accuracy: 0.7610766045548655
precision: 0.5162364696086594
recall: 0.5196982397317687
f1score: 0.5179615705931495
cohen_kappa_score: 0.3591420182090488


#########################################
I then made the number of units as 2(num_ip + num_op) = 2(24+2) = 52
loss: 0.4018 - acc: 0.8135 - val_loss: 0.4400 - val_acc: 0.7930
type(history)
<class 'tensorflow.python.keras.callbacks.History'>
4830/4830 [==============================] - 0s 17us/step
loss,accuracy 0.4141769036681509 0.8018633540866291
confusion matrix: [[3363  274]
 [ 683  510]]
accuracy: 0.801863354037267
precision: 0.6505102040816326
recall: 0.42749371332774516
f1score: 0.5159332321699543
cohen_kappa_score: 0.39800236667402067
#########################################


'''
# Import pandas 
import pandas as pd
import numpy as np

#read in parkinson's dataset csv
surgicaltable = pd.read_csv("../input/Surgical-deepnet.csv", sep=',')


#Preprocess Data

print ("unsorted surgicaltable:",surgicaltable)

#shuffle the data
surgicaltable = surgicaltable.sample(frac=1).reset_index(drop=True)

print ("sorted surgicaltable:",surgicaltable)


#let's gather details about this table
print("surgicaltable.info()", surgicaltable.info())
print("len(surgicaltable)",len(surgicaltable))
print("surgicaltable.describe()",surgicaltable.describe())
print("surgicaltable.head()",surgicaltable.head())
print("surgicaltable.tail()",surgicaltable.tail())
print("surgicaltable.sample(5)",surgicaltable.sample(5))
print("pd.isnull(surgicaltable)",pd.isnull(surgicaltable))


#Correlation Matrix
import seaborn as sns
import matplotlib.pyplot as plt
corr = surgicaltable.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()


##############creating the train and test sets
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
import numpy as np

# Specify the data 
X=surgicaltable.iloc[:,0:24]       #all rows all cols except the 22nd(last col) goes into training set

print(X.info())
print(X.describe())
# Specify the target labels and flatten the array
y= np.ravel(surgicaltable.complication)

print("type(X)",type(X))
print("type(y)",type(y))

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("type(X_train)",type(X_train))
print("type(X_test)",type(X_test))
print("type(y_train)",type(y_train))
print("type(y_test)",type(y_test))

print("X_train.shape",X_train.shape)
print("X_test.shape",X_test.shape)
print("y_train.shape",y_train.shape)
print("y_test.shape",y_test.shape)

print("X.shape",X.shape)
print("y.shape",y.shape)

#Let's normalize this data to 0 mean and unit std
# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

#Let's create a tensorflow deepneuralnet model
# Import tensorflow
import tensorflow as tf

# Import keras
import keras

# Initialize the constructor
model = tf.keras.models.Sequential()

#using heuristics for hyperparamters
'''
#number of inputs num_ip = 24
#number of outputs num_op = 2

#set the number of units in hidden layer as 2(num_ip + num_op) = 2(24+2) = 52


'''

############################### model 1: just right ########################################
# Add an input layer 
#check input data shape
print("X_train[21].shape",X_train[21].shape)

#model.add(tf.keras.layers.Dense(12, activation='relu', input_shape=(22,)))
model.add(tf.keras.layers.Dense(52, activation='relu', input_shape=X_train[21].shape))

# Add one hidden layer 
model.add(tf.keras.layers.Dense(52, activation='relu'))

# Add one hidden layer 
model.add(tf.keras.layers.Dense(52, activation='relu'))

#Add dropout
model.add(tf.keras.layers.Dropout(0.5))

# Add an output layer 
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
############################### end of model 1 ########################################


#compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
#model.fit(X_train, y_train,epochs=10, batch_size=1, verbose=1)
##############################################

#hold-out validation
num_validation_samples = 1000
x_val = X_train[:num_validation_samples]
partial_x_train = X_train[num_validation_samples:]
y_val = y_train[:num_validation_samples]
partial_y_train = y_train[num_validation_samples:]

#fit and plot
history = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))

print("type(history)")
print(type(history))

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

##########################################
#evaluate the model
(testloss,testaccuracy) = model.evaluate(X_test,y_test)
print("testloss,testaccuracy",testloss,testaccuracy)

# Model output shape
model.output_shape

# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()


#let's make predictions
y_pred = model.predict(X_test)
y_pred=y_pred.round().astype(int)
#print("type(y_pred)",type(y_pred))
#print("y_pred.shape",y_pred.shape)
print(y_pred[35:75])
print(y_test[35:75])



#testset metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
print("confusion matrix:", confusion_matrix(y_test, y_pred))

# Accuracy 
print("accuracy:", accuracy_score(y_test, y_pred))

# Precision 
print("precision:", precision_score(y_test, y_pred))

# Recall
print("recall:",recall_score(y_test, y_pred))

# F1 score
print("f1score:", f1_score(y_test,y_pred))

# Cohen's kappa
print("cohen_kappa_score:", cohen_kappa_score(y_test, y_pred))
