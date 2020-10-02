'''
August24-2018
Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for structured tabular data HR dataset csv file ternary classification with deepneural nets tensorflow, pandas and keras
Results:
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 256)               2560      
_________________________________________________________________
dense_1 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_2 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_3 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_4 (Dense)              (None, 3)                 771       
=================================================================
Total params: 200,707
Trainable params: 200,707
Non-trainable params: 0

train loss: 0.5873 - acc: 0.7212 - val_loss: 1.1604 - val_acc: 0.4936
type(history)
<class 'tensorflow.python.keras.callbacks.History'>
4500/4500 [==============================] - 0s 36us/step
test loss,accuracy 1.2001526788075765 0.5002222221957313
confusion matrix: [[ 106  120  155]
 [ 323 1111  734]
 [ 306  790  855]]
accuracy: 0.45
precision: 0.5103326612903226
recall: 0.45
f1score: 0.478271138403401
cohen_kappa_score: 0.10338946846160546




'''
# Import pandas 
import pandas as pd
from keras.utils import to_categorical

#read in parkinson's dataset csv
hrtable = pd.read_csv('../input/HR-deepnet.csv', sep=',')


#Preprocess Data

#print (hrtable)

#let's gather details about this table
print("hrtable.info()", hrtable.info())
print("len(hrtable)",len(hrtable))
print("hrtable.describe()",hrtable.describe())
print("hrtable.head()",hrtable.head())
print("hrtable.tail()",hrtable.tail())
print("hrtable.sample(5)",hrtable.sample(5))
print("pd.isnull(hrtable)",pd.isnull(hrtable))


#Correlation Matrix
import seaborn as sns
import matplotlib.pyplot as plt
corr = hrtable.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()


##############creating the train and test sets
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
import numpy as np
# Specify the data 
X=hrtable.iloc[:,0:9]       #all rows all cols except the 22nd(last col) goes into training set

print(X.info())
print(X.describe())
# Specify the target labels and flatten the array
y= np.ravel(hrtable.SalaryType)

print("type(X)",type(X))
print("type(y)",type(y))

# Split the data up in train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

print("before to_categorical:",y_train[5:15])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("after to_categorical:\n",y_train[5:15])


y_train = y_train.astype(int)
y_test = y_test.astype(int)
print("after astype(int)\n",y_train[5:15])





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

# Add an input layer 
#check input data shape
print("X_train[21].shape",X_train[21].shape)

#model.add(tf.keras.layers.Dense(12, activation='relu', input_shape=(22,)))
model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=X_train[21].shape))

# Add one hidden layer 
model.add(tf.keras.layers.Dense(256, activation='relu'))

# Add one hidden layer 
model.add(tf.keras.layers.Dense(256, activation='relu'))

# Add one hidden layer 
model.add(tf.keras.layers.Dense(256, activation='relu'))

# Add an output layer 
model.add(tf.keras.layers.Dense(3, activation='softmax'))

#compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

                   
# Model output shape
print("model.output_shape",model.output_shape)

# Model summary
print("model.summary()",model.summary())

# Model config
#print("model.get_config()",model.get_config())

# List all weight tensors 
#print("model.get_weights()",model.get_weights())

#model.fit(X_train, y_train,epochs=30, batch_size=32, verbose=1)

##############################################
#fit and plot
x_val = X_train[:5000]
partial_x_train = X_train[5000:]
y_val = y_train[:5000]
partial_y_train = y_train[5000:]

print("partial_x_train.shape:",partial_x_train.shape)
print("partial_y_train.shape:",partial_y_train.shape)
print("x_val.shape:",x_val.shape)
print("y_val.shape:",y_val.shape)

print("partial_x_train[5:10]-",partial_x_train[5:10])
print("partial_y_train[5:10]-",partial_y_train[5:10])
print("x_val[5:10]",x_val[5:10])
print("y_val[5:10]",y_val[5:10])



history = model.fit(partial_x_train,
partial_y_train,
epochs=50,
batch_size=256,
validation_data=(x_val, y_val))

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
(loss,accuracy) = model.evaluate(X_test,y_test)
print("loss,accuracy",loss,accuracy)




#let's make predictions
y_pred = model.predict(X_test)
y_pred=y_pred.round().astype(int)
#print("type(y_pred)",type(y_pred))
#print("y_pred.shape",y_pred.shape)
print(y_pred[15:30])
print(y_test[15:30])



#testset metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
print("confusion matrix:", confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Accuracy 
print("accuracy:", accuracy_score(y_test, y_pred))

# Precision 
print("precision:", precision_score(y_test, y_pred, average='micro'))

# Recall
print("recall:",recall_score(y_test, y_pred, average='micro'))

# F1 score
print("f1score:", f1_score(y_test,y_pred, average='micro'))

# Cohen's kappa
print("cohen_kappa_score:", cohen_kappa_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
