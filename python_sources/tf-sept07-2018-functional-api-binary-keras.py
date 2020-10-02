'''
Sept07-2018
Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for functional api for surgical with early stopping
Results:
##################################### 4 relu layers 256 units each #############################################

test loss: 0.44, test acc 0.78
f1score: 0.48


################################# End #####################################################
'''

# Import pandas 
import pandas as pd

#read in surgical dataset csv
surgicaltable = pd.read_csv("../input/Surgical-deepnet.csv", sep=',')
#shuffle the data
surgicaltable = surgicaltable.sample(frac=1).reset_index(drop=True)

#write shuffled data onto csv
#surgicaltable_shuffled.to_csv("/home/ducen/Desktop/datasets/Surgical-deepnet-shuffled.csv", sep=',', encoding='utf-8', index=False)

#read in shuffled surgical dataset csv
#surgicaltable = pd.read_csv("/home/ducen/Desktop/datasets/Surgical-deepnet-shuffled.csv", sep=',')

#Preprocess Data

#print (surgicaltable)

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
X=surgicaltable.iloc[:,0:24]       #all rows all cols except the (last col) goes into training set

print(X.info())
print(X.describe())
# Specify the target labels and flatten the array
y= np.ravel(surgicaltable.complication)

print("type(X)",type(X))
print("type(y)",type(y))

# Split the data up in train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#let's unset random state...gives diff samples for every run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

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

############################### Functional API Smodel ########################################
#check input data shape
print("X_train[21].shape",X_train[21].shape)

import tensorflow as tf
import keras
from keras import regularizers, Input, layers
from keras.models import Sequential, Model

input_tensor = Input(shape=X_train[21].shape)		#input layer with input shape

x = layers.Dense(256, activation='relu')(input_tensor)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu')(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu')(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu')(x)
#x = layers.Dropout(0.1)(x)

output_tensor = layers.Dense(1, activation='sigmoid')(x)	#output layer with sigmoid activation

model = Model(input_tensor, output_tensor)		#build the model by specifying the input and output tensors

'''
Alternative Sequential model
# Initialize the constructor
model = tf.keras.models.Sequential()

#input layer
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape=X_train[21].shape))

#Add dropout
model.add(tf.keras.layers.Dropout(0.2))

# Add one hidden layer 
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))

#Add dropout
model.add(tf.keras.layers.Dropout(0.2))

# Add one hidden layer 
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))

#Add dropout
model.add(tf.keras.layers.Dropout(0.2))

# Add an output layer 
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
'''
############################### end of model 1 ########################################




from keras import optimizers
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#compile the model
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
                   
#model.fit(X_train, y_train,epochs=10, batch_size=1, verbose=1)
##############################################
#fit and plot
x_val = X_train[:8000]
partial_x_train = X_train[8000:]
y_val = y_train[:8000]
partial_y_train = y_train[8000:]


#earlystopping callback
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=30,),
keras.callbacks.ModelCheckpoint(filepath='tf-sept07-Smodel-functional-api-surgical-sgd.h5',monitor='val_loss',save_best_only=True,)
]

#class weighting
from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced', np.unique(partial_y_train), partial_y_train)

print('class_weight: ', class_weight)

count0=0
count1=0
for val in partial_y_train:
	if val==0:
		count0=count0+1
	elif val==1:
		count1=count1+1
print('count 0:',count0,' count 1:',count1)
print('partial_y_train[15:45]:',partial_y_train[15:45])


#it is a good idea to set batchsize*epochs = num of training examples
history = model.fit(partial_x_train,
partial_y_train,
epochs=500,
batch_size=256,
callbacks=callbacks_list,
class_weight = class_weight, 
validation_data=(x_val, y_val))

'''
################################## without callbacks and class_weight
history = model.fit(partial_x_train,
partial_y_train,
epochs=200,
batch_size=16, 
validation_data=(x_val, y_val))

##################################
'''

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
plt.ylabel('Accuracy')
plt.legend()
plt.show()

##########################################


'''
#Directly load weights from a saved model
import keras
from keras.models import load_model
model = load_model('tf-sept03-Smodel-v1.h5')
'''


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
#print(y_pred[35:75])
#print(y_test[35:75])



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
