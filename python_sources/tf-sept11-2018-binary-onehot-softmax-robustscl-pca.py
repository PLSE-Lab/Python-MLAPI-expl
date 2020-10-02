'''
Dated: Sept11-2018
Author: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for structured tabular data surgical dataset csv file binary classification (onehot encoded - using to_categorical() - target labels with softmax output layer with pca (to handle outliers) deepneural nets tensorflow, pandas and keras

Results:
train loss: 0.4309 - acc: 0.8214 - val_loss: 0.5471 - val_acc: 0.7330
type(history)
<class 'keras.callbacks.History'>
4391/4391 [==============================] - 0s 76us/step
testloss,testaccuracy 0.5357981066623979 0.7435663858026884
model.output_shape (None, 2)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 17)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               4608      
_________________________________________________________________
dense_2 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_3 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_5 (Dense)              (None, 256)               65792     
_________________________________________________________________
dropout_2 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_7 (Dense)              (None, 256)               65792     
_________________________________________________________________
dense_8 (Dense)              (None, 2)                 514       
=================================================================
Total params: 399,874
Trainable params: 399,874
Non-trainable params: 0
_________________________________________________________________
model.summary() None
confusion matrix: [[2461  793]
 [ 333  804]]
accuracy: 0.743566385789114
precision: 0.5034439574201628
recall: 0.7071240105540897
f1score: 0.5881492318946598
cohen_kappa_score: 0.40952820211010654




################################# Model M #####################################################

################################# End #####################################################

'''
# Import pandas 
import pandas as pd
from keras.utils import to_categorical

#read in and shuffle surgicaltable's dataset csv
surgicaltable = pd.read_csv("../input/Surgical-deepnet.csv", sep=',')
surgicaltable = surgicaltable.sample(frac=1).reset_index(drop=True)
#surgicaltable.to_csv("/home/ducen/Desktop/datasets/surgicaltable-shuffled.csv", sep=',', encoding='utf-8', index=False)

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
X=surgicaltable.iloc[:,0:24]       #all rows all cols except the 22nd(last col) goes into training set

print(X.info())
print(X.describe())
# Specify the target labels and flatten the array
y= np.ravel(surgicaltable.complication)

np.savetxt('y.csv', y, delimiter=",")

print("type(X)",type(X))
print("type(y)",type(y))

# Split the data up in train and test sets...Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#class weighting
from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

print('class_weight: ', class_weight)

count0=0
count1=0
for val in y_train:
	if val==0:
		count0=count0+1
	elif val==1:
		count1=count1+1
print('count 0:',count0,' count 1:',count1)
print('y_train[15:45]:',y_train[15:45])

#convert the labels to categorical - onehot encoding (otherwise the model will see 0, 1, 2 integers)
print("before to_categorical(y_train):",y_train[5:15])
y_train = to_categorical(y_train).astype(int)
y_test = to_categorical(y_test).astype(int)
print("after to_categorical(y_train):\n",y_train[5:15])


#y_train = y_train.astype(int)
#y_test = y_test.astype(int)
#print("after astype(int)\n",y_train[5:15])





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

'''
print("X_train[5:10] before StandardScaler():\n", X_train[5:10])
##############################################################
#Let's normalize this data to 0 mean and unit std
# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)
#############################################################
print("X_train[5:10] after StandardScaler():\n", X_train[5:10])
'''

print("X_train[5:10] before RobustScaler():\n", X_train[5:10])
##############################################################
# Import `RobustScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import RobustScaler

# Define the scaler 
robustscaler = RobustScaler().fit(X_train)

# Scale the train set
X_train = robustscaler.transform(X_train)

# Scale the test set
X_test = robustscaler.transform(X_test)
#############################################################
print("X_train[5:10] after RobustScaler():\n", X_train[5:10])


#############################################################

#PCA
from sklearn.decomposition import PCA
pca = PCA(.95, whiten=True)			#select components/columns which explain 95% of the variance in the data
#pca = PCA(n_components=13)	#select k components/columns  which contribute the most to variance
pca.fit(X_train)
print("pca.n_components_:", pca.n_components_)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
'''
#PCA with svd solver as randomized
from sklearn.decomposition import PCA
pca = PCA(n_components=18, svd_solver='randomized', whiten=True)	#select k components/columns
pca.fit(X_train)
print("pca.n_components_:", pca.n_components_)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
'''
#############################################################



#Let's create a tensorflow deepneuralnet model

# Add an input layer 
#check input data shape
print("X_train[21].shape",X_train[21].shape)

#functionalAPI
import tensorflow as tf
import keras
from keras import regularizers, Input, layers
from keras.models import Sequential, Model

input_tensor = Input(shape=X_train[21].shape)		#input layer with input shape
'''
x = layers.Dense(256, activation='relu')(input_tensor)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu')(x)
#x = layers.Dropout(0.1)(x)
'''
x = layers.Dense(256, activation='relu')(input_tensor)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu')(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu')(x)
#x = layers.Dropout(0.1)(x)
x = layers.Dense(256, activation='relu')(x)
#x = layers.Dropout(0.1)(x)

output_tensor = layers.Dense(2, activation='softmax')(x)	#output layer with softmax activation

model = Model(input_tensor, output_tensor)		#build the model by specifying the input and output tensors


#compile the model
from keras import optimizers
#sgd = optimizers.SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

                  

# Model config
#print("model.get_config()",model.get_config())

# List all weight tensors 
#print("model.get_weights()",model.get_weights())

#model.fit(X_train, y_train,epochs=30, batch_size=32, verbose=1)

##############################################
#fit and plot
x_val = X_train[:2000]
partial_x_train = X_train[2000:]
y_val = y_train[:2000]
partial_y_train = y_train[2000:]

print("partial_x_train.shape:",partial_x_train.shape)
print("partial_y_train.shape:",partial_y_train.shape)
print("x_val.shape:",x_val.shape)
print("y_val.shape:",y_val.shape)

print("partial_x_train[5:10]-",partial_x_train[5:10])
print("partial_y_train[5:10]-",partial_y_train[5:10])
print("x_val[5:10]",x_val[5:10])
print("y_val[5:10]",y_val[5:10])



#earlystopping callback
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=50),
keras.callbacks.ModelCheckpoint(filepath='tf-sept11-Mmodel-functional-api-binarySurgical-softmax-onehot-adam-robustscaler-pca.h5',monitor='val_loss',save_best_only=True)
]


#it is a good idea to set batchsize*epochs = num of training examples

history = model.fit(partial_x_train,
partial_y_train,
epochs=500,
batch_size=256,
callbacks=callbacks_list,
class_weight = class_weight, 
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
(testloss,testaccuracy) = model.evaluate(X_test,y_test)
print("testloss,testaccuracy",testloss,testaccuracy)

# Model output shape
print("model.output_shape",model.output_shape)

# Model summary
print("model.summary()",model.summary())


#let's make predictions
y_pred = model.predict(X_test)
y_pred=y_pred.round().astype(int)
#print("type(y_pred)",type(y_pred))
#print("y_pred.shape",y_pred.shape)

#testset metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
print("confusion matrix:", confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Accuracy 
print("accuracy:", accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Precision 
print("precision:", precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# Recall
print("recall:",recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

# F1 score
print("f1score:", f1_score(y_test.argmax(axis=1),y_pred.argmax(axis=1)))

# Cohen's kappa
print("cohen_kappa_score:", cohen_kappa_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

#let's print out some predictions
print('y_test[10:50]: ',y_test[10:50])
print('y_pred[10:50]: ',y_pred[10:50])