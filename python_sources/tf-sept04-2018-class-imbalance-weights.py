'''
Dated: Sept04-2018
Author: Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for handling class imabalance using class_weight with heldout validation for structured tabular data surgical csv file binary classification with deepneural nets tensorflow, pandas and keras
Results:
##################################### S Model with adam 50 epochs, 16 batchsize #############################################
loss: 0.4825 - acc: 0.7829244/9244 [==============================] - 1s 93us/step - loss: 0.4821 - acc: 0.7835 - val_loss: 0.4980 - val_acc: 0.7640
type(history)
<class 'tensorflow.python.keras.callbacks.History'>
4391/4391 [==============================] - 0s 21us/step
testloss,testaccuracy 0.44835733395452465 0.8100660441812799
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 52)                1300      
_________________________________________________________________
dropout (Dropout)            (None, 52)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 52)                2756      
_________________________________________________________________
dropout_1 (Dropout)          (None, 52)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 52)                2756      
_________________________________________________________________
dropout_2 (Dropout)          (None, 52)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 53        
=================================================================
Total params: 6,865
Trainable params: 6,865
Non-trainable params: 0
_________________________________________________________________
confusion matrix: [[2993  290]
 [ 544  564]]
accuracy: 0.8100660441812799
precision: 0.6604215456674473
recall: 0.5090252707581228
f1score: 0.5749235474006117
cohen_kappa_score: 0.4552626200451665

################################# Model M without dropout layers #####################################################
I then made the number of units as 1.5 * 2(num_ip + num_op) = 1.5 * 2(24+2) = 1.5*52 = 78
train loss: 0.0382 - acc: 0.9870 - val_loss: 1.9021 - val_acc: 0.7470
type(history)
<class 'tensorflow.python.keras.callbacks.History'>
4391/4391 [==============================] - 0s 17us/step
testloss,testaccuracy 2.0573618973449865 0.7440218629149635
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 78)                1950      
_________________________________________________________________
dense_1 (Dense)              (None, 78)                6162      
_________________________________________________________________
dense_2 (Dense)              (None, 78)                6162      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 79        
=================================================================
Total params: 14,353
Trainable params: 14,353
Non-trainable params: 0
_________________________________________________________________
confusion matrix: [[2701  589]
 [ 535  566]]
accuracy: 0.7440218629013892
precision: 0.49004329004329006
recall: 0.5140781108083561
f1score: 0.50177304964539
cohen_kappa_score: 0.32967167591180835

################################# Model M with dropout layers #####################################################
train loss: 0.3753 - acc: 0.8159 - val_loss: 0.3839 - val_acc: 0.8220
type(history)
<class 'tensorflow.python.keras.callbacks.History'>
4391/4391 [==============================] - 0s 19us/step
testloss,testaccuracy 0.4159559679313717 0.8073331815076292
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 78)                1950      
_________________________________________________________________
dropout (Dropout)            (None, 78)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 78)                6162      
_________________________________________________________________
dropout_1 (Dropout)          (None, 78)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 78)                6162      
_________________________________________________________________
dropout_2 (Dropout)          (None, 78)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 79        
=================================================================
Total params: 14,353
Trainable params: 14,353
Non-trainable params: 0
_________________________________________________________________
confusion matrix: [[3139  118]
 [ 728  406]]
accuracy: 0.8073331815076292
precision: 0.7748091603053435
recall: 0.35802469135802467
f1score: 0.4897466827503016
cohen_kappa_score: 0.39020356062842043

################################# Model M with L1 regularization and dropout layers #####################################################
train loss: 0.4826 - acc: 0.7842 - val_loss: 0.4564 - val_acc: 0.8030
type(history)
<class 'tensorflow.python.keras.callbacks.History'>
4391/4391 [==============================] - 0s 19us/step
testloss,testaccuracy 0.45642075871069265 0.8025506718423149
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 78)                1950      
_________________________________________________________________
dropout (Dropout)            (None, 78)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 78)                6162      
_________________________________________________________________
dropout_1 (Dropout)          (None, 78)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 78)                6162      
_________________________________________________________________
dropout_2 (Dropout)          (None, 78)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 79        
=================================================================
Total params: 14,353
Trainable params: 14,353
Non-trainable params: 0
_________________________________________________________________
confusion matrix: [[3021  299]
 [ 568  503]]
accuracy: 0.8025506718287406
precision: 0.6271820448877805
recall: 0.4696545284780579
f1score: 0.5371062466631072
cohen_kappa_score: 0.41488957357604195

################################# Model M with L2 regularization but no dropout layers #####################################################
train loss: 0.2918 - acc: 0.9009 - val_loss: 0.6053 - val_acc: 0.7940
type(history)
<class 'tensorflow.python.keras.callbacks.History'>
4391/4391 [==============================] - 0s 19us/step
testloss,testaccuracy 0.6741637869145652 0.7681621498655442
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 78)                1950      
_________________________________________________________________
dense_1 (Dense)              (None, 78)                6162      
_________________________________________________________________
dense_2 (Dense)              (None, 78)                6162      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 79        
=================================================================
Total params: 14,353
Trainable params: 14,353
Non-trainable params: 0
_________________________________________________________________
confusion matrix: [[2795  513]
 [ 505  578]]
accuracy: 0.76816214985197
precision: 0.5297891842346472
recall: 0.5337026777469991
f1score: 0.531738730450782
cohen_kappa_score: 0.3776858128050288


################################# Model M with L2 regularization and dropout layers #####################################################
loss: 0.4208 - acc: 0.8143 - val_loss: 0.4415 - val_acc: 0.8060
type(history)
<class 'tensorflow.python.keras.callbacks.History'>
4391/4391 [==============================] - 0s 28us/step
testloss,testaccuracy 0.4417561618566676 0.8032338875243018
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 78)                1950      
_________________________________________________________________
dropout (Dropout)            (None, 78)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 78)                6162      
_________________________________________________________________
dropout_1 (Dropout)          (None, 78)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 78)                6162      
_________________________________________________________________
dropout_2 (Dropout)          (None, 78)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 79        
=================================================================
Total params: 14,353
Trainable params: 14,353
Non-trainable params: 0
_________________________________________________________________
confusion matrix: [[3086  201]
 [ 663  441]]
accuracy: 0.8032338874971533
precision: 0.6869158878504673
recall: 0.39945652173913043
f1score: 0.5051546391752577
cohen_kappa_score: 0.3929055951609418

################################# Model M with both L1 and L2 regularization and dropout layers ###############################################
train loss: 0.4863 - acc: 0.7846 - val_loss: 0.4766 - val_acc: 0.7910
type(history)
<class 'tensorflow.python.keras.callbacks.History'>
4391/4391 [==============================] - 0s 20us/step
testloss,testaccuracy 0.4616787487853499 0.7993623320563891
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 78)                1950      
_________________________________________________________________
dropout (Dropout)            (None, 78)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 78)                6162      
_________________________________________________________________
dropout_1 (Dropout)          (None, 78)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 78)                6162      
_________________________________________________________________
dropout_2 (Dropout)          (None, 78)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 79        
=================================================================
Total params: 14,353
Trainable params: 14,353
Non-trainable params: 0
_________________________________________________________________
confusion matrix: [[2944  332]
 [ 549  566]]
accuracy: 0.7993623320428148
precision: 0.6302895322939867
recall: 0.5076233183856502
f1score: 0.5623447590660705
cohen_kappa_score: 0.43414807747131845


################################# Model M #####################################################








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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

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


############################### Medium Mmodel ########################################
# Add an input layer 
#check input data shape
print("X_train[21].shape",X_train[21].shape)

from keras import regularizers

#model.add(tf.keras.layers.Dense(78, activation='relu', kernel_regularizer=regularizers.l1(0.001), input_shape=X_train[21].shape))
#model.add(tf.keras.layers.Dense(52, kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), activation='relu', input_shape=X_train[21].shape))
#model.add(tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001), input_shape=X_train[21].shape))

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
############################### end of model 1 ########################################




from keras import optimizers
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
#model.fit(X_train, y_train,epochs=10, batch_size=1, verbose=1)
##############################################
#fit and plot
x_val = X_train[:1000]
partial_x_train = X_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]


#earlystopping callback
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=30,),
keras.callbacks.ModelCheckpoint(filepath='tf-sept04-Class-Imbalance-v2-sgd.h5',monitor='val_loss',save_best_only=True,)
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
print('count 0:',count0,' count1:',count1)
print('partial_y_train[15:45]:',partial_y_train[15:45])


#it is a good idea to set batchsize*epochs = num of training examples
history = model.fit(partial_x_train,
partial_y_train,
epochs=200,
batch_size=32,
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
