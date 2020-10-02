'''
August28-2018
Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for looped kfold cross validation for structured tabular data surgical csv file binary classification with deepneural nets tensorflow, pandas and keras
Results:
######################################### overfitted model #########################################
train loss: 0.0031 - acc: 1.0000
validation_scores: [[1.8303313283273481, 0.7414584396932272], [1.6367351285417977, 0.749107598285782], [1.9858588410056044, 0.7399286079747163], [1.6844671654008225, 0.7496175421311623], [1.9843456335422762, 0.7542070373474853]]
avg_val_loss:  1.8243476193635697
avg_val_acc:  0.7468638450864746
4830/4830 [==============================] - 0s 47us/step
testloss,testaccuracy 1.68905479700669 0.7596273290938226

confusion matrix: [[3072  546]
 [ 615  597]]
accuracy: 0.7596273291925466
precision: 0.5223097112860893
recall: 0.49257425742574257
f1score: 0.5070063694267516
cohen_kappa_score: 0.34825450155962023

######################################### ideal model #########################################

###########################################################################################################################


'''
# Import pandas 
import pandas as pd

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
X=surgicaltable.iloc[:,0:24]       #all rows all cols except the (last col) goes into training set

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
number of inputs num_ip = 24
number of outputs num_op = 2

set the number of units in hidden layer as 2(num_ip + num_op) = 2(24+2) = 52




############################### model 1: just right ########################################
# Add an input layer 
#check input data shape
print("X_train[21].shape",X_train[21].shape)

#model.add(tf.keras.layers.Dense(12, activation='relu', input_shape=(22,)))
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=X_train[21].shape))

# Add one hidden layer 
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Add one hidden layer 
model.add(tf.keras.layers.Dense(128, activation='relu'))

#Add dropout
#model.add(tf.keras.layers.Dropout(0.5))

# Add an output layer 
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
############################### end of model 1 ########################################


#compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                   
#model.fit(X_train, y_train,epochs=10, batch_size=1, verbose=1)
##############################################
'''
#k-Fold Cross Validation
k=5
num_validation_samples = len(X_train) // k
#print('len(X_train) // k', len(X_train) // k)
validation_scores = []

for fold in range(k):
	x_val = X_train[(num_validation_samples * fold) : (num_validation_samples * (fold + 1))]
	#validation_data = data[num_validation_samples * fold:num_validation_samples * (fold + 1)]
	#partial_x_train1 = X_train[ : (num_validation_samples * fold)] #+ X_train[(num_validation_samples * (fold + 1)) : ]
	#partial_x_train2 = X_train[(num_validation_samples * (fold + 1)) : ]

	#print('partial_x_train1.shape',partial_x_train1.shape)
	#print('partial_x_train2.shape',partial_x_train2.shape)
	partial_x_train = np.vstack((X_train[ : (num_validation_samples * fold)],X_train[(num_validation_samples * (fold + 1)) : ]))
	print('partial_x_train.shape',partial_x_train.shape)


	#training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]
	y_val = y_train[num_validation_samples * fold:num_validation_samples * (fold + 1)]
	#validation_data = data[num_validation_samples * fold:num_validation_samples * (fold + 1)]
	partial_y_train = np.hstack((y_train[ : (num_validation_samples * fold)],y_train[(num_validation_samples * (fold + 1)) : ]))
	#training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]

	#create a brand new model for each fold
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=X_train[21].shape))
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	
	#compile
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	#fit and plot
	history = model.fit(partial_x_train, partial_y_train, epochs=100, batch_size=128, validation_data=(x_val, y_val))


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

	validation_score = model.evaluate(x_val, y_val)
	print('fold validation score:',validation_score)
	validation_scores.append(validation_score)

#outside the for loop
'''
avg_validation_loss = np.average(validation_scores[0])	#take the avg of validation scores of each fold
print("avg_validation_loss: ", avg_validation_loss)

avg_validation_acc = np.average(validation_scores[1])	#take the avg of validation scores of each fold
print("avg_validation_acc: ", avg_validation_acc)
'''

#let's train a new model on the entire training set containing all folds
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=X_train[21].shape))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
	
#compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=128)	#train the model on all data before testing it one final time

##########################################
print('validation_scores:',validation_scores)
val_loss_sum = 0
val_acc_sum = 0
for foldresult in validation_scores:
	val_loss_sum += foldresult[0]
	val_acc_sum += foldresult[1]

avg_val_loss = val_loss_sum/len(validation_scores)
avg_val_acc = val_acc_sum/len(validation_scores)

print("avg_val_loss: ", avg_val_loss)
print("avg_val_acc: ", avg_val_acc)


#evaluate the model on test data	
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
