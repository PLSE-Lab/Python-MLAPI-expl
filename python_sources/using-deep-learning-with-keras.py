#For sake of experience, i did a experience with deep learning using convolutional neural networks for 
#this dataset. Make a matrix  NxN, where N is a number of features.  This matrix form is an input 
#image of convolutional neural network:
#      row = ptrain.iloc[i].values #it get  line i of dataset
#      matrix = np.outer(row, row) #it transforms line in matrix
#I using keras...
#Since Deep Learning (DL) is good for find relationship betweenn features, i think that make a image
#represenntation of data is a good step to automatic feature extractors. But I'm realizing that the 
#training error is greater than the test error! How is this possible?
# I'm sorry, my English sucks.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import backend as K
from keras.utils import np_utils
from keras.layers import Dense, Dropout,Flatten, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras import layers, Model
from keras.optimizers import RMSprop
import os
print(os.listdir("../input"))

def make_classifier():
    # instantiate model
	SHAPE = (200, 200, 1)  # input image size to model
	ACTION_SIZE = 2
	frames_input = layers.Input(SHAPE, name='frames')
	normalized = BatchNormalization()(frames_input)
	# "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
	conv_1 = layers.convolutional.Conv2D(
		16, (8, 8), strides=(4, 4), activation='relu'
	)(normalized)
	# "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
	conv_2 = layers.convolutional.Conv2D(
		32, (4, 4), strides=(2, 2), activation='relu'
	)(conv_1)
	# Flattening the second convolutional layer.
	conv_flattened = layers.core.Flatten()(conv_2)
	# "The final hidden layer is fully-connected and consists of 256 rectifier units."
	hidden = layers.Dense(256, activation='relu')(conv_flattened)
	# "The output layer is a fully-connected linear layer with a single output for each valid action."
	output = layers.Dense(ACTION_SIZE, activation='softmax')(hidden)
	# Finally, we multiply the output by the mask!
	model = Model(inputs=[frames_input], outputs=output)
	model.summary()
	optimizer = RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
	# model.compile(optimizer, loss='mse')
	# to changed model weights more slowly, uses MSE for low values and MAE(Mean Absolute Error) for large values
	model.compile(optimizer, loss='mse', metrics=['accuracy'])
	return model
# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

print(len(train_df))

target = train_df['target']

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]  #basic features

ptrain_df = train_df[train_df['target']==1]
ntrain_df = train_df[train_df['target']==0]

ptrain = ptrain_df[features]
ntrain = ntrain_df[features]

MAX_IT = 200

model = make_classifier()
#model.load_weights('model.h5',  by_name=True)

for j in range(MAX_IT):
    examples = []
    targets = []
    positives = np.random.choice(len(ptrain), 500, False)
    negatives = np.random.choice(len(ntrain), 500, False)
    for i in positives:
        row = ptrain.iloc[i].values
        matrix = np.outer(row, row)
        examples.append(matrix.reshape(200, 200, 1))
        targets.append(np.array([0, 1]))
    for i in negatives:
        row = ntrain.iloc[i].values
        matrix = np.outer(row, row)
        examples.append(matrix.reshape(200, 200, 1))
        targets.append(np.array([1, 0]))

    h = model.fit(np.array(examples), np.array(targets), validation_split=0.1, epochs=1, batch_size=16, verbose=0, shuffle=True)
    acc = h.history['acc'][0]
    loss = h.history['loss'][0]
    val_acc = h.history['val_acc'][0]
    val_loss = h.history['val_loss'][0]
    print("%d LOSS %f, ACC %f, VAL LOSS %f, VAL ACC %f"%(j, loss, acc, val_loss, val_acc))

model.save_weights("model.h5")
#OUTPUT:
#1959 LOSS 0.024804, ACC 0.972222, VAL LOSS 0.034742, VAL ACC 0.960000