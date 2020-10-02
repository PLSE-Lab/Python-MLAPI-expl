import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
seed = 7
np.random.seed(seed)
target = train.label
train.drop(['label'],axis=1,inplace=True)
train = train/255
test = test/255
train = train.values
test = test.values
target = np_utils.to_categorical(target)
train = train.reshape(train.shape[0], 1, 28, 28).astype('float32')
test = test.reshape(test.shape[0], 1, 28, 28).astype('float32')
num_classes = target.shape[1]
num_pixels = 784
def baseline_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(60, 3, 3, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(50, 5, 5, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
model = baseline_model()
# Fit the model
model.fit(train, target,  nb_epoch=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(train, target, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
yPred = model.predict(test, batch_size=200, verbose=0)
y_index = np.argmax(yPred,axis=1)
with open('conv2l2m_out.csv', 'w') as f :
    f.write('ImageId,Label\n')
    for i in range(0,len(test)) :
        f.write("".join([str(i+1),',',str(y_index[i]),'\n']))
