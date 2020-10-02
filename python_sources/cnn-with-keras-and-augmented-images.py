import pandas
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import adam

# load training data, shuffle it and take 50% - the latter has no impact on the accuracy
train = pandas.read_csv('../input/train.csv')
test=pandas.read_csv('../input/test.csv')
train=shuffle(train)
train =train[:int(train.shape[0]*0.5)]

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs


#################################################################################
# do some basic preprocessing
#################################################################################    
labels= train.label                     # save the target column for later use
train = train.drop(['label'], axis=1)   # drop label column from data set
colnames=list(train)                    # save the columnnames

# standardise the data and change type to a smaller float to minimise mem usage
train=(train.astype('float16')/127.5-1)
test=(test.astype('float16')/127.5-1)

# split train in train and validation
train, validation, labels, labelsvalidation = train_test_split(train, labels, test_size=0.25, random_state=42)

# one hot encoding on labels
labels=to_categorical(labels)
labelsvalidation=to_categorical(labelsvalidation)

# create the CNN
model = Sequential()
model.add( Convolution2D(16, 5, 5, border_mode='valid', input_shape=(28, 28, 1)) )
model.add( BatchNormalization() )
model.add( Activation('relu') )
model.add( MaxPooling2D(2,2) )
model.add( Convolution2D(16, 5, 5) )
model.add( BatchNormalization() )
model.add( Activation('relu') )
model.add( MaxPooling2D(2,2) )
model.add( Flatten() ) 
model.add( Dense(512, activation='relu'))
model.add( Dense(output_dim=10, activation="softmax") )
    
# reshape the training and validation data
train=train.values.reshape(train.values.shape[0],28,28,1)
validation=validation.values.reshape(validation.values.shape[0],28,28,1)

# configure the compiler
model.compile(loss='categorical_crossentropy',optimizer=adam(0.001),metrics=['accuracy'])

# create generators  - training data will be augmented images
validationdatagenerator = ImageDataGenerator()
traindatagenerator = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,rotation_range=15,zoom_range=0.1 )

batchsize=8
train_generator=traindatagenerator.flow(train, labels, batch_size=batchsize) 
validation_generator=validationdatagenerator.flow(validation, labelsvalidation,batch_size=batchsize)

model.fit_generator(train_generator, steps_per_epoch=int(len(train)/batchsize), epochs=3, validation_data=validation_generator, validation_steps=int(len(validation)/batchsize))

#reshape the test data
testreshaped=test.values.reshape(test.values.shape[0],28,28,1)

# and predict on the test set
predictions = model.predict_classes(testreshaped)
    
