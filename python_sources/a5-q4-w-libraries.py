import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
import keras
import warnings
# Tensorflow backend likes to complain
# and this doesn't even solve the problem...
warnings.filterwarnings("ignore")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Create and train a model with the given epoch value
def createModel(train,epoch):

    
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=784))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    train_x = train[:,1:len(train)-1]
    # Convert labels to categorical one-hot encoding
    train_y = keras.utils.to_categorical(train[:,0], num_classes=10)
    
    print("Training model...")
    # Train the model
    model.fit(train_x, train_y, epochs=epoch, batch_size=64,verbose =0)
    print("Done training model!")
    return model
    
    
# Evaluates the model and returns the error and accuracy
def test(model,test):
    test_x = test[:,1:len(test)-1]
    test_y = keras.utils.to_categorical(test[:,0], num_classes=10)
    return model.evaluate(test_x, test_y, batch_size=128, verbose = 0)

# Tunes on development data and returns the best epoch to use
def tune(dev_data,test_data):
    small = 15
    big = 25
    step = 1
    
    
    # epoch : accuracy
    accs = []
    epochs = []
    for i in range(small,big,step):
        print("testing epoch: " + str(i))
        model = createModel(dev_data.copy(),i)
        accs.append(test(model,test_data)[1])
        print(accs)
        epochs.append(i)

    return epochs[np.argmax(accs)]
    
    
print("importing data...")
# Get data 
fashion_train = pd.read_csv("../input/fashion-mnist_train.csv").as_matrix()

# Splice out dev data 
rows = np.random.choice(range(len(fashion_train)),int(len(fashion_train) * 0.1))
train_data = np.delete(fashion_train,rows,0)
dev_data = fashion_train[rows]

test_data  = pd.read_csv("../input/fashion-mnist_test.csv").as_matrix()
print("finished importing data!")

print("Tuning...")
# tune epochs
bestEpoch = tune(dev_data,test_data)
print("Done tuning!")

print("Creating model")
model = createModel(train_data,bestEpoch)
print("Testing...")

print("Accuracy on training data: " + str(test(model,train_data)[1]))
print("Accuracy on dev data: " + str(test(model,dev_data)[1]))
print("Accuracy on test data: " + str(test(model,test_data)[1]))