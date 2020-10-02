import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import accuracy_score
# The competition datafiles are in the directory ../input

# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Preparing Training Data
train=np.asarray(train)
train_x=train[:,1:].astype('float32')
train_x=np.log(train_x+1)/10
y=train[:,0]
train_y=np.zeros((y.shape[0],10))
train_y[np.arange(y.shape[0]),y]=1

# Two Hidden Layer Model
model=Sequential()
model.add(Dense(500, input_dim=784, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer='Adagrad',loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
model.fit(train_x,train_y, shuffle=True, nb_epoch=100, batch_size=1000)

# Preparing Test Data
test=np.asarray(test)
test_x=test.astype('float32')
test_x=np.log(test_x+1)/10

# Making Predictions
test=model.predict(test_x,batch_size=10000)
test_pred=np.argmax(test,axis=1)

# Preparing CSV file
df=pd.DataFrame({'ImageId':(np.arange(test_pred.shape[0])+1).astype(int),'Label':test_pred.astype(int)})
df.to_csv("neural_net.csv",index=False)