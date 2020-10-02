import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

x_variables = test.columns
y_variable = 'label'


train[x_variables]  = train[x_variables]*1.0/255
test[x_variables]  = test[x_variables]*1.0/255

df_cv = train.sample(frac = 0.10,replace = False)
df_train = train.drop(df_cv.index)


X_train = df_train[x_variables].as_matrix()
Y_train = df_train[y_variable].as_matrix()
Y_train = np_utils.to_categorical(Y_train,10)


X_cv = df_cv[x_variables].as_matrix()
Y_cv = df_cv[y_variable].as_matrix()
Y_cv = np_utils.to_categorical(Y_cv,10)

batch_size = 200
epochs = 25

model = Sequential()
model.add(Dense(200,input_dim = 784,init = 'uniform',activation = 'relu'))
model.add(Dense(100,init = 'uniform',activation = 'relu'))
model.add(Dense(10,activation = 'softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
model.fit(X_train,Y_train,batch_size = batch_size,nb_epoch = epochs,verbose = 0)
score = model.evaluate(X_cv,Y_cv,verbose = 0)
print(score)

X_test = test[x_variables].as_matrix()
Y_predicted = model.predict(X_test)
pred_label = Y_predicted.argmax(axis = 1)
image_id = range(1,len(Y_predicted)+1)
df = {'ImageId':image_id,'Label':pred_label}
df = pd.DataFrame(df)
df.to_csv('submission.csv',index = False)

