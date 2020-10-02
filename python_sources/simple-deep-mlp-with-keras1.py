#accuracy 0.97443
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
    
seed = 7
numpy.random.seed(seed)

dataframe_train = pandas.read_csv("../input/train.csv")
dataframe_test = pandas.read_csv("../input/test.csv")
dataset_train = dataframe_train.values
dataset_test = dataframe_test.values
X_train = dataset_train[:,1:].astype('float32')
y_train = dataset_train[:,0]
y_train = to_categorical(y_train)
X_test = dataset_test[:,:].astype('float32')

X_train = X_train / 255
X_test = X_test / 255

num_pixel = X_test.shape[1]
num_classes = 10

def create_model():
    model = Sequential()
    model.add(Dense(num_pixel, input_dim=num_pixel, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
history = model.fit(X_train, y_train, batch_size=20, nb_epoch=10,validation_split=0.3, 
                    shuffle=True, verbose=2)
preds = model.predict_classes(X_test, verbose=0)

def write_preds(preds, fname):
    pandas.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-mlp-baseline.csv")
