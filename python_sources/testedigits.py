import numpy as np 
import pandas as pd 

from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split

from subprocess import check_output

train = pd.read_csv("../input/train.csv")

test_images = (pd.read_csv("../input/test.csv").values).astype('float32')

train_images = (train.ix[:,1:].values).astype('float32')
train_labels = train.ix[:,0].values.astype('int32')

train_images = train_images / 255
test_images = test_images / 255

from keras.utils.np_utils import to_categorical
train_labels = to_categorical(train_labels)
classes = train_labels.shape[1]

seed = 43
np.random.seed(seed)

model=Sequential()
model.add(Dense(32,activation='sigmoid',input_dim=(28 * 28)))
model.add(Dense(16,activation='sigmoid'))
model.add(Dense(10,activation='sigmoid'))

model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images, train_labels, nb_epoch=3)

prev = model.predict_classes(test_images, verbose=0)
resposta = pd.DataFrame({'ImagemId': list(range(1, len(prev)+1)), 'Resposta':prev})
print(resposta.Resposta)
print(train_labels)
