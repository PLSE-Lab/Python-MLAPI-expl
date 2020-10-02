import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dropout, Dense, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

K.set_image_dim_ordering('th') #input shape: (channels, height, width)

train_df = pd.read_csv("../input/train.csv")

# For checking the model params s subset of data is much faster
train_df = train_df.ix[0:2500]

x_train = train_df.drop(['label'], axis=1).values.astype('float32')

img_width, img_height = 28, 28

n_train = x_train.shape[0]
x_train = x_train.reshape(n_train,1,img_width,img_height)
x_train = x_train/255 #normalize from [0,255] to [0,1]

y_train = to_categorical(train_df['label'].values)

def build_model(dense_size=1000, dense_layers=1, dropout=0.35, conv_layers=2, conv_filters=60, conv_sz=5):
    model = Sequential()
    model.add(Conv2D(filters=conv_filters, kernel_size=(conv_sz,conv_sz), activation='relu', batch_input_shape=(None, 1, img_width, img_height)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    if conv_layers == 2:
        model.add(Conv2D(filters=conv_filters, kernel_size=(conv_sz,conv_sz), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    for i in range(0,dense_layers):
        model.add(Dense(dense_size, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax', activity_regularizer='l1_l2'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    
# Uncomment one or more of the below to find the best combinations!
param_grid = dict(
    dense_size=(600,1000),
    #dense_layers=(1,2),
    #dropout=(0.35,0.50),
    #conv_layers=(1,2),
    #conv_filters=(60),
    #conv_sz=(3,5),
)

gsc = GridSearchCV(estimator=KerasClassifier(build_fn=build_model,
                                             batch_size=128,
                                             epochs=1,
                                             verbose=2),
                   param_grid=param_grid)

grid_result = gsc.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))