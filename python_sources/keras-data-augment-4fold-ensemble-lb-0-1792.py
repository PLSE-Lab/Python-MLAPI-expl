# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.layers import GlobalMaxPooling2D, Activation, Add
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

def load_train():
    ''' channel3 = (band1+band2)/2 '''
    #Load data
    train = pd.read_json("../input/train.json")
    train.inc_angle = train.inc_angle.replace('na', 0)
    train.inc_angle = train.inc_angle.astype(float).fillna(0.0)
    
    # Train data
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
    X_train = np.concatenate([x_band1[:, :, :, np.newaxis]
                              , x_band2[:, :, :, np.newaxis]
                             , ((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)
    X_angle_train = np.array(train.inc_angle)
    y_train = np.array(train["is_iceberg"])

    return X_train, X_angle_train, y_train
#    X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(X_train
#                        , X_angle_train, y_train, random_state=123, train_size=0.8)

def load_test():  
    test = pd.read_json("../input/test.json")  
    # Test data
    x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
    x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
    X_test = np.concatenate([x_band1[:, :, :, np.newaxis]
                              , x_band2[:, :, :, np.newaxis]
                             , ((x_band1+x_band2)/2)[:, :, :, np.newaxis]], axis=-1)
    X_angle_test = np.array(test.inc_angle)
    
    return X_test, X_angle_test, test.id

X_train0, X_angle_train0, y_train0 = load_train()

'''
model fork https://www.kaggle.com/dimitrif/keras-with-data-augmentation-lb-0-1826
'''
def get_callbacks(filepath, patience=10):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]
    
def get_model():
    bn_model = 0.99
    kernel_size = (5,5)
    input_1 = Input(shape=(75, 75, 3), name="X_1")
   
    img_1 = Conv2D(32, kernel_size = kernel_size, padding = "same") (BatchNormalization(momentum=bn_model)(input_1))
    img_1 = MaxPooling2D((2,2)) (Activation('elu')(BatchNormalization(momentum=bn_model)(img_1)))
    img_1 = Dropout(0.25)(img_1)
    img_1 = Conv2D(64, kernel_size = kernel_size, padding = "same") (img_1)
    img_1 = MaxPooling2D((2,2)) (Activation('elu')(BatchNormalization(momentum=bn_model)(img_1)))
    input_cnn = Dropout(0.25)(img_1)
    
    img_2 = Conv2D(128, kernel_size = kernel_size, padding = "same") (BatchNormalization(momentum=bn_model)(input_cnn))
    img_2 = Activation('elu')(BatchNormalization(momentum=bn_model)(img_2))
    img_2 = Dropout(0.25)(img_2)
    img_2 = Conv2D(64, kernel_size = kernel_size, padding = "same") (img_2)
    input_cnn_res = Activation('elu')(BatchNormalization(momentum=bn_model)(img_2))
    
    added = Add()([input_cnn, input_cnn_res])
    img_3 = Conv2D(128, kernel_size = kernel_size, padding = "same") (added)
    img_3 = MaxPooling2D((2,2)) (Activation('elu')(BatchNormalization(momentum=bn_model)(img_3)))
    img_3 = Conv2D(256, kernel_size = kernel_size, padding = "same") (img_3)
    img_3 = Activation('elu')(BatchNormalization(momentum=bn_model)(img_3))
    img_3 = MaxPooling2D((2,2))(Dropout(0.25)(img_3))
    img_3 = Conv2D(512, kernel_size = kernel_size, padding = "same") (img_3)
    img_3 = Activation('elu')(BatchNormalization(momentum=bn_model)(img_3))
    img_3 = MaxPooling2D((2,2))(Dropout(0.25)(img_3))
    img_3 = GlobalMaxPooling2D() (img_3)
    img_3 = Dense(512, activation=None)(img_3)
    img_3 = Activation('elu')(BatchNormalization(momentum=bn_model)(img_3))
    img_3 = Dropout(0.5)(img_3)
    img_3 = Dense(256, activation=None)(img_3)
    img_3 = Activation('elu')(BatchNormalization(momentum=bn_model)(img_3))
    img_3 = Dropout(0.5)(img_3)
    output = Dense(1, activation="sigmoid")(img_3)
    
    model = Model(input_1,  output)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model
model = get_model()
model.summary()		

'''
model4+ cv
fork https://www.kaggle.com/jamesrequa/keras-k-fold-inception-v3-1st-place-lb-0-99770
'''
print("loading test data")
X_test, X_angle_test, test_id = load_test()

batch_size= 64
epochs = 100
n_fold = 4
kf = KFold(n_splits=n_fold, shuffle=True)

roc_auc = metrics.roc_auc_score
train_losses = []; valid_losses = []

i = 1

for train_index, test_index in kf.split(X_train0):
    X_train = X_train0[train_index]; X_valid = X_train0[test_index]
    y_train = y_train0[train_index]; y_valid = y_train0[test_index]

    file_path = "fold"+str(i)+ ".hdf5"
    callbacks = get_callbacks(filepath=file_path, patience=11)

    train_steps = len(X_train)*32 / batch_size
    valid_steps = len(X_valid)*32 / batch_size
    
    model = get_model()

    train_generator = ImageDataGenerator(
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                         #       shear_range=0.2,
                                zoom_range=0.1,
                                horizontal_flip=True,
                                vertical_flip = True,
                                fill_mode='nearest')
    train_generator.fit(X_train)
    valid_generator = ImageDataGenerator(
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                         #       shear_range=0.2,
#                                zoom_range=0.1,
                                horizontal_flip=True,
                                vertical_flip = True,
                                fill_mode='nearest')
    valid_generator.fit(X_valid)
    
    model.fit_generator(train_generator.flow(X_train, y_train, batch_size=64), 
                        steps_per_epoch= train_steps, epochs=epochs, verbose=1, 
                        callbacks=callbacks, validation_steps=valid_steps,
                        validation_data=valid_generator.flow(X_valid, y_valid, batch_size=64)
                        )

    model.load_weights(filepath=file_path)

    print("Train evaluate:")
    preds_train = model.evaluate([X_train0], y_train0, verbose=1, batch_size=200)
    print("valid evaluate:")
    preds_valid = model.evaluate([X_valid], y_valid, verbose=1, batch_size=200)

    valid_losses.append(preds_train[0])
    train_losses.append(preds_valid[0])
    print('Avg Train loss:{0:0.5f}, Val loss:{1:0.5f} after {2:0.5f} folds'.format
          (np.mean(train_losses), np.mean(valid_losses), i))
    
    print('Running test predictions with fold {}'.format(i))

    preds_test_fold = model.predict([X_test], verbose=1, batch_size=128)
    if i==1:
        preds_test = preds_test_fold
    else:
        preds_test += preds_test_fold

    print('\n\n')
    print("preds_test[10]", preds_test[:10])
    i += 1

    if i <= n_fold:
        print('Now beginning training for fold {}\n\n'.format(i))
    else:
        print('Finished training!')

preds_test /= n_fold

submission = pd.DataFrame({'id': test_id, 'is_iceberg': preds_test.reshape((preds_test.shape[0]))})
submission.head(10)
submission.to_csv(_"4fold.csv", index=False)