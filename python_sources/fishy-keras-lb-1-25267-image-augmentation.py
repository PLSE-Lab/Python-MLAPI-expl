__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
np.random.seed(2017)
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import os
import glob
import cv2
import datetime
import time
import gc
import warnings
warnings.filterwarnings("ignore")

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import __version__ as keras_version

def get_im_cv2(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 64), cv2.INTER_LINEAR)
    
    return img
    
def load_train():
    
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join("..", 'input', "train", fld, "*.jpg")
        print(path)
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2(fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)
           
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    gc.collect()
    return X_train, y_train, X_train_id
    
def load_test():
    path = os.path.join("..", 'input', 'test_stg1', '*.jpg')
    start_time = time.time()    
    
    files = sorted(glob.glob(path))

    X_test = []
    X_test_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im_cv2(fl)
        X_test.append(img)
        X_test_id.append(flbase)
    
    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    gc.collect()
    return X_test, X_test_id
    

def read_and_normalize_train_data():
    start_time = time.time()
    train_data, train_target, train_id = load_train()
    

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    print('Train shape:', train_data.shape)    
    #print('Reshape...')
    #train_data = train_data.transpose((0, 2, 1, 3))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)

    #print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    
    print('Read and process train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return train_data, train_target, train_id


def read_and_normalize_test_data():
    start_time = time.time()
    test_data, test_id = load_test()

    print('Convert to numpy...')
    test_data = np.array(test_data, dtype=np.uint8)
    #test_data = test_data.transpose((0, 2, 1, 3))

    #test_data = test_data.astype('float32')
    test_data = test_data / 255

    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print("Saved as: ", sub_file)
    result1.to_csv(sub_file, index=False)
    

def create_classifier():
    classifier = Sequential()
    
    classifier.add(ZeroPadding2D((1, 1), input_shape=(64, 128, 3)))
    classifier.add(Convolution2D(16, 5, 5, activation='relu'))
    classifier.add(ZeroPadding2D((1, 1)))
    classifier.add(Convolution2D(8, 5, 5, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(ZeroPadding2D((1, 1)))
    classifier.add(Convolution2D(8, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(ZeroPadding2D((1, 1)))
    classifier.add(Convolution2D(4, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 64, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(output_dim = 32, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(output_dim = 8, activation = 'softmax')) 
    
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return classifier

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()
    
def run_cross_validation_create_models(nfolds=10, augmentation=False):
    # input image dimensions
    batch_size = 16
    nb_epoch = 30
    random_state = 51

    train_data, train_target, train_id = read_and_normalize_train_data()

    yfull_train = dict()
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf.split(train_data, train_target):
        model = create_classifier()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start Train KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))
        
        callbacks = [
                EarlyStopping(monitor='val_loss', patience=3, verbose=0),
            ]
        
        if augmentation:
            
            print("Using image augmentation to fit model...")
            
            train_datagen = ImageDataGenerator(rotation_range=20,
                                       channel_shift_range=0.15,
                                       vertical_flip=True,
                                       horizontal_flip = True)
    
            training_set = train_datagen.flow(X_train, Y_train,
                                              batch_size = batch_size,
                                              seed=random_state)
    
    
            test_datagen = ImageDataGenerator()
       
            test_set = test_datagen.flow(X_valid, Y_valid,
                                         batch_size = batch_size,
                                         seed=random_state)
            
            
            model.fit_generator(training_set,
                                samples_per_epoch=6000,
                                nb_val_samples = 1800,
                                nb_epoch=nb_epoch,
                                validation_data=test_set, 
                                verbose=1, 
                                callbacks=callbacks)
                                
        else:      
            
            print("Fitting model...")
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),
                  callbacks=callbacks)
        
          
        print('Making predictions for model number {} from {}'.format(num_fold, nfolds))          
        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return info_string, models
    

def run_cross_validation_process_test(info_string, models):
    batch_size = 16
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start Test KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    print(info_string)
    print("Creating submission's file...")
    create_submission(test_res, test_id, info_string)
    
if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 3
    augmentation=True
    
    info_string, models = run_cross_validation_create_models(nfolds=num_folds, augmentation=augmentation)
    run_cross_validation_process_test(info_string, models)