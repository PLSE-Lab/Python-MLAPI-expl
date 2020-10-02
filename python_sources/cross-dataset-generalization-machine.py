# -*- coding: utf-8 -*-
'''
Cross-Dataset Generalization Machine

Key idea is to split all drivers into two sets, then in even batches use only images of first driver's set, 
and in odd batches use only images of second driver's set. 
In such settings only well-generalized features, that invariant to concrete driver, will "survive". 
Features, which learned concrete driver's appearance and works correctly only for one conrete driver, will not "survive".

This improvement give:
    - Min Loss           2.16  ->  1.83
    - Last Epoch Loss    2.47  ->  1.92
    - Average Loss       2.30  ->  2.08

This experiment is intended to solve common "Cross-Dataset Generalization" problem.
'''


import numpy as np

np.random.seed(201)



import os

import glob

import cv2

import math

import pickle

import datetime

import pandas as pd

import statistics

import random

import time



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation, Flatten

from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.optimizers import SGD, Adam

from keras.utils import np_utils

from keras.models import model_from_json

from sklearn.metrics import log_loss

from scipy.misc import imread, imresize



use_cache = 1

# color type: 1 - grey, 3 - rgb

color_type_global = 1





# color_type = 1 - gray

# color_type = 3 - RGB

def get_im_cv2(path, img_rows, img_cols, color_type=1):

    # Load as grayscale

    if color_type == 1:

        img = cv2.imread(path, 0)

    elif color_type == 3:

        img = cv2.imread(path)

    # Reduce size

    resized = cv2.resize(img, (img_cols, img_rows))

    return resized





def get_im_cv2_mod(path, img_rows, img_cols, color_type=1):

    # Load as grayscale

    if color_type == 1:

        img = cv2.imread(path, 0)

    else:

        img = cv2.imread(path)

    # Reduce size

    rotate = random.uniform(-10, 10)

    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)

    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    resized = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)

    return resized

    



def get_driver_data():

    dr = dict()

    path = os.path.join('..', 'input', 'driver_imgs_list.csv')

    print('Read drivers data')

    f = open(path, 'r')

    line = f.readline()

    while (1):

        line = f.readline()

        if line == '':

            break

        arr = line.strip().split(',')

        dr[arr[2]] = arr[0]

    f.close()

    return dr





def load_train(img_rows, img_cols, color_type=1):

    X_train = []

    y_train = []

    driver_id = []

    start_time = time.time()

    driver_data = get_driver_data()



    print('Read train images')

    for j in range(10):

        print('Load folder c{}'.format(j))

        path = os.path.join('..', 'input', 'train', 'c' + str(j), '*.jpg')

        files = glob.glob(path)

        for fl in files:

            flbase = os.path.basename(fl)

            img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)

            X_train.append(img)

            y_train.append(j)

            driver_id.append(driver_data[flbase])



    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))

    unique_drivers = sorted(list(set(driver_id)))

    print('Unique drivers: {}'.format(len(unique_drivers)))

    print(unique_drivers)

    return X_train, y_train, driver_id, unique_drivers





def load_test(img_rows, img_cols, color_type=1):

    print('Read test images')

    start_time = time.time()

    path = os.path.join('..', 'input', 'test', '*.jpg')

    files = glob.glob(path)

    X_test = []

    X_test_id = []

    total = 0

    thr = math.floor(len(files)/10)

    for fl in files:

        flbase = os.path.basename(fl)

        img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)

        X_test.append(img)

        X_test_id.append(flbase)

        total += 1

        if total%thr == 0:

            print('Read {} images from {}'.format(total, len(files)))

    

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))

    return X_test, X_test_id





def cache_data(data, path):

    if os.path.isdir(os.path.dirname(path)):

        file = open(path, 'wb')

        pickle.dump(data, file)

        file.close()

    else:

        print('Directory doesnt exists')





def restore_data(path):

    data = dict()

    if os.path.isfile(path):

        file = open(path, 'rb')

        data = pickle.load(file)

    return data





def save_model(model):

    json_string = model.to_json()

    if not os.path.isdir('cache'):

        os.mkdir('cache')

    open(os.path.join('cache', 'architecture.json'), 'w').write(json_string)

    model.save_weights(os.path.join('cache', 'model_weights.h5'), overwrite=True)





def read_model():

    model = model_from_json(open(os.path.join('cache', 'architecture.json')).read())

    model.load_weights(os.path.join('cache', 'model_weights.h5'))

    return model





def split_validation_set(train, target, test_size):

    random_state = 51

    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test





def create_submission(predictions, test_id, info):

    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])

    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)

    now = datetime.datetime.now()

    if not os.path.isdir('subm'):

        os.mkdir('subm')

    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))

    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')

    result1.to_csv(sub_file, index=False)





def read_and_normalize_train_data(img_rows, img_cols, color_type=1):

    cache_path = os.path.join('cache', 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:

        train_data, train_target, driver_id, unique_drivers = load_train(img_rows, img_cols, color_type)

        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)

    else:

        print('Restore train from cache!')

        (train_data, train_target, driver_id, unique_drivers) = restore_data(cache_path)



    train_data = np.array(train_data, dtype=np.uint8)

    train_target = np.array(train_target, dtype=np.uint8)

    train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, color_type)

    train_target = np_utils.to_categorical(train_target, 10)

    train_data = train_data.astype('float32')

    train_data /= 255

    print('Train shape:', train_data.shape)

    print(train_data.shape[0], 'train samples')

    return train_data, train_target, driver_id, unique_drivers





def read_and_normalize_test_data(img_rows, img_cols, color_type=1):

    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '.dat')

    if not os.path.isfile(cache_path) or use_cache == 0:

        test_data, test_id = load_test(img_rows, img_cols, color_type)

        cache_data((test_data, test_id), cache_path)

    else:

        print('Restore test from cache!')

        (test_data, test_id) = restore_data(cache_path)



    test_data = np.array(test_data, dtype=np.uint8)

    test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, color_type)

    test_data = test_data.astype('float32')

    test_data /= 255

    print('Test shape:', test_data.shape)

    print(test_data.shape[0], 'test samples')

    return test_data, test_id





def dict_to_list(d):

    ret = []

    for i in d.items():

        ret.append(i[1])

    return ret





def merge_several_folds_mean(data, nfolds):

    a = np.array(data[0])

    for i in range(1, nfolds):

        a += np.array(data[i])

    a /= nfolds

    return a.tolist()





def merge_several_folds_geom(data, nfolds):

    a = np.array(data[0])

    for i in range(1, nfolds):

        a *= np.array(data[i])

    a = np.power(a, 1/nfolds)

    return a.tolist()





def copy_selected_drivers(train_data, train_target, driver_id, driver_list):

    data = []

    target = []

    index = []

    for i in range(len(driver_id)):

        if driver_id[i] in driver_list:

            data.append(train_data[i])

            target.append(train_target[i])

            index.append(i)

    data = np.array(data, dtype=np.float32)

    target = np.array(target, dtype=np.float32)

    index = np.array(index, dtype=np.uint32)

    return data, target, index





def create_model_v1(img_rows, img_cols, color_type=1):

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', activation = 'relu',

                            input_shape=(img_rows, img_cols, color_type)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))



    model.add(Convolution2D(70, 3, 3, border_mode='same', init='he_normal', activation = 'relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))



    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation = 'relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))

    model.add(Convolution2D(99, 3, 3, border_mode='same', init='he_normal', activation = 'relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))
    
    
    model.add(Flatten())

    model.add(Dense(10))

    model.add(Activation('softmax'))



    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy')

    return model





def run_single():

    # input image dimensions

    img_rows, img_cols = 64, 64

    batch_size = 128    #batch_size = 8

    nb_epoch = 1

    random_state = 51



    train_data, train_target, driver_id, unique_drivers = read_and_normalize_train_data(img_rows, img_cols, color_type_global)

    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)



    yfull_train = dict()

    yfull_test = []

    #unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',

     #                'p026', 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',

      #               'p050', 'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072',

       #              'p075']

    unique_list_train = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024',

                     'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049',

                     'p051', 'p052', 'p056', 'p061', 'p064', 'p066', 'p072']

    list_train0 = ['p002',  'p014',  'p016', 'p022',

                     'p035',  'p041', 'p045',  'p049',

                     'p052', 'p061',  'p066', ]

    list_train1 = ['p012','p015','p021','p024',

                     'p039', 'p042',  'p047',

                     'p051', 'p056',  'p064', 'p072']

    X_train, Y_train, train_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_train)

    X_train0, Y_train0, train_index0 = copy_selected_drivers(train_data, train_target, driver_id, list_train0)

    X_train1, Y_train1, train_index1 = copy_selected_drivers(train_data, train_target, driver_id, list_train1)

    #unique_list_valid = ['p081']

    unique_list_valid = ['p081', 'p075', 'p050', 'p026']

    X_valid, Y_valid, test_index = copy_selected_drivers(train_data, train_target, driver_id, unique_list_valid)



    print('Start Single Run')

    #print('Split train: ', len(X_train), len(Y_train))

    print('Split valid: ', len(X_valid), len(Y_valid))

    print('Train drivers: ', unique_list_train)

    print('Test drivers: ', unique_list_valid)



    model = create_model_v1(img_rows, img_cols, color_type_global)

    #model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,

     #         verbose=1, validation_data=(X_valid, Y_valid))

    

    from numpy import random

    np.random.seed(100)

    

    min_loss = 10000

    

    average_loss = 0

    cnt = 0

    

    for i in range(300):

        

        print(i)

        print('-----------------------------------------------')

        
        
        for e in range(1):

            idxList = []

            for j in range(batch_size):

                idxList.append(np.random.randint(0, len(X_train0)))

            xt0 = X_train0[idxList]

            yt0 = Y_train0[idxList]

        

            hist = model.fit(xt0, yt0, #batch_size=batch_size, 

                  nb_epoch=1,

                  verbose=1, validation_data=(X_valid, Y_valid))#, steps_per_epoch = 1)

            average_loss += hist.history['val_loss'][0]
            cnt+=1
            
            if hist.history['val_loss'][0] < min_loss:

                min_loss = hist.history['val_loss'][0]

             

        for e in range(1):

            idxList = []

            for j in range(batch_size):

                idxList.append(np.random.randint(0, len(X_train1)))

            xt1 = X_train1[idxList]

            yt1 = Y_train1[idxList]

        

            hist = model.fit(xt1, yt1, #batch_size=batch_size, 

                nb_epoch=1,

                verbose=1, validation_data=(X_valid, Y_valid))#, steps_per_epoch = 1)

            average_loss += hist.history['val_loss'][0]
            cnt+=1   

            if hist.history['val_loss'][0] < min_loss:

                min_loss = hist.history['val_loss'][0]

        
        '''
        for e in range(1):

            idxList = []

            for j in range(batch_size):

                idxList.append(np.random.randint(0, len(X_train)))

            xt = X_train[idxList]

            yt = Y_train[idxList]

        

            hist = model.fit(xt, yt, #batch_size=batch_size, 

                  nb_epoch=nb_epoch,

                verbose=1, validation_data=(X_valid, Y_valid))#, steps_per_epoch = 1)

            if hist.history['val_loss'][0] < min_loss:

                min_loss = hist.history['val_loss'][0]

        

        average_loss += hist.history['val_loss'][0]

        cnt+=1

        

        print('min_loss = ', min_loss)

    
    '''
    """
       

    #for i in range(90):

        idxList = []

        for j in range(batch_size):

            idxList.append(np.random.randint(0, len(X_train)))

        xt = X_train[idxList]

        yt = Y_train[idxList]

        

        hist = model.fit(xt, yt, #batch_size=batch_size, 

              nb_epoch=nb_epoch,

              verbose=1, validation_data=(X_valid, Y_valid))#, steps_per_epoch = 1)

        if hist.history['val_loss'][0] < min_loss:

            min_loss = hist.history['val_loss'][0]

    

    for i in range(20):

        

        print(i)

        print('-----------------------------------------------')

        

        idxList = []

        for j in range(batch_size):

            idxList.append(np.random.randint(0, len(X_train0)))

        xt0 = X_train0[idxList]

        yt0 = Y_train0[idxList]

        

        model.fit(xt0, yt0, #batch_size=batch_size, 

              nb_epoch=nb_epoch,

              verbose=1, validation_data=(X_valid, Y_valid))#, steps_per_epoch = 1)

              

              

        idxList = []

        for j in range(batch_size):

            idxList.append(np.random.randint(0, len(X_train1)))

        xt1 = X_train1[idxList]

        yt1 = Y_train1[idxList]

        

        model.fit(xt1, yt1, #batch_size=batch_size, 

              nb_epoch=nb_epoch,

              verbose=1, validation_data=(X_valid, Y_valid))#, steps_per_epoch = 1)

          

    for i in range(60):

        

        print(i)

        

        idxList = []

        for j in range(batch_size):

            idxList.append(np.random.randint(0, len(X_train)))

        xt = X_train[idxList]

        yt = Y_train[idxList]

        

        model.fit(xt, yt, #batch_size=batch_size, 

              nb_epoch=nb_epoch,

              verbose=1, validation_data=(X_valid, Y_valid))#, steps_per_epoch = 1)

    """



    print("Average loss: ", average_loss / cnt)

    # score = model.evaluate(X_valid, Y_valid, show_accuracy=True, verbose=0)

    # print('Score log_loss: ', score[0])



    predictions_valid = model.predict(X_valid, batch_size=128, verbose=1)

    score = log_loss(Y_valid, predictions_valid)

    print('Score log_loss: ', score)



        

    print('Score min log loss = ', min_loss)    

    exit(0)

    # Store valid predictions

    for i in range(len(test_index)):

        yfull_train[test_index[i]] = predictions_valid[i]



    # Store test predictions

    test_prediction = model.predict(test_data, batch_size=128, verbose=1)

    yfull_test.append(test_prediction)



    print('Final log_loss: {}, rows: {} cols: {} epoch: {}'.format(score, img_rows, img_cols, nb_epoch))

    info_string = 'NO_CGD_CHECK___ClearCGD_loss_' + str(score)  + '_r_' + str(img_rows)  + '_c_' + str(img_cols)  + '_ep_' + str(nb_epoch)



    test_res = merge_several_folds_mean(yfull_test, 1)

    create_submission(test_res, test_id, info_string)





run_single()