import numpy as np 
import keras 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import glob
import sklearn
import cv2

map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson', 
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak', 
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}
pic_size = 64
batch_size = 32
epochs = 200
num_classes = len(map_characters)

def create_model_four_conv(input_shape):
    """
    CNN Keras model with 4 convolutions.
    :param input_shape: input shape, generally X_train.shape[1:]
    :return: Keras model, RMS prop optimizer
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    return model, opt

def training(model, X_train, X_test, y_train, y_test):
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(X_train)
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        history = model.fit_generator(datagen.flow(X_train, y_train,
                                     batch_size=batch_size),
                                    steps_per_epoch=X_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    callbacks=callbacks_list,
                                    validation_data=(X_test, y_test))
        return model, history

def load_model_from_checkpoint(weights_path, input_shape=(pic_size,pic_size,3)):
    model, opt = create_model_four_conv(input_shape)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return model
  
def load_test_set(path):
    pics, labels = [], []
    reverse_dict = {v:k for k,v in map_characters.items()}
    for pic in glob.glob(path+'*.*'):
        char_name = "_".join(pic.split('/')[3].split('_')[:-1])
        if char_name in reverse_dict:
            temp = cv2.imread(pic)
            temp = cv2.resize(temp, (pic_size,pic_size)).astype('float32') / 255.
            pics.append(temp)
            labels.append(reverse_dict[char_name])
    X_test = np.array(pics)
    y_test = np.array(labels)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    print("Test set", X_test.shape, y_test.shape)
    return X_test, y_test

def stats_every_character(y_pred, y_test):
    report = []
    true_pos_tot = 0
    space_max = np.max([len(name) for name in map_characters.values()])
    report.append('{s}{space}{s2}{space2}{s3}'.format(s='character', 
                                          space= ' ' * (space_max + 2 - len("character")), 
                                          s2='Precision',
                                          space2= '  ', 
                                          s3='Support'))
    for id_char, name in map_characters.items():
        idx_char = [i for i, x in enumerate(np.argmax(y_pred, axis= 1) == id_char) if x]
        true_pos = np.sum(np.argmax(y_test[idx_char], axis=1) == np.argmax(y_pred[idx_char], axis=1))
        true_pos_tot += true_pos
        report.append('{s}{space}{f:.2f}{space2}{s3}'.format(s=name.replace('_',' ').title(), 
                                                             space= ' ' * (space_max + 4 - len(name)), 
                                                             f=true_pos/len(idx_char),
                                                             space2 = ' '*6,
                                                             s3 = len(idx_char)))
    report.append('{s}{space}{f:.2f}{space2}{s3}'.format(s="Total", 
                                             space= ' ' * (space_max + 4 - len('Total')), 
                                             f=true_pos_tot/len(y_test),
                                             space2 = ' '*6,
                                             s3 = len(y_pred)))
    return report
 
if __name__ == '__main__':
    model = load_model_from_checkpoint('../input/weights.best.hdf5')
    X_test, y_test = load_test_set("../input/kaggle_simpson_testset/")
    y_pred = model.predict(X_test)
    
    # Need to upgrade sklearn : AttributeError: module 'sklearn' has no attribute 'metrics' 
    # print('\n', sklearn.metrics.classification_report(np.where(y_test > 0)[1], 
    #                                              np.argmax(y_pred, axis=1), 
    #                                              target_names=list(map_characters.values())), sep='')
    
    report = stats_every_character(y_pred, y_test)
    print('\n'.join(report))
    