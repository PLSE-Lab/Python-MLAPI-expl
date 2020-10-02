import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Activation, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, applications
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import backend as K 
ann_file='../input/train2019.json'
with open(ann_file) as data_file:
	train = json.load(data_file)
train_df = pd.DataFrame(train['annotations'])[['image_id','category_id']]
train_img = pd.DataFrame(train['images'])[['id','file_name']].rename(columns={'id':'image_id'})
df_train = pd.merge(train_img,train_df,on='image_id')
df_train['category_id']=df_train['category_id'].astype(str)
df_train.head()

ann_file='../input/val2019.json'
with open(ann_file) as data_file:
	val = json.load(data_file)
val_df = pd.DataFrame(val['annotations'])[['image_id','category_id']]
val_img = pd.DataFrame(val['images'])[['id','file_name']].rename(columns={'id':'image_id'})
df_val = pd.merge(val_img,val_df,on='image_id')
df_val['category_id']=df_val['category_id'].astype(str)
df_val.head()

nb_classes = 1010
batch_size = 230
img_size = 96
nb_epochs = 15

train_datagen=ImageDataGenerator(rescale=1./255, 
    validation_split=0.3,
    horizontal_flip = True,    
    zoom_range = 0.4,
    width_shift_range = 0.5,
    height_shift_range=0.5
    )
train_generator=train_datagen.flow_from_dataframe(    
    dataframe=df_train,    
    directory="../input/train_val2019",
    x_col="file_name",
    y_col="category_id",
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",    
    target_size=(img_size,img_size))

test_datagen = ImageDataGenerator(rescale=1./255)

valid_generator=test_datagen.flow_from_dataframe(    
    dataframe=df_val,    
    directory="../input/train_val2019",
    x_col="file_name",
    y_col="category_id",
    batch_size=batch_size,
    shuffle=True,
    class_mode="categorical",    
    target_size=(img_size,img_size))
model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
model.trainable = False
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.6)(x)
predictions = Dense(nb_classes, activation="softmax")(x)
model_final = Model(input = model.input, output = predictions)

model_final.compile(optimizers.rmsprop(lr=0.0002, decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy'])

checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

history = model_final.fit_generator(generator=train_generator,  
                                    
                                    steps_per_epoch=6,
                                    
                                    validation_data=valid_generator, 
                                    
                                    validation_steps=3,
                                    
                                    epochs=nb_epochs,
                                    callbacks = [checkpoint, early],
                                    verbose=2)
test_ann_file = '../input/test2019.json'
with open(test_ann_file) as data_file:
        test_anns = json.load(data_file)
test_img_df = pd.DataFrame(test_anns['images'])[['id', 'file_name']].rename(columns={'id':'image_id'})
test_img_df.head()
test_generator = test_datagen.flow_from_dataframe(      
    
        dataframe=test_img_df,    
    
        directory = "../input/test2019",    
        x_col="file_name",
        target_size = (img_size,img_size),
        batch_size = 1,
        shuffle = False,
        class_mode = None
        )
test_generator.reset()
predict=model_final.predict_generator(test_generator, steps = len(test_generator.filenames))
predicted_class_indices=np.argmax(predict,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
sam_sub_df = pd.read_csv('../input/kaggle_sample_submission.csv')
sam_sub_df.head()
filenames=test_generator.filenames
results=pd.DataFrame({"file_name":filenames,
                      "predicted":predictions})
df_res = pd.merge(test_img_df, results, on='file_name')[['image_id','predicted']]\
    .rename(columns={'image_id':'id'})

df_res.head()
df_res.to_csv("submission.csv",index=False)