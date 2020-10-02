#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import cv2
import os
import json

from PIL import Image, ImageFile, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, MaxPool1D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


# In[ ]:


# Set your own project id here
PROJECT_ID = 'your-google-cloud-project'
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)


# In[ ]:


#Use this cell to modify files in output directory
#os.remove("/kaggle/working/train_cropped/905a3c8c-21bc-11ea-a13a-137349068a90.jpg")
os.mkdir("/kaggle/working/cropped")


# In[ ]:


with open('../input/iwildcam-2020-fgvc7/iwildcam2020_megadetector_results.json', encoding='utf-8') as json_file:
    megadetector_results =json.load(json_file)
    
megadetector_results.keys()


# In[ ]:


megadetector_results_df = pd.DataFrame(megadetector_results["images"])
megadetector_results_df.head()


# In[ ]:


def draw_bboxs(detections_list, im):
    """
    detections_list: list of set includes bbox.
    im: image read by Pillow.
    """
    
    for detection in detections_list:
        x1, y1,w_box, h_box = detection["bbox"]
        ymin,xmin,ymax, xmax=y1, x1, y1 + h_box, x1 + w_box
        draw = ImageDraw.Draw(im)
        
        imageWidth=im.size[0]
        imageHeight= im.size[1]
        (left, right, top, bottom) = (xmin * imageWidth, xmax * imageWidth,
                                      ymin * imageHeight, ymax * imageHeight)
        
        draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=4, fill='Red')


# In[ ]:


def cut_bboxs(detections_list, im):
    """
    detections_list: list of set includes bbox.
    im: image read by Pillow.
    """
    
    #initialize coords as entire image  
    imageWidth=im.size[0]
    imageHeight= im.size[1]
    
    top_left_coord = [imageWidth, imageHeight]
    bottom_right_coord = [0,0]
    #print(top_left_coord[0],top_left_coord[1], bottom_right_coord[0], bottom_right_coord[1])

    if len(detections_list) == 0:
        return im
    
    for detection in detections_list:
        x1, y1,w_box, h_box = detection["bbox"]
        ymin,xmin,ymax, xmax=y1*imageHeight, x1*imageWidth, (y1 + h_box)*imageHeight, (x1 + w_box)*imageWidth
        #print(ymin,xmin,ymax, xmax)
        if ymax > bottom_right_coord[1]:
            bottom_right_coord[1] = ymax
        if xmax > bottom_right_coord[0]:
            bottom_right_coord[0] = xmax
        if ymin < top_left_coord[1]:
            top_left_coord[1] = ymin
        if xmin < top_left_coord[0]:
            top_left_coord[0] = xmin
    
    draw = ImageDraw.Draw(im)
    
    
    (left, right, top, bottom) = (top_left_coord[0] , bottom_right_coord[0] ,
                                  top_left_coord[1] , bottom_right_coord[1] )
    #print(left, top, right, bottom)
    new_img = im.crop((left, top, right, bottom))
    return new_img


# In[ ]:


# setup the directories
DATA_DIR = '../input/iwildcam-2020-fgvc7/'
TRAIN_DIR = DATA_DIR + 'train/'
TEST_DIR = DATA_DIR + 'test/'

# load the megadetector results
megadetector_results = json.load(open(DATA_DIR + 'iwildcam2020_megadetector_results.json'))
print(megadetector_results['images'][:2])

# load train images annotations
train_info = json.load(open(DATA_DIR + 'iwildcam2020_train_annotations.json'))
# split json into several pandas dataframes
train_annotations = pd.DataFrame(train_info['annotations'])
train_images = pd.DataFrame(train_info['images'])
train_categories = pd.DataFrame(train_info['categories'])

# load test images info
test_info = json.load(open(DATA_DIR + 'iwildcam2020_test_information.json'))
# split json into several pandas dataframes
test_images = pd.DataFrame(test_info['images'])
test_categories = pd.DataFrame(test_info['categories'])


# In[ ]:


print(len(train_images))
train_length = len(megadetector_results_df)
#print(train_images.head())


# In[ ]:


missing = []
for i in range(20):
    try:
        data_num = i
        try:
            im = Image.open("../input/iwildcam-2020-fgvc7/train/" + megadetector_results_df.loc[data_num]['id'] + ".jpg")
        except:
            im = Image.open("../input/iwildcam-2020-fgvc7/test/" + megadetector_results_df.loc[data_num]['id'] + ".jpg")
        #im = im.resize((500,325))
        new_img = cut_bboxs(megadetector_results_df.loc[data_num]['detections'], im)
        draw_bboxs(megadetector_results_df.loc[data_num]['detections'], im)
        new_img = new_img.resize((500,325))
        #new_img.save("test2.jpeg")
        #new_img.save("./cropped/" + megadetector_results_df.loc[data_num]['id'] + ".jpg")
        plt.imshow(im)
        plt.show()
        plt.imshow(new_img)
        plt.show()
        #os.system('cls')
        #print(train_length - i)
        #print(len(missing))
    except:
        missing.append(i)
        #print("Error at:", i)
        os.system('cls')
        print(train_length - i)
        print(len(missing))


# In[ ]:


print(len([name for name in os.listdir('./train_cropped/') if os.path.isfile(name)]))


# In[ ]:


pd.set_option('display.max_rows', 500)
train_categories


# In[ ]:


df_train = pd.merge(train_annotations, train_images, how='outer', left_on='image_id', right_on='id')
df_train = df_train.drop(['id_y'], axis=1)

df_train['Date'] = pd.to_datetime(df_train['datetime']).dt.date
df_train = df_train.astype({'Date': str})
df_train['Time'] = pd.to_datetime(df_train['datetime']).dt.time
df_train = df_train.astype({'Time': str})

df_train['Year'] = df_train['Date'].str.slice(0, 4, 1)
df_train = df_train.astype({'Year': int})
df_train['Month'] = df_train['Date'].str.slice(5, 7, 1)
df_train = df_train.astype({'Month': int})
df_train['Day'] = df_train['Date'].str.slice(8, 10, 1)
df_train = df_train.astype({'Day': int})

df_train['Hour'] = df_train['Time'].str.slice(0, 2, 1)
df_train = df_train.astype({'Hour': int})
df_train['Min'] = df_train['Time'].str.slice(3, 5, 1)
df_train = df_train.astype({'Min': int})
df_train['Sec'] = df_train['Time'].str.slice(6, 8, 1)
df_train = df_train.astype({'Sec': int})

df_train = df_train.drop(['Date', 'Time'], axis=1)

df_train.columns = ['animal_cnt', 'image_id', 'id', 'category_id', 'seq_num_frames', 'location', 'datetime', 'frame_num', 'seq_id', 'width', 'height', 'file_name', 'year',
                    'month', 'day', 'hour', 'min', 'sec']

df_train = df_train[['id', 'seq_id', 'image_id', 'file_name', 'width', 'height', 'seq_num_frames', 'frame_num', 'datetime', 'location', 'animal_cnt', 'year',
                     'month', 'day', 'hour', 'min', 'sec', 'category_id']]

df_train['category_id'] = df_train['category_id'].apply(str)

df_train.head(10)


# In[ ]:


batch_size = 16
epochs = 1
IMG_HEIGHT = 325
IMG_WIDTH = 500


# In[ ]:


corrupted_files = ['87022118-21bc-11ea-a13a-137349068a90.jpg',
                   '8f17b296-21bc-11ea-a13a-137349068a90.jpg',
                   '883572ba-21bc-11ea-a13a-137349068a90.jpg',
                   '896c1198-21bc-11ea-a13a-137349068a90.jpg',
                   '8792549a-21bc-11ea-a13a-137349068a90.jpg',
                   '99136aa6-21bc-11ea-a13a-137349068a90.jpg']

for corrupted_file in corrupted_files:
    df_train = df_train[df_train['file_name'] != corrupted_file]


# In[ ]:


total_train_image_generator = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15, width_shift_range=0.2, 
                                                 height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, 
                                                 fill_mode="nearest")
total_train_data_gen = total_train_image_generator.flow_from_dataframe(dataframe=df_train, directory=TRAIN_DIR, x_col='file_name', y_col='category_id', 
                                                            class_mode="categorical", target_size=(IMG_HEIGHT,IMG_WIDTH), batch_size=batch_size)


# In[ ]:


from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(total_train_data_gen.classes), 
                total_train_data_gen.classes)


# In[ ]:


total_train_data_gen.class_indices
REVERSE_CLASSMAP = dict([(v, k) for k, v in total_train_data_gen.class_indices.items()])
REVERSE_CLASSMAP


# In[ ]:


sample_training_images, _ = next(total_train_data_gen)


# In[ ]:


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[ ]:


plotImages(sample_training_images[:5])


# In[ ]:


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(total_train_data_gen.class_indices))
])


# In[ ]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(
    total_train_data_gen,
    epochs=epochs,
    class_weight=class_weights
)


# In[ ]:


acc = history.history['accuracy']

loss = history.history['loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('TrainingAccuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('TrainingLoss')
plt.show()


# In[ ]:


df_test = df_train.sample(15)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


# In[ ]:


def predict_image(model, file_name, probability_model):
    img = Image.open(file_name)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    test_img = np.array([np.asarray(img)])

    prediction = probability_model.predict(test_img)
    pred_class = REVERSE_CLASSMAP[np.argmax(prediction)]
    
    return pred_class


# In[ ]:


def plot_image(pred_label, true_label, img_filename):  
    img = Image.open(img_filename)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    if pred_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.title("{} ({})".format(train_categories.loc[train_categories['id'] == int(true_label), 'name'].to_string(index=False),
                                true_label))
    
    plt.xlabel("{} ({})".format(train_categories.loc[train_categories['id'] == int(pred_label), 'name'].to_string(index=False),
                                pred_label)
               ,color=color)
    plt.show()


# In[ ]:


count = 0
for filename in df_test['file_name']:
    pred_class = predict_image(model, TRAIN_DIR + filename, probability_model)
    
    plot_image(pred_class, df_test['category_id'].iloc[count], TRAIN_DIR + filename)
    
    count += 1


# In[ ]:


submission = pd.read_csv(DATA_DIR + 'sample_submission.csv')


# In[ ]:


submission.head()


# In[ ]:


count = 0
for ids in submission['Id']:
    if count % 1000 == 0:
        print('Through ' + str(count) + ' images')
    
    filename = TEST_DIR + ids + '.jpg'
    pred_cat = predict_image(model, filename, probability_model)
    
    submission['Category'][count] = int(pred_cat)
    
    count += 1


# In[ ]:


submission.to_csv('E:/submission_004.csv', index=False)

