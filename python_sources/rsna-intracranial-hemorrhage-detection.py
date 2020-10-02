#!/usr/bin/env python
# coding: utf-8

# ## Table of contents
# 1. [Import libraries and datasets](#Import)
# 2. [Exploratory analysis](#Exploratory-analysis)
# 
#     * [Number of samples](#Number-of-samples)
#     * [Data Visualization](#Data-visualization) 
#     * [Preprocess images](#Prepocess-Images)
# 
# 
# 3. [Model](#Model) 
# 4. [Submission](#Submission)
# 

# ## Import

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pydicom
import cv2
from tqdm import tqdm
import json
import os


# In[ ]:


directory = '../input/rsna-intracranial-hemorrhage-detection/'
dir_train = '../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/'
dir_test = '../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/'


# In[ ]:


train_df =  pd.read_csv(directory+'stage_1_train.csv')
test_df = pd.read_csv(directory+'stage_1_sample_submission.csv')
train_images = os.listdir(dir_train)
test_images = os.listdir(dir_test)


# ## Exploratory analysis

# ### Number of samples

# In[ ]:


print("Train CSV :",train_df.shape)
print("Test CSV :",test_df.shape)
print("Train Images:",len(train_images))
print("Test Images:",len(test_images))


# In[ ]:


display(train_df.head())


# In[ ]:


display(train_df.tail())


# In[ ]:


display(train_df.head())


# In[ ]:


print("Train: \n",train_df.count())
print("Test: \n",test_df.count())


# In[ ]:


train_df['Image_ID'] = train_df['ID'].str.rsplit(pat='_',n=1,expand=True)[0]
train_df['Hemorrhage'] = train_df['ID'].str.rsplit(pat='_',n=1,expand=True)[1]
train_df = train_df[['Image_ID','Hemorrhage','Label']]


# In[ ]:


#Nombre d'images uniques
print("Number of images :",train_df['Image_ID'].nunique())
print("Number of Hemorraghes :",train_df['Hemorrhage'].nunique())


# In[ ]:


pd.DataFrame(train_df['Image_ID'].value_counts()).reset_index().head(10)


# In[ ]:


display(test_df.head())


# In[ ]:


test_df['Image_ID'] = test_df['ID'].str.rsplit(pat='_',n=1,expand=True)[0]
test_df['Image_ID'] = test_df['Image_ID']+".png"
test_df = test_df['Image_ID'].drop_duplicates().reset_index()[['Image_ID']]


# In[ ]:


display(test_df.head())


# ### Data visualization

# ##### An additional label for any, which should always be true if any of the sub-type labels is true. We could know the number of images that have any kind of hemorrhage with this variable

# In[ ]:


def graph_hemorrhage():
    percentage = train_df[(train_df['Hemorrhage']=='any')&(train_df['Label']==1)]['Image_ID'].count()/train_df['Image_ID'].nunique()*100
    print("percentage of images with hemorrhage : ",round(percentage,2),'%')

    pd.DataFrame([percentage,100-percentage],columns=['percentage']).plot(kind='pie',
                                                                          y='percentage',
                                                                          labels=['Hemorrhage','Non_Hemorrhage'],
                                                                          title='Hemorraghe distribution',
                                                                          autopct='%.1f%%',
                                                                          legend=False,
                                                                          figsize=(6,6),
                                                                          shadow=True,
                                                                          startangle=90)
    plt.show()


# In[ ]:


Hemorrhage = pd.DataFrame(train_df[(train_df['Label']==1)&(train_df['Hemorrhage']!='any')]['Hemorrhage'].value_counts()).reset_index()
Hemorrhage.columns = ['Hemorrhage','Number_Pictures']

display(Hemorrhage)

Hemorrhage.plot(kind='pie',y='Number_Pictures',labels=Hemorrhage['Hemorrhage'].unique(),title='Hemorrhage distribution',               legend=False,autopct='%.1f%%',figsize=(6,6),shadow=True, startangle=90,fontsize=11)

plt.show()


# In[ ]:


graph_hemorrhage()


# In[ ]:


display(train_df.head())


# ###  Transform dataset

# In[ ]:


# Remove image with damaged pixels
train_df = train_df[train_df['Image_ID']!='ID_6431af929']
train_images.remove('ID_6431af929.dcm')


# In[ ]:


def undersample(dataset): 
    ID_hemorrhage = train_df[(train_df['Hemorrhage']=='any')&(train_df['Label']==1)][['Image_ID']].reset_index(drop=True)
    n_samples = len(ID_hemorrhage)
    ID_Healthy = train_df[(train_df['Hemorrhage']=='any')&(train_df['Label']==0)][['Image_ID']].reset_index(drop=True).sample(n=n_samples,random_state=1)
    
    if dataset =='train_df':
        result = pd.concat([train_df.merge(ID_hemorrhage,how='inner',on='Image_ID'),                            train_df.merge(ID_Healthy,how='inner',on='Image_ID')]).sample(frac=1).                            sort_values('Image_ID').reset_index(drop=True)
    
    elif dataset == 'train_images':
        data_frame = pd.DataFrame(train_images,columns=['Image_ID'])
        result = pd.concat([data_frame.merge(ID_hemorrhage+'.dcm',how='inner',on='Image_ID'),                            data_frame.merge(ID_Healthy+'.dcm',how='inner',on='Image_ID')])['Image_ID'].tolist()
    return(result)
    


# In[ ]:


train_df = undersample('train_df')
train_images = undersample('train_images')


# In[ ]:


graph_hemorrhage()


# In[ ]:


pivot_df = train_df.drop_duplicates().pivot(index='Image_ID', columns='Hemorrhage', values='Label').reset_index()
pivot_df['Image_ID'] = pivot_df['Image_ID']+'.png'
display(pivot_df.head())


# ### Prepocess Images

# In[ ]:


def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_metadata(image):
    metadata = {
        "window_center": image.WindowCenter,
        "window_width": image.WindowWidth,
        "intercept": image.RescaleIntercept,
        "slope": image.RescaleSlope
    }
    return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}

def window_image(img, window_center, window_width, intercept, slope):
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img

def normalize(image):
    min_image = image.min()
    max_image = image.max()
    return (image - min_image) / (max_image - min_image)

def resize(image,width,weight):
    resized = cv2.resize(image, (width, weight))
    return resized

def save(directory,image,image_normalized_resized):
    save_dir = '/kaggle/tmp/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    path = directory+image
    new_path = save_dir + image.replace('.dcm', '.png')        
    res = cv2.imwrite(new_path, image_normalized_resized)
    
def normalize_resize_save(dataset,width,weight,directory):
    for i in tqdm(dataset):
        image=pydicom.read_file(directory+i)
        image_windowed = window_image(image.pixel_array, ** get_metadata(image))
        image_normalized_resized = resize(normalize(image_windowed),width,weight)
        save(directory,i,image_normalized_resized)


# #### Visualize first image in the Data Set

# In[ ]:


image=pydicom.read_file(dir_train+train_df['Image_ID'][0]+".dcm")
image_windowed = window_image(image.pixel_array, ** get_metadata(image))

display(image)
plt.imshow(image_windowed, cmap=plt.cm.bone)


# #### Visualize images with hemorraghes

# In[ ]:


def view_images(data_frame,hemorraghe):
    width = 5
    height = 1
    fig, axs = plt.subplots(height, width, figsize=(20,5))

    list_hem = pd.DataFrame(train_df[(train_df['Label']==1)&(train_df['Hemorrhage']==hemorraghe)][['Image_ID']].                            head(width*height)+".dcm").reset_index()
    
    for i in range(0,width*height):
        image=pydicom.read_file(dir_train+list_hem['Image_ID'][i])
        image_windowed = window_image(image.pixel_array, ** get_metadata(image))
        fig.add_subplot(height,width, i+1)
        axs[i].set_title(list_hem['Image_ID'][i])
        plt.imshow(image_windowed, cmap=plt.cm.bone)
        
    plt.suptitle("Images with "+hemorraghe,fontsize = 20)
    plt.show()


# In[ ]:


for i in train_df['Hemorrhage'].unique():
    view_images(train_df,i)


# #### Normalize, resize and save new images in png format[](http://)

# In[ ]:


normalize_resize_save(train_images,224,224,dir_train)
normalize_resize_save(test_images,224,224,dir_test)


# In[ ]:


print("Number of files resized and saved: {}".format(len(os.listdir("/kaggle/tmp/"))))


# ## Model

# In[ ]:


from keras import layers
import tensorflow as tf
from keras.applications import DenseNet121
# from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
import torch
import keras

EPOCHS = 11
BATCH_SIZE = 32
model_nn='densenet'


# In[ ]:


def pick_nn(neu_net):
    if neu_net == 'densenet':
        neural_network = DenseNet121(
            weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
            include_top=False,
            input_shape=(224,224,3)
        )
        
        
    elif neu_net=='resnet':
        neural_network = ResNet50 (
            weights = 'imagenet',
            include_top=False,
            input_shape =(224,224,3)
        )
        
    return(neural_network)


# In[ ]:


def DataGenerator():
    return(ImageDataGenerator(zoom_range=0.1,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images,
        validation_split=0.1))

def create_generator(dataset,batch_size):
    if dataset == 'test':
        generator = DataGenerator().flow_from_dataframe(
                        test_df,
                        directory='/kaggle/tmp/',
                        x_col='Image_ID',
                        class_mode=None,
                        target_size=(224, 224),
                        batch_size=batch_size,
                        shuffle=False
                    )
    else:
        generator = DataGenerator().flow_from_dataframe(dataframe=pivot_df, 
                                            directory="/kaggle/tmp/",
                                            x_col="Image_ID",
                                            y_col=['any', 'epidural', 'intraparenchymal', 
                                           'intraventricular', 'subarachnoid', 'subdural'],
                                            class_mode='other',
                                            target_size=(224,224),
                                            batch_size=batch_size,
                                            subset = dataset)
    
    return(generator)


# In[ ]:


def build_model(nn):
    model = Sequential()
    
    model.add(pick_nn(neu_net=nn))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(6, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=0.001),
        metrics=['accuracy']
    )
    
    return model


# In[ ]:


model = build_model(model_nn)
model.summary()


# In[ ]:


checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor='val_loss', 
    verbose=0, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)


history = model.fit_generator(
    create_generator(dataset = 'training', batch_size=BATCH_SIZE),
    steps_per_epoch=3500,
    validation_data=create_generator(dataset = 'validation', batch_size=BATCH_SIZE),
    validation_steps=3000,
    callbacks=[checkpoint],
    epochs=EPOCHS
)


# In[ ]:


history_df = pd.DataFrame(history.history)

plt.plot(history_df[['loss', 'val_loss']] )
plt.title('Value of loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend(('Loss_train', 'Loss_Val'))
plt.grid()
plt.show()

plt.plot(history_df[['accuracy', 'val_accuracy']])
plt.title('Value of accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.legend(('Acc', 'Acc_Val'))
plt.grid()
plt.show()


# In[ ]:


model.load_weights('model.h5')
create_data_test = create_generator(dataset = 'test',batch_size=BATCH_SIZE)
y_test = model.predict_generator(create_data_test,
    steps=len(create_data_test),
    verbose=1
)


# ## Submission

# In[ ]:


test_df = test_df.join(pd.DataFrame(y_test, columns = ['any', 'epidural', 'intraparenchymal', 
         'intraventricular', 'subarachnoid', 'subdural']))


# In[ ]:


test_df = test_df.melt(id_vars=['Image_ID'])


# In[ ]:


test_df['ID'] = test_df.Image_ID.apply(lambda x: x.replace('.png', '')) + '_' + test_df.variable
test_df['Label'] = test_df['value']

test_df[['ID', 'Label']].to_csv('submission.csv', index=False)


# In[ ]:


test_df[['ID', 'Label']].to_csv('submission.csv', index=False)


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np

def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='submission.csv')


# #### References
# 
# [Data_Visualisation + Model ](https://www.kaggle.com/jesucristo/rsna-introduction-eda-models) <br>
# [Data_Visualisation ](https://www.kaggle.com/marcovasquez/basic-eda-data-visualization)
# 
