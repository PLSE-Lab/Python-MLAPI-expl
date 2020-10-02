#!/usr/bin/env python
# coding: utf-8

# # <font color='Teal'> Clothing Item </font> <font color='red'>Recognition</font>
# > ---

# > In this notebook, a multi-label problem has been formulated to identify attributes of clothing items, labels have distinct classes against them. As a whole the model describes attributes of any clothing item within visual and dataset related constraints
# 
# ---
# #### Part 1
# * Initialization & Data Loading
# * Data Exploration
# 
# ---
# #### Part 2
# * Centralized CNN Model 
# * Training with Callbacks
# 
# ---
# #### Part 3
# * Prediction Scripts

# ---
# ## <font color='teal'> Part 1</font>

# ### Initialization

# In[ ]:


import os
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization, add, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model


# ### Loading available data

# In[ ]:


get_ipython().system('pwd')
path = '/kaggle/input/fashion-small/fashion_small/'
table = 'styles.csv'
base = path+'resized_images/'

print('\nDirectory contains:\n', os.listdir(path))

data = pd.read_csv(path+table, error_bad_lines=False, warn_bad_lines=False) ;

print('\n CSV contains {} entries, with attributes\n {}'.format(len(data),data.keys().values))


# In[ ]:


# Check for exsisting images in database

im_path = []  

for item_id in data['id']:
    #
    tmp_path = base + str(item_id) +'.jpg'
    #
    if os.path.exists(tmp_path):
        im_path.append(tmp_path)
    else:
        data.drop(data[data.id == item_id].index, inplace=True)
        print('Item {} Doesn\'t exists'.format(item_id))

data['img_path'] = im_path
data = data.sample(frac=1, random_state=10)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
data.tail()


# ### Exploratory Data Analysis

# In[ ]:


# > Unique values per column

data.nunique()


# > Analysing Apparels from masterCategory
# 
# * Removing Duplicate entries
# 
# * Removing _'year'_                     :: Can't be learnt visually
# 
# * Removing _'productDisplayName'_       :: Can't be learnt visually

# In[ ]:


df = data[data['masterCategory'] == 'Apparel']

df.drop_duplicates(subset='img_path', inplace=True)

df.drop(columns=['masterCategory', 'year', 'productDisplayName'], inplace=True)

df.reset_index(inplace=True, drop=True)
df.tail(5)


# > Temp functions
# 1. Display Categorical charts
# 2. Clean dataFrame as per minimum requirements for modelling

# In[ ]:


def disp_category(category, df, min_items=500):
    '''
    Display Categorical charts
    category  :: dtpye=str, Category (attribute/column name)
    df        :: DataFrame to be loaded
    min_items :: dtype=int, minimum rows to qualify as class
    
    returns classes to be selected for the analysis
    '''
    dff = df.groupby(category)
    class_ = dff.count().sort_values(by='id')['id'].reset_index()
    class_.columns=[category,'count']
    
    class_= class_[class_['count']>=min_items][category]
    df = df[df[category].isin(class_)]

    labels = df[category]
    
    counts = defaultdict(int)
    for l in labels:
         counts[l] += 1

    counts_df = pd.DataFrame.from_dict(counts, orient='index')
    counts_df.columns = ['count']
    counts_df.sort_values('count', ascending=False, inplace=True)

    fig, ax = plt.subplots()
    ax = sns.barplot(x=counts_df.index, y=counts_df['count'], ax=ax)
    fig.set_size_inches(10,5)
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=-90);
    ax.set_title('Label: '+category.upper())
    plt.show()
    return class_


def reduce_df(df, category, class_):
    '''
    Remove noise points from dataFrame
    category  :: dtpye=str, Category (attribute/column name)
    df        :: DataFrame to be loaded
    class_    :: list, classes to be seleted for the analysis
    
    returns: clean dataFrame
    '''
    print('Analysing {} as feature'.format(category))
    print('Prior Number of datapoints: {}  ::  Classes: {}'.format(len(df),
                                                                   set(df[category])))
    df = df[df[category].isin(class_)]
    df.reset_index(drop=True, inplace=True)
    print('Posterior Number of datapoints: {}  ::  Classes: {}\n'.format(len(df),
                                                                         set(df[category])))
    return df


# ## Analysing labels
# 
# > Minmum requirement: set minimum samples per class

# In[ ]:


for cate in df.keys()[1:-1]:
    class_ = disp_category(cate, df, min_items=700)
    df = reduce_df(df, cate, class_)


# In[ ]:


# > Posterior Unique values per column

df.nunique()


# ### EDA outcome

# > Sample images (distinct labels with all classes)

# In[ ]:


def grid_plot(title, grouped_df, samples=2):
    samples= len(grouped_df)
    item_id, img_path = grouped_df['id'].values, grouped_df['img_path'].values
    plt.figure(figsize=(7,15))
    
    for i in range(samples):
        plt.subplot(len(item_id) / samples + 1, 3, i + 1, title=title)
        plt.axis('off')
        plt.imshow(mpimg.imread(img_path[i]))
        

for cat in df.keys()[1:-1]:
    tmp_df = df.groupby(cat)
    for group_objects in tmp_df:
        title = '{}:\n  {}'.format(cat,group_objects[0])
        grouped_df = group_objects[1].sample(n=3)
        grid_plot(title, grouped_df)


# > Ready to use Dataframe

# In[ ]:


df.tail()


# ---
# ## <font color='teal'> Part 2</font>

# ### <font color='orange'>Modelling (CNN Based)</font>

# > Selection a fraction of data for further anaysis

# In[ ]:


analysis_df = df.sample(frac=0.95, random_state=10)
analysis_df.reset_index(drop=True, inplace=True)

labels = analysis_df.keys()[1:-1].values
N = len(analysis_df)

print('Total nuber of Data_points {}\nLabels {}'.format(N, labels))


# > Sample Image description

# In[ ]:


randm = np.random.randint(0, N)
img = mpimg.imread(analysis_df['img_path'][randm])

plt.figure(figsize=(7,7))
plt.imshow(img)
plt.title(str(analysis_df[labels].values[randm]))
plt.xlabel('Product_id :: {} \n Image shape :: {}'.format(str(analysis_df['id'][randm]), img.shape))
plt.show()


# > Functions to --
# 
# * Load & preprocess images
# 
# * Categorical conversions for labels

# In[ ]:


def load_image(path, shape=(112,112,3)):
    image_list = np.zeros((len(path), shape[0], shape[1], shape[2]))
    for i, fig in enumerate(path):
        img = image.load_img(fig, color_mode='rgb', target_size=shape)
        x = image.img_to_array(img).astype('float16')
        x = x / 255.0
        image_list[i] = x
    return image_list


def load_attr(df, attr, N=None, det=False, one_hot=True):
    le = LabelEncoder()
    le.fit(df[attr])
    target = le.transform(df[attr])
    if N is None:
        N = len(set(target))
    if one_hot:
        target = to_categorical(target, num_classes=N)
    if det:
        print('\n{}:: \n{}'.format(attr,le.classes_))
        print('Target shape', target.shape)
    return le.classes_, N, target


# > Data Partitioning

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nt = int(0.9*N)\n\n#----------------------------------------------------#----------------------------------------------------\n\nx_train = load_image(analysis_df['img_path'][:t])\n\ngen_names, num_gen, gender_tr = load_attr(analysis_df[:t], 'gender', det=True)\nsub_names, num_sub, subCategory_tr = load_attr(analysis_df[:t], 'subCategory', det=True)\nart_names, num_art, articleType_tr = load_attr(analysis_df[:t], 'articleType', det=True)\ncol_names, num_col, baseColour_tr = load_attr(analysis_df[:t], 'baseColour', det=True)\nsea_names, num_sea, season_tr = load_attr(analysis_df[:t], 'season', det=True)\nuse_names, num_use, usage_tr = load_attr(analysis_df[:t], 'usage', det=True)\n\n#----------------------------------------------------#----------------------------------------------------\nx_val = load_image(analysis_df['img_path'][t:-100])\n\n_, _, gender_val = load_attr(analysis_df[t:-100], 'gender', N=num_gen)\n_, _, subCategory_val = load_attr(analysis_df[t:-100], 'subCategory', N=num_sub)\n_, _, articleType_val = load_attr(analysis_df[t:-100], 'articleType', N=num_art)\n_, _, baseColour_val = load_attr(analysis_df[t:-100], 'baseColour', N=num_col)\n_, _, season_val = load_attr(analysis_df[t:-100], 'season', N=num_sea)\n_, _, usage_val = load_attr(analysis_df[t:-100], 'usage', N=num_use)\n\n#----------------------------------------------------#----------------------------------------------------\n# Last 100 images to be testset\n\n#----------------------------------------------------#----------------------------------------------------\n\ndict_ = {'gen_names' : gen_names.tolist(),\n         'sub_names' : sub_names.tolist(),\n         'art_names' : art_names.tolist(),\n         'col_names' : col_names.tolist(),\n         'sea_names' : sea_names.tolist(),\n         'use_names' : use_names.tolist()}\n\njson.dump(dict_, open('label_map.json', 'w'))\n\nprint('\\n Distinct classes (Per label):',num_gen, num_sub, num_art, num_col, num_sea, num_use)\nprint('Shape:: Train: {}, Val: {}'.format(x_train.shape, x_val.shape))")


# ## <font color='Teal'>Architecture</font>

# In[ ]:


class Classifier():
    '''
    Created on Fri Oct 18 18:02:02 2019
    Modified on Wed Oct 23 19:46:26 2019
    
    @author: Bhartendu
    
    Contains Multi-label Multi-class architecture
    
    ***Arguments***
    input_shape    :: Input shape, format : (img_rows, img_cols, channels), type='tuple'
    pre_model      :: Pretrained model file path, file extension : '.h5', type='str'
    
    ***Fuctions***
    build_model()  :: Define the CNN model (can be modified & tuned as per usecase)
    train_model()  :: Model Training (optimizers and metrics can be modified)
    eval_model()   :: Predict classes (in one-hot encoding)
    '''
    
    def __init__(self, input_shape=(112,112,3), pre_model=None):
        self.img_shape = input_shape

        # Load/Build Model
        if pre_model is None:
            self.cnn_model = self.build_model()
        else:
            try:
                self.cnn_model = load_model(pre_model)
            except OSError:
                print('Unable to load {}'.format(pre_model))

        # Compile Model
        self.cnn_model.compile(loss='categorical_crossentropy',
                               optimizer='adam',
                               metrics=['accuracy'])
        
        plot_model(self.cnn_model, to_file='consolidated_model.png')
        self.cnn_model.summary()


    def custom_residual_block(self, r, channels):
        r = BatchNormalization()(r)
        r = Conv2D(channels, (3, 3), activation='relu')(r)
        h = Conv2D(channels, (3, 3), padding='same', activation='relu')(r)
        h = Conv2D(channels, (3, 3), padding='same', activation='relu')(h)
        h = Conv2D(channels, (3, 3), padding='same', activation='relu')(h)
        return add([h, r])


    def build_model(self):
        input_layer = Input(shape=self.img_shape)
        
        # Conv Layers
        en_in = Conv2D(16, (5, 5), activation='relu')(input_layer)
        h = MaxPooling2D((3, 3))(en_in)
        
        h = self.custom_residual_block(h, 32)
        h = self.custom_residual_block(h, 32)
        
        h = self.custom_residual_block(h, 48)
        h = self.custom_residual_block(h, 48)

        h = self.custom_residual_block(h, 48)
        h = self.custom_residual_block(h, 48)
        
        h = self.custom_residual_block(h, 54)
        h = self.custom_residual_block(h, 54)
        en_out = self.custom_residual_block(h, 54)

        
        # Dense gender
        gen = self.custom_residual_block(en_out, 48)
        gen = GlobalAveragePooling2D()(gen)
        gen = Dense(100, activation='relu')(gen)
        gen = Dropout(0.1)(gen)
        gen = Dense(50, activation='relu')(gen)
        gen_out = Dense(num_gen, activation='softmax', name= 'gen_out')(gen)

        # Dense subCategory
        sub = self.custom_residual_block(en_out, 48)
        sub = GlobalAveragePooling2D()(sub)
        sub = Dense(100, activation='relu')(sub)
        sub = Dropout(0.1)(sub)
        sub = Dense(50, activation='relu')(sub)
        sub_out = Dense(num_sub, activation='softmax', name= 'sub_out')(sub)
        
        # Dense articleType
        art = self.custom_residual_block(en_out, 48)
        art = GlobalAveragePooling2D()(art)
        art = Dense(100, activation='relu')(art)
        art = Dropout(0.1)(art)
        art = Dense(50, activation='relu')(art)
        art_out = Dense(num_art, activation='softmax', name= 'art_out')(art)

        # Dense baseColour
        col = self.custom_residual_block(en_out, 48)
        col = GlobalAveragePooling2D()(col)
        col = Dense(100, activation='relu')(col)
        col = Dropout(0.1)(col)
        col = Dense(50, activation='relu')(col)
        col_out = Dense(num_col, activation='softmax', name= 'col_out')(col)
        
        # Dense season
        sea = self.custom_residual_block(en_out, 48)
        sea = GlobalAveragePooling2D()(sea)
        sea = Dense(100, activation='relu')(sea)
        sea = Dropout(0.1)(sea)
        sea = Dense(50, activation='relu')(sea)
        sea_out = Dense(num_sea, activation='softmax', name= 'sea_out')(sea)

        # Dense usage
        use = self.custom_residual_block(en_out, 48)
        use = GlobalAveragePooling2D()(use)
        use = Dense(100, activation='relu')(use)
        use = Dropout(0.1)(use)
        use = Dense(50, activation='relu')(use)
        use_out = Dense(num_use, activation='softmax', name='use_out')(use)

        return Model(input_layer, [gen_out, sub_out, art_out, col_out, sea_out, use_out])
    
    
    def train_model(self, x_train, y_train, x_val, y_val, save_path, epochs=25, batch_size=16):
        #
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                       patience=5, verbose=1,
                                       mode='auto', restore_best_weights=True)
        #
        check_pointer = ModelCheckpoint(save_path, monitor='val_loss',
                                        verbose=1, save_best_only=True,
                                        save_weights_only=False,
                                        mode='auto', period=1)
        #
        history = self.cnn_model.fit(x_train, y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    validation_data=(x_val, y_val),
                                    callbacks=[early_stopping, check_pointer])
        return history
    
    
    def eval_model(self, x_test):
        preds = self.cnn_model.predict(x_test)
        return preds


# ## <font color='Orange'>Training Starts..</font>
# 
# > Set Epochs & Batch Size

# In[ ]:


ce = Classifier(pre_model=None)

history  = ce.train_model(x_train, [gender_tr, subCategory_tr, articleType_tr,
                                    baseColour_tr, season_tr, usage_tr],
                          x_val, [gender_val, subCategory_val, articleType_val,
                                  baseColour_val, season_val, usage_val],
                          save_path='consolidated_model.h5',
                          epochs=25,
                          batch_size=30)


# ### Visualize Custom Model

# > Clothing Item Recognition Model
# ![Clothing Item Recognition Model](consolidated_model.png 'Consolidated Model')

# ### Learning Curves

# In[ ]:


def learning_curves(hist):
       '''
       Learing curves included (losses and accuracies per label)
       '''
       plt.style.use('ggplot')
       
       epochs = range(1,len(hist['loss'])+1)
       colors = ['b', 'g' ,'r', 'c', 'm', 'y']
       
       loss_ = [s for s in hist.keys() if 'loss' in s and 'val' not in s]
       val_loss_ = [s for s in hist.keys() if 'loss' in s and 'val' in s]
       acc_ = [s for s in hist.keys() if 'acc' in s and 'val' not in s]
       val_acc_ = [s for s in hist.keys() if 'acc' in s and 'val' in s]
       
       # Loss (per label)
       plt.figure(1, figsize=(20,10))
       for tr, val ,c in zip(loss_, val_loss_, colors):
           plt.plot(epochs, hist[tr], c,
                    label='train_{} : {}'.format(tr, str(format(hist[tr][-1],'.3f'))))
           
           plt.plot(epochs, hist[val], '-.'+c,
                    label='{} : {}'.format(val, str(format(hist[val][-1],'.3f'))))
           
       plt.title('Model Loss (Distinct labels)')
       plt.xlabel('Epochs')
       plt.ylabel('Loss')
       plt.legend(loc='upper right')
       plt.show()
       
       # Accuracy (per label)
       plt.figure(2, figsize=(20,10))
       for tr, val, c in zip(acc_, val_acc_, colors):
           plt.plot(epochs, hist[tr], c,
                    label='train_{} : {}'.format(tr, str(format(hist[tr][-1],'.3f'))))
           
           plt.plot(epochs, hist[val], '-.'+c,
                    label='{} : {}'.format(val, str(format(hist[val][-1],'.3f'))))
           
       plt.title('Model Accuracy (Distinct labels)')
       plt.xlabel('Epochs')
       plt.ylabel('Accuracy')
       plt.legend(loc='lower right')
       plt.show()
       
learning_curves(history.history)


# ---
# ## <font color='teal'> Part 3</font>

# ### Script for prediction

# In[ ]:


class Recognize_Item(object):
    '''
    Created on Wed Oct 23 21:38:51 2019

    @author: Bhartendu

    Summary:

        Ready-to-deploy class file for clothing-item-recogintion 

    Required:
        image_path: path or url of the image file [Required]
        model file: to be saved/updated at consolidated_model.h5
        label_dict: to be saved/updated at label_map.json
    '''

    def __init__(self, model_path='consolidated_model.h5'):
        self.model_path = model_path


    def load_model(self):
        '''
        Load Consolidated model from utils
        '''
        try:
            item_reco_model = load_model(self.model_path)
            return item_reco_model
        except Exception:
            raise Exception('#-----Failed to load model file-----#')


    def process_img(self, image_path):
        '''
        image: load & preprocess image file
        '''
        img = image.load_img(image_path, color_mode='rgb', target_size=(112,112,3))
        x = image.img_to_array(img).astype('float32')
        return x / 255.0


    def tmp_fn(self, one_hot_labels):
        '''
        tmp function to manpulate all one-hot encoded values labels to --
        -- a vector of predicted categories per labels
        '''
        flatten_labels = []
        for i in range(len(one_hot_labels)):
            flatten_labels.append(np.argmax(one_hot_labels[i], axis=-1)[0])
        return self.class_map(flatten_labels)


    def class_map(self, e):
        '''
        To convert class encoded values to actual label attributes
        '''
        label_map = json.load(open('label_map.json'))
        
        dict_ = {'gen_names' : gen_names.tolist(),
         'sub_names' : sub_names.tolist(),
         'art_names' : art_names.tolist(),
         'col_names' : col_names.tolist(),
         'sea_names' : sea_names.tolist(),
         'use_names' : use_names.tolist()}
        
        gender = label_map['gen_names'][e[0]]
        subCategory = label_map['sub_names'][e[1]]
        articleType = label_map['art_names'][e[2]]
        baseColour = label_map['col_names'][e[3]]
        season = label_map['sea_names'][e[4]]
        usage = label_map['use_names'][e[5]]
        #
        return [gender, subCategory, articleType, baseColour, season, usage]


    def predict_all(self, model, image):
        '''
        Predict all labels
        '''
        x_image = np.expand_dims(image, axis=0)
        return self.tmp_fn(model.predict(x_image))


    def demo_results(self, image_path, ground_truth=None):
        '''
        To demo sample predictions
        '''
        image = self.process_img(image_path)
        predicted_labels  = self.predict_all(model, image)
        #
        plt.figure(figsize=(5,7))
        plt.title('Fashion: Clothing Item recognition module')
        plt.text(0, 133, str('Predictions    : '+' | '.join(predicted_labels)),
                 fontsize=12, color='teal')
        #
        if ground_truth is None:
            pass
        else:
            plt.text(0, 142, str('Ground Truth : '+' | '.join(ground_truth)),
                     fontsize=12, color='b')
        # 
        plt.axis('off')
        plt.imshow(image)
        plt.show()


# ### Final Outcome

# In[ ]:


reco = Recognize_Item()

global model
model = reco.load_model()


# > Predictions vs ground_truth (from testset)

# In[ ]:


test_id = N-np.random.randint(0, 100)
reco.demo_results(analysis_df['img_path'][test_id],
                  ground_truth=analysis_df.ix[test_id,'gender':'usage'],)


# > Prediction for random image from testset

# In[ ]:


test_id = N-np.random.randint(0, 100)
reco.demo_results(analysis_df['img_path'][test_id])


# ##### Data courtesy:
# > [Small-Fashion Dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)
# 
# #### Bhartendu ([matrixB](www.linkedin.com/in/bhartendu-thakur-56bb6285/)), Machine Learning & Computing 
# ---
