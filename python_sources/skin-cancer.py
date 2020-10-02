#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
get_ipython().system('pip install keras')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import itertools


# In[ ]:


import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import keras.backend.common as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding


# In[ ]:


from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.applications.resnet50 import ResNet50
#keras.backend.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf')
from keras.layers import Input
keras.__version__


# In[ ]:


#1. Function to plot model's validation loss and validation accuracy
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()


# In[ ]:


base_skin_dir = '../input/skin-cancer-mnist-ham10000'

# Merging images from both folders HAM10000_images_part1.zip and HAM10000_images_part2.zip into one dictionary

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

# This dictionary is useful for displaying more human-friendly labels later on

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
lesion_danger = {
    'nv': 0, # 0 for benign
    'mel': 1, # 1 for malignant
    'bkl': 0, # 0 for benign
    'bcc': 1, # 1 for malignant
    'akiec': 1, # 1 for malignant
    'vasc': 0,
    'df': 0
}


# In[ ]:


skin_df = pd.read_csv('../input/skin-cancer-mnist-ham10000/HAM10000_metadata.csv')

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes


# In[ ]:


#brief overview of table
skin_df.head()


# In[ ]:


#check for missing values on each column
skin_df.isnull().sum()


# In[ ]:


##For the null values, use mean values
skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)


# In[ ]:


#Resizing
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))


# In[ ]:


skin_df["Malignant"] = skin_df["dx"].map(lesion_danger.get)


# In[ ]:


skin_df.sample(3)


# In[ ]:


skin_df.iloc[0]["image"]


# In[ ]:


skin_df.iloc[6500]["path"]


# In[ ]:


skin_df["image"].map(lambda x: x.shape).value_counts()


# In[ ]:


n_samples = 5 # choose 5 samples for each cell type
fig, m_axs = plt.subplots(7, n_samples, figsize=(4*n_samples, 3 * 7))

for n_axs, (type_name, type_rows) in zip(m_axs, skin_df.sort_values(["cell_type"]).groupby("cell_type")):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=0).iterrows()):
        c_ax.imshow(c_row["image"])
        c_ax.axis("off")
fig.savefig("category_samples.png", dpi=300)


# In[ ]:


# create a pandas dataframe to store mean value of Red, Blue and Green for each picture
rgb_info_df = skin_df.apply(lambda x: pd.Series({'{}_mean'.format(k): v for k, v 
                                                 in zip(["Red", "Blue", "Green"], 
                                                        np.mean(x["image"], (0, 1)))}), 1)


gray_col_vec = rgb_info_df.apply(lambda x: np.mean(x), 1) # take the mean value across columns of rgb_info_df
for c_col in rgb_info_df.columns:
    rgb_info_df[c_col] = rgb_info_df[c_col]/gray_col_vec 
rgb_info_df["Gray_mean"] = gray_col_vec
rgb_info_df.sample(3)


# In[ ]:


for c_col in rgb_info_df.columns:
    skin_df[c_col] = rgb_info_df[c_col].values


# In[ ]:


# let's draw a plot showing the distribution of different cell types over colors!
sns.pairplot(skin_df[["Red_mean", "Green_mean", "Blue_mean", "Gray_mean", "cell_type"]], 
             hue="cell_type", plot_kws = {"alpha": 0.5})


# In[ ]:


skin_df.head()


# In[ ]:


import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import os
import shutil
import warnings
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix, f1_score, precision_score,                             recall_score, accuracy_score, classification_report

import seaborn as sns; sns.set()

from skimage import io, exposure, morphology, filters, color,                     segmentation, feature, measure, img_as_float, img_as_ubyte
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
warnings.simplefilter("ignore")
from scipy import ndimage as ndi


# In[ ]:


def img_segmentation(image):
  idx = 3
  plot_limit = 10015
  
#image_name = 'bresto'
  #segmented_img_set = []
# read image
  #image = io.imread(img_path)
        # convert the original image into grayscale
  gray_img = color.rgb2gray(image)

        # 1] Apply Sobel filter
  elevation_map = filters.sobel(gray_img)

        # 2] Build image markers using the threshold obtained through the ISODATA filter
  markers = np.zeros_like(gray_img)
  threshold = filters.threshold_isodata(gray_img)
  markers[gray_img > threshold] = 1
  markers[gray_img < threshold] = 2
        
# 3] Apply Wathershed algorithm in order to segment the image filtered
#    using the markers
  segmented_img = morphology.watershed(elevation_map, markers)
# 4] Improve segmantation:
#    >  Fill small holes 
  segmented_img = ndi.binary_fill_holes(segmented_img - 1)
#    > Remove small objects that have an area less than 800 px:
        #      this could be useful to exclude some regions that does not represent a lesion
  segmented_img = morphology.remove_small_objects(segmented_img, min_size=800)
        #    > Clear regions connected to the image borders.
        #      This operation is very useful when there are contour regions have a
        #      big area and so they can be exchanged with the lesion.
        #      However, this can also create some issues when the lesion region is
        #      connected to the image borders. In order to (try to) overcome this
        #      issue, we use a lesion identification algorithm (see below)
  img_border_cleared = segmentation.clear_border(segmented_img)

# 5] Apply connected components labeling algorithm:
#    it assigns labels to a pixel such that adjacent pixels of the same
#    features are assigned the same label.
# labeled_img, _ = ndi.label(segmented_img)
  labeled_img = morphology.label(img_border_cleared)
# 6] Lesion identification algorithm:
        # Compute properties of labeled image regions:
        # it will be used to automatically select the region that contains
        # the skin lesion according to area and extent
  props = measure.regionprops(labeled_img)
        # num labels -> num regions
  num_labels = len(props)
# Get all the area of detected regions
  areas = [region.area for region in props]

        # If we have at least one region and the area of the region having the
        # biggest area is at least 1200 px, we choose it as the region that
        # contains the leson because if properly segmented (i.e., after removing
        # small objects and regions on the image contours (since in most of the
        # images, the lesion is in the center))
  if num_labels > 0 and areas[np.argmax(areas)] >= 1200:
    
    target_label = props[np.argmax(areas)].label
  else:
    
    labeled_img = morphology.label(segmented_img)
  # Get new region properties
    props = measure.regionprops(labeled_img)
  # Get the new list of areas
    areas = [region.area for region in props]
  # List of regions' extent.
  # Each extent is defined as the ratio of pixels in the region  to pixels
  # in the total bounding box (computed as: area / (rows * cols))
    extents = [region.extent for region in props]
  # Get the index of the region having the largest area and if there are
  # more than one or two regions, find also the index of the second and
  # third most largest regions.
    region_max1 = np.argmax(areas)
    if len(props) > 1:
      
      areas_copy = areas.copy()
      areas_copy[region_max1] = 0
      region_max2 = np.argmax(areas_copy)
    if len(props) > 2:
    
      areas_copy[region_max2] = 0
      region_max3 = np.argmax(areas_copy)

            # If the largest region has an extent greater than 0.50, it is our target region
    if extents[region_max1] > 0.50:

      target_label = props[region_max1].label

            # ... else check if the extent of the second largest region is greater than 0.5,
            # and if so we have found our target region
    elif len(props) > 1 and extents[region_max2] > 0.50:
      
      target_label = props[region_max2].label
            # ... else if the third largest region has an extent greater than 0.50,
            # it is (more probably) the one containing the lesion
    elif len(props) > 2 and extents[region_max3] > 0.50:

      target_label = props[region_max3].label
            # ... otherwise we choose the largest region
    else:
      target_label = props[region_max1].label
# assign label 0 to all the pixels that are not in the target region (that is
        # the ragion that more probably contains the lesion)
  for row, col in np.ndindex(labeled_img.shape):

    if labeled_img[row, col] != target_label:

      labeled_img[row, col] = 0
        # Convert the labeled image into its RGB version
  image_label_overlay = color.label2rgb(labeled_img, gray_img)
  if idx < plot_limit:
    
    print('Chosen label: {}'.format(target_label))
  # Plot the original image ('image') in which the contours of all the
  # segmented regions are highlighted
    #fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    #axes[0].imshow(image)
    #axes[0].contour(segmented_img, [0.5], linewidths=1.2, colors='y')
    #axes[0].axis('off')
  # Plot 'image_label_overlay' that contains the target region highlighted
    #axes[1].imshow(image_label_overlay)
    #axes[1].axis('off')

    #plt.tight_layout()
    #plt.show();

  #segmented_img_set.append(props[target_label - 1])
  
  return props[target_label - 1]
  


# In[ ]:


#sk_df = skin_df.drop([skin_df.index[340],skin_df.index[403],skin_df.index[426],skin_df.index[1074],skin_df.index[1118],skin_df.index[1134]])


# In[ ]:


sk_df.iloc[5870]["path"]


# In[ ]:


sk_df = skin_df.drop([skin_df.index[340]])


# In[ ]:


sk_df.drop(sk_df.index[403], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[424], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[425], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[1073], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[1117], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[1133], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[1133], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[2161], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[9727], inplace=True)


# In[ ]:


sk_df.index[sk_df['path'] =='../input/skin-cancer-mnist-ham10000/HAM10000_images_part_1/ISIC_0027662.jpg' ]


# In[ ]:


sk_df.drop(sk_df.index[9695], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[6923], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[6902], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[6842], inplace=True)


# In[ ]:


sk_df.drop(sk_df.index[6804], inplace=True)


# In[ ]:


sg = []
label = []
for index in sk_df.index.values.tolist()[5867:]:
    path = sk_df.iloc[index]["path"]
    img = np.asarray(Image.open(path))
    img = img_segmentation(img)
    cell_type_id = sk_df.iloc[index]["cell_type_idx"]
    sg.append(img)
    label.append(cell_type_id)
    print(index)
    print(path)


# In[ ]:


len(names_list)


# In[ ]:


names_list = []
for index in sk_df.index.values.tolist()[5867:]:
    pato = sk_df.iloc[index]["path"]
    names_list.append(pato)


# In[ ]:


len(sg)


# In[ ]:


len(label)


# In[ ]:


names_list[4117]


# In[ ]:


len(names_list)


# In[ ]:


dictionary = dict(zip(names_list, sg))


# In[ ]:


len(dictionary)


# In[ ]:


skin_df.iloc[1]["path"]


# In[ ]:


iterator = iter(dictionary.items())
for i in range(3):
    print(next(iterator))


# In[ ]:


len(label)


# In[ ]:


skin_df['cell_type_idx'].value_counts()


# In[ ]:


def features_extraction(dictionaryyy, image_paths ,out_df=True ):

    print('-- FEATURE EXTRACTION --')
    plot_limit = 10015
    if out_df:
        train_list = [] 
        train_index = []
        
    

    

    segmented_regions = {**dictionaryyy}  # Python 3.5
    # segmented_regions = {k: v for d in (segmented_region_train, segmented_region_te) for k, v in d.items()}
    for idx, image_name in enumerate(segmented_regions):
        if idx < plot_limit:
            print('{:_<100}'.format(''))
            print('Image name: {}'.format(image_name))
        elif idx == plot_limit:
            print('\nContinuing feature extraction without printing the results ...')
        if image_name in image_paths:
          image_path = image_name

        image = io.imread(image_path)
        gray_img = color.rgb2gray(image)

        lesion_region = segmented_regions[image_name]

        # 1] ASYMMETRY
        area_total = lesion_region.area
        img_mask = lesion_region.image

        horizontal_flip = np.fliplr(img_mask)
        diff_horizontal = img_mask * ~horizontal_flip

        vertical_flip = np.flipud(img_mask)
        diff_vertical = img_mask * ~vertical_flip

        diff_horizontal_area = np.count_nonzero(diff_horizontal)
        diff_vertical_area = np.count_nonzero(diff_vertical)
        asymm_idx = 0.5 * ((diff_horizontal_area / area_total) + (diff_vertical_area / area_total))
        ecc = lesion_region.eccentricity
        # mmr = lesion_region.minor_axis_length / lesion_region.major_axis_length

        if idx < plot_limit:
            print('-- ASYMMETRY --')
            print('Diff area horizontal: {:.3f}'.format(np.count_nonzero(diff_horizontal)))
            print('Diff area vertical: {:.3f}'.format(np.count_nonzero(diff_vertical)))
            print('Asymmetric Index: {:.3f}'.format(asymm_idx))
            print('Eccentricity: {:.3f}'.format(ecc))
            # print('Minor-Major axis ratio: {}'.format(mmr))
        
            #imshow_all(img_mask, horizontal_flip, diff_horizontal,
                       #titles=['image mask', 'horizontal flip', 'difference'], size=4, cmap='gray')
            #imshow_all(img_mask, vertical_flip, diff_vertical,
                       #titles=['image mask', 'vertical flip', 'difference'], size=4, cmap='gray')
            #plt.show();

        # 2] Border irregularity:
        compact_index = (lesion_region.perimeter ** 2) / (4 * np.pi * area_total)
        if idx < plot_limit:
            print('\n-- BORDER IRREGULARITY --')
            print('Compact Index: {:.3f}'.format(compact_index))

        # 3] Color variegation:
        sliced = image[lesion_region.slice]
        lesion_r = sliced[:, :, 0]
        lesion_g = sliced[:, :, 1]
        lesion_b = sliced[:, :, 2]

        C_r = np.std(lesion_r) / np.max(lesion_r)
        C_g = np.std(lesion_g) / np.max(lesion_g)
        C_b = np.std(lesion_b) / np.max(lesion_b)

        if idx < plot_limit:
            print('\n-- COLOR VARIEGATION --')
            print('Red Std Deviation: {:.3f}'.format(C_r))
            print('Green Std Deviation: {:.3f}'.format(C_g))
            print('Blue Std Deviation: {:.3f}'.format(C_b))
            # imshow_all(lesion_r, lesion_g, lesion_b)
            # plt.show();

        # Alternative method to compute colorfulness:
        # https://www.pyimagesearch.com/2017/06/05/computing-image-colorfulness-with-opencv-and-python/
        # compute rg = Red - Green
        # rg = np.absolute(lesion_r - lesion_g)
        # compute yb = 0.5 * (Red + Green) - Blue
        # yb = np.absolute(0.5 * (lesion_r + lesion_g) - lesion_b)
        #
        # compute the mean and standard deviation of both `rg` and `yb`
        # (rb_mean, rb_std) = (np.mean(rg), np.std(rg))
        # (yb_mean, yb_std) = (np.mean(yb), np.std(yb))
        #
        # combine the mean and standard deviations
        # std_root = np.sqrt((rb_std ** 2) + (yb_std ** 2))
        # mean_root = np.sqrt((rb_mean ** 2) + (yb_mean ** 2))
        #
        # derive the "colorfulness" metric (color index)
        # color_index = std_root + (0.3 * mean_root)

        # 4] Diameter:
        eq_diameter = lesion_region.equivalent_diameter
        if idx < plot_limit:
            print('\n-- DIAMETER --')
            print('Equivalent diameter: {:.3f}'.format(eq_diameter))
            # optionally convert the diameter in mm, knowing that 1 px = 0.265 mm:
            # 1 px : 0.265 mm = diam_px : diam_mm -> diam_mm = diam_px * 0.265
            print('Diameter (mm): {:.3f}'.format(eq_diameter * 0.265))

        # 5] Texture:
        glcm = feature.greycomatrix(image=img_as_ubyte(gray_img), distances=[1],
                                    angles=[0, np.pi/4, np.pi/2, np.pi * 3/2],
                                    symmetric=True, normed=True)

        correlation = np.mean(feature.greycoprops(glcm, prop='correlation'))
        homogeneity = np.mean(feature.greycoprops(glcm, prop='homogeneity'))
        energy = np.mean(feature.greycoprops(glcm, prop='energy'))
        contrast = np.mean(feature.greycoprops(glcm, prop='contrast'))

        if idx < plot_limit:

            print('\n-- TEXTURE --')
            print('Correlation: {:.3f}'.format(correlation))
            print('Homogeneity: {:.3f}'.format(homogeneity))
            print('Energy: {:.3f}'.format(energy))
            print('Contrast: {:.3f}'.format(contrast))

        if image_name in dictionaryyy:
          
          if out_df:
            data_list = train_list
            train_index.append(image_name)
            
        if out_df:
            data_list.append([asymm_idx, ecc, compact_index, C_r, C_g, C_b,
                              eq_diameter, correlation, homogeneity, energy, contrast])
        
    if out_df:
        attr = ['AsymIdx', 'Eccentricity', 'CI', 'StdR', 'StdG', 'StdB', 
                'Diameter', 'Correlation', 'Homogeneity', 'Energy', 'Contrast']
        sk_df = pd.DataFrame(data=data_list, index=train_index, columns=attr)
        
        return sk_df


# In[ ]:


ski_df = features_extraction(dictionary, names_list)


# In[ ]:


len(ski_df)


# In[ ]:


ski_df.head()


# In[ ]:


se = pd.Series(label)
ski_df['label'] = se.values


# In[ ]:


ski_df.head()


# In[ ]:


ski_df['label'].value_counts()


# In[ ]:


out_path = "features_extracted_vsuite.csv"
ski_df.to_csv(out_path, index=True)


# In[ ]:


skino_df = pd.read_csv('../input/met-data/features_extracted_v5871.csv')


# In[ ]:


skia_df = pd.read_csv('../input/metaaaa/features_extracted_vsuite.csv')


# In[ ]:


skia_df.rename(columns={'Unnamed: 0': 'path'}, inplace=True)


# In[ ]:


skino_df.rename(columns={'Unnamed: 0': 'path'}, inplace=True)


# In[ ]:


skino_df.iloc[5700]


# In[ ]:


frames = [skino_df,skia_df]


# In[ ]:


result_df = pd.concat(frames)


# In[ ]:


result_df.columns


# In[ ]:


result_df.shape


# In[ ]:


result_df['label'].value_counts()


# In[ ]:


X = result_df.drop(columns=['label','path'])
y = result_df['label']


# In[ ]:


X.head()


# In[ ]:


from collections import Counter
print(sorted(Counter(y_resampled).items()))


# In[ ]:


X_resampled.shape()


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1234)


# In[ ]:


from imblearn.over_sampling import SMOTE, ADASYN
X_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)


# In[ ]:


X_resampled.shape


# In[ ]:


y_resampled.shape


# In[ ]:


y_resampled = to_categorical(y_resampled, num_classes = 7)
#y_test = to_categorical(y_test, num_classes = 7)


# In[ ]:


x_train.shape


# In[ ]:


X_resampled, x_validate, y_resampled, y_validate = train_test_split(X_resampled, y_resampled, test_size = 0.1, random_state = 2)


# In[ ]:


from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


# In[ ]:


n_cols = X_resampled.shape[1]
def create_model(dropout_rate,init):
    model = Sequential()
    model.add(Dense(256,input_dim = 11,kernel_initializer = init,activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256,kernel_initializer = init,activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(7,activation = 'softmax'))
    
    adam = Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model

# Create the model

model = KerasClassifier(build_fn = create_model,verbose = 0)


# In[ ]:


n_cols


# In[ ]:


# Define the grid search parameters

batch_size = [10,20,40]
epochs = [50,100]
#learning_rate = [0.001,0.01,0.1]
dropout_rate = [0.0,0.1,0.2]
#activation_function = ['softmax','relu','tanh','linear']
init = ['uniform','normal','zero']
#neuron1 = [128,256]
#neuron2 = [128,256]

# Make a dictionary of the grid search parameters

param_grids = dict(batch_size = batch_size,epochs = epochs,dropout_rate = dropout_rate,
                   init = init)

# Build and fit the GridSearchCV

grid = GridSearchCV(estimator = model,param_grid = param_grids,cv = KFold(),verbose = 10)
grid_result = grid.fit(X_resampled,y_resampled)

# Summarize the results

print('Best : {}, using {}'.format(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{},{} with: {}'.format(mean, stdev, param))


# In[ ]:


#create model
model = Sequential()

#get number of columns in training data
n_cols = X_resampled.shape[1]

#add layers to model
model.add(Dense(250, activation='relu', input_shape=(n_cols,)))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.summary()


# In[ ]:


optimizer = Adam(lr=0.001)


# In[ ]:


model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=['Recall'])


# In[ ]:


from keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
#train model
model.fit(X_resampled, y_resampled, validation_data=(x_validate,y_validate), epochs=100,verbose = 1)
#callbacks=[early_stopping_monitor]


# In[ ]:


# Function to plot confusion matrix    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(x_validate)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_validate,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

 

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(7)) 


# In[ ]:


get_ipython().system('pip install imutils')


# In[ ]:


from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import scipy.misc as im
import seaborn as sns
import numpy as np
#import imutils
import cv2
from matplotlib import pyplot as plt


# In[ ]:


def crop_cancer_contour(image, plot=False):
    
    
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]] 
    #img = cv2.resize(new_image, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
    #img = img / 255.
    
    return new_image


# In[ ]:


result_df.iloc[9500]["path"]


# In[ ]:


s = np.asarray(Image.open('../input/skin-cancer-mnist-ham10000/HAM10000_images_part_2/ISIC_0034229.jpg'))
plt.imshow(s)


# In[ ]:


m = crop_cancer_contour(s)


# In[ ]:


from skimage import io, color


# In[ ]:



lab = color.rgb2yiq(s)
plt.imshow(lab)
lab.shape


# In[ ]:


X = []
y = []
for index in result_df.index.values.tolist():
    path = result_df.iloc[index]["path"]
    labe = result_df.iloc[index]["label"]
    img = np.asarray(Image.open(path))
    #img = crop_cancer_contour(img)
    img = color.rgb2hsv(img)
    img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    #img = img / 255.
    X.append(img)
    y.append(labe)


# In[ ]:


X = np.array(X)
y = np.array(y)
y.shape


# In[ ]:


plt.imshow(X[200])
X.shape


# In[ ]:


features = model.predict(X, batch_size=32)


# In[ ]:


features[0]


# In[ ]:


np.save("features_map_RGB_resnet50.npy",features)


# In[ ]:


features_flatten = features.reshape((features.shape[0], 4 * 4 * 512))


# In[ ]:


features_flatten[0]


# In[ ]:


f_rgb = np.load("../input/feature-map-v2/features_map_rgb.npy")


# In[ ]:


f_rgb[0]


# In[ ]:


features_flatten_rgb = f_rgb.reshape((f_rgb.shape[0], 4 * 4 * 512))


# In[ ]:


features_flatten_hsv = f_hsv.reshape((f_hsv.shape[0], 4 * 4 * 512))


# In[ ]:


features_flatten_lab = f_lab.reshape((f_lab.shape[0], 4 * 4 * 512))


# In[ ]:


f_map = np.concatenate((features_flatten_rgb,features_flatten_hsv,features_flatten_lab), axis=1)


# In[ ]:


f_map.shape


# In[ ]:


abcdt_feature = result_df.drop(columns=['label','path'])


# In[ ]:


abcdt_feature = np.array(abcdt_feature)


# In[ ]:


abcdt_feature.shape[1]


# In[ ]:


f_map_all = np.concatenate((f_map,abcdt_feature),axis = 1)


# In[ ]:


model = LogisticRegression(class_weight='balanced', multi_class="auto",max_iter=200, random_state=1,C = 0.01, solver = 'newton-cg',penalty ='l2')
model.fit(x_train, y_train)


# In[ ]:


model.score(x_test, y_test)
preds = model.predict(x_test)
print("\nAccuracy on Test Data: ", accuracy_score(y_test, preds))
print("\nNumber of correctly identified imgaes: ",
accuracy_score(y_test, preds, normalize=False),"\n")
confusion_matrix(y_test, preds, labels=range(0,7))


# In[ ]:


f_hsv = np.load("../input/feature-map-v2/features_map_hsv.npy")


# In[ ]:


f_lab = np.load("../input/feature-map-v2/features_map_lab.npy")


# In[ ]:


f_map = f_map.reshape((f_map.shape[0],12,4,512))


# In[ ]:


f_map_all = f_map_all.reshape((f_map_all.shape[0],12,4,512))


# In[ ]:


y.shape


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.20,random_state=1234)


# In[ ]:


x_train.shape


# In[ ]:


y_train = to_categorical(y_train,num_classes=7)


# In[ ]:


x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)


# In[ ]:


x_train.shape[1:]


# In[ ]:


model_transfer = Sequential()
model_transfer.add(GlobalAveragePooling2D(input_shape=x_train.shape[1:]))
model_transfer.add(Dropout(0.2))
model_transfer.add(Dense(512, activation='relu'))
model_transfer.add(Dropout(0.3))
model_transfer.add(Dense(100, activation='relu'))
model_transfer.add(Dropout(0.5))
model_transfer.add(Dense(7, activation='softmax'))
model_transfer.summary()


# In[ ]:


checkpointer = ModelCheckpoint(filepath='mobile_rgb.best.hdf5',
                               verbose=1,save_best_only=True)
optimizer = Adam(lr=0.01)


# In[ ]:


model_transfer.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])
history = model_transfer.fit(x_train, y_train, batch_size=32, epochs=30,
          validation_data=(x_validate, y_validate), callbacks=[checkpointer]
          ,verbose=1)


# In[ ]:


preds = np.argmax(model_transfer.predict(x_test), axis=1)
print("\nAccuracy on Test Data: ", accuracy_score(y_test, preds))
print("\nNumber of correctly identified imgaes: ",
      accuracy_score(y_test, preds, normalize=False),"\n")
confusion_matrix(y_test, preds, labels=range(0,7))


# In[ ]:


plt.imshow(X[0])


# In[ ]:


from keras.applications import VGG16


# In[ ]:


from keras.applications.inceptio import InceptionV3
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2


# In[ ]:


model = ResNet50(weights="imagenet",input_shape=(128,128,3), include_top=False)
model.summary()


# In[ ]:


for layer in model.layers:
    print(layer.name)
    if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
        layer.trainable = True
        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
    else:
        layer.trainable = False

print(len(model.layers))


# In[ ]:


last_layer = model.get_layer('conv5_block3_out')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


# In[ ]:


from keras import layers
from keras import Model


# In[ ]:


# Flatten the output layer to 1 dimension
x = layers.GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(7, activation='softmax')(x)

# Configure and compile the model

model_1 = Model(model.input, x)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model_1.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1234)


# In[ ]:


y_train = to_categorical(y_train,num_classes=7)


# In[ ]:


x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)


# In[ ]:


train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, fill_mode='nearest')
train_datagen.fit(x_train)

val_datagen = ImageDataGenerator()
val_datagen.fit(x_validate)


# In[ ]:


batch_size = 64
epochs = 3
history = model_1.fit_generator(train_datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(x_validate, y_validate),
                              verbose = 1, steps_per_epoch=(x_train.shape[0] // batch_size), 
                              validation_steps=(x_validate.shape[0] // batch_size))


# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; 
 
# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
 
# Importing hypopt library for grid search
#from hypopt import GridSearch
 
# Importing Keras libraries
from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


IMG_SHAPE = X[0].shape
IMG_SHAPE


# In[ ]:


# load the VGG16 network
print("[INFO] loading network...")
 
# chop the top dense layers, include_top=False
model = VGG16(weights="imagenet",input_shape=(128,128,3), include_top=False)
model.summary()


# In[ ]:


def create_features(dataset, pre_model):
 
    x_scratch = []
 
    # loop over the images
    for i in range(len(dataset)):
 
        # load the input image and image is resized to 224x224 pixels
        #image = load_img(imagePath, target_size=(224, 224))
        #image = img_to_array(image)
 
        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = dataset[i]
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
 
        # add the image to the batch
        x_scratch.append(image)
 
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 4 * 4 * 512))
    return x, features, features_flatten


# In[ ]:


data, X_features, X_features_flatten = create_features(X, model)


# In[ ]:


print(data.shape, X_features.shape, X_features_flatten.shape)


# In[ ]:


out_vec = np.stack(X_features_flatten, 0)


# In[ ]:


out_vec.shape


# In[ ]:


out_df = pd.DataFrame(out_vec)


# In[ ]:


out_path = "features_128_128_HSV.csv"
out_df.to_csv(out_path, index=True)


# In[ ]:


result_df.columns


# In[ ]:


StdG = result_df['StdG']
out_df = out_df.join(StdG)


# In[ ]:


out_df.columns


# In[ ]:


X_train_orig, X_test, y_train_orig, y_test = train_test_split(out_vec, y, test_size=0.1,random_state=0)


# In[ ]:


out_vec = out_vec.astype("float32")


# In[ ]:


np.save("256_192_test.npy", X_test)
np.save("test_labels.npy", y_test)


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train_orig, y_train_orig, test_size=0.1, random_state=1)


# In[ ]:


np.save("256_192_val.npy", X_val)
np.save("val_labels.npy", y_val)


# In[ ]:


np.save("256_192_train.npy", X_train)
#np.save("train_labels.npy", y_train)


# In[ ]:


import os
from glob import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[ ]:


X_train = np.load("./256_192_train.npy")


# In[ ]:


y_train = np.load("../input/moredata/train_labels.npy")


# In[ ]:


X_val = np.load("../input/moredata/256_192_val.npy")
y_val = np.load("../input/moredata/val_labels.npy")


# In[ ]:


X_train.shape, X_val.shape


# In[ ]:


y_train.shape, y_val.shape


# In[ ]:


y_train = to_categorical(y_train)
y_val = to_categorical(y_val)


# In[ ]:


pre_trained_model = InceptionV3(input_shape=(192, 256, 3), include_top=False, weights="imagenet")


# In[ ]:


for layer in pre_trained_model.layers:
    print(layer.name)
    layer.trainable = False
    
print(len(pre_trained_model.layers))


# In[ ]:


last_layer = pre_trained_model.get_layer('mixed10')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


# In[ ]:


# Flatten the output layer to 1 dimension
x = layers.GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)
# Add a dropout rate of 0.7
x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(7, activation='softmax')(x)

# Configure and compile the model

model = Model(pre_trained_model.input, x)
optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])


# In[ ]:


train_datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, fill_mode='nearest')

train_datagen.fit(X_train)

val_datagen = ImageDataGenerator()
val_datagen.fit(X_val)


# In[ ]:


batch_size = 64
epochs = 3
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size), 
                              validation_steps=(X_val.shape[0] // batch_size))


# In[ ]:


for layer in pre_trained_model.layers:
    layer.trainable = True


# In[ ]:


optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])


# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, 
                                            min_lr=0.000001, cooldown=2)


# In[ ]:


batch_size = 64
epochs = 20
history = model.fit_generator(train_datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = val_datagen.flow(X_val, y_val),
                              verbose = 1, steps_per_epoch=(X_train.shape[0] // batch_size),
                              validation_steps=(X_val.shape[0] // batch_size),
                              callbacks=[learning_rate_reduction])


# In[ ]:


get_ipython().system('pip install imutils')


# In[ ]:


from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import scipy.misc as im
import seaborn as sns
import numpy as np
import imutils
import cv2
from matplotlib import pyplot as plt


# In[ ]:


def crop_cancer_contour(image, plot=False):
    
    
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]] 
    #img = cv2.resize(new_image, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
    #img = img / 255.
    
    return new_image


# In[ ]:


#reshaped_image = skin_df["path"].map(lambda x: crop_cancer_contour(np.asarray(Image.open(x))))


# In[ ]:


skin_df.head()


# In[ ]:


X = []
y = []
for index in skin_df.index.values.tolist():
    path = skin_df.iloc[index]["path"]
    cell_type_id = skin_df.iloc[index]["cell_type_idx"]
    img = np.asarray(Image.open(path))
    img = crop_cancer_contour(img)
    img = cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
    #img = img / 255.
    X.append(img)
    y.append(cell_type_id)


# In[ ]:


X = np.array(X)
y = np.array(y)
X.shape


# In[ ]:


np.save("224_224_data.npy", X)
np.save("224_224_label.npy", y)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1234)


# In[ ]:


y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)


# In[ ]:


x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)


# In[ ]:


from keras.applications import VGG16


# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; 
 
# Importing sklearn libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
 
# Importing hypopt library for grid search
#from hypopt import GridSearch
 
# Importing Keras libraries
from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


IMG_SHAPE = x_train[0].shape
IMG_SHAPE


# In[ ]:


# load the VGG16 network
print("[INFO] loading network...")
 
# chop the top dense layers, include_top=False
model = VGG16(weights="imagenet",input_shape=IMG_SHAPE, include_top=False)
model.summary()


# In[ ]:


import tensorflow as tf

base_model = tf.keras.applications.MobileNet(input_shape=IMG_SHAPE, include_top=False)
base_model.summary()


# In[ ]:


def create_features(dataset, pre_model):
 
    x_scratch = []
 
    # loop over the images
    for i in range(len(dataset)):
 
        # load the input image and image is resized to 224x224 pixels
        #image = load_img(imagePath, target_size=(224, 224))
        #image = img_to_array(image)
 
        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = dataset[i]
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
 
        # add the image to the batch
        x_scratch.append(image)
 
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 2 * 2 * 512))
    return x, features, features_flatten


# In[ ]:


val_x, val_features, val_features_flatten = create_features(x_validate, model)


# In[ ]:


test_x, test_features, test_features_flatten = create_features(x_test, model)


# In[ ]:


train_x, train_features, train_features_flatten = create_features(x_train, model)


# In[ ]:





# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1234)


# In[ ]:


x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std


# In[ ]:


y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)


# In[ ]:


x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)


# In[ ]:


x_train = x_train.reshape(x_train.shape[0], *(64, 64, 3))
x_test = x_test.reshape(x_test.shape[0], *(64, 64, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(64, 64, 3))


# feature map of RGB SPACE COLOR FROM MOBILE V2 MODEL

# In[ ]:





# In[ ]:


# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
input_shape = (64, 64, 3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[ ]:


# Define the optimizer
optimizer = Adam(lr=0.001)


# In[ ]:


# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:


# With data augmentation to prevent overfitting 

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)


# In[ ]:


# Fit the model
epochs = 50 
batch_size = 10
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_validate,y_validate),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df['cell_type_idx']


# In[ ]:


skin_df['image'].map(lambda x: x.shape).value_counts()


# In[ ]:


x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234)


# In[ ]:


x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std


# In[ ]:


# Perform one-hot encoding on the labels
y_train = to_categorical(y_train_o, num_classes = 7)
y_test = to_categorical(y_test_o, num_classes = 7)


# In[ ]:


x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)


# In[ ]:


# Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))


# In[ ]:


# Set the CNN model 
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
input_shape = (75, 100, 3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape))
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[ ]:


# Define the optimizer
optimizer = Adam(lr=0.001)


# In[ ]:


# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# In[ ]:





# In[ ]:


# With data augmentation to prevent overfitting 

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)


# In[ ]:


# Fit the model
epochs = 50 
batch_size = 10
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_validate,y_validate),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)


# In[ ]:


loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(x_validate, y_validate, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
model.save("model.h5")


# In[ ]:


plot_model_history(history)


# In[ ]:


# Function to plot confusion matrix    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(x_validate)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_validate,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

 

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(7)) 


# In[ ]:


feature_rgb = pd.read_csv('../input/feature-map-from-spaces/fetures_128_128_RBG.csv')


# In[ ]:


feature_hsv = pd.read_csv('../input/feature-map-from-spaces/features_128_128_HSV.csv')


# In[ ]:


feature_lab = pd.read_csv('../input/feature-map-from-spaces/features_128_128_LAB.csv')


# In[ ]:


feature_rgb.columns


# In[ ]:


del feature_rgb['Unnamed: 0']


# In[ ]:


del feature_lab['Unnamed: 0']


# In[ ]:


del feature_hsv['Unnamed: 0']


# In[ ]:


feature_rgb.columns = [str(col) + '_RGB' for col in feature_rgb.columns]


# In[ ]:


feature_lab.columns = [str(col) + '_lab' for col in feature_lab.columns]


# In[ ]:


feature_hsv.columns = [str(col) + '_hsv' for col in feature_hsv.columns]


# In[ ]:


feature_hsv.columns


# In[ ]:


feature_spaces = pd.concat([feature_rgb,feature_hsv,feature_lab], axis=1)


# In[ ]:


labe = result_df['label'].to_list()


# In[ ]:


feature_rgb['label'] = labe


# In[ ]:


feature_spaces['label'] = labe


# In[ ]:


feature_spaces.shape 


# In[ ]:


feature_spaces.columns


# In[ ]:


out_path = "all_space_feature_comb.csv"
feature_spaces.to_csv(out_path, index=False)


# In[ ]:


X = feature_rgb.drop(columns=['label'])
y = feature_rgb['label']


# In[ ]:


X = feature_spaces.drop(columns=['label'])
y = feature_spaces['label']


# In[ ]:


X = np.array(X)
y = np.array(y)
X.shape


# In[ ]:


type(y)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=1234)


# In[ ]:


from collections import Counter
print(Counter(y_train))


# In[ ]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks


# In[ ]:


over = SMOTE()


# In[ ]:


x_train, y_train = over.fit_resample(x_train, y_train)


# In[ ]:


under = RandomUnderSampler()


# In[ ]:


x_train, y_train = under.fit_resample(x_train, y_train)


# In[ ]:


print(Counter(y_train))


# In[ ]:


y_train = to_categorical(y_train, num_classes = 7)
#y_test = to_categorical(y_test, num_classes = 7)


# In[ ]:


y_train


# In[ ]:


x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.1, random_state = 2)


# In[ ]:


x_train.shape


# In[ ]:


x_train = x_train.reshape((x_train.shape[0],4,4,512))


# In[ ]:


x_train = x_train.reshape((x_train.shape[0],12,4,512))


# In[ ]:


model_transfer = Sequential()
model_transfer.add(GlobalAveragePooling2D(input_shape=x_train.shape[1:]))
model_transfer.add(Dropout(0.2))
model_transfer.add(Dense(100, activation='relu'))
model_transfer.add(Dropout(0.2))
model_transfer.add(Dense(64, activation='relu'))
model_transfer.add(Dropout(0.5))
model_transfer.add(Dense(7, activation='softmax'))
model_transfer.summary()


# In[ ]:


x_validate = x_validate.reshape((x_validate.shape[0],12,4,512))


# In[ ]:


x_validate = x_validate.reshape((x_validate.shape[0],4,4,512))


# In[ ]:


checkpointer = ModelCheckpoint(filepath='vgg.best.hdf5',
                               verbose=1,save_best_only=True)
optimizer = Adam(lr=0.01)


# In[ ]:


model_transfer.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])
history = model_transfer.fit(x_train, y_train, batch_size=32, epochs=100,
          validation_data=(x_validate, y_validate), callbacks=[checkpointer]
          ,verbose=1)


# In[ ]:


def plot_acc_loss(history):
    fig = plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
 
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
 
plot_acc_loss(history)


# In[ ]:


x_test = x_test.reshape((x_test.shape[0],4,4,512))


# In[ ]:


x_test = x_test.reshape((x_test.shape[0],12,4,512))


# In[ ]:


preds = np.argmax(model.predict(x_test), axis=1)
print("\nAccuracy on Test Data: ", accuracy_score(y_test, preds))
print("\nNumber of correctly identified imgaes: ",
      accuracy_score(y_test, preds, normalize=False),"\n")
confusion_matrix(y_test, preds, labels=range(0,7))


# In[ ]:


get_ipython().system('pip install hypopt')


# In[ ]:


from hypopt import GridSearch
from sklearn.linear_model import LogisticRegression


# In[ ]:


#param_grid = [{'C': [0.1,1,10],'solver': ['newton-cg','lbfgs']}]
 
# Grid-search all parameter combinations using a validation set.
#opt = GridSearch(model = LogisticRegression(class_weight='balanced', multi_class="auto",
                        #max_iter=200, random_state=1),param_grid = param_grid)
model = LogisticRegression(class_weight='balanced', multi_class="auto",max_iter=200, random_state=1,C = 0.01, solver = 'lbfgs',penalty ='l2')
model.fit(x_train, y_train)
#print(opt.get_best_params())


# In[ ]:


model.score(x_test,y_test)
preds = model.predict(x_test)
print("\nAccuracy on Test Data: ", accuracy_score(y_test, preds))
print("\nNumber of correctly identified imgaes: ",
accuracy_score(y_test, preds, normalize=False),"\n")
confusion_matrix(y_test, preds, labels=range(0,7))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




