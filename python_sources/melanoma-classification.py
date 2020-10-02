#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pydicom
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from tqdm import tqdm, tqdm_notebook
import gc
from keras.applications.densenet import preprocess_input, DenseNet169
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K

sns.set()

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# # **Load Data**

# We load data from csv file

# In[ ]:


train_df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
test_df = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")


# Let's check shape of our train dataset

# In[ ]:


train_df.shape


# Take a look at first 5 rows of our train dataset

# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# # EDA (Exploratory Data Analysis)

# We see that some values are missing. Most of the missing values are in column representing site where growth is

# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# .describe doesn't help us too much, because we have only 2 numeric features

# In[ ]:


train_df.describe() #Two numeric features


# In[ ]:


train_df['sex'].hist(figsize=(10, 4))


# We see that there are more men in our train dataset than women

# In[ ]:


train_df['sex'].value_counts(normalize = True)


# Age distribution looks normal

# In[ ]:


plt.hist(train_df['age_approx'], bins = 10)


# Since age is one of our numeric features, let's make a boxplot for it

# In[ ]:


sns.boxplot(x = 'age_approx', data = train_df)


# We see outliers with age of 0

# In[ ]:


train_df[train_df['age_approx'] < 5].head(10)


# There are two patients with age 0. Well, it may be actually newborn babys, but they are obwiously outliers, who can worse our predictions

# In[ ]:


train_df['age_approx'].max()


# The oldest patient is 90 years old

# In[ ]:


train_df['anatom_site_general_challenge'].value_counts(normalize = True)


# In[ ]:


test_df['anatom_site_general_challenge'].value_counts(normalize = True)


# More than a half of patients have growths on torso and the same picture we see in test dataset

# In[ ]:


fig, ax = plt.subplots(figsize = (10, 8))
sns.countplot(y = 'anatom_site_general_challenge', data = train_df)
plt.ylabel("Anatom site")


# In[ ]:


train_df['diagnosis'].value_counts(normalize = True)


# So most of diagnosis are labeled as 'unknown'. Let's see real diadnosis distribution

# In[ ]:


train_df[train_df['diagnosis'] != 'unknown']['diagnosis'].value_counts(normalize = True)


# So nevus is actually benign growth, while melanoma is malignant

# In[ ]:


real_diagn = train_df[train_df['diagnosis'] != 'unknown']['diagnosis']


# In[ ]:


fig, ax = plt.subplots(figsize = (10, 8))
sns.countplot(y = real_diagn)
plt.ylabel("Diagnosis")


# We see very imbalanced train dataset

# In[ ]:


train_df['benign_malignant'].value_counts()


# In[ ]:


pd.crosstab(train_df['benign_malignant'], train_df['target']) 


# In[ ]:


train_df['target'].value_counts(normalize = True)


# So benign/malignant is actually our target. Of course, we don't have this feature in our test dataset

# In[ ]:


test_df.head()


# In[ ]:


test_df.isnull().sum()


# So we have some missing values in test dataset as well

# Let's compare some features with each other to see if we can find something interesting

# In[ ]:


train_df.groupby(['sex'])['target'].agg([np.mean, np.sum])


# So men are actually more likely to get malignant growth

# In[ ]:


fig, ax = plt.subplots(figsize = (10, 8))
sns.countplot(x = 'target', hue = 'sex', data = train_df)


# In[ ]:


sns.boxplot(x = 'target', y = 'age_approx', data = train_df)


# So older people are more likely to get malignant growth

# In[ ]:


fig, ax = plt.subplots(figsize = (10, 8))
sns.countplot(y = 'anatom_site_general_challenge', hue = 'target', data = train_df)
plt.ylabel("Anatom site")


# In[ ]:


pd.crosstab(train_df['anatom_site_general_challenge'], train_df['target'], normalize = True)


# So malignant growths are more likely to be on torso

# Amount of unique patients in both train and test datasets:

# In[ ]:


train_df['patient_id'].nunique(), test_df['patient_id'].nunique()


# In[ ]:


jpeg_dir_train = "../input/siim-isic-melanoma-classification/jpeg/train"


# Let's display images for first patient in train dataset

# In[ ]:


first = train_df['patient_id'].unique()[0]
first


# There are 115 images for first patient. Let's display only few of them

# In[ ]:


train_df[train_df['patient_id'] == first]['image_name']


# In[ ]:


img_names = train_df[train_df['patient_id'] == first]['image_name'][:12]
img_names = img_names.apply(lambda x: x + '.jpg')
img_names


# Style of displaying images was taken from this [kernel](https://www.kaggle.com/parulpandey/melanoma-classification-eda-starter)

# In[ ]:


plt.figure(figsize = (12,10))
for i in range (len(img_names)):
    plt.subplot(4,4, i + 1)
    img = plt.imread(os.path.join(jpeg_dir_train, img_names.iloc[i]))
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    
plt.tight_layout()


# Let's compare different types of growths with images:

# In[ ]:


img_names = train_df[train_df['target'] == 0]['image_name'][:8]
img_names = img_names.apply(lambda x: x + '.jpg')
print ('Benign growths:')

plt.figure(figsize = (12,10))
for i in range (len(img_names)):
    plt.subplot(4,4, i + 1)
    img = plt.imread(os.path.join(jpeg_dir_train, img_names.iloc[i]))
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    
plt.tight_layout()


# Now let's display malignant growths:

# In[ ]:


img_names = train_df[train_df['target'] == 1]['image_name'][:8]
img_names = img_names.apply(lambda x: x + '.jpg')
print ('Malignant growths:')

plt.figure(figsize = (12,10))
for i in range (len(img_names)):
    plt.subplot(4,4, i + 1)
    img = plt.imread(os.path.join(jpeg_dir_train, img_names.iloc[i]))
    plt.imshow(img, cmap = 'gray')
    plt.axis('off')
    
plt.tight_layout()


# # Dicom files overview

# In addition we have .dcm files in our data, which are DICOM type of files. This format is actually widely used in different medical competitions. We already loaded pydicom library in the start, so now we can try to use it for overviewing data we have

# Interesting and helpful kernels about DICOM are [here](https://www.kaggle.com/schlerp/getting-to-know-dicom-and-the-data) and [here](https://www.kaggle.com/gpreda/visualize-ct-dicom-data). For now let's just find out if we can get valuable information from this data

# In[ ]:


train_dcm_path = "../input/siim-isic-melanoma-classification/train"
test_dcm_path = "../input/siim-isic-melanoma-classification/test"


# Let's see first dcm file

# In[ ]:


os.listdir(train_dcm_path)[0]


# In[ ]:


first_file_path = os.path.join(train_dcm_path, os.listdir(train_dcm_path)[0]) 
dicom_file = pydicom.dcmread(first_file_path)
dicom_file


# We see a lot of new data, but some of it we already know from .csv file

# In[ ]:


def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)


# In[ ]:


def plot_pixel_array(dataset, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


# These methods were taken from this [kernel ](https://www.kaggle.com/schlerp/getting-to-know-dicom-and-the-data)

# Here we display information about patient and an image of growth

# In[ ]:


n_files = 5
filenames = os.listdir(train_dcm_path)[:n_files]
for file_name in filenames:
    file_path = os.path.join(train_dcm_path, file_name)
    dataset = pydicom.dcmread(file_path)
    show_dcm_info(dataset)
    plot_pixel_array(dataset)


# Method for extracting data from DICOM files was taken from [this](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154658) discussion

# In[ ]:


def extract_DICOM_attributes():
    images = list(os.listdir(PATH))
    df = pd.DataFrame()
    for image in images:
        image_name = image.split(".")[0]
        dicom_file_path = os.path.join(PATH, image)
        dicom_file_dataset = pydicom.read_file(dicom_file_path)
        study_date = dicom_file_dataset.StudyDate
        modality = dicom_file_dataset.Modality
        age = dicom_file_dataset.PatientAge
        sex = dicom_file_dataset.PatientSex
        body_part_examined = dicom_file_dataset.BodyPartExamined
        patient_orientation = dicom_file_dataset.PatientOrientation
        photometric_interpretation = dicom_file_dataset.PhotometricInterpretation
        rows = dicom_file_dataset.Rows
        columns = dicom_file_dataset.Columns

        df = df.append(pd.DataFrame({'image_name': image_name, 
                        'dcm_modality': modality,'dcm_study_date':study_date, 'dcm_age': age, 'dcm_sex': sex,
                        'dcm_body_part_examined': body_part_examined,'dcm_patient_orientation': patient_orientation,
                        'dcm_photometric_interpretation': photometric_interpretation,
                        'dcm_rows': rows, 'dcm_columns': columns}, index=[0]))
    return df


# This takes a lot of time and memory!

# In[ ]:


PATH = train_dcm_path
dcm_df = extract_DICOM_attributes()


# In[ ]:


dcm_df.head()


# Let's check dataframe that we got from dcm 

# In[ ]:


dcm_df.columns


# In[ ]:


dcm_df['dcm_modality'].value_counts() 


# So modality column doesn't help us too much

# In[ ]:


dcm_df['dcm_sex'].value_counts()


# Rememdering train.csv we understand, what blank space actually is, but what X means isn't clear

# In[ ]:


train_df.sex.value_counts()


# In[ ]:


train_df.sex.isnull().sum()


# Also we got some new features like photometric interpretation, let's check them

# In[ ]:


dcm_df['dcm_photometric_interpretation'].value_counts()


# Another not too informative feature

# So what is actually rows and columns features? Let's check our first image

# In[ ]:


img_to_compare = cv2.imread('../input/siim-isic-melanoma-classification/jpeg/train/ISIC_0015719.jpg')
img_to_compare.shape


# In[ ]:


rows_to_disp = ['dcm_rows', 'dcm_columns']
dcm_df[dcm_df['image_name'] == 'ISIC_0015719'][rows_to_disp]


# So it's just size of our image. Duh

# After all, we didn't find any helpful information from .dmc files, but at least we tried. Next step is going to be data preprocessing. But before it let's clear our memory

# In[ ]:


del dcm_df, rows_to_disp, img_to_compare, filenames


# # Data preprocessing

# Back to missing values:

# In[ ]:


train_df.isnull().sum()


# Let's take a look at patients with missing values in features

# In[ ]:


missing_ids_sex = train_df[train_df['sex'].isnull()]['patient_id']
missing_ids_sex.unique()


# Two patients have missing sex column

# In[ ]:


train_df.loc[train_df['patient_id'].isin(missing_ids_sex.unique()), ['sex']]['sex'].value_counts()


# As we see any rows with their ID have missing sex. Let's check the same thing for age:

# In[ ]:


missing_ids_age = train_df[train_df['age_approx'].isnull()]['patient_id']
missing_ids_age.unique()


# Three patients have missing age

# In[ ]:


train_df.loc[train_df['patient_id'].isin(missing_ids_age.unique()), ['age_approx']]['age_approx'].value_counts()


# So we have same situation with ages. There's no data about age of these patients in dataset

# In[ ]:


set(missing_ids_age) - set(missing_ids_sex)


# Two patients (IP_5205991 and IP_9835712) have no age nor sex in dataset. They miss pretty helpful features for us

# In[ ]:


missing_ids_site = train_df[train_df['anatom_site_general_challenge'].isnull()]['patient_id']
missing_ids_site.unique() 


# We have a quite large number of patients missing anatom site where growth is

# Let's drop two patients, who don't have age and sex in dataset

# In[ ]:


ind_to_drop = train_df[train_df['patient_id'].isin(missing_ids_sex.unique())].index
train_df.drop(index = ind_to_drop, inplace = True)


# In[ ]:


train_df.isnull().sum()


# Remembering two patients with age zero, we should change their age as well

# In[ ]:


id_w_zero = train_df[train_df['age_approx'] < 5]['patient_id']
id_w_zero.values


# In[ ]:


ind_zero = train_df.loc[train_df['patient_id'].isin(id_w_zero.values)].index
train_df.loc[train_df['patient_id'].isin(id_w_zero.values)]


# So there's obviously mistake in data, patient IP_1300691 is actually 10 years  old, but in two rows his age is zero. Let's fix it

# In[ ]:


train_df.loc[ind_zero, 'age_approx'] = 10.0
train_df.loc[ind_zero]


# Good, we fixed incorrect data about patient. Now we can input mean column values instead of missing values of age in dataset

# In[ ]:


val = {'age_approx' : train_df['age_approx'].mean()}
train_df.fillna(val, inplace = True)


# In[ ]:


train_df.isnull().sum()


# Now let's deal with categorical data

# First replace 'male' and 'female' values with numbers

# In[ ]:


mapping = {'male' : 1, 'female' : 0}
train_df['sex'] = train_df['sex'].map(mapping)
test_df['sex'] = test_df['sex'].map(mapping)
train_df.head()


# Looks like we have need to somehow imput missing values in anatom_site feature both in train and in test datasets

# In[ ]:


train_df[train_df['anatom_site_general_challenge'].isna()]['target'].value_counts()


# So I feel like just dropping those 518 instances of benign growths won't harm our dataset too much, while rows with malignant growths will have median value in anatom_site feature 

# In[ ]:


ind_to_drop = train_df[(train_df['anatom_site_general_challenge'].isna()) & (train_df['target'] == 0)].index
ind_to_drop


# In[ ]:


train_df.drop(ind_to_drop, inplace = True)


# In[ ]:


train_df.isnull().sum() ,train_df.shape


# In[ ]:


train_df = train_df.fillna('torso')


# In[ ]:


train_df.isnull().sum()


# It would be a good idea to drop useless columns from train dataset, since test dataset doesn't have such columns as diagnosis and benign_malignant

# In[ ]:


train_df = train_df.drop(['diagnosis', 'benign_malignant'], axis = 1)
train_df.head()


# Extract target from train dataset

# In[ ]:


X_train, y_train = train_df.drop('target', axis = 1), train_df['target']


# In[ ]:


X_train


# In[ ]:


y_train


# Things get much more complex with test dataset

# In[ ]:


test_df.isnull().sum()


# Probably taking median value from train dataset will be fine, since distribution of anatom site feature is similar in train and test datasets

# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
imp.fit(X_train)
ind, col = test_df.index, test_df.columns
X_test = pd.DataFrame(imp.transform(test_df), index = ind, columns = col)
X_test.head()


# In[ ]:


X_test.isnull().sum()


# Now our train and test datasets don't have missing values. It's time to make categorical features one-hot encoded. We have only one such feature, which is anatom_site

# In[ ]:


X_train.head()


# In[ ]:


cat_features = ["anatom_site_general_challenge"]
encoded = pd.get_dummies(X_train[cat_features])
encoded.set_index(X_train.index)
X_train.drop(cat_features, inplace = True, axis = 1)
X_train_encoded = pd.concat([X_train,encoded], axis = 1)


# In[ ]:


X_train_encoded.head()


# Now we will do the same thing for test dataset

# In[ ]:


encoded = pd.get_dummies(X_test[cat_features])
encoded.set_index(X_test.index)
X_test.drop(cat_features, inplace = True, axis = 1)
X_test_encoded = pd.concat([X_test,encoded], axis = 1)


# In[ ]:


X_test_encoded.head()


# In[ ]:


train_df_clean = pd.concat([X_train_encoded, y_train], axis = 1)
test_df_clean = X_test_encoded


# In[ ]:


train_df_clean.head()


# Let's save preprocessed data

# In[ ]:


train_df_clean.to_csv('train_clean.csv', index=False)
test_df_clean.to_csv('test_clean.csv', index=False)


# # Getting features from images

# Define methods for resize and loading images

# In[ ]:


img_size = 256

#Paths to train and test images
train_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_img_path = '/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

def resize_image(img):
    old_size = img.shape[:2]
    ratio = float(img_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = cv2.resize(img, (new_size[1],new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0,0,0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img

def load_image(path, img_id):
    path = os.path.join(path,img_id+'.jpg')
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_img = resize_image(img)
    new_img = preprocess_input(new_img)
    return new_img


# Image processing and resizing was taken from [this](https://www.kaggle.com/anshuls235/siim-isic-melanoma-analysis-eda-prediction#2.-Studying-the-data) kernel

# In[ ]:


fig = plt.figure(figsize=(16, 16))
for i,image_id in enumerate(np.random.choice(train_df[train_df['target']== 0].image_name,5)):
    image = load_image(train_img_path,image_id)
    fig.add_subplot(1,5,i+1)
    plt.imshow(image)


# In[ ]:


fig = plt.figure(figsize=(16, 16))
for i,image_id in enumerate(np.random.choice(train_df[train_df['target']== 1].image_name,5)):
    image = load_image(train_img_path,image_id)
    fig.add_subplot(1,5,i+1)
    plt.imshow(image)


# In[ ]:


train_df.head()


# We load a DenseNet neural network, which can be used to get features from our image data.  

# In[ ]:


batch_size = 16

train_img_ids = train_df.image_name.values
n_batches = len(train_img_ids) // batch_size + 1

#Model to extract image features
inp = Input((256,256,3))
backbone = DenseNet169(input_tensor = inp, include_top = False)
x = backbone.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
x = AveragePooling1D(4)(x)
out = Lambda(lambda x: x[:,:,0])(x)

m = Model(inp,out)


# In a loop we get batches of images and extract features from them

# In[ ]:


features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_ids = train_img_ids[start:end]
    batch_images = np.zeros((len(batch_ids),img_size,img_size,3))
    for i,img_id in enumerate(batch_ids):
        try:
            batch_images[i] = load_image(train_img_path,img_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,img_id in enumerate(batch_ids):
        features[img_id] = batch_preds[i]


# We convert features in a Dataframe

# In[ ]:


train_feats = pd.DataFrame.from_dict(features, orient = 'index')
train_feats.to_csv('train_img_features.csv')
train_feats.head()


# Do the same thing for test dataset

# In[ ]:


test_img_ids = test_df.image_name.values
n_batches = len(test_img_ids) // batch_size + 1


# In[ ]:


features = {}
for b in tqdm_notebook(range(n_batches)):
    start = b*batch_size
    end = (b+1)*batch_size
    batch_ids = test_img_ids[start:end]
    batch_images = np.zeros((len(batch_ids),img_size,img_size,3))
    for i,img_id in enumerate(batch_ids):
        try:
            batch_images[i] = load_image(test_img_path,img_id)
        except:
            pass
    batch_preds = m.predict(batch_images)
    for i,img_id in enumerate(batch_ids):
        features[img_id] = batch_preds[i]


# In[ ]:


test_feats = pd.DataFrame.from_dict(features, orient='index')
test_feats.to_csv('test_img_features.csv')
test_feats.head()


# This takes lots of time, so to make it easier already did it and loaded into dataset 

# In[ ]:


train_feat_img = pd.read_csv ('../input/melanoma-dataset-for-images/train_img_features.csv')
test_feat_img = pd.read_csv ('../input/melanoma-dataset-for-images/test_img_features.csv')


# In[ ]:


test_feat_img.head()


# Let's make combine two of our datasets

# In[ ]:


train_feat_img = train_feat_img.set_index('Unnamed: 0')


# In[ ]:


test_feat_img = test_feat_img.set_index('Unnamed: 0')


# In[ ]:


train_feat_img.head()


# In[ ]:


test_feat_img.head()


# In[ ]:


train_data = pd.read_csv('./train_clean.csv')
test_data = pd.read_csv('./test_clean.csv')


# In[ ]:


train_data.head()


# In[ ]:


X_train_encoded = train_data.drop('target', axis = 1)
y_train = train_data['target']


# In[ ]:


X_train_encoded.head()


# In[ ]:


y_train.head()


# In[ ]:


X_train_full =  X_train_encoded.merge (train_feat_img, 
                       how = 'inner',
                      left_on = 'image_name', 
                      right_index = True,
                      )


# Now we have combined two datasets into a new full one

# In[ ]:


X_train_full.head()


# Let's do the same thing for test set

# In[ ]:


X_test_full = test_data.merge (test_feat_img, 
                      how = 'inner',
                      left_on = 'image_name', 
                      right_index = True,
                      )


# In[ ]:


X_test_full.head()


# Let's drop not useful features so they won't bother our training

# In[ ]:


X_train_full.drop(['image_name', 'patient_id'], inplace = True, axis = 1)


# In[ ]:


X_train_full.head()


# In[ ]:


X_test_full.head()


# In[ ]:


X_test_full.drop('patient_id', axis = 1, inplace = True)
X_test_full = X_test_full.set_index('image_name')


# In[ ]:


X_test_full.head()


# In[ ]:


import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# For this task I will use gradient boosting from xgboost library. After spending couple hours of training, I found this hyperparameters the best

# In[ ]:


boosting = xgb.XGBClassifier(max_depth = 8, 
                            reg_lambda = 1.2,
                            subsample = 0.8, 
                            n_estimators = 400, 
                            min_child_weight = 2, 
                            learning_rate = 0.1)


# Make stratified folds, so there will be equal part of benign and malignant growths in each fold

# In[ ]:


skf = StratifiedKFold(n_splits = 3)
score_cv = cross_val_score(boosting, X_train_full, y_train, cv = skf, scoring = 'roc_auc')


# See cross validation scores 

# In[ ]:


score_cv


# In[ ]:


boosting.fit(X_train_full, y_train)


# After fitting make predictions. We need to predict probability, since scores in the competition is ROC-AUC

# In[ ]:


y_test = boosting.predict_proba(X_test_full)[:, 1]


# Convert them to pandas Dataframe

# In[ ]:


submission = pd.DataFrame({
    'image_name': X_test_full.index,  
    'target' : y_test
})


# Get our submissions!

# In[ ]:


submission.to_csv('submission.csv', index = False)

