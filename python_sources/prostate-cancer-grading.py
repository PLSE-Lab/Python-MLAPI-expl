#!/usr/bin/env python
# coding: utf-8

# # **About this notebook**
# This notebook consists of domain knowledge about the prostatce cancer as well predictive analytics.
# 
# **If you find this kernel useful, Please consider upvoting it, It motivates me to write more quality content.**
# # **What is prostate cancer?**
# Prostate cancer is the most common cancer after skin cancer in men in the United States. Also, it is the third leading cause of cancer death, behind lung cancer and colon cancer, in men in the United States. About 1 man in 41 will die of prostate cancer. In some men, it is slow growing and unlikely to cause serious problems. In others, the disease is very aggressive. About 1 man in 9 will be diagnosed with prostate cancer during his lifetime. 
# 
# Prostate cancer starts in the prostate, a gland located below the bladder and in front of the rectum. 
# * The prostate contains several types of cells, but nearly all prostate cancers develop from glandular cells, which make fluid that becomes part of semen.
# * Prostate cancer cells can spread by invading nearby organs and tissues, such as the bladder or rectum, or by travelling through the blood or lymph to other parts of the body. This is known as metastatic prostate cancer.
# * Other than the lymph nodes near the prostate, the most common site of prostate cancer spread, or metastatis, is the bones, especially in the spine.
# 
# Your prostate makes and stores seminal fluid - a milkey liquid that protects and nourishes sperm. Your prostate surrounds part of your urethra, the tube that carries urine and semon out of your body. Many men develop a noncancerous condition called bengin prostatic hyperplasia (BPH), or enlargement of the prostate. If the prostate, which is normally about the size of a walnut, grows too large, it can slow or block the flow of urine. 
# 
# # **Symptoms of prostate cancer**
# Prostate cancer symptoms typically don't appear early in the disease. In many men, doctors first detect signs of prostate cancer during routine check-up. More advanced prostate cancer symptoms may include:
# * Weak or interrupted flow of urine
# * Urinating often (especially at night)
# * Difficulty getting or sustaining an erection (impotance)
# * Painful ejaculation
# * Frequent pain or stiffness in the lower back, hips or upper thighs
# 
# # **Diagnosing prostate cancer**
# Prostate cancer is curable, when it is diagnosed early. Good prostate cancer screening tests, like the prostate-specific antigen (PSA) test, have resulted in early diagnosis in about 80 percent of men with the disease. According to the American Cancer Society (ACS), all of these men survive at least five years. Whether cancer is suspected based on symptoms or a digital rectal exam or PSA test, the actual diagnosis is made with a prostate biposy, a procedure in which samples of your prostate are removed and examined under a microscope.
# 
# A core needle biopsy, the main method for diagnosing prostate cancer, is typically performed in a doctor's office by a urologist as follows:
# * Using transrectal ultrasound (TRUS) and a local anesthetic, the doctor inserts a needle into your prostate through a probe in your rectum.
# * The doctor uses the needle to take about 12 samples of cells.
# * The procedure typically takes no more than 5 to 10 minutes, and you should have very little discomfort.
# 
# An imaging test called an Axumin positron emission tomography (PET) scan may assist in detecting prostate cancer that has come back in men whose PSA levels rise after they've had treatment. Before the scan, you receive an injection of fluciclovine F18 (Axumin), a radioactive agent that tends to collect in areas of cancer activity, which then light up on your scan.
# 
# # **What causes prostate cancer?**
# While the exact cause of prostate cancer is unknown, generally speaking it results from mutations in cell DNA. DNA is the chemical that makes up your genes, which control when your cells grow, divide into new cells and die. DNA mutations that turn on oncogenes which help cells growand divide, or that turn off tumor-suppressor genes (which slow cell division or make cells die when they should) can cause prostate cells to grow abnormally and lead to cancer. 
# ## *Prostate cancer risk factors*
# Numerous factors may contribute to prostate cancer risk. The main risk factors (variables) are as follows:
# * Age: Although prostate cancer can occur at any age, it is most often found in men over age 50, and more than two-thirds of men diagnosed with the disease are over 65. About 6 cases in 10 are diagnosed in men who are 65 or older, and it is rare in men under 40.
# * Family history and genetics: A family history of prostate cancer may increase your risk, particularly if you have a number of close relatives who were younger than 60 when they were diagnosed. If your father or brother had prostate cancer, your risk is two to three times greater than if you had no family history of the disease. 
# * Race or ethnicity: African-American men are more likely than men of other races to develop prostate according to different research conducted in this area. The disease is less common among men of Asian or Hispanic/Latino descent than among those of European descent.
# * Nationality: Prostate cancer is most common in North America, northwestern Europe, Australia and the Caribbean and less common in Asia, Africa and Central and South America. 
# * Hormone levels: Research suggests that the development of prostate cancer is linked to higher levels of certain hormones, such as testosterone, the main male sex hormone. Testosterone is changed into dihydrotestostrone (DHT) by an enzyme in the body. DHT is important for normal prostate growth but can also cause the prostate to get bigger and may play a role in development of prostate cancer. 
# * Diet: Scientists believe that diet is a critical factor in prostate cancer risk. A diet high in red meat, dairy foods and calcium and low in fruits and vegetables may play a part. Vitamin E and folic acid are also thought to increase the risk. 

# # **Analysis**
# The dataset consists of around 11,000 whole-side images (WSI) of digitized H&E-stained prostate biopsies originating from Radboud University Medical Center and the Karolinska Institute. "isup_grade" is the target variable which illustrats the severity of the cancer on a 0-5 scale. "gleason_score" is an alternative cancer severity rating system with more levels than the ISUP scale. 
# 

# In[ ]:


import openslide
import skimage.io
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import PIL
from IPython.display import Image, display
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
from tqdm import tqdm_notebook as tqdm
import math
import torch.utils.model_zoo as model_zoo
import cv2
import openslide
import skimage.io
import random
from sklearn.metrics import cohen_kappa_score
import albumentations
from PIL import Image
import os
from fastai import *
from fastai.vision import *
import openslide
from PIL import Image as pil_image


# # **Importing Dataset**

# In[ ]:


BASE_PATH = '../input/prostate-cancer-grade-assessment'
data_dir = f'{BASE_PATH}/train_images'
mask_dir = f'{BASE_PATH}/train_label_masks'
train = pd.read_csv(f'{BASE_PATH}/train.csv').set_index('image_id')
test = pd.read_csv(f'{BASE_PATH}/test.csv')
submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
display(train.head())
print("Shape of training data :", train.shape)
display(test.head())
print("Shape of training data :", test.shape)
train.isna().sum()


# # **Exploratory Data Analysis**

# In[ ]:


def plot_count(df, feature, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.3f}%'.format(100*height/total),
                ha="center") 
    plt.show()
def plot_relative_distribution(df, feature, hue, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    total = float(len(df))
    sns.countplot(x=feature, hue=hue, data=df, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()
def display_images(slides): 
    f, ax = plt.subplots(5,3, figsize=(18,22))
    for i, slide in enumerate(slides):
        image = openslide.OpenSlide(os.path.join(data_dir, f'{slide}.tiff'))
        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)
        patch = image.read_region((1780,1950), 0, (256, 256))
        ax[i//3, i%3].imshow(patch) 
        image.close()       
        ax[i//3, i%3].axis('off')
        
        image_id = slide
        data_provider = train.loc[slide, 'data_provider']
        isup_grade = train.loc[slide, 'isup_grade']
        gleason_score = train.loc[slide, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")

    plt.show() 
plot_count(df=train, feature='data_provider', title = 'Data Provider Frequency Percentage Plot')
plot_count(df=train, feature='isup_grade', title = 'ISUP Grade Frequency Percentage Plot')
plot_count(df=train, feature='gleason_score', title = 'Gleason Score Frequency Percentage Plot', size=3)
plot_relative_distribution(df=train, feature='isup_grade', hue='data_provider', title = 'Relative Count Plot of ISUP Grade with Data Provider', size=2)
plot_relative_distribution(df=train, feature='gleason_score', hue='data_provider', title = 'Relative Count Plot of Gleason Score with Data Provider', size=3)
images = [
    '07a7ef0ba3bb0d6564a73f4f3e1c2293',
    '037504061b9fba71ef6e24c48c6df44d',
    '035b1edd3d1aeeffc77ce5d248a01a53',
    '059cbf902c5e42972587c8d17d49efed',
    '06a0cbd8fd6320ef1aa6f19342af2e68',
    '06eda4a6faca84e84a781fee2d5f47e1',
    '0a4b7a7499ed55c71033cefb0765e93d',
    '0838c82917cd9af681df249264d2769c',
    '046b35ae95374bfb48cdca8d7c83233f',
    '074c3e01525681a275a42282cd21cbde',
    '05abe25c883d508ecc15b6e857e59f32',
    '05f4e9415af9fdabc19109c980daf5ad',
    '060121a06476ef401d8a21d6567dee6d',
    '068b0e3be4c35ea983f77accf8351cc8',
    '08f055372c7b8a7e1df97c6586542ac8'
]

display_images(images)
def display_masks(slides): 
    f, ax = plt.subplots(5,3, figsize=(18,22))
    for i, slide in enumerate(slides):
        
        mask = openslide.OpenSlide(os.path.join(mask_dir, f'{slide}_mask.tiff'))
        mask_data = mask.read_region((0,0), mask.level_count - 1, mask.level_dimensions[-1])
        cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])

        ax[i//3, i%3].imshow(np.asarray(mask_data)[:,:,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5) 
        mask.close()       
        ax[i//3, i%3].axis('off')
        
        image_id = slide
        data_provider = train.loc[slide, 'data_provider']
        isup_grade = train.loc[slide, 'isup_grade']
        gleason_score = train.loc[slide, 'gleason_score']
        ax[i//3, i%3].set_title(f"ID: {image_id}\nSource: {data_provider} ISUP: {isup_grade} Gleason: {gleason_score}")
        f.tight_layout()
        
    plt.show()
display_masks(images)


# In[ ]:


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_torch(seed=42)
class config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    TEST_BATCH_SIZE = 1
    CLASSES = 6

