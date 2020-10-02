#!/usr/bin/env python
# coding: utf-8

# ## Conditions to detect outliers
# - `Quantity` == 1 and `Description` contains "puppies", "kitties", "kittens"
# - `Quantity` > 10 and `PhotoAmt` == 1
# - `Type` == "Dog" and `ImageDescription` == "cat"
# - `Type` == "Cat" and `ImageDescription` == "dog"
# 
# ## Outliers I have found so far
# |Pet ID|Reason|
# |:-|:-|
# |6a72cfda7|cat but breed2 is Akita which is dog breed|
# |3c778df64|type is dog but it's obviously a cat|
# |06634513c|gender should be mixed because description says "3 female, 1 male"|
# |1ef39cee1|4 dogs but quantity is 3|
# |f8619af42|1 dog but quantity is 20|
# |500327aed|1 dog but quantity is 20|
# |eff316e87|1 dog but quantity is 20|
# |fe481f81c|1 dog but quantity is 20|
# |140ef35de|1 dog but quantity is 20|
# |e9e4424a6|1 dog but quantity is 3|
# |5b5b2c882|multiple cats but quantity is 1|
# |ab13611c9|1 dog but quantity is 10|
# |ae14a91dc|Is she really brown or black?|
# |0ffe99b25|3 kittens and 1 mama cat but age is 2|
# |f102632f6|description says "he" is "2" years old but gender is female and age is 8|
# |9883a048e|puppies but quantity is 1|
# |3cc84c2f8|4 puppies but quantity is 1|
# |c0ab24656|These puppies|
# |ed6684f3a|4 puppies but quantity is 1|

# In[ ]:


import os
import sys
import re
import json
from glob import glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from plotly.colors import DEFAULT_PLOTLY_COLORS as colors
py.init_notebook_mode(connected=True)

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.display import display, HTML

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

pd.options.display.max_columns = None


# In[ ]:


get_ipython().system('ls ../input')


# In[ ]:


get_ipython().system('ls ../input/train_images | head -3')


# In[ ]:


get_ipython().system('ls ../input/train_metadata | head -3')


# ### Data Fields
# 
# |Feature|Description|
# |:-|:-|
# |**PetID**|Unique hash ID of pet profile|
# |**AdoptionSpeed**|Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.|
# |**Type**|Type of animal (1 = Dog, 2 = Cat)|
# |**Name**|Name of pet (Empty if not named)|
# |**Age**|Age of pet when listed, in months|
# |**Breed1**|Primary breed of pet (Refer to BreedLabels dictionary)|
# |**Breed2**|Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)|
# |**Gender**|Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)|
# |**Color1**|Color 1 of pet (Refer to ColorLabels dictionary)|
# |**Color2**|Color 2 of pet (Refer to ColorLabels dictionary)|
# |**Color3**|Color 3 of pet (Refer to ColorLabels dictionary)|
# |**MaturitySize**|Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)|
# |**FurLength**|Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)|
# |**Vaccinated**|Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)|
# |**Dewormed**|Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)|
# |**Sterilized**|Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)|
# |**Health**|Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)|
# |**Quantity**|Number of pets represented in profile|
# |**Fee**|Adoption fee (0 = Free)|
# |**State**|State location in Malaysia (Refer to StateLabels dictionary)|
# |**RescuerID**|Unique hash ID of rescuer|
# |**VideoAmt**|Total uploaded videos for this pet|
# |**PhotoAmt**|Total uploaded photos for this pet|
# |**Description**|Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.|

# ### Adoption Speed
# |Value|Description|
# |:-|:-|
# |**0**|Pet was adopted on the same day as it was listed. |
# |**1**|Pet was adopted between 1 and 7 days (1st week) after being listed. |
# |**2**|Pet was adopted between 8 and 30 days (1st month) after being listed. |
# |**3**|Pet was adopted between 31 and 90 days (2nd & 3rd month) after being listed. |
# |**4**|No adoption after 100 days of being listed. (There are no pets in this dataset that waited between 90 and 100 days).|

# In[ ]:


# read data
df_train = pd.read_csv("../input/train/train.csv")
df_breed = pd.read_csv("../input/breed_labels.csv")
df_color = pd.read_csv("../input/color_labels.csv")
df_state = pd.read_csv("../input/state_labels.csv")

# cleaning
df_train.loc[df_train["Name"].isnull(), "Name"] = np.nan

df_train.loc[df_train["Description"].isnull(), "Description"] = ""

df_train["PhotoAmt"] = df_train["PhotoAmt"].astype(int)

is_breed1_zero = df_train["Breed1"] == 0
df_train["Breed1"][is_breed1_zero] = df_train["Breed2"][is_breed1_zero]
df_train["Breed2"][is_breed1_zero] = 0

# merge dataframes
df_breed = df_breed.append({"BreedID": 0, "Type": 0, "BreedName": ""}, ignore_index=True).replace("", np.nan)
df_color = df_color.append({"ColorID":0, "ColorName": ""}, ignore_index=True).replace("", np.nan)

# Decode categorical features
df_train["Breed1"] = df_breed.set_index("BreedID").loc[df_train["Breed1"]]["BreedName"].values
df_train["Breed2"] = df_breed.set_index("BreedID").loc[df_train["Breed2"]]["BreedName"].values
df_train["Color1"] = df_color.set_index("ColorID").loc[df_train["Color1"]]["ColorName"].values
df_train["Color2"] = df_color.set_index("ColorID").loc[df_train["Color2"]]["ColorName"].values
df_train["Color3"] = df_color.set_index("ColorID").loc[df_train["Color3"]]["ColorName"].values
df_train["State"] = df_state.set_index("StateID").loc[df_train["State"]]["StateName"].values

mapdict = {
    "Type"        : ["", "Dog", "Cat"],
    "Gender"      : ["", "Male", "Female", "Mixed"],
    "MaturitySize": ["Not Specified", "Small", "Meidum", "Large", "Extra Large"],
    "FurLength"   : ["Not Specified", "Short", "Medium", "Long"],
    "Vaccinated"  : ["", "Yes", "No", "Not Sure"],
    "Dewormed"    : ["", "Yes", "No", "Not Sure"],
    "Sterilized"  : ["", "Yes", "No", "Not Sure"],
    "Health"      : ["Not Specified", "Healthy", "Minor Injury", "Serious Injury"]
}

for k, v in mapdict.items():
    dummy_df = pd.DataFrame({k: v})
    df_train[k] = dummy_df.loc[df_train[k]][k].values


# In[ ]:


def read_json(fpath):
    with open(fpath) as f:
        return json.load(f)

def get_sentiment(pet_id, dir_):
    fpath = f"../input/{dir_}/{pet_id}.json"
    if not os.path.exists(fpath):
        return np.nan, np.nan
    data = read_json(fpath)
    result = data["documentSentiment"]
    return result["magnitude"], result["score"]

def get_image_meta(pet_id, dir_):
    fpath = f"../input/{dir_}/{pet_id}-1.json"
#     print(fpath)
    if not os.path.exists(fpath):
        return np.nan, np.nan
    
    data = read_json(fpath)
    
    if not "labelAnnotations" in data:
        return np.nan, np.nan
    
    result = data["labelAnnotations"][0]
    return result["description"], result["score"]


# In[ ]:


# merge image metadata
df_train["ImageDescription"], df_train["ImageDescriptionScore"] = zip(*df_train["PetID"].map(lambda pet_id: get_image_meta(pet_id, "train_metadata")))


# In[ ]:


df_train.sample()


# In[ ]:


def rand_pet_id():
    return df_train["PetID"].sample(1).values[0]

def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)

def show_pics(pet_id):
    img_paths = glob(f"../input/train_images/{pet_id}*.jpg")
    npics = len(img_paths)
    if npics == 0:
        print("No picture found")
        return
    max_ncols = 5
    ncols = max_ncols if npics > max_ncols else npics
    nrows = int(np.ceil(npics / max_ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, nrows * 4))        
    axes = [axes] if npics == 1 else axes.ravel()
    for ax in axes[npics:]: fig.delaxes(ax)
    
    for i, img_path, ax in zip(range(npics), img_paths, axes):
        ax.imshow(plt.imread(img_path))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

def show_pet(pet_id):
    row_match = df_train[df_train["PetID"] == pet_id]
    d = {col: ser.iloc[0] for col, ser in row_match.items()}
    description = d.pop("Description")
    text = ""
    
    for k in d.keys():
        text += f"{k:22}: {d[k]}\n"
    print(text)
    print(f"< Description >\n{description}\n")

    show_pics(pet_id)


# In[ ]:


show_pet(rand_pet_id())


# In[ ]:


df_train[(df_train["Quantity"] > 10 ) & (df_train["PhotoAmt"] == 1)]


# In[ ]:


df_train[(df_train["Quantity"] == 1 ) & (df_train["Description"].str.lower().str.contains("puppies"))]


# In[ ]:


df_train[(df_train["Type"] == "Cat") & (df_train["ImageDescription"].str.contains("dog"))]

