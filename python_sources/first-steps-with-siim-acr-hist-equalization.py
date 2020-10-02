#!/usr/bin/env python
# coding: utf-8

# # Train, Test and uniques ImageId
# 
# Thanks [@jeesper](https://www.kaggle.com/jesperdramsch) for the [SIIM ACR Pneumothorax Segmentation Data](https://www.kaggle.com/jesperdramsch/siim-acr-pneumothorax-segmentation-data)
# 
# Forked from https://www.kaggle.com/steubk/first-steps-with-siim-acr-pneumothorax-data/ to explore image histogram equalization

# In[ ]:


import numpy as np  
import pandas as pd 
import os
import pydicom
from tqdm import tqdm_notebook as tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import exposure

# import mask utilities
import sys
sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation')
from mask_functions import rle2mask


# In[ ]:


def extract_dcm_pixel_array(file_path):
    return pydicom.dcmread(file_path).pixel_array

def extract_dcm_metadata(file_path):
    ds = pydicom.dcmread(file_path)
    d = {}
    for elem in ds.iterall():
        if elem.name != 'Pixel Data' and elem.name != "Pixel Spacing" :
            d[elem.name.lower().replace(" ","_").replace("'s","")] = elem.value
        elif elem.name == "Pixel Spacing" :
            d["pixel_spacing_x"] = elem.value[0]
            d["pixel_spacing_y"] = elem.value[1]
            
    return d

def create_metadataset (df):

    ImageIds = []
    data  = []
    all_feats = set()    

    for index, row in tqdm ( df[["ImageId", "path"] ].drop_duplicates().iterrows() ) :            
        path = row["path"] 
        ImageId =  row["ImageId"]       
        feature_dict = extract_dcm_metadata (path)
        data.append(feature_dict)
        ImageIds.append(ImageId)
        feats = set (feature_dict.keys())
        if len ( feats - all_feats ) > 0:
            all_feats = all_feats.union(feats)


    df_meta = pd.DataFrame(columns=["ImageId"])
    df_meta["ImageId"]=ImageIds

    for feat in sorted(all_feats):
        df_meta[feat]=[ d[feat] for d in data ]

    df_meta['patient_age'] =  df_meta['patient_age'].map (lambda x: int(x))   
    return df_meta

DATA_PATH = "../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/"
SAMPLE_SUBMISSION = "../input/siim-acr-pneumothorax-segmentation/sample_submission.csv"

df_train  = pd.DataFrame([(name.replace(".dcm",""),  os.path.join(root, name)) for root, dirs, files in os.walk(DATA_PATH + "/dicom-images-train" )
             for name in files if name.endswith((".dcm"))], columns = ['ImageId','path']) 

df_test = pd.DataFrame([(name.replace(".dcm",""), os.path.join(root, name)) for root, dirs, files in os.walk(DATA_PATH + "/dicom-images-test" )
             for name in files if name.endswith((".dcm"))], columns = ['ImageId','path']) 

df_sub = pd.read_csv(SAMPLE_SUBMISSION)


df_rle = pd.read_csv(DATA_PATH + "/train-rle.csv")  
df_rle = df_rle.rename ( columns =  { ' EncodedPixels': 'EncodedPixels' })
df_rle ["EncodedPixels"] = df_rle ["EncodedPixels"].map(lambda x: x[1:])
df_train = df_train.merge(df_rle, on="ImageId", how="left")

not_pneumothorax_ImageId = set(df_train.query( "EncodedPixels == '-1' or EncodedPixels.isnull()",  engine='python') ["ImageId"])
df_train["pneumothorax"] = df_train["ImageId"].map(lambda x: 0 if x  in not_pneumothorax_ImageId else 1)


df_train["rle_count"] = df_train["ImageId"].map(df_rle.groupby(["ImageId"]).size())
df_train["rle_count"] = df_train["rle_count"].fillna(-1)  

## adding dicom metadata
df_train = df_train.merge(create_metadataset ( df_train ), on="ImageId", how='left') 
df_test = df_test.merge(create_metadataset ( df_test ), on="ImageId", how='left')

## removing dicom metadata with no variance
df_all = df_train.append(df_test, sort=False)
cols = [ c for c in  df_all.columns if len(df_all[c].unique()) != 1]
df_train= df_train[cols]
cols = [ c for c in  cols if c not in ["EncodedPixels", "rle_count", "pneumothorax"]]
df_test = df_test [cols]

df_train.to_csv("train.csv",index=False)
df_test.to_csv("test.csv",index=False)

df_sub["entries"] = df_sub["ImageId"].map( df_sub.groupby(['ImageId']).size() )



print ( "train-rle: {}, unique ImageId: {}".format(len(df_rle), len(df_rle["ImageId"].unique()))) 
print ( "train: {}, unique ImageId: {}".format(len(df_train), len(df_train["ImageId"].unique()))) 
print("train ImageId not in rle: {}".format( 
    len( df_train.query ( "EncodedPixels.isnull()",  engine='python') )))
print("train ImageId with multiple rle: {}".format( 
    len( df_train.query ( "rle_count > 1",  engine='python')["ImageId"].unique() )))

print ( "sample_submission: {}, unique ImageId: {}, ImegeId with multiple entries: {}".format(
    len(df_sub), 
    len(df_sub["ImageId"].unique()), 
    len ( df_sub.query ( "entries > 1")["ImageId"].unique() )
    )) 

print ( "test: {}, unique ImageId: {}".format(len(df_test), len(df_test["ImageId"].unique())))
print("test ImageId not in sample_submission: {}".format( 
    len( df_test [ ~ df_test["ImageId"].isin(df_sub["ImageId"])])))


# ## X-Ray visualization
# 

# In[ ]:


pneumothorax = df_train.query ( "pneumothorax == 1 and rle_count == 1",  engine='python').sample(n=5).reset_index()

fig, axs = plt.subplots(2, 5, figsize=(30,10))
fig.suptitle("samples with pneumothorax (train)", fontsize=30)
for j, row in pneumothorax.iterrows():
    img = extract_dcm_pixel_array (row['path'])
    x = 0
    y = j % 5
    axs[x,y].imshow(img, cmap='bone')
    axs[x,y].axis('off')
    
    rle_mask = rle2mask(row["EncodedPixels"] , 1024, 1024).T
    x = 1
    axs[x,y].imshow(img, cmap='bone')
    axs[x,y].imshow(rle_mask, alpha=0.5, cmap="Blues")    
    axs[x,y].axis('off')
    
    
    
fig.subplots_adjust(top=0.9)


plt.show()


# A visual explanation here [https://www.youtube.com/watch?v=0vZ9gVyWreo](https://www.youtube.com/watch?v=0vZ9gVyWreo)

# In[ ]:


images = df_train.query ( "pneumothorax == 0",  engine='python')["path"].values
np.random.shuffle(images)


fig, axs = plt.subplots(2, 5, figsize=(30,10))
fig.suptitle("samples without pneumothorax (train)", fontsize=30)
for j, path in enumerate(images[:10]):
    img = extract_dcm_pixel_array (path)
    x = j // 5
    y = j % 5
    axs[x,y].imshow(img, cmap='bone')
    axs[x,y].axis('off')
fig.subplots_adjust(top=0.9)


plt.show()


# In[ ]:


images = df_train.query ( "EncodedPixels.isnull()",  engine='python')["path"].values
np.random.shuffle(images)


fig, axs = plt.subplots(2, 5, figsize=(30,10))
fig.suptitle("samples without a mask in train-rle.csv (train)", fontsize=30)
for j, path in enumerate(images[:10]):
    img = extract_dcm_pixel_array (path)
    x = j // 5
    y = j % 5
    axs[x,y].imshow(img, cmap='bone')
    axs[x,y].axis('off')
fig.subplots_adjust(top=0.9)


plt.show()


# in train set there are 37 images without a mask in train-rle.csv

# In[ ]:


images = df_test [ ~ df_test["ImageId"].isin(df_sub["ImageId"])]["path"].values
np.random.shuffle(images)


fig, axs = plt.subplots(1, 5, figsize=(30,10))
fig.suptitle("not in sample submission (test)", fontsize=30)
for j, path in enumerate(images[:5]):
    img = extract_dcm_pixel_array (path)
    y = j % 5
    axs[y].imshow(img, cmap='bone')
    axs[y].axis('off')
fig.subplots_adjust(top=1.0)


plt.show()


# in test set there are 5 images not in sample_submission.csv

# # Metadata distributions

# In[ ]:


train_data = df_train.drop(["EncodedPixels"],axis=1).drop_duplicates()
train_data["dataset"] = "train"
test_data = df_test
test_data["dataset"] = "test"
all_data = train_data.append(test_data, sort=False)

g = sns.FacetGrid(all_data.query ( "patient_age < 100"  ), 
                  hue='dataset', row="patient_sex", col='view_position',  margin_titles=True)
g.map(sns.distplot, "patient_age").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Train and Test distribution")
plt.show()



# In[ ]:


pvt_train = train_data.groupby(["view_position", "patient_sex"]).agg({"ImageId":'count'}).reset_index().pivot(index="patient_sex",columns="view_position", values="ImageId")
pvt_test = test_data.groupby(["view_position", "patient_sex"]).agg({"ImageId":'count'}).reset_index().pivot(index="patient_sex",columns="view_position", values="ImageId")

f, axes = plt.subplots(1, 2, figsize=(12, 6))

g = sns.heatmap(pvt_train, annot=True, fmt="d", linewidths=.5, ax=axes[0])
axes[0].title.set_text("Train distribution")
sns.heatmap(pvt_test, annot=True, fmt="d", linewidths=.5, ax=axes[1])
axes[1].title.set_text("Test distribution")

plt.show()



# In[ ]:


g = sns.FacetGrid(train_data.query ( "patient_age < 100"  ), palette= {0:"green", 1:"gray"},
                  hue='pneumothorax', row="patient_sex", col='view_position',  margin_titles=True)
g.map(sns.distplot, "patient_age").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Train distribution")
plt.show()


# # The bimodal mean_pixel_value distribution
# see [@Giulia Savorgnan's](https://www.kaggle.com/giuliasavorgnan) [discussion](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/97525)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_data["mean_pixel_value"] = train_data["path"].map(lambda x : extract_dcm_pixel_array(x).mean() )\ntest_data["mean_pixel_value"] = test_data["path"].map(lambda x : extract_dcm_pixel_array(x).mean() )')


# In[ ]:


sns.distplot(train_data["mean_pixel_value"], label="train")
sns.distplot(test_data["mean_pixel_value"], label="test")
plt.legend()
plt.show()


# In[ ]:


g = sns.FacetGrid(train_data,
                  hue='pneumothorax', row="patient_sex", col='view_position',  margin_titles=True)
g.map(sns.distplot, "mean_pixel_value").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Train distribution")
plt.show()


# In[ ]:


high_means = train_data.query( "mean_pixel_value > 140 and  mean_pixel_value < 170" )["path"].values
np.random.shuffle(high_means)
low_means = train_data.query( "mean_pixel_value > 90 and  mean_pixel_value < 110" )["path"].values
np.random.shuffle(low_means)

fig, axs = plt.subplots(2, 10, figsize=(20,5))

for j,path in enumerate(low_means[:10]):
    img = extract_dcm_pixel_array (path)
    axs[0,j].imshow(img, cmap='bone')
    axs[0,j].axis('off')
    axs[0,j].title.set_text(str(round(img.mean(),2)))

for j, path in enumerate(high_means[:10]):
    img = extract_dcm_pixel_array (path)
    axs[1,j].imshow(img, cmap='bone')
    axs[1,j].axis('off')
    axs[1,j].title.set_text(str(round(img.mean(),2)))


plt.show()


# ### Extremes of the distribution

# In[ ]:


train_data = train_data.sort_values(by=["mean_pixel_value"])

fig, axs = plt.subplots(2, 10, figsize=(20,5))

for j, path in enumerate ( train_data["path"].values[:10] ):
    img = extract_dcm_pixel_array (path)
    axs[0,j].imshow(img, cmap='bone')
    axs[0,j].axis('off')
    axs[0,j].title.set_text(str(round(img.mean(),2)))

for j,path in enumerate(train_data["path"].values[-10:] ):
    img = extract_dcm_pixel_array (path)
    axs[1,j].imshow(img, cmap='bone')
    axs[1,j].axis('off')
    axs[1,j].title.set_text(str(round(img.mean(),2)))

plt.show()
    


# #### Extremes of the distribution: rescaling intensities

# In[ ]:


# Rescaling intensities doesn't seem to solve the issue
fig, axs = plt.subplots(2, 10, figsize=(20,5))

for j, path in enumerate ( train_data["path"].values[:10] ):
    img = extract_dcm_pixel_array (path)
    img = exposure.rescale_intensity(img, in_range=tuple(np.percentile(img, (2, 98))))
    axs[0,j].imshow(img, cmap='bone')
    axs[0,j].axis('off')
    axs[0,j].title.set_text(str(round(img.mean(),2)))

for j,path in enumerate(train_data["path"].values[-10:] ):
    img = extract_dcm_pixel_array (path)
    img = exposure.rescale_intensity(img, in_range=tuple(np.percentile(img, (2, 98))))
    axs[1,j].imshow(img, cmap='bone')
    axs[1,j].axis('off')
    axs[1,j].title.set_text(str(round(img.mean(),2)))

plt.show()


# #### Extremes of the distribution: histogram equalization

# In[ ]:


# Histogram equalization!
fig, axs = plt.subplots(2, 10, figsize=(20,5))

for j, path in enumerate ( train_data["path"].values[:10] ):
    img = extract_dcm_pixel_array (path)
    img = exposure.equalize_hist(img)
    axs[0,j].imshow(img, cmap='bone')
    axs[0,j].axis('off')
    axs[0,j].title.set_text(str(round(img.mean(),2)))

for j,path in enumerate(train_data["path"].values[-10:] ):
    img = extract_dcm_pixel_array (path)
    img = exposure.equalize_hist(img)
    axs[1,j].imshow(img, cmap='bone')
    axs[1,j].axis('off')
    axs[1,j].title.set_text(str(round(img.mean(),2)))

plt.show()


# #### Plot distribution again after histogram equalization 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_data["eq_mean_pixel_value"] = train_data["path"].map(lambda x : exposure.equalize_hist(extract_dcm_pixel_array(x)).mean() )\ntest_data["eq_mean_pixel_value"] = test_data["path"].map(lambda x : exposure.equalize_hist(extract_dcm_pixel_array(x)).mean() )')


# In[ ]:


sns.distplot(train_data["eq_mean_pixel_value"], label="eq_train")
sns.distplot(test_data["eq_mean_pixel_value"], label="eq_test")
plt.legend()
plt.show()


# # Images with multiple pneumothorax masks

# In[ ]:


multiple_masks = train_data.query ( "rle_count > 1",  engine='python').sort_values(by="rle_count", ascending=False)
multiple_masks =  multiple_masks[["ImageId","path","rle_count"]][:5]

for i, row in multiple_masks.iterrows():
    path = row["path"]
    image_id = row["ImageId"]
    rle = df_train.query( "ImageId == '" + image_id + "'" )
    
    img = extract_dcm_pixel_array (path)
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(20,10))
    
    #plt.set_title(image_id)
    ax[0].imshow(img, cmap='bone')
    ax[1].imshow(img, cmap='bone')

    rle_count = row["rle_count"]
    
    rle_mask = np.zeros ( (1024, 1024) )    
    for i, row in rle.iterrows():
        mask =  row["EncodedPixels"] 
        rle_mask += rle2mask(mask, 1024, 1024).T 

    ax[1].imshow(rle_mask, alpha=0.5, cmap="Blues")    

    plt.axis('off')
    plt.show()

    


# in train set there are 624 images with multiple masks
# (in sample_submission there are 78 images with multiple entries)
