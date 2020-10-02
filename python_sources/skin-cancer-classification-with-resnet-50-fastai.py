#!/usr/bin/env python
# coding: utf-8

# # Skin Cancer Detencion
# 
# 
# Training of neural networks for automated diagnosis of pigmented skin lesions is hampered by the small size and lack of diversity of available dataset of dermatoscopic images. We tackle this problem by releasing the HAM10000 ("Human Against Machine with 10000 training images") dataset. We collected dermatoscopic images from different populations, acquired and stored by different modalities. The final dataset consists of 10015 dermatoscopic images which can serve as a training set for academic machine learning purposes. Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).
# 
# More than 50% of lesions are confirmed through histopathology (histo), the ground truth for the rest of the cases is either follow-up examination (follow_up), expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal). The dataset includes lesions with multiple images, which can be tracked by the lesion_id-column within the HAM10000_metadata file.
# 
# ![skin cancer](http://www.justscience.in/wp-content/uploads/2017/12/what-causes-skin-cancer.jpg)
# 
# 
# 
# 

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve

import os
print(os.listdir("../input"))


# ## Exploratory Data Analysis

# In[ ]:


# Paths and roots to the important files
path='../input/'
csv_file='../input/HAM10000_metadata.csv'


# In[ ]:


df=pd.read_csv(csv_file).set_index('image_id')
df.head()


# In[ ]:


# Categories of the diferent diseases
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


# In[ ]:


df.dx=df.dx.astype('category',copy=True)
df['labels']=df.dx.cat.codes # Convert the labels to numbers
df['lesion']= df.dx.map(lesion_type_dict)
df.head()


# In[ ]:


print(df.lesion.value_counts())


# In[ ]:


df.loc['ISIC_0027419','lesion']


# ## Countplot
# Here we notice tha we have data imbalance 

# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))
sns.countplot(y='lesion',data=df, hue="lesion",ax=ax1)


# ## Dataset

# In[ ]:


class CustomImageItemList(ImageItemList):
    def custom_label(self,df, **kwargs)->'LabelList':
        """Custom Labels from path"""
        file_names=np.vectorize(lambda files: str(files).split('/')[-1][:-4])
        get_labels=lambda x: df.loc[x,'lesion']
        #self.items is an np array of PosixPath objects with each image path
        labels= get_labels(file_names(self.items))
        y = CategoryList(items=labels)
        res = self._label_list(x=self,y=y)
        return res


# In[ ]:


def get_data(bs, size):
    train_ds = (CustomImageItemList.from_folder('../input', extensions='.jpg')
                    .random_split_by_pct(0.15)
                    .custom_label(df)
                    .transform(tfms=get_transforms(flip_vert=True),size=size)
                    .databunch(num_workers=2, bs=bs)
                    .normalize(imagenet_stats))
    return train_ds


# In[ ]:


data=get_data(16,224)


# In[ ]:


data.classes=list(np.unique(df.lesion))  
data.c= len(np.unique(df.lesion))  


# In[ ]:


data.show_batch(rows=3)


# ## Model ResNet50 

# In[ ]:


learner=create_cnn(data,models.resnet50,metrics=[accuracy], model_dir="/tmp/model/")


# In[ ]:


learner.loss_func=nn.CrossEntropyLoss()


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


learner.fit_one_cycle(10, 3e-3)


# In[ ]:


learner.unfreeze()


# In[ ]:


learner.lr_find()
learner.recorder.plot()


# In[ ]:


lr=1e-6
learner.fit_one_cycle(3, slice(3*lr,10*lr))


# In[ ]:


learner.save('stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)


# In[ ]:


interp.plot_confusion_matrix(figsize=(10,8))


# In[ ]:


interp.most_confused()


# ## Inference

# In[ ]:


pred_data=get_data(16,224)


# In[ ]:


pred_data.classes=list(np.unique(df.lesion))  
pred_data.c= len(np.unique(df.lesion)) 


# In[ ]:


pred_data.single_from_classes(path, pred_data.classes)


# In[ ]:


predictor = create_cnn(pred_data, models.resnet50, model_dir="/tmp/model/").load('stage-1')


# In[ ]:


img = open_image('../input/ham10000_images_part_2/ISIC_0029886.jpg')
img


# In[ ]:


pred_class,pred_idx,outputs = predictor.predict(img)
pred_class


# ## Predictions

# In[ ]:


# Predictions of the validation data
preds_val, y_val=learner.get_preds()


# ### Roc Curve
# With the ROC curve we will mesuare how good it's our model

# In[ ]:


#  ROC curve
fpr, tpr, thresholds = roc_curve(y_val.numpy(), preds_val.numpy()[:,1], pos_label=1)

#  ROC area
pred_score = auc(fpr, tpr)
print(f'ROC area is {pred_score}')


# In[ ]:


plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % pred_score)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")


# ## Heatmap

# In[ ]:


x,y = data.valid_ds[2]
x.show()
data.valid_ds.y[2]


# In[ ]:


def heatMap(x,y,data, learner, size=(0,224,224,0)):
    """HeatMap"""
    
    # Evaluation mode
    m=learner.model.eval()
    
    # Denormalize the image
    xb,_ = data.one_item(x)
    xb_im = Image(data.denorm(xb)[0])
    xb = xb.cuda()
    
    # hook the activations
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(y)].backward()

    # Activations    
    acts=hook_a.stored[0].cpu()
    
    # Avg of the activations
    avg_acts=acts.mean(0)
    
    # Show HeatMap
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(avg_acts, alpha=0.6, extent=size,
              interpolation='bilinear', cmap='magma')
    


# In[ ]:


heatMap(x,y,pred_data,learner)

