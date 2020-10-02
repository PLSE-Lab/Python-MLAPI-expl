#!/usr/bin/env python
# coding: utf-8

# # Histopathologic Cancer Detection
# 
# This challenge will focus on the detection and classification of breast cancer metastases in lymph nodes. Lymph nodes are small glands that filter lymph, the fluid that circulates through the lymphatic system. 
# ![bcancer](https://camelyon17.grand-challenge.org/media/CAMELYON17/public_html/breast_lymph_node.png)
# The lymph nodes in the axilla are the first place breast cancer is likely to spread. Metastatic involvement of lymph nodes is one of the most important prognostic factors in breast cancer. Prognosis is poorer when cancer has spread to the lymph nodes. This is why lymph nodes are surgically removed and examined microscopically. However, the diagnostic procedure for pathologists is tedious and time-consuming. But most importantly, small metastases are very difficult to detect and sometimes they are missed.

# In[ ]:


get_ipython().run_cell_magic('bash', '', "pip install git+https://github.com/fastai/fastai2\npip install torch torchvision feather-format pillow=='6.2.0' kornia pyarrow --upgrade   > /dev/null")


# In[ ]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from fastai2.data.all import *
from fastai2.vision.all import *
from fastai2.vision.core import *
from fastai2.vision.data import *
from sklearn.metrics import auc, roc_curve, precision_recall_curve, classification_report

import os
print(os.listdir("../input"))


# ## Exploratory Data Analysis
# In this dataset, you are provided with a large number of small pathology images to classify. Files are named with an image id. The train_labels.csv file provides the ground truth for the images in the train folder. You are predicting the labels for the images in the test folder. A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable fully-convolutional models that do not use zero-padding, to ensure consistent behavior when applied to a whole-slide image.

# In[ ]:


path=Path('../input')
trn_path = path/'train'
tst_path = path/'test'
labels = pd.read_csv(path/'train_labels.csv').set_index('id')
labels.head()


# In[ ]:


print(f'Number of labels {len(labels)}')
sns.countplot(x='label',data=labels)


# In[ ]:


img_fn = trn_path.ls()[0]
get_labels = lambda x: labels.loc[x.name[:-4],'label']
print(f'Label: {get_labels(img_fn)}')
PILImage.create(img_fn)


# In[ ]:


splitter = RandomSplitter()
item_tfms = [Resize(224)]
batch_tfms=[*aug_transforms(flip_vert=True,max_zoom=1.2), Normalize.from_stats(*imagenet_stats)]


# In[ ]:


data_block = DataBlock(blocks=[ImageBlock, CategoryBlock],
                  get_items=get_image_files,
                  get_y=get_labels,
                  splitter=splitter,
                  item_tfms=item_tfms,
                  batch_tfms=batch_tfms)


# In[ ]:


data = data_block.dataloaders(trn_path, bs=64)
data.show_batch()


# In[ ]:


learner= cnn_learner(data, xresnet50, metrics=[accuracy]).to_fp16()


# In[ ]:


learner.lr_find()


# In[ ]:


# learner.fine_tune(20)
learner.fine_tune(1, 1e-2)


# In[ ]:


learner.save('stage1')


# In[ ]:


learner.fine_tune(5)


# ##  Confusion Matrix

# In[ ]:


interp = ClassificationInterpretation.from_learner(learner)
interp.plot_confusion_matrix(figsize=(10,8))


# ## Production 

# # Heatmap

# ## Test Prediction

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


# In[ ]:


tst_files = get_image_files(tst_path)
data.test_dl(tst_files)


# In[ ]:


dl = learner.dls.test_dl(tst_files) 


# In[ ]:


preds_tst, y_tst=learner.get_preds(dl=dl)


# In[ ]:


# Predictions on the Test data
# preds_test,y_test = learner.TTA(ds_type=DatasetType.Test)
# preds_test, y_test=learner.get_preds(ds_type=DatasetType.Test)


# ## Submision

# In[ ]:


sub=pd.read_csv(f'{path}/sample_submission.csv').set_index('id')
sub.head()


# In[ ]:


names=np.vectorize(lambda img_name: str(img_name).split('/')[-1][:-4]) 
file_names= names(data.test_ds.items).astype(str)


# In[ ]:


sub.loc[file_names,'label']=preds_test.numpy()[:,1]
sub.to_csv(f'submission_{pred_score}.csv')


# In[ ]:


sub.head()


# In[ ]:




