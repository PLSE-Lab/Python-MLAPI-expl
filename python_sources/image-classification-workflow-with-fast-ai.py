#!/usr/bin/env python
# coding: utf-8

# # Introduction
# `fastai` is a free deep learning API built on [PyTorch V1](https://pytorch.org/). The [fast.ai team](https://www.fast.ai/2018/10/02/fastai-ai/) incorporates their reseach breakthroughs into the software, enabling users to achieve more accurate results faster and with fewer lines of code.
# 
# This kernel illustrates the simplicity of deploying the `fastai.vision` package for image classification tasks. I am in no way a domain expert in this topic, in fact having no domain knowledge at all before this competition! I will heavily rely on published kernels (which are all referenced under [Acknowledgements](#Acknowledgements)) in guidance for setting hyperparameters in this task.
# 
# I will be deploying standard techniques taught in the fast.ai course to see how well these techniques can perform without needing expert knowledge. The techniques are:
# 1. Learning rate finder
# 2. 1-cycle learning
# 3. Differential learning rates for model finetuning
# 4. Data augmentation
# 5. Test time augmentation
# 6. Transfer learning via low-resolution images
# 
# This kernel had the previous name of **Minimal fast.ai kit for image classification**, which is a slight misnomer now, considering the detailed techniques being deployed in this image classification task.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Contents
# 1. [Skeleton code](#Skeleton-code)
# * [Import packages](#Import-packages)
# * [Exploratory data analysis](#Exploratory-data-analysis)
# * [Data loading and preparation](#Data-loading-and-preparation)
# * [Model creation](#Model-creation)
# * [Model training](#Model-training).
# * [Model interpretation](#Model-interpretation)
# * [Transfer learning](#Transfer-learning)
# * [Generating submission](#Generating-submission)
# * [Future work](#Future-work)
# * [Acknowledgements](#Acknowledgements)

# # Skeleton code

# Following from the initial idea of showing the simplicity of using the `fastai` library, below is a code snippet containing 27 lines of code using default settings for a base model generation. Of these 27 lines, 10 lines are used to generate the submission file for the required format.
# ```
# from fastai import *
# from fastai.vision import *
# 
# data = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",
#                                test = 'test',suffix=".tif", size = 36, ds_tfms = get_transforms())
# data.path = pathlib.Path('.')
# data.normalize(imagenet_stats)
# 
# learn = create_cnn(data,resnet50,pretrained = True,metrics = accuracy)
# learn.fit_one_cycle(5)
# 
# learn.unfreeze()
# learn.lr_find()
# learn.recorder.plot()
# learn.fit_one_cycle(3,max_lr = slice(1e-6,3e-4))
# 
# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_top_losses(9)
# interp.plot_confusion_matrix()
# preds,y = learn.TTA()
# acc = accuracy(preds, y)
# print('The validation accuracy is {} %.'.format(acc * 100))
# 
# def generateSubmission(learner):
#     submissions = pd.read_csv('../input/sample_submission.csv')
#     id_list = list(submissions.id)
#     preds,y = learner.TTA(ds_type=DatasetType.Test)
#     pred_list = list(preds[:,1])
#     pred_dict = dict((key, value.item()) for (key, value) in zip(learner.data.test_ds.items,pred_list))
#     pred_ordered = [pred_dict[Path('../input/test/' + id + '.tif')] for id in id_list]
#     submissions = pd.DataFrame({'id':id_list,'label':pred_ordered})
#     submissions.to_csv("submission_{}.csv".format(pred_score),index = False)
#  
#  generateSubmission(learn)
# ```

# # Import packages

# In[ ]:


from fastai import *
from fastai.vision import *
from torchvision.models import * 

import os
import matplotlib.pyplot as plt


# # Exploratory data analysis

# Exploratory data analysis should be the first step of every data science task. Due to the lack of domain knowledge, I will only check for the number of classes and the number of items per class. Imbalanced datasets may require resampling of the data to ensure proper training.

# In[ ]:


path = Path("../input")
labels = pd.read_csv(path/"train_labels.csv")
labels.head()


# In[ ]:


print(labels["label"].nunique()); classes = list(set(labels["label"])); classes


# In[ ]:


for i in classes:
    print("Number of items in class {} is {}".format(i,len(labels[labels["label"] == i])))


# # Data loading and preparation

# In[ ]:


tfms = get_transforms(do_flip = True,flip_vert = True,max_zoom = 1.1)


# In[ ]:


np.random.seed(123)
sz = 32
data = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",
                               test = 'test',suffix=".tif", size = sz,bs = 256,
                               ds_tfms = tfms)
data.path = pathlib.Path('.')
data.normalize(imagenet_stats)


# In[ ]:


print(data.classes); data.c


# In[ ]:


data.show_batch(rows = 3)


# # Model creation

# Submissions into the competition are [evaluated on the area under the ROC curve](https://www.kaggle.com/c/histopathologic-cancer-detection#evaluation) between the predicted probability and the observed target. Since we have a limited number of submissions per day, implementing a metric for the ROC AUC (which is non-standard in the fast.ai v1 library) allows us to run as many experiments we want.
# 
# At this point, I am not sure if changing the metric changes the loss function in the `Learner` to optimize the metric. I will be doing more reading up in that area. If anyone knows the answer to this, leave something in the comments below!

# In[ ]:


from sklearn.metrics import roc_auc_score

def auc_score(y_pred,y_true,tens=True):
    score = roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    if tens:
        score = tensor(score)
    return score


# In[ ]:


arch = models.densenet121
learn = cnn_learner(data,arch,pretrained = True,ps = 0.45,metrics = [auc_score,accuracy])


# # Model training

# The most important hyperparameter in training neural networks in general is the **learning rate**. Unfortunately as of now, there is no way of finding a good learning rate without trial-and-error. 
# 
# The library has made it convenient to test different learning rates. We find a good learning rate using the method `lr_find`, then plotting the graph of learning rates against losses. As a rule of thumb, the learning rate is chosen from a part of the graph where it is **steepest** and **most consistent**.

# `fit_one_cycle`is a method implemented by the library and proposed in [this paper](https://arxiv.org/pdf/1803.09820.pdf) to produce more accurate results and faster convergence. [This post](https://sgugger.github.io/the-1cycle-policy.html) is a great explanation of why `fit_one_cycle`works over the standard `fit`.

# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8,1e-2)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# A [simple rule of thumb suggested](https://github.com/hiromis/notes/blob/master/Lesson1.md) by Jeremy when selecting differential learning rates is to:
# 1. Set the first part of the slice (corresponding to the earlier layers of the model) to a learning rate much smaller than where the loss starts increasing.
# 2. Set the final slice to a learning rate 0.1x of that used when training the frozen model.

# In[ ]:


learn.fit_one_cycle(2,max_lr = slice(1e-5,1e-3))


# # Transfer learning

# We will now train the model on the same dataset, except we are using images of higher resolution. Intuitively, the 'concepts' learnt by the neural network will continue to be applied in training with the new set of images.

# ## 64x64 images

# In[ ]:


newTfms = get_transforms(do_flip = True,flip_vert = True,max_zoom = 1.25)
newSz = 64
newData = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",
                               test = 'test',suffix=".tif", size = newSz, ds_tfms = newTfms)
newData.path = pathlib.Path('.')
newData.normalize(imagenet_stats)
learn.data = newData


# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(8,1e-2)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2,max_lr = slice(1e-3/3,1e-3))


# In[ ]:


learn.save('stage-2')


# In[ ]:


preds,y = learn.TTA()
acc = accuracy(preds, y)
print('The validation accuracy is {} %.'.format(acc * 100))
pred_score = auc_score(preds,y).item()
print('The validation AUC is {}.'.format(pred_score))


# ## 96x96 images

# In[ ]:


newTfms = get_transforms(do_flip = True,flip_vert = True,max_zoom = 1.5)
newSz = 96
newData = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",
                               test = 'test',suffix=".tif", size = newSz, ds_tfms = newTfms)
newData.path = pathlib.Path('.')
newData.normalize(imagenet_stats)
learn.data = newData


# In[ ]:


learn.freeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4,1e-4)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2,max_lr = slice(1e-5/3,1e-5))


# In[ ]:


learn.save('stage-3')


# In[ ]:


preds,y = learn.TTA()
acc = accuracy(preds, y)
print('The validation accuracy is {} %.'.format(acc * 100))
pred_score = auc_score(preds,y).item()
print('The validation AUC is {}.'.format(pred_score))


# # Model interpretation

# At this stage, we would like to check the effectiveness of the `learn` model against our validation set (which is automatically generated by the `ImageDataBunch` object). We will use the following methods to evaluate the effectiveness.
# 1. Confusion matrix.
# 2. Accuracy.
# 3. ROC-AUC, as dictated in the competition evaluation.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_top_losses(6)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


preds,y = learn.TTA()
acc = accuracy(preds, y)
print('The validation accuracy is {} %.'.format(acc * 100))
pred_score = auc_score(preds,y).item()
print('The validation AUC is {}.'.format(pred_score))


# # Generating submission

# As the order of the images loaded into `data` is not necessarily the same order as that in the required submission, we will need to rearrange the predictions on the test set.
# 
# The desired order for submission is that of `sample_submission.csv`. while the order of the test set loaded into `data` can be accessed by calling `learn.data.test_ds.items`. We will first create a dictionary assigning each image in the test to its prediction by the model, then call the keys in the order in  `sample_submission.csv`.

# In[ ]:


def generateSubmission(learner):
    submissions = pd.read_csv('../input/sample_submission.csv')
    id_list = list(submissions.id)
    preds,y = learner.TTA(ds_type=DatasetType.Test)
    pred_list = list(preds[:,1])
    pred_dict = dict((key, value.item()) for (key, value) in zip(learner.data.test_ds.items,pred_list))
    pred_ordered = [pred_dict[Path('../input/test/' + id + '.tif')] for id in id_list]
    submissions = pd.DataFrame({'id':id_list,'label':pred_ordered})
    submissions.to_csv("submission_transferLearning_{}.csv".format(pred_score),index = False)


# In[ ]:


generateSubmission(learn)


# # Future work
# 
# 1. [DONE] Generate sample submission to ensure functional code. (0.6007)
# 2. [DONE] Implement skeleton model for baseline (0.9106).
# 3. [DONE] Prepare AUC metric (0.9233).
# 4. [DONE] Deploy reasoned data augmentation (0.9241).
# 5. [DONE] Deploy test-time augmentation (0.9364).
# 6. Test other architectures (which are found [here in the Pytorch docs](https://pytorch.org/docs/stable/torchvision/models.html))
#     1. [DONE] ResNet-34 (0.9362)
#     2. [DONE] DenseNet-169 (0.9370)
# 7. [DONE] Implement weight decay. (0.9368)
# 8. [DONE] Retrain model with higher resolution images. (0.9573)

# # Acknowledgements
# 
# * [qitvision](https://www.kaggle.com/qitvision/) for his [extremely well-explained kernel](https://www.kaggle.com/qitvision/a-complete-ml-pipeline-fast-ai) on the same competition and for answering my questions on data loading.
# * [Gunther](https://www.kaggle.com/guntherthepenguin) for providing the implementation of the AUC metric in [his kernel in the same competition.](https://www.kaggle.com/guntherthepenguin/fastai-v1-densenet169)
# * The [fast.ai team](https://www.fast.ai/about/) for creating the [library](https://docs.fast.ai/index.html) and [the v3 course](https://course.fast.ai/index.html) for teaching deep learning in a very accessible manner.
