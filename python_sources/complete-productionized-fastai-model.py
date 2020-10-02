#!/usr/bin/env python
# coding: utf-8

# # Histopathological Cancer Detection 

# ## RoadMap
# - Import Libraries
# - Check GPU
# - EDA
# - Model Building
# - Model Improvement
# - Model Validation
# - Submission

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from fastai.vision import *
import fastai
from fastai.metrics import *
from fastai import *
from os import *
import seaborn as sns
from sklearn.metrics import auc,roc_curve,accuracy_score, roc_auc_score
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import random
np.random.seed(42)
from glob import glob 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


model_path='.'
path='/kaggle/input/histopathologic-cancer-detection/'
train_folder=f'{path}train'
test_folder=f'{path}test'
train_lbl=f'{path}train_labels.csv'

bs=64
num_workers=None 
sz=96


# ## Check GPU 

# In[ ]:


# Programming framework behind the scenes of NVIDIA GPU is CUDA
print(torch.cuda.is_available())
# Check if gpu is enabled
print(torch.backends.cudnn.enabled)


# # EDA

# In[ ]:


df_train = pd.read_csv(train_lbl)
print(f'Number of labels {len(df_train)}')


# In[ ]:


# Proportion of classes 
df_train['label'].value_counts(normalize=True)


# In[ ]:


sns.countplot(x='label',data=df_train)


# ### Analyze cancer and non-cancer cell

# In[ ]:


cancer_cell = df_train[df_train['label']==1].head()
cancer_cell


# In[ ]:


non_cancer_cell = df_train[df_train['label']==0].head()
non_cancer_cell


# In[ ]:


plt.subplot(1 , 2 , 1)
img = np.asarray(plt.imread(train_folder+'/'+cancer_cell.iloc[1][0]+'.tif'))
plt.title('METASTATIC CELL TISSUE')
plt.imshow(img)

plt.subplot(1 , 2 , 2)
img = np.asarray(plt.imread(train_folder+'/'+ non_cancer_cell.iloc[1][0]+'.tif'))
plt.title('NON-METASTATIC CELL TISSUE')
plt.imshow(img)

plt.show()


# In[ ]:


list = os.listdir(test_folder) # dir is your directory path
len(list)


# In[ ]:


list = os.listdir(train_folder) # dir is your directory path
len(list)


# ## Model Building
# - Model Training requires the objects of DataBunch and Learner

# ### Data Augmentation

# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=1.1,max_lighting=0.05, max_warp=0.)


# ### Create DataBunch object

# In[ ]:


data = ImageDataBunch.from_csv(path,folder='train',valid_pct=0.3,csv_labels=train_lbl,ds_tfms=tfms, size=90, suffix='.tif',test=test_folder,bs=64)


# In[ ]:


data.classes


# In[ ]:


print(data.c, len(data.train_ds), len(data.valid_ds))


# ### Data Preprocesing (Normalization)

# In[ ]:


stats=data.batch_stats()        
data.normalize(stats)
#data.normalize(imagenet_stats)


# In[ ]:


#See the classes and labels
data.show_batch(rows=3, figsize=(8,5))


# ### Model Training
# - Model Training requires DataBunch object and learner

# In[ ]:


model_dir = "/kaggle/working/tmp/models/"
os.makedirs('/kaggle/working/tmp/models/')


# #### Create Learning Classifier

# In[ ]:


#fastai comes with various models
dir(fastai.vision.models)


# In[ ]:


#create learner object by passing data bunch, specifying model architecture and metrics to use to evaluate training stats
learner_resnet50 = cnn_learner(data=data, base_arch=models.resnet50,model_dir=model_dir, metrics=[accuracy,error_rate], ps=0.5) #densenet201


# #### Classifier Training

# In[ ]:


lr_find(learner_resnet50)


# In[ ]:


learner_resnet50.recorder.plot()


# In[ ]:


defaults.device = torch.device('cuda') # makes sure the gpu is used


# ### Transfer learning
# - Allows you to train nets with 1/100th less time using 1/100 less data.

# In[ ]:


learner_resnet50.fit_one_cycle(1, 1e-02)


# In[ ]:


learner_resnet50.recorder.plot(return_fig=True)


# In[ ]:


#See how the learning rate and momentum varies with the training and losses
learner_resnet50.recorder.plot_lr(show_moms=True)


# In[ ]:


learner_resnet50.recorder.plot_losses(show_grid=True)


# In[ ]:


learner_resnet50.show_results(alpha=1)


# In[ ]:


#save weights in a file
learner_resnet50.save('stage-1',return_path=True)


# ## Model Improvement
# - Generally, when you call fit_one_cycle it only trains the last or last few layers. To improve this better, you need to call learn.unfreeze() to unfreeze the model and train it again.

# In[ ]:


#Unfreeze the encoder resnet
learner_resnet50.unfreeze()


# In[ ]:


lr_find(learner_resnet50)
learner_resnet50.recorder.plot()


# In[ ]:


#slice suggests is, train the initial layers at start value specified and last layer at the end value specified and interpolate for the rest of the layers
learner_resnet50.fit_one_cycle(1,slice(1e-06,1e-05),pct_start=0.8)


# In[ ]:


learner_resnet50.recorder.plot_losses()


# In[ ]:


learner_resnet50.save('stage-2',return_path=True)


# ## Model Validation
#     - Plot top losses images
#     - Confusion Matrix
#     - Validate across validation set by auc_score and accuracy
#     - Plot roc_curve

# In[ ]:


#create interpreter object
interp = ClassificationInterpretation.from_learner(learner_resnet50)


# In[ ]:


#Plot the biggest losses of the model
interp.plot_top_losses(9,figsize=(12,12),heatmap=False)


# In[ ]:


losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(15,5))


# In[ ]:


#To view the list of classes most misclassified as a list
#Sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences
interp.most_confused(min_val=2)


# #### AUC SCORE

# In[ ]:


pred_val ,y_val = learner_resnet50.get_preds()

def auc_score(y_pred,y_true,tens=True):
    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    if tens:
        score=tensor(score)
    else:
        score=score
    return score

pred_score=auc_score(pred_val ,y_val)
pred_score


# #### ACCURACY

# In[ ]:


pred_score_acc=accuracy(pred_val ,y_val)
pred_score_acc


# #### ROC CURVE

# In[ ]:


fpr, tpr, thresholds = roc_curve(y_val.numpy(), pred_val.numpy()[:,1], pos_label=1)
pred_score_auc = auc(fpr, tpr)
print(f'ROC area: {pred_score_auc}')


# In[ ]:


plt.figure()
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % pred_score_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")


# ## Predictions
#     - Export the final learner object for productionizing
#     - Prediction using above saved model on:
#         - Train and validation images
#         - Uploaded single image
#         - Test images        

# In[ ]:


learner_resnet50.export('/kaggle/working/tmp/models/export.pkl')


# #### Prediction using saved model on training and validation images

# In[ ]:


loaded_learner = load_learner(Path(model_dir))
loaded_learner.data.classes


# #### Predict on train data

# In[ ]:


img, cat = data.train_ds[0]
img.show()
print(cat)


# In[ ]:


pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
print(pred_class, pred_idx,pred_probs)


# #### Predict on validation data

# In[ ]:


img, cat = data.valid_ds[1]
img.show()
print(cat)


# In[ ]:


pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
print(pred_class, pred_idx,pred_probs)


# #### Prediction using saved model on testing single image

# In[ ]:


img = open_image(Path('../input/test-image/test_img.tif'))
pred_class,pred_idx,pred_probs = loaded_learner.predict(img)
img.show()
targets = ['Non-Cancerous','Cancerous'] #since sequence of classes in data is as 0,1
print("Tissue cell is identified as" , targets[pred_idx] , "with probability of", float(pred_probs[pred_idx]*100))


# #### Prediction using saved model on testing data for submission purpose

# In[ ]:


loaded_learner_val = load_learner(Path(model_dir),test=ImageList.from_folder(Path(test_folder)))


# In[ ]:


pred_test ,y_test = loaded_learner_val.get_preds(ds_type=DatasetType.Test)


# ## Submission

# In[ ]:


sub=pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv').set_index('id')


# In[ ]:


clean_names = np.vectorize(lambda imgname: str(imgname).split('/')[-1][:-4])
cleaned_names = clean_names(data.test_ds.items).astype(str)


# In[ ]:


sub.loc[cleaned_names,'label']=pred_test.numpy()[:,1]
sub.to_csv(f'/kaggle/working/submission_{int(pred_score_auc*100)}auc.csv')


# In[ ]:


predicted_prob_test = pd.read_csv('./submission_98auc.csv')
predicted_prob_test.head(10)


# ### Feel free to share doubts, feedbacks or concerns. Also, fuel some motivation by upvoting if notebook has enhanced your learning. 
# 
# <b> Happy Learning! 
# 
