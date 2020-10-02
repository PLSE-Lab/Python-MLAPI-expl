#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Let's start the data modelling process which is the second phase of our project.

# # Copying the datasets
# Yes. You read that right. I'm copying the datasets from ``input`` directory to ``working`` directory in order to get rid of the **read-only** filesystem errors.

# Check the directory of the datasets (currently)

# In[ ]:


get_ipython().system('ls ../input/pap-smear-datasets')


# Let's copy them to the ``working`` directory

# In[ ]:


get_ipython().system('cp -r ../input/pap-smear-datasets ../working/')


# Let's see if they actually get copied into the ``working`` directory

# In[ ]:


get_ipython().system('ls ../working/')


# Oh you beauty...

# # Imports
# We need the dependencies in order to build a model using the Fastai library.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from fastai import *
from fastai.vision import *
from fastai.utils.ipython import *
from fastai.metrics import *
from fastai.callbacks.tracker import SaveModelCallback


# # Data pipeline
# As observed from my previous tutorial, let's build a data pipeline of the Sipakmed pap smear dataset using the data-block API of the fastai library

# In[ ]:


path = Path("../working/pap-smear-datasets")
path.ls()


# In[ ]:


classes = ["normal", "abnormal"]

def labelling_func(fname):
    c = fname.parent.name
    if "abnormal" in c or "benign" in c:
        return classes.index("abnormal")
    else:
        return classes.index("normal")

tfms = get_transforms(flip_vert=True, max_zoom=0.5, max_warp=0.0)

with gpu_mem_restore_ctx():
    data = (ImageList.from_folder(path/"sipakmed_fci_pap_smear")
           .split_by_rand_pct(seed=19)
           .label_from_func(labelling_func)
           .transform(tfms, size=32)
           .databunch(bs=8)
           .normalize(imagenet_stats))


# Let's check whether the pipeline was built as per our requirements...

# In[ ]:


print(data)
data.show_batch(rows=2, figsize=(9, 9))


# # Defining our metrics for performance evaluation
# We need some metrics to evaluate our model's performance on the validation set. With these, we can compare our model with rest of the models seen in research papers.

# We will be using the following metrics :-
# * Accuracy
# * Sensitivity (or Recall)
# * Specificity
# * H-mean
# * F1-score
# * AUROC score

# In[ ]:


class PapMetrics(ConfusionMatrix):
    
    def on_epoch_end(self, last_metrics, **kwargs):
        self.tn, self.fn, self.fp, self.tp = self.cm.flatten().numpy().ravel()
    
    def calc_prec(self):
        return self.tp / (self.tp + self.fp)
    
    def calc_recall(self):
        return self.tp / (self.tp + self.fn)
    
    def calc_fbeta(self):
        prec = self.calc_prec()
        recall = self.calc_recall()
        return 2 * ((prec * recall) / (prec + recall))
    
    def calc_spec(self):
        return self.tn / (self.tn + self.fp)
    
    def calc_hmean(self):
        sens = self.calc_recall()
        spec = self.calc_spec()
        return 2 * ((sens * spec) / (sens + spec))
    
    def calc_acc(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
    
class Accuracy(PapMetrics):
    
    def on_epoch_end(self, last_metrics, **kwargs):
        super().on_epoch_end(last_metrics, **kwargs)
        return add_metrics(last_metrics, self.calc_acc())
    
class Sensitivity(PapMetrics):
    
    def on_epoch_end(self, last_metrics, **kwargs):
        super().on_epoch_end(last_metrics, **kwargs)
        return add_metrics(last_metrics, self.calc_recall())
    
class Specificity(PapMetrics):
    
    def on_epoch_end(self, last_metrics, **kwargs):
        super().on_epoch_end(last_metrics, **kwargs)
        return add_metrics(last_metrics, self.calc_spec())
    
class HMean(PapMetrics):
    
    def on_epoch_end(self, last_metrics, **kwargs):
        super().on_epoch_end(last_metrics, **kwargs)
        return add_metrics(last_metrics, self.calc_hmean())
    
class F1Score(PapMetrics):
    
    def on_epoch_end(self, last_metrics, **kwargs):
        super().on_epoch_end(last_metrics, **kwargs)
        return add_metrics(last_metrics, self.calc_fbeta())
    
our_metrics = [Accuracy(), Sensitivity(), Specificity(), HMean(), F1Score(), AUROC()]


# # Building our model with some tricks..
# Let's build a simple started model using Fastai library

# In[ ]:


with gpu_mem_restore_ctx():
    learner = cnn_learner(data, models.resnet34,
                          metrics=our_metrics).to_fp16()
    callback=[SaveModelCallback(learner, every='improvement', monitor='accuracy', name='best-rn34-stage1')]


# Let's train it for a couple of epochs in order to see whether all our efforts truly pay-off

# In[ ]:


with gpu_mem_restore_ctx():
    learner.fit_one_cycle(2, callbacks=callback)
    learner.save("rn34-stage1-last")


# ## Note
# The models will be saved inside the ``models`` directory in the path used to build your data pipeline

# In[ ]:


get_ipython().system('ls ../working/pap-smear-datasets/sipakmed_fci_pap_smear/models')


# # Conclusion
# That's a wrap for this tutorial. Next time, we'll be training our models on the Herlev and SipakMed datasets respectively.
