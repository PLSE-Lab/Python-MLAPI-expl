#!/usr/bin/env python
# coding: utf-8

# # ULMFiT - the first place solution 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


import fastai
fastai.__version__


# In[ ]:


from fastai.text import *
from fastai.callbacks import *


# ## Load the data 

# In[ ]:


data_train = pd.read_csv("../input/train.csv")
data_valid = pd.read_csv("../input/valid.csv")
data_test = pd.read_csv("../input/test.csv")


# Some rows have missing article titles or texts - let's fix that.

# In[ ]:


data_train.fillna("xxempty", inplace=True)
data_valid.fillna("xxempty", inplace=True)
data_test.fillna("xxempty", inplace=True)


# Let's join them together, putting titles at the end.

# In[ ]:


data_train["full"] = data_train["text"].apply(lambda x: x + " xxtitle ") + data_train["title"]
data_valid["full"] = data_valid["text"].apply(lambda x: x + " xxtitle ") + data_valid["title"]
data_test["full"] = data_test["text"].apply(lambda x: x + " xxtitle ") + data_test["title"]


# In[ ]:


data_train["is_valid"] = False
data_valid["is_valid"] = True


# ## Language model fine-tuning 

# In[ ]:


data_lm = (TextList.from_df(pd.concat([data_train, data_valid]), cols=["full"])
           .split_from_df("is_valid")
           .label_for_lm()
           .databunch())


# In[ ]:


lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3, pretrained=True)


# In[ ]:


lm.lr_find()


# In[ ]:


lm.recorder.plot(suggestion=True)


# In[ ]:


lm.fit_one_cycle(1, 4e-3, callbacks=[SaveModelCallback(lm, name="best_lm")], moms=(0.8,0.7))


# In[ ]:


lm.unfreeze()


# In[ ]:


lm.fit_one_cycle(5, 4e-3, callbacks=[SaveModelCallback(lm, name="best_lm")], moms=(0.8,0.7))


# In[ ]:


lm.load("best_lm")


# In[ ]:


lm.save_encoder("enc")


# ## Classification

# In[ ]:


data_clf = (TextList.from_df(pd.concat([data_train, data_valid]), vocab=data_lm.vocab, cols=["full"]).
           split_from_df("is_valid").
           label_from_df("label").
           add_test(data_test["full"]).
           databunch()
          )


# In[ ]:


clf = text_classifier_learner(data_clf, AWD_LSTM, drop_mult=0.3)


# A Kaggle kernel's memory is not infinite, so let's do some clean up.

# In[ ]:


del lm
torch.cuda.empty_cache()


# In[ ]:


clf.load_encoder("enc")


# In[ ]:


clf.lr_find()


# In[ ]:


clf.recorder.plot()


# In[ ]:


clf.fit(3, 2e-3, callbacks=[SaveModelCallback(clf, name="best_clf")])


# In[ ]:


clf.load("best_clf")


# In[ ]:


clf.unfreeze()


# In[ ]:


clf.fit(1, 3e-4, callbacks=[SaveModelCallback(clf, name="best_clf_ft1")])


# In[ ]:


clf.fit(1, 3e-4, callbacks=[SaveModelCallback(clf, name="best_clf_ft2")])


# ## Validation 

# In[ ]:


pred_val = clf.get_preds(DatasetType.Valid, ordered=True)


# In[ ]:


pred_val_l = pred_val[0].argmax(1)


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(pred_val[1], pred_val_l))


# ## Make a submission 

# In[ ]:


pred_test, label_test = clf.get_preds(DatasetType.Test, ordered=True)


# In[ ]:


pred_test_ = pred_test.argmax(1)
pred_test_l = [data_clf.train_ds.y.classes[n] for n in pred_test_]


# In[ ]:


res = pd.Series(pred_test_l, index=data_test.index, name="label")


# In[ ]:


res.index.name = "id"


# In[ ]:


pd.DataFrame(res).to_csv("submission.csv")

