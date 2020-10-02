#!/usr/bin/env python
# coding: utf-8

# # Import the data

# In[ ]:


import pandas as pd

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(train.info())


# In[ ]:


print(test.info())


# In[ ]:


train["id"] = [int(i.split("_")[1]) for i in train.ID_code]
label = train.target
train.drop(["target", "ID_code"], axis=1, inplace=True)


# # Baseline Lightgbm

# In[ ]:


import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


model  = LGBMClassifier(num_iterations=500, learning_rate=0.1)
gbm = cross_val_score(model, train, label, scoring="roc_auc", cv=5)
print([round(i, 4) for i in gbm])
print(np.mean(gbm), "+/-", np.std(gbm))


# In[ ]:


model  = LGBMClassifier(num_iterations=1000, learning_rate=0.05)
gbm = cross_val_score(model, train, label, scoring="roc_auc", cv=5)
print([round(i, 4) for i in gbm])
print(np.mean(gbm), "+/-", np.std(gbm))


# In[ ]:


model  = LGBMClassifier(num_iterations=2000, learning_rate=0.01)
gbm = cross_val_score(model, train, label, scoring="roc_auc", cv=5)
print([round(i, 4) for i in gbm])
print(np.mean(gbm), "+/-", np.std(gbm))


# In[ ]:


model  = LGBMClassifier(num_iterations=2000, learning_rate=0.03)
gbm = cross_val_score(model, train, label, scoring="roc_auc", cv=5)
print([round(i, 4) for i in gbm])
print(np.mean(gbm), "+/-", np.std(gbm))


# # Save the model

# In[ ]:


model = LGBMClassifier(num_iterations=2000, learning_rate=0.03).fit(train, label)


# In[ ]:


ids = test.ID_code
test["id"] = [int(i.split("_")[1]) for i in test.ID_code]
test.drop(["ID_code"], axis=1, inplace=True)
probabilities = model.predict_proba(test)

submission = pd.DataFrame({
    "ID_code": ids,
    "target": probabilities[:,1]
})
submission.to_csv("submission_lightgbm.csv", index=False)


# In[ ]:


import pickle

pickle.dump(model, open("model_gbm.pickle", "wb"))

