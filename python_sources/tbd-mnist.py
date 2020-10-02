#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
df.head()


# In[ ]:


sample = df.iloc[4].drop("label")


# In[ ]:


sample = sample.values.reshape(28, 28)


# In[ ]:


plt.matshow(sample, cmap="gray")
plt.show()


# In[ ]:


y = df["label"].values
X = df.drop("label", axis=1)


# In[ ]:


print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


# In[ ]:


predictions = rf_model.predict(X_test)


# In[ ]:


print(f1_score(y_test, predictions, average="weighted"))
print("Accuracy:", accuracy_score(y_test, predictions))


# In[ ]:


y_test[15]


# In[ ]:


plt.matshow(X_test.iloc[15].values.reshape(28, 28), cmap="gray")


# In[ ]:


rf_model.predict(X_test.iloc[15].values.reshape(1, -1))

