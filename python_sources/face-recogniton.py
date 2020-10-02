#!/usr/bin/env python
# coding: utf-8

# # Face Recognition
# 
# See: https://github.com/paoloripamonti/face-recognition.git

# In[ ]:


get_ipython().system('pip install -U git+https://github.com/paoloripamonti/face-recognition.git')


# In[ ]:


from face_recognition import FaceRecognition

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, roc_auc_score, accuracy_score

import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import random
import numpy as np
import cv2
import base64
from tqdm import tqdm
import requests
from pprint import pprint


# ### Settings

# In[ ]:


ROOT_FOLDER ="/kaggle/input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/"
MODEL_PATH = "lfw_model.pkl"


# ### Read dataset

# In[ ]:


dataset = []
for path in glob.iglob(os.path.join(ROOT_FOLDER, "**", "*.jpg")):
    person = path.split("/")[-2]
    dataset.append({"person":person, "path": path})
    
dataset = pd.DataFrame(dataset)
dataset = dataset.groupby("person").filter(lambda x: len(x) > 10)
dataset.head(10)


# In[ ]:


dataset.groupby("person").count()[:200].plot(kind='bar', figsize=(20,5))


# In[ ]:


plt.figure(figsize=(20,10))
for i in range(20):
    idx = random.randint(0, len(dataset))
    img = plt.imread(dataset.path.iloc[idx])
    plt.subplot(4, 5, i+1)
    plt.imshow(img)
    plt.title(dataset.person.iloc[idx])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()


# ### Split train & test

# In[ ]:


train, test = train_test_split(dataset, test_size=0.1, random_state=0)
print("Train:",len(train))
print("Test:",len(test))


# ### Fit Face Recognition

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fr = FaceRecognition()')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'fr.fit_from_dataframe(train)')


# In[ ]:


fr.save(MODEL_PATH)


# ### Evaluate

# In[ ]:


get_ipython().run_cell_magic('time', '', 'y_test, y_pred, y_scores = [],[],[]\nfor idx in tqdm(range(len(test))):\n    path = test.path.iloc[idx]\n    result = fr.predict(path)\n    for prediction in result["predictions"]:\n        y_pred.append(prediction["person"])\n        y_scores.append(prediction["confidence"])\n        y_test.append(test.person.iloc[idx])')


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


print("Accuracy: %f" % accuracy_score(y_test, y_pred))


# In[ ]:


person = "George_W_Bush"
path = test[test.person==person]["path"].iloc[0]
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = fr.predict(path)
file_bytes = np.fromstring(base64.b64decode(result["frame"]), np.uint8)
output = cv2.imdecode(file_bytes,1)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title(person)
plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title("%s (%f)" % (result["predictions"][0]["person"], result["predictions"][0]["confidence"]))
plt.tight_layout()
plt.show()


# In[ ]:


person = "Jennifer_Aniston"
path = test[test.person==person]["path"].iloc[0]
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = fr.predict(path)
file_bytes = np.fromstring(base64.b64decode(result["frame"]), np.uint8)
output = cv2.imdecode(file_bytes,1)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title(person)
plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title("%s (%f)" % (result["predictions"][0]["person"], result["predictions"][0]["confidence"]))
plt.tight_layout()
plt.show()


# In[ ]:


person = "Michael_Schumacher"
path = test[test.person==person]["path"].iloc[0]
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = fr.predict(path)
file_bytes = np.fromstring(base64.b64decode(result["frame"]), np.uint8)
output = cv2.imdecode(file_bytes,1)

plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title(person)
plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title("%s (%f)" % (result["predictions"][0]["person"], result["predictions"][0]["confidence"]))
plt.tight_layout()
plt.show()


# In[ ]:


get_ipython().system('wget -O img.jpg https://i.pinimg.com/originals/9e/1c/c9/9e1cc9329b82ad0084d5c4c30757d469.jpg')


# In[ ]:


path = "img.jpg"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = fr.predict(path, threshold=0.3)
file_bytes = np.fromstring(base64.b64decode(result["frame"]), np.uint8)
output = cv2.imdecode(file_bytes,1)

plt.figure(figsize=(20,20))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Source Image")
plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title("Output Image")
plt.tight_layout()
plt.show()


pprint(result["predictions"])


# In[ ]:




