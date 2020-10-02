#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os
print(os.listdir("../input/fruits/fruits-360_dataset/fruits-360/Training"))


# In[ ]:


import numpy as np
import cv2
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import glob
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.cluster import KMeans


# In[ ]:


fruit_images = []
labels = [] 
for fruit_dir_path in glob.glob("../input/fruits/fruits-360_dataset/fruits-360/Training/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        fruit_images.append(image)
        labels.append(fruit_label)
fruit_images = np.array(fruit_images)
labels = np.array(labels)


# In[ ]:


print(labels)


# In[ ]:


label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}


# In[ ]:


id_to_label_dict


# In[ ]:


label_ids = np.array([label_to_id_dict[x] for x in labels])


# In[ ]:


def plot_image_grid(images, rows, columns):
    figure = plt.figure(figsize=(columns * 3, rows * 3))
    for i in range(columns * rows):
        figure.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i])
    plt.show()


# In[ ]:


plot_image_grid(fruit_images[0:100], 10, 10)


# In[ ]:


scaler = StandardScaler()


# In[ ]:


images_scaled = scaler.fit_transform([i.flatten() for i in fruit_images])


# In[ ]:


pca = PCA(n_components=50)
pca_result = pca.fit_transform(images_scaled)


# In[ ]:


tsne = TSNE(n_components=2, perplexity=40.0)
tsne_result = tsne.fit_transform(pca_result)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)


# In[ ]:


tsnedf = pd.DataFrame()
tsnedf['x'] = list(tsne_result_scaled[:,0])
tsnedf['y'] = list(tsne_result_scaled[:,1])
tsnedf['label'] = labels
tsnedf.head()


# In[ ]:


nb_classes = len(np.unique(label_ids))
sns.set_style('white')
#120 for 120 fruits, so 120 different colors
cmap = plt.cm.get_cmap("Spectral", 120) 

plt.figure(figsize=(20,20))
for i, label_id in enumerate(np.unique(label_ids)):
    
    #plot matching labels to tsne results so labels are accurate
    plt.scatter(tsne_result_scaled[np.where(label_ids == label_id), 0],
                tsne_result_scaled[np.where(label_ids == label_id), 1],
                marker = '.',
                c = cmap(i),
                linewidth = '5',
                alpha=0.8,
                label = id_to_label_dict[label_id])
plt.title('T-SNE Plot (PCA 50 Components)', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5), ncol = 2)
plt.show()


# In[ ]:


import seaborn as sns


# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))
for df, i in zip(tsnedf.iterrows(), fruit_images):
    x = df[1]['x']
    y = df[1]['y']
    img = OffsetImage(i, zoom = .4)
    ab = AnnotationBbox(img, (x,y), xycoords = 'data', frameon = False)
    ax.add_artist(ab)
ax.update_datalim(tsnedf[['x', 'y']].values)
ax.autoscale()
plt.title('T-SNE Plot (PCA 50 Components)', fontsize = 15)
plt.xlabel('X', fontsize = 15)
plt.ylabel('Y', fontsize = 15)
plt.show()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(pca_result, label_ids, test_size=0.25, random_state=42)


# In[ ]:


forest = RandomForestClassifier(n_estimators=10)
forest = forest.fit(X_train, y_train)


# In[ ]:


test_predictions = forest.predict(X_test)


# In[ ]:


precision = accuracy_score(test_predictions, y_test) * 100
print("Accuracy with RandomForest: {0:.6f}".format(precision))


# In[ ]:


svm_clf = svm.SVC()
svm_clf = svm_clf.fit(X_train, y_train) 


# In[ ]:


test_predictions = svm_clf.predict(X_test)


# In[ ]:


precision = accuracy_score(test_predictions, y_test) * 100
print("Accuracy with SVM: {0:.6f}".format(precision))


# In[ ]:


validation_fruit_images = []
validation_labels = [] 
for fruit_dir_path in glob.glob("../input/fruits/fruits-360_dataset/fruits-360/Test/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        validation_fruit_images.append(image)
        validation_labels.append(fruit_label)
validation_fruit_images = np.array(validation_fruit_images)
validation_labels = np.array(validation_labels)


# In[ ]:


print(validation_labels)


# In[ ]:


validation_label_ids = np.array([label_to_id_dict[x] for x in validation_labels])


# In[ ]:


validation_images_scaled = scaler.transform([i.flatten() for i in validation_fruit_images])


# In[ ]:


validation_pca_result = pca.transform(validation_images_scaled)


# In[ ]:


test_predictions = forest.predict(validation_pca_result)


# In[ ]:


precision = accuracy_score(test_predictions, validation_label_ids) * 100
print("Validation Accuracy with Random Forest: {0:.6f}".format(precision))


# In[ ]:


test_predictions = svm_clf.predict(validation_pca_result)


# In[ ]:


precision = accuracy_score(test_predictions, validation_label_ids) * 100
print("Validation Accuracy with SVM: {0:.6f}".format(precision))


# In[ ]:




