#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


np.random.seed(42)


# In[ ]:


df = pd.read_csv('../input/mushrooms.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


train_set, test_set = train_test_split(df, stratify=df['class'])


# In[ ]:


Xs, ys = train_set.drop(columns=['class']), train_set['class']
test_Xs, test_ys = test_set.drop(columns=['class']), test_set['class']

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoder.fit(pd.concat([Xs, test_Xs]))
Xs_encoded = encoder.transform(Xs)

ys_encoded, ys_categories = ys.factorize()


# In[ ]:


knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [5, 10, 20],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='recall')
grid_search.fit(Xs_encoded, ys_encoded)


# In[ ]:


test_Xs_encoded = encoder.transform(test_Xs)
test_ys_encoded = list(map(lambda c: np.where(ys_categories == c)[0], test_ys))


# In[ ]:


best_estimator = grid_search.best_estimator_

test_predictions = best_estimator.predict(test_Xs_encoded)
print('recall=', recall_score(test_ys_encoded, test_predictions))
confusion_matrix(test_ys_encoded, test_predictions)


# In[ ]:


grid_search.best_params_


# In[ ]:


pca_3d = PCA(n_components=3)
Xs_reduced_3d = pca_3d.fit_transform(Xs_encoded)

colors = ['#D11F0E', '#A5DD88']
colors = [colors[y] for y in ys_encoded]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xs_reduced_3d[:,0], Xs_reduced_3d[:,1], Xs_reduced_3d[:,2], color=colors, alpha=0.1)
plt.show()

