#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np


# In[ ]:


df = read_csv("../input/opcode_frequency_benign.csv")


# In[ ]:


# df.head()


# In[ ]:


df_mal = read_csv("../input/opcode_frequency_malware.csv")


# In[ ]:


# df_mal.head()


# In[ ]:


# df.drop('FileName', axis=1)
# df_mal.drop('FileName', axis=1)


# In[ ]:


# df


# In[ ]:


# import pandas as pd
dff = pd.concat([df, df_mal])


# In[ ]:


# df.shape[0]
# # df_mal.shape[0]
# # dff.shape[0]


# In[ ]:


# dff['2'][0]


# In[ ]:


# df_final = dff


# In[ ]:


dff = dff.reset_index()


# In[ ]:


Y = [0]*1400
Y = Y + [1]*1999


# In[ ]:


len(Y)


# In[ ]:


dff = dff.drop('FileName', axis=1)


# In[ ]:


dff = dff.drop('index', axis=1)


# In[ ]:


# dff
X = dff.values


# In[ ]:


model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)


# In[ ]:


arr = np.array(model.feature_importances_)
inds = np.argsort(arr)
ar = arr
np.sort(ar)


# In[ ]:


# arr[0]


# In[ ]:


from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# knn_cv = KNeighborsClassifier(n_neighbors=3)
# cv_scores = cross_val_score(knn_cv, X, Y, cv=5)


# In[ ]:


# arr[926]


# In[ ]:


# inds[-404:]


# In[ ]:


# dff = df_final
inds_int = inds[-404:]
# inds_int


# In[ ]:


# dff_copy = dff


# In[ ]:


# dff


# In[ ]:


# dff = dff_copy
for x in range(1,1809):
    if x not in inds_int:
        dff = dff.drop(str(x), axis=1)


# In[ ]:


# dff


# In[ ]:


X = dff.values
# len(X[0])


# In[ ]:


# knn_cv = KNeighborsClassifier(n_neighbors=3)
# cv_scores = cross_val_score(knn_cv, X, Y, cv=5)


# In[ ]:


# print(cv_scores)
# print(np.mean(cv_scores))


# In[ ]:


# from sklearn.model_selection import GridSearchCV

# knn2 = KNeighborsClassifier()

# param_grid = {'n_neighbors': np.arange(1, 25)}

# knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

# knn_gscv.fit(X, Y)


# In[ ]:


# knn_gscv.best_params_


# In[ ]:


test_df = read_csv('../input/Test_data.csv')
# test_df
# file = df['FileName']


# In[ ]:


test_df = test_df.drop('Unnamed: 1809', axis=1)


# In[ ]:


test_df = test_df.drop('FileName', axis=1)


# In[ ]:


# test_df


# In[ ]:


# %matplotlib inline
# from matplotlib import pyplot as plt
# from sklearn.decomposition import PCA
# pca = PCA().fit(dff)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
ranfor = RandomForestClassifier()


# In[ ]:


# len(X[0])
ranfor = RandomForestClassifier()
ranfor.fit(X, Y)


# In[ ]:


for x in range(1,1809):
    if x not in inds_int:
        test_df = test_df.drop(str(x), axis=1)


# In[ ]:


# test_df


# In[ ]:


X_val = test_df.values
predicted = ranfor.predict(X_val)


# In[ ]:


# predicted[0:18]


# In[ ]:


sample = read_csv('../input/sample_submission.csv')
sample['Class'] = predicted
# sample


# In[ ]:


sample = sample.set_index('FileName')


# In[ ]:


sample.to_csv('submission.csv')


# In[ ]:





# In[ ]:


# from sklearn.linear_model import LogisticRegression


# In[ ]:


# logreg = LogisticRegression()
# cv_scores = cross_val_score(logreg, X, Y, cv=5)


# In[ ]:


# print(cv_scores)
# print(np.mean(cv_scores))


# In[ ]:


# X = dff_404.values


# In[ ]:


# from sklearn.ensemble import RandomForestClassifier
# ranfor = RandomForestClassifier()
# cv_scores = cross_val_score(ranfor, X, Y, cv=5)


# In[ ]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode())
payload = b64.decode()
html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
html = html.format(payload=payload,title=title,filename=filename)
return HTML(html)
create_download_link(sample)

