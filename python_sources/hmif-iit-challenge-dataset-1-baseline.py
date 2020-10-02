#!/usr/bin/env python
# coding: utf-8

# #### Bagi yang bingung cara mengolah dataset tersebut, kernel ini bisa dijadikan panduan. Kernel ini hanya baseline, banyak hal yang bisa dilakukan lagi untuk meningkatkan performa modelnya.

# In[ ]:


import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# from xgboost import XGBClassifier # install dahulu

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Dataset

# In[ ]:


df = pd.read_csv('../input/train-data-1.csv')


# In[ ]:


df.columns


# In[ ]:


plt.title('Proporsi kelas')
sns.countplot(df['akreditasi'])


# data akreditasi 2 (B) jauh lebih banyak dibanding kelas lainnya. Penting untuk memperhatikan performa model untuk data sekolah dengan akreditasi minoritas (A dan C).

# ## Split data

# In[ ]:


X = df.drop(['id', 'akreditasi'], axis=1)
y = df['akreditasi']


# In[ ]:


X_dummy = pd.get_dummies(X) # disc: bukan cara terbaik, ini hanya agar mudah


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_dummy, y, test_size=0.2, random_state=496)


# ## Feature Engineering
# Apakah ada fitur yang aneh?
# 
# Ada banyak feature engineering yang bisa dilakukan, kira-kira apa saja fitur baru yang bisa dibuat? Lalu apakah semua fitur penting? Bagian ini sangat menentukan performa model yang kamu buat.

# In[ ]:





# ## Modeling

# In[ ]:


clf = LogisticRegression()


# In[ ]:


clf.fit(X_train, y_train)


# In[ ]:


y_train_pred = clf.predict(X_train)


# In[ ]:


print(classification_report(y_train, y_train_pred))
print('accuracy', accuracy_score(y_train, y_train_pred))
print('mae', mean_absolute_error(y_train, y_train_pred))


# In[ ]:


y_test_pred = clf.predict(X_test)


# In[ ]:


print(classification_report(y_test, y_test_pred))
print('accuracy', accuracy_score(y_test, y_test_pred))
print('mae', mean_absolute_error(y_test, y_test_pred))


# Hasilnya belum memuaskan, model masih cenderung memprediksi ke akreditasi mayoritas (B). Mungkin kamu harus melakukan feature engineering, eksperimen dengan algoritma lain juga perlu dilakukan, cobalah algoritma XGBoost.

# ## Full train

# In[ ]:


X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])


# In[ ]:


clf.fit(X_full, y_full)


# In[ ]:


y_full_pred = clf.predict(X_full)


# In[ ]:


print(classification_report(y_full, y_full_pred))
print('accuracy', accuracy_score(y_full, y_full_pred))
print('mae', mean_absolute_error(y_full, y_full_pred))


# ## Submit to Kaggle

# In[ ]:


# test_data = pd.read_csv('../input/test-data-1.csv')


# In[ ]:


# test_data.head()


# In[ ]:


# test_data = test_data.drop(['id'], axis=1)


# In[ ]:


# test_data_dummy = pd.get_dummies(test_data)


# In[ ]:


# dummy_absent = set(X_full.columns) - set(test_data_dummy.columns)


# In[ ]:


# for col in dummy_absent:
#     test_data_dummy[col] = 0


# In[ ]:


# test_data_dummy = test_data_dummy[X_full.columns]


# In[ ]:


# test_data_dummy.head()


# In[ ]:


# test_data_pred = clf.predict(test_data_dummy)


# In[ ]:


# test_data_pred


# In[ ]:


# submission = pd.read_csv('../input/sample-submission-1.csv')


# In[ ]:


# submission['akreditasi'] = test_data_pred


# In[ ]:


# submission.to_csv('submission-1.csv', index=False)


# In[ ]:




