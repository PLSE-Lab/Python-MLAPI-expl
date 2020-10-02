#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


get_ipython().system('pip install jcopml')


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease


# # Import Data

# In[ ]:


df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df


# # Nomenklatur Data
# 
# -age: Umur <br>
# 
# -sex: jenis kelamin-> (1 = male, 0 = female)<br>
# 
# -cp: chest pain. Ada 4 tipe-> (1: typical angina, 2: atypical angina,3: non-anginal pain,4: asymptomatic)<br>
# 
# -trestbps: Tekanan darah (kondisi resting) [mmHg]<br>
# 
# -chol: serum cholestoral [mg/dl]<br>
# 
# -fbs: kadar gula (kondisi berpuasa) -> (1: artinya > 120 mg/dl, 0: sebaliknya)<br>
# 
# -restecg: electrocardiographic/ kondisi resting -> (0: normal, 1: ada ST-T wave abnormality, 2: ada indikasi left ventricular hypertrophy menurut kriteria Estes)<br>
# 
# -thalach: detak jantung maximum<br>
# 
# -exang: mengalami chest pain tipe angina ketika olahraga -> (1: artinya iya, 0 artinya tidak)<br>
# 
# -oldpeak: mengalami ST depression ketika olahraga dibandingkan saat diam<br>
# 
# -slope: kemiringan pada peak exercise ST segment -> (1: naik,2: datar,3: turun)<br>
# 
# -ca: banyaknya saluran saluran darah utama (0-3) dilihat dengan flourosopy)<br>
# 
# -thal -> (3 = normal, 6 = cacat permanen, 7 = cacat reversibel)<br>
# 
# -target -> (0: tidak ada indikasi penyakit jantung;  1,2,3,4: ada indikasi penyakit jantung)
#    

# In[ ]:


df.replace("?", np.nan, inplace=True)


# In[ ]:


plot_missing_value(df)


# > # Mengubah Terget
# Kita hanya ingin mendeteksi apakah ada indikasi penyakit jantung atau tidak. Jadi:
# - 0 -> `False` (tidak ada indikasi)
# - 1, 2, 3, 4 -> `True` (ada indikasi)

# In[ ]:


df.target = df.target.apply(lambda x: int(x>0))
df.head()


# # Analisa Numeric vs Target (histogram)

# In[ ]:


plt.figure(figsize=(7, 6))
sns.distplot(df.age[df.target ==0], bins=[0, 5, 12, 18, 40, 120], color="g", label="tidak ada indikasi")
sns.distplot(df.age[df.target ==1], bins=[0, 5, 12, 18, 40, 120], color="r", label="ada indikasi")
plt.legend();


# # Analisa Kategorik vs Target

# In[ ]:


cat_var = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

fig, axes = plt.subplots(2, 4, figsize=(15,10))
for cat, ax in zip(cat_var, axes.flatten()):
    sns.countplot(cat, data=df, hue="target", ax=ax)


# # Dataset Splitting
# Split data menggunakan `train_test_split` dari `sklearn.model_selection`.
# Pastikan:
# - memakai stratified shuffle split<br>
# ya karena ini klasifikasi. Kita mau soal ujiannya serepresentatif mungkin.
# - test size yang sesuai<br>
# Hati-hati, pastikan soal ujian tidak terlalu sedikit agar nilainya pun tidak sensitif

# In[ ]:


X = df.drop(columns="target")
y = df.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# # Training

# In[ ]:


X_train.head()


# In[ ]:


X_train.columns


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from jcopml.tuning import grid_search_params as gsp


# In[ ]:


preprocessor = ColumnTransformer([
    ('numeric', num_pipe(),  ["age", "trestbps", "chol", "thalach", "oldpeak"]),
    ('categoric', cat_pipe(encoder='onehot'), ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]),
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', RandomForestClassifier(n_jobs=-1, random_state=42))
])



model = GridSearchCV(pipeline, gsp.rf_params, cv=3, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))


# In[ ]:


df_imp = mean_score_decrease(X_train, y_train, model, plot=True)


# # Save Model

# In[ ]:


from jcopml.utils import save_model


# In[ ]:


save_model(model.best_estimator_, "rf_heart.pkl")

