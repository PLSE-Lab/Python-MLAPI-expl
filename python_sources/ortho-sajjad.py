#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# 1. Importing packages

# In[ ]:


import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt


# 2. Importing the dataset

# In[ ]:


Ortho = pd.read_csv('Orthopedic_Normality.csv')
Ortho = pd.DataFrame(Ortho)
Ortho.dtypes
Ortho


# 3. Checking for missing values

# In[ ]:


Ortho.dropna()
Ortho


# 4. changing class variable to binary

# In[ ]:


Ortho['class'] = np.where(Ortho['class'] == 'Abnormal', 1, 0)


# In[ ]:


Ortho['class'].sum()


# 5. Checkign variable pelvic_incidence

# In[ ]:


print('pelvic_incidence:',Ortho['pelvic_incidence'].unique())
print('pelvic_incidence_max:', Ortho['pelvic_incidence'].max())
print('pelvic_incidence_min:', Ortho['pelvic_incidence'].min())
print('pelvic_incidence_mean:', Ortho['pelvic_incidence'].mean())
print('pelvic_incidence_mode:', Ortho['pelvic_incidence'].mode())
plt.boxplot(Ortho['pelvic_incidence'])


# In[ ]:


Ortho[Ortho['pelvic_incidence'] > 100]


#     There seems to be some information in pelvic_incidence more than 100. Thus, we don't treat them as outliers.

# 6. Checking variable pelvic_tilt

# In[ ]:


print('pelvic_tilt:',Ortho['pelvic_tilt'].unique())
print('pelvic_tilt_max:', Ortho['pelvic_tilt'].max())
print('pelvic_tilt_min:', Ortho['pelvic_tilt'].min())
print('pelvic_tilt_mean:', Ortho['pelvic_tilt'].mean())
print('pelvic_tilt_mode:', Ortho['pelvic_tilt'].mode())
print('pelvic_tilt_max:', Ortho['pelvic_tilt'].max())
plt.boxplot(Ortho['pelvic_tilt'])


# In[ ]:


Ortho[Ortho['pelvic_tilt'] > 45]


#     Don't cut any outliers here as well.

# 7. Checking variable lumber_lordosis_angle

# In[ ]:


print('lumbar_lordosis_angle:',Ortho['lumbar_lordosis_angle'].unique())
print('lumbar_lordosis_angle_max:', Ortho['lumbar_lordosis_angle'].max())
print('lumbar_lordosis_angle_min:', Ortho['lumbar_lordosis_angle'].min())
print('lumbar_lordosis_angle_mean:', Ortho['lumbar_lordosis_angle'].mean())
print('lumbar_lordosis_angle_mode:', Ortho['lumbar_lordosis_angle'].mode())
print('lumbar_lordosis_angle_max:', Ortho['lumbar_lordosis_angle'].max())
plt.boxplot(Ortho['lumbar_lordosis_angle'])


#     Cutting one outlier here

# In[ ]:


Ortho = Ortho[Ortho['lumbar_lordosis_angle'] < 120]


# 8. Checking variable sacral_slope

# In[ ]:


print('sacral_slope:',Ortho['sacral_slope'].unique())
print('sacral_slope_max:', Ortho['sacral_slope'].max())
print('sacral_slope_min:', Ortho['sacral_slope'].min())
print('sacral_slope_mean:', Ortho['sacral_slope'].mean())
print('sacral_slope_mode:', Ortho['sacral_slope'].mode())
print('sacral_slope_max:', Ortho['sacral_slope'].max())
plt.boxplot(Ortho['sacral_slope'])


# Another outlier here

# In[ ]:


Ortho = Ortho[Ortho['sacral_slope'] < 120]
plt.boxplot(Ortho['sacral_slope'])


# 9. Checking variable pelvic_radius

# In[ ]:


print('pelvic_radius:',Ortho['pelvic_radius'].unique())
print('pelvic_radius_max:', Ortho['pelvic_radius'].max())
print('pelvic_radius_min:', Ortho['pelvic_radius'].min())
print('pelvic_radius_mean:', Ortho['pelvic_radius'].mean())
print('pelvic_radius_mode:', Ortho['pelvic_radius'].mode())
print('pelvic_radius_max:', Ortho['pelvic_radius'].max())
#plt.boxplot(Ortho['pelvic_radius'])
plt.hist(Ortho['pelvic_radius'])


# In[ ]:


Ortho[Ortho['pelvic_radius'] > 150]


# 8. Checking variable degree_spondylolisthesis

# In[ ]:


print('degree_spondylolisthesis:',Ortho['degree_spondylolisthesis'].unique())
print('degree_spondylolisthesis_max:', Ortho['degree_spondylolisthesis'].max())
print('degree_spondylolisthesis_min:', Ortho['degree_spondylolisthesis'].min())
print('degree_spondylolisthesis_mean:', Ortho['degree_spondylolisthesis'].mean())
print('degree_spondylolisthesis_mode:', Ortho['degree_spondylolisthesis'].mode())
print('degree_spondylolisthesis_max:', Ortho['degree_spondylolisthesis'].max())
plt.boxplot(Ortho['degree_spondylolisthesis'])


# In[ ]:


Ortho[Ortho['degree_spondylolisthesis'] > 100]


# In[ ]:


Ortho.index = np.arange(0, len(Ortho))


# In[ ]:


Ortho.index


# 9. Creating features and labels

# In[ ]:


features = Ortho[['pelvic_incidence','pelvic_tilt','lumbar_lordosis_angle',
                  'sacral_slope','pelvic_radius','degree_spondylolisthesis']]
labels = Ortho['class']


# 10. Splitting the dataset

# In[ ]:


from sklearn.model_selection import train_test_split 
tr_features, test_features, tr_labels, test_labels = train_test_split(features, labels,
random_state=0)


# 11. Cross Validation and fitting 

# In[ ]:


#ignoring the warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(random_state=0)
Parameters = {'max_depth':[1,2,4,8,16,32, 64, None]}
CV = GridSearchCV(DT, Parameters, cv=5)


# 12. Fitting the CV object

# In[ ]:


CV.fit(tr_features, tr_labels.values.ravel())
CV.best_params_


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
predict = CV.predict(test_features)
print('accuracy:',accuracy_score(test_labels, predict))
print('precision:',precision_score(test_labels, predict))
print('recall:',recall_score(test_labels, predict))
print('f1:',f1_score(test_labels, predict))
print('total predicted 1s:',predict.sum())
print('total actual 1s:',test_labels.sum())


# accuracy: 0.8181818181818182
# precision: 0.9148936170212766
# recall: 0.8113207547169812
# f1: 0.8600000000000001
# total predicted 1s: 47
# total actual 1s: 53

# In[ ]:




