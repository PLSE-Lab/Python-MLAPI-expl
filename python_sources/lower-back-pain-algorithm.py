#!/usr/bin/env python
# coding: utf-8

# <h1>Lower Back Pain Classification Algorithm </h1>
# 
# <p>This dataset contains the anthropometric measurements of the curvature of the spine to support the model towards a more accurate classification.
# <br />
# Lower back pain affects around 80% of individuals at some point in their life. If this model becomes robust enough, then these measurements may soon become predictive and treatable measures. 
# <br /> 
# <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.471.4845&rep=rep1&type=pdf">This study</a> asserts the validity of the manual goniometer measurements as a valid clinical tool. </p>

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns

# read data into dataset variable
data = pd.read_csv("../input/Dataset_spine.csv")

# Drop the unnamed column in place (not a copy of the original)#
data.drop('Unnamed: 13', axis=1, inplace=True)

# Concatenate the original df with the dummy variables
data = pd.concat([data, pd.get_dummies(data['Class_att'])], axis=1)

# Drop unnecessary label column in place. 
data.drop(['Class_att','Normal'], axis=1, inplace=True)


# In[ ]:


data.info()


# In[ ]:



import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <h1>Exploratory Data Analysis </h1>

# In[ ]:


data.columns = ['Pelvic Incidence','Pelvic Tilt','Lumbar Lordosis Angle','Sacral Slope','Pelvic Radius', 'Spondylolisthesis Degree', 'Pelvic Slope', 'Direct Tilt', 'Thoracic Slope', 'Cervical Tilt','Sacrum Angle', 'Scoliosis Slope','Outcome']
data.drop(['Pelvic Radius','Direct Tilt','Thoracic Slope', 'Scoliosis Slope'], axis=1, inplace=True)

corr = data.corr()

# Set up the matplot figure
f, ax = plt.subplots(figsize=(12,9))

#Draw the heatmap using seaborn
sns.heatmap(corr, cmap='inferno')


# In[ ]:


sns.residplot(x="Spondylolisthesis Degree", y="Pelvic Incidence", data=data, scatter_kws={"s":80})


# In[ ]:


sns.lmplot(x="Spondylolisthesis Degree", y="Pelvic Incidence",hue="Outcome", data=data, markers=["o", "x"], palette="Set1")


# In[ ]:


training = data.drop('Outcome', axis=1)
testing = data['Outcome']


# In[ ]:


training.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(training, testing, test_size=0.33, random_state=22)


# In[ ]:


estimators = [('clf', LogisticRegression())]

pl = Pipeline(estimators)

pl.fit(X_train, y_train)

accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data",accuracy)


# In[ ]:


ypred = pl.predict(X_test)

pl.score(X_test, y_test)


# In[ ]:


pl.predict(X_test)
pl.score(X_test, y_test)


# In[ ]:


report = classification_report(y_test, ypred)
print(report)


# <h1> That's it! </h1>
# <p> 88%+ prediction accuracy feels like a good start. To increase the accuracy of the model, feature engineering is a suitable solution - as well as creating new variables based on domain knowledge.</p>

# In[ ]:




