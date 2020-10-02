#!/usr/bin/env python
# coding: utf-8

# # Sloan Digital Sky Survey Classification
# ## Classification of Galaxies, Stars and Quasars based on the RD14 from the SDSS

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# In[ ]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# ### About the notebook

# In this notebook we will try to classify observations of space to be either stars, galaxies or quasars. We will try to have a complete cycle of the data science workflow including querying the database to get the dataset, data analysis and building machine learning models to predict for new data.
# 
# We are using data from the Sloan Digital Sky Survey (**Release 14**). 
# 
# **I followed Niklas Donges' Titanic classification approach for this notebook as I found it really useful and comprehensive!**  

# ### About the SDSS

# The Sloan Digital Sky Survey is a project which offers public data of space observations. Observations have been made since 1998 and have been made accessible to everyone who is interested. 
# 
# For this purpose a special 2.5 m diameter telescope was built at the Apache Point Observatory in New Mexico, USA. The telescope uses a camera of 30 CCD-Chips with 2048x2048 image points each. The chips are ordered in 5 rows with 6 chips in each row. Each row observes the space through different optical filters (u, g, r, i, z) at wavelengths of approximately 354, 476, 628, 769, 925 nm.
# 
# The telescope covers around one quarter of the earth's sky - therefore focuses on the northern part of the sky.
# 
# **For more information about this awesome project - please visit their website:**
# 
# http://www.sdss.org/
# 
# ![alt text](http://www.fingerprintdigitalmedia.com/wp-content/uploads/2014/08/sdss1.jpg)

# ### Importing Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
sns.set_style('whitegrid')
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
get_ipython().run_line_magic('matplotlib', 'inline')

SMALL_SIZE = 10
MEDIUM_SIZE = 12

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rcParams['figure.dpi']=150


# ## Data Acquisition

# Public data from the SDSS can be accessed through multiple ways - I used the **CasJobs** website which offers a **SQL-based interface** which lets you query their database which contains the released data.
# 
# For more information about how to get data from the SDSS see their Data Access Guide:
# 
# http://www.sdss.org/dr14/data_access/
# 
# I used the sample query given by the **CasJobs** to receive the data. Find the exact query below:

# ### Query

# **SELECT TOP 10000** <br/>
# p.objid,p.ra,p.dec,p.u,p.g,p.r,p.i,p.z, p.run, p.rerun, p.camcol, p.field,  <br/>
# s.specobjid, s.class, s.z as redshift, s.plate, s.mjd, s.fiberid  <br/>
# **FROM** PhotoObj **AS** p <br/>
#    **JOIN** SpecObj **AS** s **ON** s.bestobjid = p.objid <br/>
# **WHERE** <br/>
#    p.u **BETWEEN** 0 **AND** 19.6 <br/>
#    **AND** g **BETWEEN** 0 **AND** 20 <br/>
# 
# 

# The above query joins two tables (actually views): The image table (PhotoObj) which contains all image objects and the spectral table (SpecObj) which contains corresponding spectral data. 

# ### Feature Description

# #### View "PhotoObj"
# * objid = Object Identifier
# * ra = J2000 Right Ascension (r-band)
# * dec = J2000 Declination (r-band)
# 
# Right ascension (abbreviated RA) is the angular distance measured eastward along the celestial equator from the Sun at the March equinox to the hour circle of the point above the earth in question. When paired with declination (abbreviated dec), these astronomical coordinates specify the direction of a point on the celestial sphere (traditionally called in English the skies or the sky) in the equatorial coordinate system.
# 
# Source: https://en.wikipedia.org/wiki/Right_ascension
# 
# * u = better of DeV/Exp magnitude fit
# * g = better of DeV/Exp magnitude fit
# * r = better of DeV/Exp magnitude fit
# * i = better of DeV/Exp magnitude fit
# * z = better of DeV/Exp magnitude fit
# 
# The Thuan-Gunn astronomic magnitude system. u, g, r, i, z represent the response of the 5 bands of the telescope.
# 
# Further education: https://www.astro.umd.edu/~ssm/ASTR620/mags.html
# 
# * run = Run Number
# * rereun = Rerun Number
# * camcol = Camera column
# * field = Field number
# 
# Run, rerun, camcol and field are features which describe a field within an image taken by the SDSS. A field is basically a part of the entire image corresponding to 2048 by 1489 pixels. A field can be identified by:
# - **run** number, which identifies the specific scan,
# - the camera column, or "**camcol**," a number from 1 to 6, identifying the scanline within the run, and
# - the **field** number. The field number typically starts at 11 (after an initial rampup time), and can be as large as 800 for particularly long runs.
# - An additional number, **rerun**, specifies how the image was processed. 
# 
# #### View "SpecObj"
# 
# * specobjid = Object Identifier
# * class = object class (galaxy, star or quasar object)
# 
# The class identifies an object to be either a galaxy, star or quasar. This will be the response variable which we will be trying to predict.
# 
# * redshift = Final Redshift
# * plate = plate number
# * mjd = MJD of observation
# * fiberid = fiber ID
# 
# In physics, **redshift** happens when light or other electromagnetic radiation from an object is increased in wavelength, or shifted to the red end of the spectrum. 
# 
# Each spectroscopic exposure employs a large, thin, circular metal **plate** that positions optical fibers via holes drilled at the locations of the images in the telescope focal plane. These fibers then feed into the spectrographs. Each plate has a unique serial number, which is called plate in views such as SpecObj in the CAS.
# 
# **Modified Julian Date**, used to indicate the date that a given piece of SDSS data (image or spectrum) was taken.
# 
# The SDSS spectrograph uses optical fibers to direct the light at the focal plane from individual objects to the slithead. Each object is assigned a corresponding **fiberID**. 
# 
# **Further information on SDSS images and their attributes:** 
# 
# http://www.sdss3.org/dr9/imaging/imaging_basics.php
# 
# http://www.sdss3.org/dr8/glossary.php

# In[ ]:


#sdss_df = pd.read_csv('Skyserver_SQL2_27_2018 6_51_39 PM.csv', skiprows=1)
sdss_df = pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv', skiprows=0)


# ## Data Exploration

# ### Basic stats about our dataset

# Let's take a first look at our dataset to see what we're working with!

# In[ ]:


sdss_df.head()


# We can tell that we have all the features as described in the above query. 
# 
# We notice that there are no categorical features at all - besides the class column. As some machine learning models can't handle categorical feature columns at all, we will encode this column to be a numerical column later on.

# Let's find out about the types of columns we have:

# In[ ]:


sdss_df.info()


# The dataset has 10000 examples, 17 feature columns and 1 target column. 8 of the 17 features are 64 bit integers, 1 feature is an unsigned 64 bit integer, 8 are 64 bit floats and the target column is of the type object. 

# In[ ]:


sdss_df.describe()


# From the above table we can tell that are no missing values at all. This means: **no imputing!**
# 
# We also notice that most of the features stay within a reasonable scale when comparing values within **only one** column. We can recognize this from the min, max and quartil rows.

# In[ ]:


sdss_df['class'].value_counts()


# The most objects (50%) are galaxies, a little less (40%) are stars and only around (10%) of the rows are classified as QSOs.

# ### First Data Filtering

# In[ ]:


sdss_df.columns.values


# There is no need to know everything about stars, galaxy or quasars - yet we can already tell which features are **unlikely** to be related to the target variable 'class'.
# 
# **objid** and **specobjid** are just identifiers for accessing the rows back when they were stored in the original databank. Therefore we will not need them for classification as they are not related to the outcome.
# 
# Even more: The features 'run', 'rerun', 'camcol' and 'field' are values which describe parts of the camera at the moment when making the observation, e.g. 'run' represents the corresponding scan which captured the oject.
# 
# Source: http://www.sdss3.org/dr9/imaging/imaging_basics.php
# 
# We will drop these columns as any correlation to the outcome would be coincidentally.

# In[ ]:


sdss_df.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid'], axis=1, inplace=True)
sdss_df.head(1)


# ### Univariate Analysis

# #### Redshift

# To start the univariate analysis we will plot histograms for the 'redshift' feature column for each class.
# 
# This will tell us how the redshift values are distributed over their range.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
ax = sns.distplot(sdss_df[sdss_df['class']=='STAR'].redshift, bins = 30, ax = axes[0], kde = False)
ax.set_title('Star')
ax = sns.distplot(sdss_df[sdss_df['class']=='GALAXY'].redshift, bins = 30, ax = axes[1], kde = False)
ax.set_title('Galaxy')
ax = sns.distplot(sdss_df[sdss_df['class']=='QSO'].redshift, bins = 30, ax = axes[2], kde = False)
ax = ax.set_title('QSO')


# This is an interesting result.
# 
# We can cleary tell that the redshift values for the classes quite differ. 
# 
# * **Star:** The histogram looks like a truncated zero-centered normal distribution.
# 
# * **Galaxy:** The redshift values may come from a slightly right-shifted normal distribution which is centered around 0.075.
# 
# * **QSO:** The redshift values for QSOs are a lot more uniformly distributed than for Stars or Galaxies. They are roughly evenly distributed from 0 to 3, than the occurences decrease drastically. For 4 oder ~5.5 there are some outliers.
# 
# **The redshift can be an estimate(!) for the distance from the earth to a object in space.**
# 
# Hence the distplot tells us that most of the stars observed are somewhat closer to the earth than galaxies or quasars. Galaxies tend to be a little further away and quasars are distant from very close to very far.  
# 
# Possible rookie explanation: Since galaxies and quasars radiate stronger due to their size and physical structure, they can be observed from further away than "small" stars.
# 
# As we can distinct the classes from each other just based on this column - 'redshift' is very likely to be helping a lot classifying new objects.

# #### dec

# Let's lvplot the values of dec (Recall: position on celestial equator)!

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(16, 4))
ax = sns.lvplot(x=sdss_df['class'], y=sdss_df['dec'], palette='coolwarm')
ax.set_title('dec')


# **First of all: what does this plot tell us?**
# 
# The Letter value (LV) Plot show us an estimate of the distribution of the data. It shows boxes which relate to the amount of values within the range of values inside the box.
# 
# In this case we can observe a clear distinction between Stars and the other two classes. The difference between Galaxies and Quasars is smaller.
# 
# * **Star:** The largest part of the data points lay within a 0 to 10 range. Another large part consists of values between about 10 to 55. Only small amounts of the data are lower or higher than these ranges.
# 
# * **Galaxy:** The largest part of values lays between 0 and 45. There is a smaller amount of values in the range of 45 to 60. The rest of the data has smaller or higher values.
# 
# * **QSO:** This plot looks quite similiar to the GALAXY plot. Only the amount of data points in the range of 0 to 60 is even bigger.
# 
# Side Note: The fact that the distribution of dec values of galaxies und quasar objects is almost the same might indicate that one can find both galaxies and quasars at smiliar positions in the night sky.

# ### Multivariate Analysis

# #### u,g,r,i,z filters

# Recall: u, g, r, i, z represent the different wavelengths which are used to capture the observations.
# 
# Let's find out how much they are correlated.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
fig.set_dpi(100)
ax = sns.heatmap(sdss_df[sdss_df['class']=='STAR'][['u', 'g', 'r', 'i', 'z']].corr(), ax = axes[0], cmap='coolwarm')
ax.set_title('Star')
ax = sns.heatmap(sdss_df[sdss_df['class']=='GALAXY'][['u', 'g', 'r', 'i', 'z']].corr(), ax = axes[1], cmap='coolwarm')
ax.set_title('Galaxy')
ax = sns.heatmap(sdss_df[sdss_df['class']=='QSO'][['u', 'g', 'r', 'i', 'z']].corr(), ax = axes[2], cmap='coolwarm')
ax = ax.set_title('QSO')


# Right of the top we observe that the correlation matrices look very similiar for every class.
# 
# We can tell that there are high correlations between the different bands. This feels not really suprising - intuitively one would think that if one of the bands captures some object, the other bands should capture something aswell.
# 
# Therefore it is interesting to see that band 'u' is less correlated to the other bands. 
# 
# Remember: u, g, r, i, z capture light at wavelengths of 354, 476, 628, 769 and 925 nm.
# 
# This might indicates that galaxies, stars and quasar objects shine brighter at wavelengths from 476 - 925 nm. Don't quote me on that though.
# 
# **But:** as we can see - the correlation is roughly the same for every class...the different bands behave the same for the different classes!

# #### Right ascension (ra) and declination (dec) 

# We will now plot the right ascension versus the declination depending on the class 

# In[ ]:


sns.lmplot(x='ra', y='dec', data=sdss_df, hue='class', fit_reg=False, palette='coolwarm', size=6, aspect=2)
plt.title('Equatorial coordinates')


# As we can clearly observe the equatorial coordinates do not differ significantly between the 3 classes. There are some outliers for stars and galaxies but for the bigger part the coordinates are within the same range.
# 
# Why is that?
# 
# All SDSS images cover the same area of the sky. The plot above tells us that stars, galaxies and quasars are observed equally at all coordinates within this area. So whereever the SDSS "looks" - the chance of observing a star or galaxy or quasar is always the same.  
# 
# **This contradicts our interpretation of the letter value plot of dec from the univariate analysis.**

# ## Feature Engineering

# ### u, g, r, i, z

# We will now reduce the amount of dimensions by replacing the different bands 'u', 'g', 'r', 'i' and 'z' by a linear combination with only 3 dimensions using **Principal Component Analysis**.
# 
# **Principal Component Analysis:**
# 
# n observations with p features can be interpreted as n points in a p-dimensional space. PCA aims to project this space into a q-dimensional subspace (with q<p) with as little information loss as possible. 
# 
# It does so by finding the q directions in which the n points vary the most (--> the principal components). It then projects the original data points into the q-dimensional subspace. PCA returns a n x q dimensional matrix. 
# 
# Using PCA on our data will decrease the amount of operations during training and testing.

# In[ ]:


sdss_df_fe = sdss_df

# encode class labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(sdss_df_fe['class'])
sdss_df_fe['class'] = y_encoded

# Principal Component Analysis
pca = PCA(n_components=3)
ugriz = pca.fit_transform(sdss_df_fe[['u', 'g', 'r', 'i', 'z']])

# update dataframe 
sdss_df_fe = pd.concat((sdss_df_fe, pd.DataFrame(ugriz)), axis=1)
sdss_df_fe.rename({0: 'PCA_1', 1: 'PCA_2', 2: 'PCA_3'}, axis=1, inplace = True)
sdss_df_fe.drop(['u', 'g', 'r', 'i', 'z'], axis=1, inplace=True)
sdss_df_fe.head()


# ## Machine Learning Models - Training

# #### Feature Scaling

# We will now train different models on this dataset. 
# 
# Scaling all values to be within the (0, 1) interval will reduce the distortion due to exceptionally high values and make some algorithms converge faster.

# In[ ]:


scaler = MinMaxScaler()
sdss = scaler.fit_transform(sdss_df_fe.drop('class', axis=1))


# We will  split the data into a training and a test part. The models will be trained on the training data set and tested on the test data set

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(sdss, sdss_df_fe['class'], test_size=0.33)


# #### K Nearest Neighbors

# In[ ]:


knn = KNeighborsClassifier()
training_start = time.perf_counter()
knn.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = knn.predict(X_test)
prediction_end = time.perf_counter()
acc_knn = (preds == y_test).sum().astype(float) / len(preds)*100
knn_train_time = training_end-training_start
knn_prediction_time = prediction_end-prediction_start
print("Scikit-Learn's K Nearest Neighbors Classifier's prediction accuracy is: %3.2f" % (acc_knn))
print("Time consumed for training: %4.3f seconds" % (knn_train_time))
print("Time consumed for prediction: %6.5f seconds" % (knn_prediction_time))


# #### Naive Bayes

# Thanks to [Adithya Raman's](https://www.kaggle.com/christodieu) comment as he proposed to use a MaxAbsScaler for the Naive Bayes classifier. Naive Bayes assumes the data to be normally distributed which can be achieved by scaling using the MaxAbsScaler. Many thanks!

# In[ ]:


from sklearn.preprocessing import MaxAbsScaler
scaler_gnb = MaxAbsScaler()
sdss = scaler_gnb.fit_transform(sdss_df_fe.drop('class', axis=1))
X_train_gnb, X_test_gnb, y_train_gnb, y_test_gnb = train_test_split(sdss, sdss_df_fe['class'], test_size=0.33)

gnb = GaussianNB()
training_start = time.perf_counter()
gnb.fit(X_train_gnb, y_train_gnb)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = gnb.predict(X_test_gnb)
prediction_end = time.perf_counter()
acc_gnb = (preds == y_test_gnb).sum().astype(float) / len(preds)*100
gnb_train_time = training_end-training_start
gnb_prediction_time = prediction_end-prediction_start
print("Scikit-Learn's Gaussian Naive Bayes Classifier's prediction accuracy is: %3.2f" % (acc_gnb))
print("Time consumed for training: %4.3f seconds" % (gnb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (gnb_prediction_time))


# #### XGBoost

# In[ ]:


xgb = XGBClassifier(n_estimators=100)
training_start = time.perf_counter()
xgb.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = xgb.predict(X_test)
prediction_end = time.perf_counter()
acc_xgb = (preds == y_test).sum().astype(float) / len(preds)*100
xgb_train_time = training_end-training_start
xgb_prediction_time = prediction_end-prediction_start
print("XGBoost's prediction accuracy is: %3.2f" % (acc_xgb))
print("Time consumed for training: %4.3f" % (xgb_train_time))
print("Time consumed for prediction: %6.5f seconds" % (xgb_prediction_time))


# #### Scitkit-Learn's Random Forest Classifier

# In[ ]:


rfc = RandomForestClassifier(n_estimators=10)
training_start = time.perf_counter()
rfc.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = rfc.predict(X_test)
prediction_end = time.perf_counter()
acc_rfc = (preds == y_test).sum().astype(float) / len(preds)*100
rfc_train_time = training_end-training_start
rfc_prediction_time = prediction_end-prediction_start
print("Scikit-Learn's Random Forest Classifier's prediction accuracy is: %3.2f" % (acc_rfc))
print("Time consumed for training: %4.3f seconds" % (rfc_train_time))
print("Time consumed for prediction: %6.5f seconds" % (rfc_prediction_time))


# #### Support Vector Machine Classifier

# In[ ]:


svc = SVC()
training_start = time.perf_counter()
svc.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = svc.predict(X_test)
prediction_end = time.perf_counter()
acc_svc = (preds == y_test).sum().astype(float) / len(preds)*100
svc_train_time = training_end-training_start
svc_prediction_time = prediction_end-prediction_start
print("Scikit-Learn's Support Vector Machine Classifier's prediction accuracy is: %3.2f" % (acc_svc))
print("Time consumed for training: %4.3f seconds" % (svc_train_time))
print("Time consumed for prediction: %6.5f seconds" % (svc_prediction_time))


# Let's compare the results. We will create a table for a more comprehensive overview.

# In[ ]:


results = pd.DataFrame({
    'Model': ['KNN', 'Naive Bayes', 
              'XGBoost', 'Random Forest', 'SVC'],
    'Score': [acc_knn, acc_gnb, acc_xgb, acc_rfc, acc_svc],
    'Runtime Training': [knn_train_time, gnb_train_time, xgb_train_time, rfc_train_time, 
                         svc_train_time],
    'Runtime Prediction': [knn_prediction_time, gnb_prediction_time, xgb_prediction_time, rfc_prediction_time,
                          svc_prediction_time]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Model')
result_df


# We can see that both XGBoost and Scikit-Learn's Random Forest Classifier could achieve very high accuracy.
# 
# Gaussian Naive Bayes achieves just a little less accuracy but needs a very little amount of time to both train and predict data.
# 
# KNN performs about 5% worse than Naive Bayes.
# 
# The Support Vector Machine Classifier has the worst accuracy, plus takes the most of time for its operations.

# ### K Fold Cross Validation

# We will no perform k fold cross valdiation for the top 2 classifiers, i.e. XGBoost & Random Forest.
# 
# We do this to get a more realistic result by testing the performance for 10 different train and test datasets and averaging the results. 
# 
# Cross validation ensures that the above result is not arbitary and gives a more reliable performance check.

# #### Scikit-Learn's Random Forest Classifier

# In[ ]:


from sklearn.model_selection import cross_val_score
rfc_cv = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rfc_cv, X_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# #### XGBoost

# In[ ]:


xgb_cv = XGBClassifier(n_estimators=100)
scores = cross_val_score(xgb_cv, X_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# Cross validating the models showed that the accuracy values were in fact not arbitary and proofed that both models are performing very well. 
# 
# XGBoost showed a higher mean and lower standard deviation than the Scikit-Learn RFC.
# 
# A high mean corresponds to a more stable performance and a low standard deviation corresponds to smaller range of results. 

# ### Feature Importance

# Decision Trees have the unique property of being able to order features by their ability to split between the classes.
# 
# We will now visualize the features and their splitting ability.

# In[ ]:


importances = pd.DataFrame({
    'Feature': sdss_df_fe.drop('class', axis=1).columns,
    'Importance': xgb.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)
importances = importances.set_index('Feature')
importances


# In[ ]:


importances.plot.bar()


# Here we can clearly see how PCA helped to improve the performance of our predictors as 2 of the principal components are in the top 3 features.
# 
# The best (in terms of being able to split classes) is redshift.
# 
# Mjd is the feature with the lowest importance during the classification process, we will therefore drop it from the dataframe.

# Let's drop the column from the dataframe and rescale it. Since XGBoost requires the class to be discrete, we will re-add it manually afterwards.

# In[ ]:


scaler = MinMaxScaler()
sdss = pd.DataFrame(scaler.fit_transform(sdss_df_fe.drop(['mjd', 'class'], axis=1)), columns=sdss_df_fe.drop(['mjd', 'class'], axis=1).columns)
sdss['class'] = sdss_df_fe['class']


# In[ ]:


sdss.head()


# ### Summary

# We trained different machine learning models to solve this classification problems. Without any further hyperparameter tuning XGBoost and Scikit-Learn's Random Forest Classifier performed the best.
# 
# As XGBoost showed a little higher accuracy in most of the tests, we will continue only with this classifier.

# ## XGBoost - Finding the best hyperparameters

# Now it's time to look for the optimal hyperparameters - what does this mean?
# 
# We will test our chosen model with different values for (almost) each of its tuning parameters and give back the parameters with which the model performed best.
# 
# **The actual searching for optimal parameters is not done in this notebook since the operations can take some time and parallel editing would not be possible.
# 
# We will write our transformed data set to disk so the tuning scripts can access it. 

# In[ ]:


sdss.to_csv('sdss_data.csv')


# The best parameters for prediction as found by the tuning tests are:
# 
# - max_depth = 5
# - min_child_weight = 1
# - gamma = 0
# - subsample = 0.8
# - colsample_bytree = 0.8
# - reg_alpha = 0.005

# ## XGBoost - Testing optimal hyperparameters

# The following model implements the best performing model with optimal parameters evaluated by the hyperparameter tuning. We will expect the model to perform even better than before.
# 
# Analytics Vidhya presented a really nice guide for tuning XGBoost. 
# 
# Please read more: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(sdss.drop('class', axis=1), sdss['class'],
                                                   test_size=0.33)


# In[ ]:


xgboost = XGBClassifier(max_depth=5, learning_rate=0.01, n_estimators=100, gamma=0, 
                        min_child_weight=1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005)

xgboost.fit(X_train, y_train)
preds = xgboost.predict(X_test)

accuracy = (preds == y_test).sum().astype(float) / len(preds)*100

print("XGBoost's prediction accuracy WITH optimal hyperparameters is: %3.2f" % (accuracy))


# The parameter tuning did not improve the accuracy as excpected. We will therefore do a cross validation to test to get a more reliable result.

# In[ ]:


xgb_cv = XGBClassifier(n_estimators=100)
scores = cross_val_score(xgb_cv, X_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# Depending on the run the cross validation results vary from a little lower and a little better than before. 
# 
# This indicates that the parameter tuning was not as effective as expected - this could mean that XGBoost was actually close to its maximum performance capability on this data set.
# 
# As we still have a good performance we will now continue with further evaluation of the performance of our model!

# ## XGBoost - Evaluation

# ### Confusion Matrix

# In[ ]:


unique, counts = np.unique(sdss['class'], return_counts=True)
dict(zip(unique, counts))


# In[ ]:


predictions = cross_val_predict(xgb, sdss.drop('class', axis=1), sdss['class'], cv=3)
confusion_matrix(sdss['class'], predictions)


# The first row shows that out of 4998 stars, **4962 were classified correctly as stars**. 29 stars were classified incorrectly as galaxies and 7 stars were classified incorrectly as quasars.
# 
# The second row shows out of 850 quasars **827 were classified correctly**. 22 qsos were classified incorrectly as stars and 1 quasar was classified as galaxy.
# 
# The last row tells us that out of 4152 galaxies **4147 were classified correctly.**. 5 galaxies were classified incorrectly as star.
# 
# In total: 
# 
# We have only 64 objects which were classified incorrectly. Most of the objects were recognized as what they are.

# ### Precision & Recall

# In[ ]:


print("Precision:", precision_score(sdss['class'], predictions, average='micro'))
print("Recall:",recall_score(sdss['class'], predictions, average='micro'))


# Precision is the fraction of events where the algorithm classified an object of type **t** correctly out of all occurences of the algorithm classifying objects of type **t**.
# 
# Recall is the fraction of events where the algorithm classified an object of type **t** correctly when the true type of that object was actually **t**.
# 
# Precision in our case:
# 
# For every class its calculated how many objects were classified as stars (or galaxies or quasars) in relation to the amount of correct star (or galaxies or quasars) predictions. The results are averaged --> 99.36%.
# 
# Recall in our case:
# 
# For every class its calculated how many objects were classified as stars (or galaxies or quasars) in relation to the total amount of predictions where the object actually was a star (or galaxy or quasar). The results are averaged --> 99.36%.
# 
# In both cases our algorithm did a very good job. The highest precision or recall value a predictor can have is 1.0.

# ### F1-Score

# One can combine precision and recall into one score, which is called the F-score. The F-score is computed with the harmonic mean of precision and recall. Note that it assigns much more weight to low values. As a result of that, the classifier will only get a high F-score, if both recall and precision are high.

# In[ ]:


print("F1-Score:", f1_score(sdss['class'], predictions, average='micro'))


# As precision and recall have the same value the F1-Score has automatically the same value too. Again, we are very close to 1.0 which indicates strong performance.

# ## Summary

# In this notebook learned how to get data from the SDSS RD14, analyze the data (we learned some very interesting facts about our space along the way), how to build a machine learning model to predict for unseen data from this data set and how to improve its performance (even though there was only a slight improvent).
# We used XGBoost for predicting and evaluated its result.
# 
# This project was very interesting to work on as I'm also interested in space and astronomy.

# In[ ]:





# In[ ]:




