#!/usr/bin/env python
# coding: utf-8

# ## 1.- Introduction

# I made this as a small work for a master degree. It's not a big deal but as long it helps somebody it will be fine (:
# 
# You can found further detail of the dataset, among a very interesting example, in the work of Lennart Grosser -> https://www.kaggle.com/lucidlenn/data-analysis-and-classification-using-xgboost

# The personal approach in this notebook will be to avoid using data that doesnt belong to the objects itselves, like characteristics of the telescope and other identifiers.
# From this features we will try to classify between: <b> Stars</b>, <b> Galaxys </b> and <b> Quasars </b>.
# 
# I'm nothing near to a specialist in astronomy so feel free to correct me!

# ## 2.- Data analysis
# 

# The characteristics to be taken into account for the analysis are:
# 
# <b>ra</b> y <b>dec</b>: Respectively, right ascension and declination. Representing the position of the object in the celestial sphere
# 
# <b>u</b>, <b>g</b>, <b>r</b>, <b>i</b>, <b>z</b>: Thuan-Gunn magnitudes. They refer to the response of the object to the different frequency bands of the telescope.
# 
# <b>redshift</b>: Effect that happens when light from a very distant object is increased in wavelength.
# 
# <b>mjd</b>: Modified Julian Date on which the image was taken.

# In[3]:


#Load of the librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# We load the data. Those that have nothing to do with the features of the objects are ignored.
sdss_data=pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv', skiprows=1)
sdss_data.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid', 'plate', 'fiberid'], axis=1, inplace=True)


# In[5]:


#We have a general look at the features
sdss_data.head(3)


# In[6]:


sdss_data.describe()


# In[7]:


sdss_data['class'].value_counts()

star_color='#4a7dce'
galaxy_color='#7e0087'
qso_color='#870000'

array_color=[star_color, galaxy_color, qso_color]

sdss_data['class'].value_counts().plot(kind='bar',title='Number of samples', color=['#7e0087','#4a7dce','#870000']);


# **This color legend will be maintained through all the notebook**

# ####              How does the data relate to each other?

# In[20]:


#We visualize the correlation matrix between the characteristics of each class...

fig, axes = plt.subplots(nrows=3, ncols=1,figsize=(8, 16))
fig.set_dpi(100)
ax = sns.heatmap(sdss_data[sdss_data['class']=='GALAXY'][['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift', 'mjd']].corr(), ax = axes[0], cmap='coolwarm', annot=True)
ax.set_title('Galaxy')
ax = sns.heatmap(sdss_data[sdss_data['class']=='STAR'][['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift', 'mjd']].corr(), ax = axes[1], cmap='coolwarm', annot=True)
ax.set_title('Star')
ax = sns.heatmap(sdss_data[sdss_data['class']=='QSO'][['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift', 'mjd']].corr(), ax = axes[2], cmap='coolwarm', annot=True)
ax = ax.set_title('Quasar')


# In[21]:


#...and with all the classes together
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(8, 5))
fig.set_dpi(100)
ax = sns.heatmap(sdss_data[['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift', 'mjd']].corr(), cmap='coolwarm', annot=True)
ax.set_title('Stars, Galaxys and Quasars');


# ### Analysis of color curves
# 
# The average, minimum and maximum of the Thuan-Gunn magnitudes are observed in search of some particularity.

# In[22]:


fig=plt.figure(1,figsize=(16,16))

ugriznames_array=['u','g','r','i','z']

for i in range(5):

    ax = fig.add_subplot(3, 2, i+1)
    ax.set_ylim(10,30)
    ax.xaxis.grid(False)
    # construct some data like what you have:
    u_errorbar= [sdss_data[sdss_data['class']=='STAR'][ugriznames_array[i]],
                 sdss_data[sdss_data['class']=='GALAXY'][ugriznames_array[i]],
                 sdss_data[sdss_data['class']=='QSO'][ugriznames_array[i]]]

    x=[0,1,2];

    mins = np.array([u_errorbar[0].min(0),u_errorbar[1].min(0),u_errorbar[2].min(0)])
    maxes = np.array([u_errorbar[0].max(0),u_errorbar[1].max(0),u_errorbar[2].max(0)])
    means = np.array([u_errorbar[0].mean(0),u_errorbar[1].mean(0),u_errorbar[2].mean(0)])
    std = np.array([u_errorbar[0].std(0),u_errorbar[1].std(0),u_errorbar[2].std(0)])

    # create stacked errorbars:
    plt.errorbar(np.arange(3), means, std, fmt='ok', linewidth=21, ecolor=array_color)
    plt.errorbar(np.arange(3), means, [means - mins, maxes - means],
                 fmt='.k', ecolor=array_color, lw=17)
    plt.tick_params( axis='x', which='both', bottom=False,top=False,labelbottom=False)

    plt.title(ugriznames_array[i])


# ### Dimensionality reduction
# The variables related to the Thuan-Gunn magnitudes will be subject to a reduction in their dimensionality. Its very high correlation will allow us to reduce the number of dimensions without much loss of information. We will go from 5 to 2 characteristics, as reducing to 1 may be excessive as the correlation decreases when the variables are not close in the spectrum.

# In[12]:


from sklearn.preprocessing import StandardScaler


# In[23]:


features = ['u','g','r', 'i', 'z'];

# Normalization of the features
x = sdss_data.loc[:, features].values
x = StandardScaler().fit_transform(x)


# In[24]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

prinComp = pca.fit_transform(x)

ugriz_pca_Df = pd.DataFrame(data = prinComp, columns = ['ugriz_pca1','ugriz_pca2'])

#Se crea el conjunto final de datos
sdss_finaldata=pd.concat([ sdss_data[['ra']], sdss_data[['dec']], ugriz_pca_Df, sdss_data[['redshift']], sdss_data[['class']], sdss_data[['mjd']] ], axis = 1)


# In[25]:


sdss_finaldata.head()


# In[26]:


#The data is displayed according to the two main ugriz components

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 1', fontsize = 15)
ax.set_ylabel('PC 2', fontsize = 15)
ax.set_title('PCA result', fontsize = 20)

clases = ['STAR', 'GALAXY', 'QSO']
colors = [star_color, galaxy_color, qso_color]
for clases, color in zip(clases,colors):
    indicesToKeep = sdss_finaldata['class'] == clases
    ax.scatter(ugriz_pca_Df.loc[indicesToKeep, 'ugriz_pca1'], ugriz_pca_Df.loc[indicesToKeep, 'ugriz_pca2']
               , c = color, s = 15,alpha=0.45)
ax.legend(['STAR', 'GALAXY', 'QSO'])
ax.grid()


# ## 3.- Classification

# In[27]:


from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score
import time


# In[28]:


# Some algorithms don't support categorical classes so we'll have to replace them with numbers
d=pd.DataFrame(sdss_finaldata)


class_num=pd.DataFrame(LabelEncoder().fit_transform(d['class']), columns=['class'])
d.drop(['class'], axis=1, inplace=True)
names=list(d)


# In[29]:


#Data are normalized for better conditioning of the problem

scaler = MinMaxScaler()
d=pd.DataFrame(scaler.fit_transform(d), columns=names)


d=pd.concat([d, class_num], axis=1)

d.head(3)


# In[30]:


#A cross validation will be performed to ensure the reliability of the results.

#In addition, an isolated training will serve to measure the times and extract a matrix of confusion than will give us a general idea.

x=d.drop('class',axis=1);
y=d['class']

x_train, x_test, y_train, y_test = train_test_split(d.drop('class',axis=1), d['class'], test_size=0.4)


# For the classification, simple sklearn algorithms will be used.
# 
# ### ---Logistic Regression

# In[31]:


from sklearn import linear_model, datasets


lr = linear_model.LogisticRegression() 

training_start = time.perf_counter()
lr.fit(x_train, y_train)#Training
training_end = time.perf_counter()

predict_start = time.perf_counter()
preds = lr.predict(x_test)#Prediction
predict_end = time.perf_counter()
acc_lreg = (preds == y_test).sum().astype(float) / len(preds)*100

print("The first iteration of the Logistic Regression gives an accuracy of the %3.2f %%" % (acc_lreg))

from numpy import linalg as LA
mc=confusion_matrix(y_test, preds)
mc_norm = mc / np.linalg.norm(mc, axis=1, keepdims=True)
sns.heatmap(pd.DataFrame(mc_norm), cmap=sns.cm.rocket_r, annot=True, fmt='.5g',);

print("[0=Galaxy 1=Quasar 2=Star]")


# In[32]:


lr_train_t=training_end-training_start;
lr_predict_t=predict_end-predict_start;

scores = cross_val_score(lr, x, y, cv=10, scoring = "accuracy")
score_lr=scores.mean()
print("The 10 cross validations of Logistic Regression have had an average success rate of  %3.2f %%" %(score_lr*100))
std_lr=scores.std()
print("..and a standar deviation of %8.5f" %(std_lr))


# ### ---K-Nearest Neighbours

# In[35]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

training_start = time.perf_counter()
knn.fit(x_train, y_train)
training_end = time.perf_counter()

predict_start = time.perf_counter()
preds = knn.predict(x_test)
predict_end = time.perf_counter()
acc_knn = (preds == y_test).sum().astype(float) / len(preds)*100

print("The first iteration of the K-Nearest Neighbours gives an accuracy of the %3.2f %%" % (acc_knn))



mc=confusion_matrix(y_test, preds)
mc_norm = mc / np.linalg.norm(mc, axis=1, keepdims=True)
sns.heatmap(pd.DataFrame(mc_norm), cmap=sns.cm.rocket_r, annot=True, fmt='.5g')


print("[0=Galaxy 1=Quasar 2=Star]")


# In[37]:


knn_train_t=training_end-training_start;
knn_predict_t=predict_end-predict_start;

scores = cross_val_score(knn, x, y, cv=10, scoring = "accuracy")
score_knn=scores.mean()
print("The 10 cross validations of K- Nearest Neighbours have had an average success rate of %3.2f %%" %(score_knn*100))
std_knn=scores.std()
print("..and a standar deviation of %8.5f" %(std_knn))


# ### ---Naive-bayes

# In[39]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

training_start = time.perf_counter()
gnb.fit(x_train, y_train)
training_end = time.perf_counter()

predict_start=time.perf_counter()
preds = gnb.predict(x_test)
predict_end = time.perf_counter()
acc_gnb = (preds == y_test).sum().astype(float) / len(preds)*100


print("The first iteration of the naive Bayes gives an accuracy of the %3.2f %%" % (acc_gnb))


mc=confusion_matrix(y_test, preds)
mc_norm = mc / np.linalg.norm(mc, axis=1, keepdims=True)
sns.heatmap(pd.DataFrame(mc_norm), cmap=sns.cm.rocket_r, annot=True, fmt='.5g');


# In[40]:


gnb_train_t=training_end-training_start;
gnb_predict_t=predict_end-predict_start;

scores = cross_val_score(gnb, x, y, cv=10, scoring = "accuracy")
score_gnb=scores.mean()
print("The 10 cross validations of naive Bayes have had an average success rate of %3.2f %%" %(score_gnb*100))
std_gnb=scores.std()
print("..and a standar deviation of %8.6f" %(std_gnb))


# ### ---Random Forest

# In[41]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10)

training_start = time.perf_counter()
rfc.fit(x_train, y_train)
training_end = time.perf_counter()

predict_start=time.perf_counter()
preds = rfc.predict(x_test)
predict_end = time.perf_counter()

acc_rfc = (preds == y_test).sum().astype(float) / len(preds)*100


print("The first iteration of the Random Forest gives an accuracy of the %3.2f %%" % (acc_rfc))


mc=confusion_matrix(y_test, preds)
mc_norm = mc / np.linalg.norm(mc, axis=1, keepdims=True)
sns.heatmap(pd.DataFrame(mc_norm), cmap=sns.cm.rocket_r, annot=True, fmt='.5g');


# In[44]:


rfc_train_t=training_end-training_start;
rfc_predict_t=predict_end-predict_start;

scores = cross_val_score(rfc, x, y, cv=10, scoring = "accuracy")
score_rfc=scores.mean()
print("The 10 cross validations of Random Forest have had an average success rate of %3.2f" %(score_rfc*100))
std_rfc=scores.std()
print("..and a standar deviation of %8.6f" %(std_rfc))


# ### ---Support Vector Machines

# In[47]:


from sklearn.svm import SVC

svm = SVC(kernel='sigmoid', gamma='auto')

training_start = time.perf_counter()
svm.fit(x_train, y_train)
training_end = time.perf_counter()

predict_start = time.perf_counter()
preds = svm.predict(x_test)
predict_end = time.perf_counter()

acc_svm = (preds == y_test).sum().astype(float) / len(preds)*100


print("The first iteration of the SVM gives an accuracy of the %3.2f %%" % (acc_svm))


mc=confusion_matrix(y_test, preds)
mc_norm = mc / np.linalg.norm(mc, axis=1, keepdims=True)
sns.heatmap(pd.DataFrame(mc_norm), cmap=sns.cm.rocket_r, annot=True, fmt='.5g');


# In[48]:


svm_train_t=training_end-training_start;
svm_predict_t=predict_end-predict_start;

scores = cross_val_score(svm, x, y, cv=10, scoring = "accuracy")
score_svm=scores.mean()
print("The 10 cross validations of SVM have had an average success rate of %3.2f %%" %(score_svm*100))
std_svm=scores.std()
print("..and a standar deviation of %8.6f" %(std_svm))


# ### ---Redes Neuronales

# In[49]:


from sklearn.neural_network import MLPClassifier

nnc = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 3), random_state=1)

training_start = time.perf_counter()
nnc.fit(x_train, y_train)                         
training_end = time.perf_counter()

predict_start = time.perf_counter()
preds=nnc.predict(x_test)
predict_end=time.perf_counter()

acc_nnc = (preds == y_test).sum().astype(float) / len(preds)*100
print("The first iteration of the Neural Networks gives an accuracy of the %3.2f %%" % (acc_nnc))
mc=confusion_matrix(y_test, preds)
mc_norm = mc / np.linalg.norm(mc, axis=1, keepdims=True)
sns.heatmap(pd.DataFrame(mc_norm), cmap=sns.cm.rocket_r, annot=True, fmt='.5g');


# In[50]:


nnc_train_t=training_end-training_start;
nnc_predict_t=predict_end-predict_start;

scores = cross_val_score(nnc, x, y, cv=10, scoring = "accuracy")
score_nnc=scores.mean()
print("The 10 cross validations of Neural Networks have had an average success rate of %3.2f %%" %(score_nnc*100))
std_nnc=scores.std()
print("..and a standar deviation of %8.6f" %(std_nnc))


# ### Results analysis

# In[52]:


scores_df=pd.DataFrame([score_lr , score_knn, score_gnb, score_rfc, score_svm, score_nnc])
std_df=pd.DataFrame([std_lr , std_knn, std_gnb, std_rfc, std_svm, std_nnc])
train_t=pd.DataFrame([lr_train_t , knn_train_t, gnb_train_t, rfc_train_t, svm_train_t, nnc_train_t])
predict_t=pd.DataFrame([lr_predict_t , knn_predict_t, gnb_predict_t, rfc_predict_t, svm_predict_t, nnc_predict_t])

names=['lreg','knn','gnb','rfc','svm','nnc']

fig, axes = plt.subplots(figsize=(12,12), nrows=2, ncols=2)

ax=scores_df.plot(kind='bar',title='Accuracy', ax=axes[0,0], legend=False)
ax.set_xticklabels(names)
ax=std_df.plot(kind='bar',title='Deviation', ax=axes[0,1], legend=False)
ax.set_xticklabels(names)
ax=train_t.plot(kind='bar',title='t Train', ax=axes[1,0], legend=False)
ax.set_xticklabels(names)
ax=predict_t.plot(kind='bar',title='t Prediction', ax=axes[1,1], legend=False)
ax.set_xticklabels(names);
     


# ## 4.- Conclusion
# <b>Random Forest</b> scores the best accuracy and the minimun deviation, so we can consider it as the best classfier for this problem.
# 
# 
# 
# ### **Thanks for reading!!**

# In[ ]:




