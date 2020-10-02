#!/usr/bin/env python
# coding: utf-8

# # Gender Recognition by Voice

# 
#    **meanfreq:** mean frequency (in kHz)
#    
#    **sd:** standard deviation of frequency
#    
#    **median:** median frequency (in kHz)
#    
#    **Q25:** first quantile (in kHz)
#    
#    **Q75:** third quantile (in kHz)
#    
#    **IQR:** interquantile range (in kHz)
#    
#    **skew:** skewness (see note in specprop description)
#    
#    **kurt:** kurtosis (see note in specprop description)
#    
#    **sp.ent:** spectral entropy
#    
#    **sfm:** spectral flatness
#    
#    **mode:** mode frequency
#    
#    **centroid:** frequency centroid (see specprop)
#    
#    **peakf:** peak frequency (frequency with highest energy)
#    
#    **meanfun:** average of fundamental frequency measured across acoustic signal
#    
#    **minfun:** minimum fundamental frequency measured across acoustic signal
#    
#    **maxfun:** maximum fundamental frequency measured across acoustic signal
#    
#    **meandom:** average of dominant frequency measured across acoustic signal
#    
#    **mindom:** minimum of dominant frequency measured across acoustic signal
#    
#    **maxdom:** maximum of dominant frequency measured across acoustic signal
#    
#    **dfrange:** range of dominant frequency measured across acoustic signal
#    
#    **modindx:** modulation index.
#    
#    Calculated as the accumulated absolute difference between adjacent measurements of fundamental fequencies divided by the frequency range. 
#    
#    **label:** male or female
# 

# In[ ]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd
import math 
# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

import keras
from keras import backend as K

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# In[ ]:


train = pd.read_csv('../input/voice.csv')


# In[ ]:


df = train.copy()


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# **No imputations needed.**

# In[ ]:


temp = []
for i in df.label:
    if i == 'male':
        temp.append(1)
    else:
        temp.append(0)
df['label'] = temp


# ** 0 - FEMALE , 1 - MALE** 

# In[ ]:


df.label.value_counts()


# In[ ]:


correlation_map = df.corr()
obj = np.array(correlation_map)
obj[np.tril_indices_from(obj)] = False
fig,ax= plt.subplots()
fig.set_size_inches(15,15)
sns.heatmap(correlation_map, mask=obj,vmax=.7, square=True,annot=True)


# **This clearly shows that Q25 and meanfun highly correlates with label inversely which is our are target variable.**

# **Some other variables are sd , IQR , sp.ent that correlates directly with label.**

# In[ ]:


kws = dict(s=50, linewidth=.5, edgecolor="w")
g = sns.FacetGrid(df, col="label", palette="Set1")
g = (g.map(plt.scatter, "meanfun", "IQR", **kws).add_legend())


# ** This shows that a MALES have comparatively LOW IQR and LOW meanfun and their respective mappings as well when compared to FEMALES.**

# In[ ]:


kws = dict(s=50, linewidth=.5, edgecolor="g")
g = sns.FacetGrid(df, col="label", palette="Pal")
g = (g.map(plt.scatter, "sp.ent", "Q25", **kws).add_legend())


# **The above plot depicts the outliers and that we can actually see the instances of the dataframe that are filled with LOW sp.ent and a HIGH Q25 are labeled as FEMALE whereas the males fail to show the above stated trend.**

# In[ ]:


kws = dict(s=50, linewidth=.5, edgecolor="y")
g = sns.FacetGrid(df, col="label")
g = (g.map(plt.scatter, "Q25", "meanfun", **kws).add_legend())


# In[ ]:


kws = dict(s=50, linewidth=.5, edgecolor="m")
g = sns.FacetGrid(df, col="label", palette="Set1")
g = (g.map(plt.scatter, "IQR", "sd", **kws).add_legend())


# **The below plots will clearly give you the idea of where the points have been accumulated greater than others or their densities.**

# In[ ]:


sns.set(style="white", color_codes=True)
sns.jointplot("meanfun", "IQR", data=df, kind="reg" , color='k')
sns.set(style="darkgrid", color_codes=True)
sns.jointplot("sp.ent", "Q25", data=df, kind="hex" , color='g')
sns.set(style="whitegrid", color_codes=True)
g = (sns.jointplot(df.Q25 , df.meanfun , kind="hex", stat_func=None).set_axis_labels("Q25", "meanfun"))


# **Combining above all to have one last glance at all important features**

# In[ ]:


g = sns.PairGrid(df[["meanfun" , "Q25" , "sd" , "meanfreq" , "IQR" , "sp.ent" , "centroid", "label"]] , palette="Set2" , hue = "label")
g = g.map(plt.scatter, linewidths=1, edgecolor="r", s=60)
g = g.add_legend()


# **Though we don't have any values to impute , but if we have to ,then we have to go with the inferences from the plots above to impute the more accurate value into their respective columns**

# In[ ]:


from sklearn import svm
X = df[df.columns[0:20]]
y = df.label
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2)
clf = svm.SVC()
clf.fit(X_train , y_train)


# In[ ]:


pred = clf.predict(X_test)


# In[ ]:


print(accuracy_score(y_test, pred))


# **Non crossvalidated accuracy without any standardization or normalization is 0.7634 which is very less. **

# **Now , Standardizing the data for greater performance by the model.**

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
df = scaler.transform(df)


# In[ ]:


X = df[: , 0:20]
y = df[: , 20]
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2)
clf = svm.SVC()
clf.fit(X_train , y_train)


# In[ ]:


pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))


# **The above accuracy clearly depicts that how much fruitful is doing the standardization for this dataframe.**

# **It achieves a pretty much good accuracy of 0.9810 only after standardization i.e. making the mean of the each and every panda series in the dataframe as 0 , variance as 1.**

# ## Now , tuning the parameters and trying different models.

# In[ ]:


from sklearn.model_selection import cross_val_score
clf = svm.SVC()
scores = cross_val_score(clf , X , y , cv=10)
scores


# **Mean cross-validation score for default parameters.**

# In[ ]:


scores.mean()


# **Different cross-validation scores with default parameters of SVC** 

# In[ ]:


temp = np.arange(11)[1 : 11]
plt.plot(temp , scores)
plt.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV
model = svm.SVC()
param_grid = {
"C" : np.linspace(0.01,1,10)
}
grid = GridSearchCV( model , param_grid , cv = 10 , scoring = "accuracy")
grid.fit(X,y)


# **For tuning only C , we got a rise in the mean cross-validation than before.**

# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# In[ ]:


grid.grid_scores_


# **Cross-validation scores for each value of C**

# In[ ]:


for i in grid.grid_scores_:
    print(i[2])


# **Variation of mean cross-validation scores wrt the change in C parameter alone. Notice the shoot in curve**

# In[ ]:


gridscores = []
for i in grid.grid_scores_:
    gridscores.append(i[1])
    
plt.xlabel('C')
plt.ylabel('Mean cross-validation Accuracy')
plt.plot(np.linspace(0.01,1,10) , gridscores , 'r')
plt.show()


# In[ ]:


from sklearn.model_selection import GridSearchCV
model = svm.SVC()
param_grid = {
"gamma" : [0.0001,0.001,0.01,0.1,1,10,100,300,600]
}
grid = GridSearchCV( model , param_grid , cv = 10 , scoring = "accuracy")
grid.fit(X,y)


# **Again, with tuning of only gamma gave us the best of the mean cross-validation scores than the two cases before.**

# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# In[ ]:


grid.grid_scores_


# **The plot at the below shows that how Mean cross-validation Accuracy decreases with increase in gamma parameter.**

# In[ ]:


gridscores = []
for i in grid.grid_scores_:
    gridscores.append(i[1])
    
plt.xlabel('Gamma')
plt.ylabel('Mean cross-validation Accuracy')
plt.plot([0.0001,0.001,0.01,0.1,1,10,100,300,600] , gridscores , 'k')
plt.show()


# In[ ]:


model = svm.SVC()
param_grid = {
"C" : [0.50 , 0.55 , 0.59 , 0.63] ,
"gamma" : [0.005, 0.008, 0.010, 0.012 , 0.015]
}
grid = GridSearchCV( model , param_grid , cv = 10 , scoring = "accuracy")
grid.fit(X,y)


# **The so far best score with SVC is 0.9681 with rbf kernel.**

# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# In[ ]:


model = svm.SVC(kernel='linear')
param_grid = {
"C" : [0.50 , 0.55 , 0.59 , 0.63] ,
"gamma" : [0.005, 0.008, 0.010, 0.012 , 0.015]
}
grid = GridSearchCV( model , param_grid , cv = 10 , scoring = "accuracy")
grid.fit(X,y)


# **With linear kernel we got 0.96969 or ~97% accuracy.**

# In[ ]:


grid.best_score_


# In[ ]:


grid.best_params_


# **Still tuning in proximity gave 0.9700481 for C = 0.1 and gamma = 0.447**

# In[ ]:


model = svm.SVC(kernel='linear',C=0.1 , gamma=0.447)
scores = cross_val_score(model , X, y, cv=10, scoring='accuracy')
print(scores.mean())


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 61 , max_depth = 37)
scores = cross_val_score(model , X , y , cv=10)
scores


# In[ ]:


scores.mean()


# **RandomForestClassifier is also giving accuracy close to SVC.**

# In[ ]:


model.fit(X , y)
print(model.feature_importances_)


# **Feature Importance**

# In[ ]:


ranks = np.argsort(-model.feature_importances_)
f, ax = plt.subplots(figsize=(12, 7))
sns.barplot(x=model.feature_importances_[ranks], y=train.columns.values[ranks], orient='h')
ax.set_xlabel("Importance Of Features in RandomForestClassifier")
plt.tight_layout()
plt.show()


# **With Default parameters GradientBoostingClassifier is giving 0.9681.**

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
scores = cross_val_score(model, X , y , cv=5)
scores


# In[ ]:


scores.mean()


# **With Default parameters XGBClassifier is giving 0.9674.**

# In[ ]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
scores = cross_val_score(model, X , y , cv=10)
scores


# In[ ]:


scores.mean()


# In[ ]:


xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30)
xgb.fit(X_train , y_train)
pred = xgb.predict(X_test)


# In[ ]:


print(accuracy_score(y_test , pred))


# ## XGBClassifier is able to give the whooping accuracy of 99.52 %

# **But the above accuracy is not cross-validated.**

# **The best cross-validated score obtained so far is by Support Vector Machines and is 97.004 %.**

# In[ ]:


ranks = np.argsort(-xgb.feature_importances_)
f, ax = plt.subplots(figsize=(12, 7))

sns.barplot(x=xgb.feature_importances_[ranks], y=train.columns.values[ranks], orient='h')
ax.set_xlabel("Importance Of Features in XGBClassifier")
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




