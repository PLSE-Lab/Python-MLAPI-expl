#!/usr/bin/env python
# coding: utf-8

# # A brief introduction to particle physics 
# This dataset from https://www.kaggle.com/naharrison/particle-identification-from-detector-responses/home is a simulation of electron-proton inelastic scattering measured by a particle detector system. 
# In order to analyse these data is necessary to recall some concepts.
# ## The Standard Model
# The building blocks of matter are elementary particles. These particles are divided in two major types: quarks and leptons. The Standard Model also studies the interaction of these particles through fundamental forces (strong, weak and electromagnetic).
# For every type of particle there also exists a corresponding antiparticle.
# ### Quarks
# Quarks are fundamental constituents of matter because they combine to form hadrons. There are six quarks paired in three groups: "up/down", "charm/strange" and "top/bottom". They are held together through strong forces.
# #### Hadrons 
# They divide in Baryons and Mesons. Baryons are made of three quarks. For example **protons** are made of (uud) quarks and  neutrons are made of (udd) quarks.
# Mesons contain one quark and one antiquark. An example of a meson is a **pion** which is made of an up quark and a down antiquark. Another example of a meson is **kaon**, it is formed by a up or down quark and a anti-strange quark.
# ### Leptons
# Leptons have a 1/2 spin and do not undergo strong interactions. There are six leptons, three of wich have an electrical charge. These are: electron, muon and tau. The three remaining are neutrinos. A **positron** is the antiparticle counterpart of an electron. It possess the same mass and spin but positive charge.
# 
#  

# # Inelastic scattering
# Is a process used to probe the inside structure of hadrons, in this case protons. In this process a incident particle (photoelectron) collides with a target proton. The kinetic energy of the incident particle is not conserved after the collision. During inelastic scattering a proton can break up into its constituent quarks which then form a hadronic jet. The angles of the deflection gives information about the nature of the process.

# # Import modules

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd


# # Read dataset

# In[ ]:


df = pd.read_csv('../input/pid-5M.csv')


# # Data Visualization

# In[ ]:


df.head()
#The id means: positron (-11), pion (211), kaon (321), and proton (2212)
#p is momentum (GeV/c)
#theta and beta are angles (rad)
#nphe is the number of photoelectons
#ein is the inner energy (GeV)
#eout is the outer energy (GeV)


# In[ ]:


df.describe()


# In[ ]:


df.shape #number of rows and columns


# In[ ]:


sns.set(style='darkgrid')
sns.distplot(df['p'], hist=True, kde=True, color='c')
plt.xlabel('Momentum of Particles')
plt.ylabel('Feature Value')
plt.title('Momentum Distribution')


# In[ ]:


#correlation heat map
sns.set(style='darkgrid')
corr = df[['id', 'p','ein', 'eout','nphe', 'theta', 'beta']].corr()
sns.heatmap(corr)


# In[ ]:


f1 = df['p'].values
f2 = df['beta'].values
plt.scatter(f1, f2, c='black', s=7)
plt.xlabel('Momentum of the Particle')
plt.ylabel('beta angle')


# # Preprocessing
# ## Removing null values

# In[ ]:


df.isnull().sum() #there are no null values


# ## Split dataset into train and test set

# In[ ]:


features = df.drop('id', axis=1)
labels = df['id']


# In[ ]:


#test and train split using sklearn.model_selection
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.22, random_state = 1)


# In[ ]:


y_train.unique()


# # Applying Models
# 

# In[ ]:


from sklearn.metrics import accuracy_score


# ## SGDClassifier

# In[ ]:


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(x_train, y_train)
pred_sgd = clf.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_sgd))


# ## AdaBoostClassifier

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
clf_abc = AdaBoostClassifier()
clf_abc.fit(x_train, y_train)
pred_abc = clf_abc.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_abc))


# ## XGBoost
# 

# In[ ]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
pred_xgb = xgb.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_xgb))


# ## RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier()
clf_rfc.fit(x_train, y_train)
pred_rfc = clf_rfc.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_rfc))

