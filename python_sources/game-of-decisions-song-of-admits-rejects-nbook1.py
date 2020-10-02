#!/usr/bin/env python
# coding: utf-8

# Hi! I applied to some highly ambitious grad schools this year and was completely baffled by their admission criteria.  
# To relieve my boredom while waiting for the decisions, what better thing to do than trying to find some shred of sanity in this insane game of decisions.  
# Lets load the dataset and take a look

# In[35]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

df = pd.read_csv("../input/Admission_Predict.csv",sep = ",",index_col='Serial No.')
df.head()


# Hmm, column names seem a bit off, lets have one worder column names

# In[36]:


columnMap = {key: key.split()[0] for key in df.head(0)}
df.rename(columns = columnMap, inplace = True)
df.head(1)


# Better, now let's check if there are any missing values

# In[37]:


df.isna().any()


# No missing values, cool, lets analyze our data through histograms and correlation

# In[38]:


df.hist(layout=(2,4),figsize=(16,6))
df.corr()


# The Highest correlation of Chance is with CGPA. While [correlation doesn't imply causation](https://www.mathtutordvd.com/public/Why-Correlation-does-not-Imply-Causation-in-Statistics.cfm), this aligns with my personal observation that people with high CGPA breeze through the admissions even if their SOP or GRE are quite lackluster. Later we will find out the importance of each feature in deciding the admit.

# Now, there are two ways to work with this data.  
# * **Classification**  
#     Since we are given the chance of acceptance, we can assume that if Chance>P its an Admit, otherwise a Reject.  
#     This will convert the problem into a classification problem with classes Admit and Reject and the parameter P will allow us some measure of control for the ranking of the institution you are applying to.  
#     For eg. P = .9 for highly selective uni's like Stanford and Berkeley while P = .6 for some of the saner ones
# * **Regression**  
#     We can also do the vanilla regression where we predict the probability of gaining an Admit.
#     Since there exist numerous other kernels with regression analysis, I won't be handling it here

# Let's begin by preprocesings the data by standardizing it and splitting into train and test sets

# In[100]:


X= df.drop("Chance",axis=1).apply(zscore)
X.Research = df.Research
Y = df.Chance
x_train, x_test,y_train, y_test = train_test_split(X,Y,test_size = 0.20,random_state = 0)

P=.9
yc1_train = np.where(y_train>=P, 1, 0)
yc1_test = np.where(y_test>=P, 1, 0)

P =.6
yc2_train = np.where(y_train>=P, 1, 0)
yc2_test = np.where(y_test>=P, 1, 0)


# Now, let's train a random forest model to predict the decisions as well as finding out the importance of each feature on the decision

# In[111]:


rf1 = RandomForestClassifier(n_estimators=100,random_state=0)
rf1.fit(x_train, yc1_train)

rf2 = RandomForestClassifier(n_estimators=100,random_state=0)
rf2.fit(x_train, yc2_train)

yc1_out = rf1.predict(x_test)
yc2_out = rf2.predict(x_test)

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

importances1 = pd.DataFrame(rf1.feature_importances_,index = x_train.columns,columns=['importance']).sort_values('importance',ascending=False)
importances1.plot(kind='bar',title='Selective University',ax=ax1,ylim=(.05,.35))

importances2 = pd.DataFrame(rf2.feature_importances_,index = x_train.columns,columns=['importance']).sort_values('importance',ascending=False)
importances2.plot(kind='bar',title='Sane University',ax=ax2,ylim=(.05,.35))

print("Selective University's Report")
print(mt.classification_report(yc1_test, yc1_out,target_names=['Reject','Admit']))

print("Sane University's Report")
print(mt.classification_report(yc2_test, yc2_out,target_names=['Reject','Admit']))


# Here we see that CGPA is the single most important factor in deciding your admission which is self-evident as it is a good indication of one's hardwork and perserverence if not intelligence.   
# 
# What is not evident is the fact that for the selective universities only CGPA and GRE scores matter, your University, LOR and SOP plays a much smaller role.  
# 
# Finally in both cases, your university has a minimal impact in your decision, even if you are from a tier 2 or 3 college, as long as you have a high enough CGPA, you are likely to get an admit. Well, perserverence pays :)
# 
# Disclaimer: I can not verify the authenticity of this data set, please use your own head while reading this analysis
# 
# Ciao !
