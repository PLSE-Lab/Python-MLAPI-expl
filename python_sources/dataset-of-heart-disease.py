#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size = "7">**HEART DISEASE**</font></center></h1>
# <h2><center><font size = "6">**DATA VISUALIZATION and PREDICTIVE MODELS**</font></center></h2>
# <a id='0'><font size = "5">**CONTENT**</font></a>
# - <a href='#1'>INTRODUCTION</a>
# - <a href='#2'>LOAD PACKAGES and DATA</a>
# - <a href='#3'>CHECK and EXAMINATION of the DATA</a>
# - <a href='#4'>DATA EXPLORATION and DATA VISUALIZATION</a>
#     - <a href='#41'>PIE PLOT and BAR PLOT</a> 
#     - <a href='#42'>HISTOGRAM</a>
#     - <a href='#43'>LINE PLOT</a> 
#     - <a href='#44'>SCATTER PLOT</a> 
#     - <a href='#45'>PLOT and SUBPLOT</a>
#     - <a href='#46'>BAR PLOT</a>
#     - <a href='#47'>SUBPLOT</a>
#     - <a href='#48'>BOX PLOT</a>
#     - <a href='#49'>BAR PLOT and COUNT PLOT</a> 
#     - <a href='#410'>HEATMAP</a>
#     - <a href='#411'>JOINT PLOT</a> 
#     - <a href='#412'>LM PLOT</a> 
#     - <a href='#413'>POINT PLOT</a>
#     - <a href='#414'>SWARM PLOT</a>
#     - <a href='#415'>VIOLIN PLOT</a>
# - <a href='#5'>PREDICTIVE MODELS</a>
#     - <a href='#51'>MULTIPLE LINEAR REGRESSION</a> 
#     - <a href='#52'>DECISION TREE</a>
#     - <a href='#53'>K-NEAREST NEIGHBORS</a> 
#     - <a href='#54'>SUPPORT VECTOR MACHINE</a> 
#     - <a href='#55'>NAIVE BAYES</a>
#     - <a href='#56'>RANDOM FOREST</a>
#     - <a href='#57'>COMPARISON of ALGORITHMS</a>
# - <a href='#6'>CONCLUCIONS</a>
# - <a href='#7'>REFERANCES</a>

# ## <a href='#1'>Introduction</a>
# * ## Context
# <div>This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to
# this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.</div>
# * ## Content 
# 1. age
# 2. sex
# 3. chest pain type (4 values)
# 4. resting blood pressure
# 5. serum cholestoral in mg/dl
# 6. fasting blood sugar > 120 mg/dl
# 7. resting electrocardiographic results (values 0,1,2)
# 8. maximum heart rate achieved
# 9. exercise induced angina
# 10. oldpeak = ST depression induced by exercise relative to rest
# 11. the slope of the peak exercise ST segment
# 12. number of major vessels (0-3) colored by flourosopy
# 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
#     
# The names and social security numbers of the patients were recently removed from the database, replaced with dummy values. One file has been "processed", that one containing the Cleveland database. All four unprocessed files also exist in this directory. To see Test Costs (donated by Peter Turney), please see the folder "Costs"
# 
# * ##Acknowledgements
# 1. Creators:
# Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
# University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
# University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
# V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
# Donor: David W. Aha (aha '@' ics.uci.edu) (714) 856-8779
# 
# 2. Inspiration
# Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0).
# 
# See if you can find any other trends in heart data to predict certain cardiovascular events or find any clear indications of heart health.

# ## <a href='#2'>Load Packages and Data</a>

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.shape
data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.nunique()


# ## <a href='#3'>Check and Examination of the Data</a>
# ## <a href='#31'>Overview the Data</a>
# I want to find relation between features. I want to find relation between features. Then, I want to visualize these relations with using seaborn and matplotlib. 

# ## <a href='#4'>DATA VISUALIZATION</a>
# ## <a href='#41'>PIE PLOT and BAR PLOT</a>

# **CHEST PAIN TYPES - CP**
# 

# In[ ]:


labels_cp = data.cp.value_counts().index
values_cp = data.cp.value_counts().values
print(data.cp.describe())
explode = [0.05,0.01,0.01,0.01]
plt.figure(figsize = (7,7))
plt.pie(values_cp, explode = explode, labels = labels_cp, autopct = '%1.1f%%')
plt.title("Chest Pain Types")
plt.show()


# In[ ]:


pd.crosstab(data.cp,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for CP')
plt.xlabel('CP')
plt.ylabel('Frequency')
plt.show()


# **CA : Number of major vessels (0-3) colored by flourosopy**

# In[ ]:


labels_ca = data.ca.value_counts().index
values_ca = data.ca.value_counts().values
print(data.ca.describe())
explode = [0,0,0,0,0.1]
plt.figure(figsize = (7,7))
plt.pie(values_ca, explode = explode, labels = labels_ca, autopct = '%1.1f%%')
plt.title("number of major vessels (0-3) colored by flourosopy")
plt.show()


# In[ ]:


pd.crosstab(data.ca,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for CA')
plt.xlabel('CA')
plt.ylabel('Frequency')
plt.show()


# **THAL : 3 = normal; 6 = fixed defect; 7 = reversable defect**

# In[ ]:


labels_thal = data.thal.value_counts().index
values_thal = data.thal.value_counts().values
print(data.thal.describe())
explode = [0,0,0,0.1]
plt.figure(figsize = (7,7))
plt.pie(values_thal, explode = explode, labels = labels_thal, autopct = '%1.1f%%')
plt.title("thalach")
plt.show()


# In[ ]:


pd.crosstab(data.thal,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Thalach')
plt.xlabel('Thalach')
plt.ylabel('Frequency')
plt.show()


# **EXANG : exercise induced angina (1 = yes; 0 = no)**

# In[ ]:


labels_exang = data.exang.value_counts().index
labels_exang = ['No','Yes']
values_exang = data.exang.value_counts().values
print(data.exang.describe())
explode = [0,0.1]
plt.figure(figsize = (7,7))
plt.pie(values_exang, explode = explode, labels = labels_exang, autopct = '%1.1f%%')
plt.title("Exercise Induced Angina")
plt.show()


# In[ ]:


pd.crosstab(data.exang,data.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Exercise Induced Angina')
plt.xlabel('Exercise Induced Angina')
plt.ylabel('Frequency')
plt.show()


# **SLOPE : The slope of the peak exercise ST segment**

# In[ ]:


labels_slope = data.slope.value_counts().index
values_slope = data.slope.value_counts().values
print(data.slope.describe())
explode = [0,0,0.1]
plt.figure(figsize = (7,7))
plt.pie(values_slope, explode = explode, labels = labels_slope, autopct = '%1.1f%%')
plt.title("the slope of the peak exercise ST segment")
plt.show()


# In[ ]:


pd.crosstab(data.slope,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('Slope')
plt.ylabel('Frequency')
plt.show()


# **FBS : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)**

# In[ ]:


labels_fbs = ['fbs < 120 mg/dl','fbs > 120 mg/dl']
values_fbs = data.fbs.value_counts().values
print(data.fbs.describe())
explode = [0,0.1]
plt.figure(figsize = (7,7))
plt.pie(values_fbs, explode = explode, labels = labels_fbs, autopct = '%1.1f%%')
plt.title("Fasting Blood Sugar")
plt.show()


# In[ ]:


pd.crosstab(data.fbs,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Fasting Blood Sugar')
plt.xlabel('Fasting Blood Sugar')
plt.ylabel('Frequency')
plt.show()


# **RESTECG : Resting electrocardiographic results**

# In[ ]:


labels_restecg = data.restecg.value_counts().index
values_restecg = data.restecg.value_counts().values
print(data.restecg.describe())
explode = [0.02,0.02,0.05]
plt.figure(figsize = (7,7))
plt.pie(values_restecg, explode = explode, labels = labels_restecg, autopct = '%1.1f%%')
plt.title("Resting Electrocardiographic Results")
plt.show()


# In[ ]:


pd.crosstab(data.restecg,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Restecg')
plt.xlabel('Restecg')
plt.ylabel('Frequency')
plt.show()


# **HEART DISEASE FREQUENCY for AGES**

# In[ ]:


pd.crosstab(data.age,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# **SEX : (1 = male; 0 = female)**

# In[ ]:


pd.crosstab(data.sex,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for SEX')
plt.xlabel('Sex (0 = Female, 1 = Male)' )
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **TrestBps : Resting blood pressure (in mm Hg on admission to the hospital)**

# In[ ]:


pd.crosstab(data.trestbps,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for TrestBps')
plt.xlabel('TrestBps' )
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **CHOL : Serum cholestoral in mg/dl**

# In[ ]:


pd.crosstab(data.chol,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for CHOL')
plt.xlabel('Chol' )
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **OLDPEAK : ST depression induced by exercise relative to rest**

# In[ ]:


pd.crosstab(data.oldpeak,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for OldPeak')
plt.xlabel('OldPeak' )
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# **THALACH : Maximum heart rate achieved**

# In[ ]:


pd.crosstab(data.thalach,data.target,normalize=True).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Thalach')
plt.xlabel('Thalach')
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()


# <div>**RESULT**</div>
# Missing value is none but ca and oldpeak could be a problem for me because both of them have 0 values.While oldpeak has 32.7% 0 values, ca has 57.8% 0 values. ca means number of major vessels (0-3) colored by fluoroscopy. If your value of ca is 0, you can't have major vessels colored by fluoroscopy. This situation is not a problem. On the other hand, oldpeak means ST depression induced by exercise relative to rest. This situation is not a problem. As a conclusion, everything looks fine and I have not a problem with dataset. Besides, I have some ideas, firstly I believe that target could have a strong relation with cp,  slope, thalach. As a second, I think target might be having relation with exang, thal, restecg. I will investigate this situation and while I create a model, I will use results of this investigate. 

# ## <a href='#42'>HISTOGRAM</a>

# In[ ]:


data['AgeBin'] = 0 #creates a column of 0
data.loc[((data['age'] > 28) & (data['age'] < 30)) , 'AgeBin'] = 1
data.loc[((data['age'] >= 30) & (data['age'] < 40)) , 'AgeBin'] = 2
data.loc[((data['age'] >= 40) & (data['age'] < 50)) , 'AgeBin'] = 3
data.loc[((data['age'] >= 50) & (data['age'] < 60)) , 'AgeBin'] = 4
data.loc[((data['age'] >= 60) & (data['age'] < 70)) , 'AgeBin'] = 5
data.loc[((data['age'] >= 70) & (data['age'] < 78)) , 'AgeBin'] = 6
plt.figure()
plt.title('Age --- (29,77)')
data.AgeBin.hist()
plt.show()


# ## <a>Normalization</a>
# I define a function called norm to use scaling when needed. In simple terms, if the value is divided by the difference between the maximum value and the minimum value, it will be scaled from 0 to 1.

# In[ ]:


def norm(data):
    return (data)/(max(data)-min(data))


# ## <a href='#43'>LINE PLOT</a>

# In[ ]:


norm(data.chol).plot(kind = 'line', color = 'r',label = 'Target',linewidth=1,alpha = 0.9,grid = True,linestyle = ':')
norm(data.trestbps).plot(color = 'g',label = 'Sex',linewidth=1, alpha = 0.4,grid = True,linestyle = '-.')
norm(data.thal).plot(color = 'b',label = 'Thalach',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('x axis')            
plt.ylabel('y axis')
plt.title('Line Plot')           
plt.show()


# ## <a href='#44'>SCATTER PLOT</a>

# In[ ]:


data.plot(kind='scatter', x='chol', y='trestbps',alpha = 0.5,color = 'red')
plt.xlabel('Chol')              # label = name of label
plt.ylabel('Trestbps')
plt.title('Chol-TrestBps Scatter Plot')            # title = title of plot
plt.show()


# In[ ]:


data.plot(kind='scatter', x='chol', y='thal',alpha = 0.5,color = 'blue')
plt.xlabel('Chol')              # label = name of label
plt.ylabel('Thalach')
plt.title('Chol-Thalach Scatter Plot')            # title = title of plot
plt.show()


# ## <a href='#45'>PLOT and SUBPLOT</a>

# In[ ]:


data1 = data.loc[:,["chol","trestbps","thalach"]]
data1.plot()

plt.show()


# In[ ]:


data1.plot(subplots = True)
plt.show()


# ## <a href='#46'>BAR PLOT</a>

# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x=data.trestbps, y=data.thalach)
plt.xticks(rotation= 90)
plt.xlabel('TrestBps')
plt.ylabel('Thalach')
plt.title('TrestBps-Thalach')
plt.show()


# In[ ]:


plt.figure(figsize=(15,10))
ax= sns.barplot(x=data.age, y=data.target)
plt.xlabel('Age')
plt.ylabel('Target')
plt.title('Target-Age Seaborn Bar Plot')
plt.show()


# ## <a href='#47'>SUBPLOT</a>

# In[ ]:


f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=data.sex,y=data.target,color='blue',alpha = 1.0,label='Sex')
sns.barplot(x=data.exang,y=data.target,color='red',alpha = 1.0,label='Exang')
sns.barplot(x=data.fbs,y=data.target,color='green',alpha = 0.5,label='Fbs')

ax.legend(loc='best',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Percentage of Features', ylabel='Target',title = "Features-Target")
plt.show()


# ## <a href='#48'>BOX PLOT</a>

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x="age", y="chol", hue="target", data=data, palette="PRGn")
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x="trestbps", y="chol", hue="target", data=data, palette="PRGn")
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(x="sex", y="cp", hue="target", data=data, palette="PRGn")
plt.show()


# ## <a href='#49'>BAR PLOT and COUNT PLOT</a>

# In[ ]:


plt.figure(figsize=(10,7))
sns.barplot(x=data['age'].index,y=data['target'].values)
plt.xlabel('Ages')

sns.countplot(x=data.age)
plt.ylabel('Number of Target People')
plt.title('Age of target people',color = 'blue',fontsize=15)
plt.show()


# ## <a href='#410'>HEAT MAP - CORRELATION MATRICE</a>

# In[ ]:


features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca',
            'thal','target']
data = data[features]
f,ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()


# ## <a href='#411'>JOINT PLOT</a>

# In[ ]:


sns.jointplot(data.cp, data.target, kind="kde", size=5, color ="g")
plt.show()


# In[ ]:


sns.jointplot(data.thalach, data.target, kind="kde", size=5, color="r")
plt.show()


# In[ ]:


sns.jointplot(data.slope, data.target, kind="kde", size=5, color="b")
plt.show()


# In[ ]:


sns.jointplot(data.restecg, data.target, kind="kde", size=5, color="cyan")
plt.show()


# In[ ]:


sns.kdeplot(data.age, data.target, shade=True, cut=3)
plt.show()


# ## <a href='#412'>LM PLOT</a>

# In[ ]:


sns.lmplot(x="slope", y="target", data=data)
plt.show()


# In[ ]:


sns.lmplot(x="cp", y="target",data=data)
plt.show()


# In[ ]:


sns.lmplot(x="cp", y="slope",hue="target",data=data,markers=["o", "x"])
plt.show()


# In[ ]:


sns.lmplot(x="cp", y="thalach", hue="target", col="slope",
               data=data,markers=["o", "x"], height=6, aspect=.4, x_jitter=.1)
plt.show()


# In[ ]:


sns.lmplot(x="cp", y="thalach", row="slope", col="restecg",hue="target",
               markers=["o", "x"],data=data, height=3)
plt.show()


# ## <a href='#413'>POINT PLOT</a>

# In[ ]:


ax = sns.pointplot(x="cp", y="thalach", hue="target",data=data,dodge=True)
plt.show()


# In[ ]:


ax = sns.pointplot(x="cp", y="slope", hue="target",data=data,dodge=True,
                   markers=["o", "x"],linestyles=["-", "--"])
plt.show()


# In[ ]:


ax = sns.pointplot(x="slope", y="thalach", hue="target",data=data,dodge=True,
                   markers=["o", "x"],linestyles=["-", "--"])
plt.show()


# ## <a href='#414'>SWARM PLOT</a>

# In[ ]:


sns.swarmplot(x="cp", y="thalach",hue="target", data=data)
plt.show()


# In[ ]:


sns.swarmplot(x="slope", y="thalach",hue="target", data=data)
plt.show()


# In[ ]:


sns.swarmplot(x="exang", y="thalach",hue="target", data=data)
plt.show()


# In[ ]:


sns.swarmplot(x="restecg", y="thalach",hue="target", data=data)
plt.show()


# In[ ]:


sns.swarmplot(x="thal", y="thalach",hue="target", data=data)
plt.show()


# ## <a href='#415'>VIOLIN PLOT</a>

# In[ ]:


plt.figure(figsize=(10,7))
sns.violinplot(x="cp", y="slope", hue="target",data=data, palette="muted")
plt.show()


# In[ ]:


sns.violinplot(x="cp", y="slope", hue="target",data=data, palette="muted", split=True)
plt.show()


# In[ ]:


sns.violinplot(x="cp", y="thalach", hue="target",data=data, palette="muted", split=True)
plt.show()


# In[ ]:


sns.violinplot(x="slope", y="thalach", hue="target",data=data, palette="muted", split=True)
plt.show()


# In[ ]:


sns.violinplot(x="exang", y="restecg", hue="target",data=data, palette="muted", split=True)
plt.show()


# In[ ]:


sns.violinplot(x="exang", y="thal", hue="target",data=data, palette="muted", split=True)
plt.show()


# In[ ]:


sns.violinplot(x="restecg", y="thal", hue="target",data=data, palette="muted", split=True)
plt.show()


# ## <a href='#5'>PREDICTED MODELS</a>

# ## <a href='#51'>MULTIPLE LINEAR REGRESSION</a>

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,roc_curve

features = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang',
            'oldpeak','slope','ca','thal']
target = ['target']
x = data[features]
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
mlreg = LinearRegression()
mlreg.fit(x_train, y_train)
predictionmlreg = mlreg.predict(x_test)
test_set_rmse = (np.sqrt(mean_squared_error(y_test, predictionmlreg)))
print("Intercept: \n", mlreg.intercept_)
print("Root Mean Square Error \n", test_set_rmse)
rocaucscore=roc_auc_score(y_test.values, predictionmlreg)
print('Roc Score: ',rocaucscore)


# ## <a href='#52'>DECISION TREE</a>

# In[ ]:


dtclassifieroptimal = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
                       max_features=None, max_leaf_nodes=10,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=42, splitter='best')
dtclassifieroptimal.fit(x_train, y_train)
dtprediction = dtclassifieroptimal.predict(x_test)
print('Accuracy of Decision Tree:', accuracy_score(dtprediction,y_test))

from sklearn.metrics import confusion_matrix
cmdt=confusion_matrix(y_test, dtprediction, labels=None, sample_weight=None)
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cmdt, fmt = "d",
            xticklabels=['Have not Disease', 'Have Disease'],
            yticklabels=['Have not Disease', 'Have Disease'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Purple", cmap="Purples")
plt.title('Confusion Matrix in DT', fontsize=14)
plt.show()

rocaucscore=roc_auc_score(y_test.values, dtprediction)
print('Roc Score: ',rocaucscore)


# ## <a href='#53'>KNN</a>

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knnclassifier=KNeighborsClassifier(n_neighbors=6,algorithm='auto',
                                    leaf_size=30,metric='manhattan')
knnclassifier.fit(x_train, y_train)
trainaccuracy=knnclassifier.score(x_train, y_train)
testaccuracy=knnclassifier.score(x_test, y_test)
knnprediction=knnclassifier.predict(x_test)
print('train accuracy: {}\ntest accuracy: {}\n'.format(trainaccuracy,testaccuracy))

cmknn=confusion_matrix(y_test, knnprediction, labels=None, sample_weight=None)
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cmknn, fmt="d",
            xticklabels=['Have not Disease', 'Have Disease'],
            yticklabels=['Have not Disease', 'Have Disease'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Blue", cmap="Blues")
plt.title('Confusion Matrix in KNN', fontsize=14)
plt.show()


# ## <a href='#54'>SUPPORT VECTOR MACHINE</a>

# In[ ]:


from sklearn.svm import SVC
svm = SVC(kernel='rbf',random_state = 42, tol=0.001, shrinking=True, probability=True,
          C=1.0,  degree=3, gamma='auto', coef0=0.0, cache_size=200, class_weight=None, 
          verbose=False, max_iter=-1)
svm.fit(x_train,y_train)
svmprediction = svm.predict(x_test)
print("print accuracy of svm algo: ",svm.score(x_test,y_test))

cmsvm=confusion_matrix(y_test, knnprediction, labels=None, sample_weight=None)
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cmknn, fmt="d",
            xticklabels=['Have not Disease', 'Have Disease'],
            yticklabels=['Have not Disease', 'Have Disease'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Green", cmap="Greens")
plt.title('Confusion Matrix in SVM', fontsize=14)
plt.show()

rocaucscore=roc_auc_score(y_test.values, svmprediction)
print(rocaucscore)


# ## <a href='#55'>NAIVE BAYES</a>

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
nbprediction = nb.predict(x_test)
print("print accuracy of naive bayes algo: ",nb.score(x_test,y_test))
cmnb=confusion_matrix(y_test, nbprediction, labels=None, sample_weight=None)
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cmnb, fmt="d",
            xticklabels=['Have not Disease', 'Have Disease'],
            yticklabels=['Have not Disease', 'Have Disease'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Orange", cmap="Oranges")
plt.title('Confusion Matrix in NaiveBayes', fontsize=14)
plt.show()

rocaucscore=roc_auc_score(y_test.values, nbprediction)
print(rocaucscore)


# ## <a href = '#56'>RANDOM FOREST CLASSIFIER</a>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfclassifieroptimal=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=4, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=200,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
rfclassifieroptimal.fit(x_train, y_train)
rfprediction = rfclassifieroptimal.predict(x_test)
print('Accuracy of Random Forest Classifier:', accuracy_score(rfprediction,y_test))
cmrf=confusion_matrix(y_test, rfprediction, labels=None, sample_weight=None)
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cmrf, fmt="d",
            xticklabels=['Have not Disease', 'Have Disease'],
            yticklabels=['Have not Disease', 'Have Disease'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Red", cmap="Reds")
plt.title('Confusion Matrix in Random Forest', fontsize=14)
plt.show()

rocaucscore=roc_auc_score(y_test.values, rfprediction)
print(rocaucscore)


# ## <a href = '#57'>COMPARISON</a>

# In[ ]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
y_pred_proba_DT = dtclassifieroptimal.predict_proba(x_test)[::,1]
fpr1, tpr1, _ = roc_curve(y_test,  y_pred_proba_DT)
auc1 = roc_auc_score(y_test, y_pred_proba_DT)

y_pred_proba_KNN = knnclassifier.predict_proba(x_test)[::,1]
fpr2, tpr2, _ = roc_curve(y_test,  y_pred_proba_KNN)
auc2 = roc_auc_score(y_test, y_pred_proba_KNN)

y_pred_proba_SVM = svm.predict_proba(x_test)[::,1]
fpr3, tpr3, _ = roc_curve(y_test,  y_pred_proba_SVM)
auc3 = roc_auc_score(y_test, y_pred_proba_SVM)

y_pred_proba_NB = nb.predict_proba(x_test)[::,1]
fpr4, tpr4, _ = roc_curve(y_test,  y_pred_proba_NB)
auc4 = roc_auc_score(y_test, y_pred_proba_NB)

y_pred_proba_RF = svm.predict_proba(x_test)[::,1]
fpr5, tpr5, _ = roc_curve(y_test,  y_pred_proba_RF)
auc5 = roc_auc_score(y_test, y_pred_proba_RF)

plt.figure(figsize=(10,7))
plt.title('ROC', size=15)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr1,tpr1,label="DT, auc="+str(round(auc1,2)))
plt.plot(fpr2,tpr2,label="KNearest Neighbor, auc="+str(round(auc2,2)))
plt.plot(fpr3,tpr3,label="SVM, auc="+str(round(auc3,2)))
plt.plot(fpr4,tpr4,label="NB, auc="+str(round(auc4,2)))
plt.plot(fpr5,tpr5,label="RF, auc="+str(round(auc5,2)))
plt.legend(loc='best', title='Models', facecolor='white')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.box(False)
plt.show()


# ## <a href='#6'>CONCLUSION</a>
# In this study, I did data visualization and algorithm implementation. It is an application about how some tools can be used.
