#!/usr/bin/env python
# coding: utf-8

# ## Breast Cancer Prediction

# **Dataset info**

# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
# n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].
# 
# This database is also available through the UW CS ftp server:
# ftp ftp.cs.wisc.edu
# cd math-prog/cpo-dataset/machine-learn/WDBC/
# 
# Also can be found on UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
# 
# 
# **Attribute Information:**
# 
# 1) ID number
# 
# 2) Diagnosis (M = malignant, B = benign)
# 
# 
# **Ten real-valued features are computed for each cell nucleus:**
# 
# a) radius (mean of distances from center to points on the perimeter)
# 
# b) texture (standard deviation of gray-scale values)
# 
# c) perimeter
# 
# d) area
# 
# e) smoothness (local variation in radius lengths)
# 
# f) compactness (perimeter^2 / area - 1.0)
# 
# g) concavity (severity of concave portions of the contour)
# 
# h) concave points (number of concave portions of the contour)
# 
# i) symmetry
# 
# j) fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three
# largest values) of these features were computed for each image,
# resulting in 30 features. For instance, field 3 is Mean Radius, field
# 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# **Class distribution: 357 benign, 212 malignant**

# ### Import Libraries

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

#Showing full path of datasets
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Disable warnings 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


breast=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")


# In[ ]:


breast.head()


# In[ ]:


breast.info()


# In[ ]:


breast.drop(columns=["id","Unnamed: 32"],axis=1,inplace=True)


# In[ ]:


breast.columns


# In[ ]:


breast.describe()


# Checking missing values

# In[ ]:


breast.isna().sum()


# As we can see there are no missing values

# In[ ]:


breast.skew(axis=0)


# As most of the features are skewed , we can have a further look at the visualization for more information

# **Let's have a look at the diagnosis feature**

# In[ ]:


breast['diagnosis'].value_counts()


# In[ ]:


sns.countplot(x="diagnosis",data=breast)


# ### Univariate Analysis

# **We'll separate the 30 features into 10 each i.e mean radius, se radius and worst radius and analyze each one separately**

# In[ ]:


diagnosis = breast['diagnosis']
breast_mean = breast.iloc[:,0:11]
breast_se = pd.concat([breast.iloc[:,11:21],diagnosis],axis=1)
breast_worst = pd.concat([breast.iloc[:,21:31],diagnosis],axis=1)

display(breast_mean)
display(breast_se)
display(breast_worst)


# **Breast mean**

# In[ ]:


import plotly.graph_objects as go
fig = make_subplots(rows=5,cols=2,subplot_titles=("Area_mean",'Texture_mean',
                                                 "radius_mean","compactness_mean",
                                                 "perimeter_mean","concavity_mean",
                                                 "concave points_mean","symmetry_mean",
                                                 "fractal_dimension_mean","smoothness_mean"))
fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['area_mean'],name='area_mean'),row=1,col=1)
fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['texture_mean'],name='texture_mean'),row=1,col=2)
fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['radius_mean'],name='radius_mean'),row=2,col=1)
fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['compactness_mean'],name='compactness_mean'),row=2,col=2)
fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['perimeter_mean'],name='perimeter_mean'),row=3,col=1)
fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['concavity_mean'],name='concavity_mean'),row=3,col=2)
fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['concave points_mean'],name='concave points_mean'),row=4,col=1)
fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['symmetry_mean'],name='symmetry_mean'),row=4,col=2)
fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['fractal_dimension_mean'],name='fractal_dimension_mean'),row=5,col=1)
fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['smoothness_mean'],name='smoothness_mean'),row=5,col=2)

# Update title and height
fig.update_layout(title_text="Breast mean Visualizations", height=1500,width=1000)


# In[ ]:


for col in breast_mean.columns:
    if col != 'diagnosis':
        print(col+' vs diagnosis')
        fig = px.box(breast_mean,x='diagnosis',y=col,color='diagnosis',width=500,height=500)
        fig.show()


# **Breast squared error**

# In[ ]:


for col in breast_se.columns:
    if col != 'diagnosis':
        print(col+' vs diagnosis')
        fig = px.box(breast_se,x='diagnosis',y=col,color='diagnosis',width=500,height=500)
        fig.show()
        


# **Breast worst**

# In[ ]:


for col in breast_worst.columns:
    if col != 'diagnosis':
        print(col+' vs diagnosis')
        fig = px.box(breast_worst,x='diagnosis',y=col,color='diagnosis',width=500,height=500)
        fig.show()


# In[ ]:


breast.isna().sum()

No null values
# ### Correlation
# 
# Heatmap tells us about the correlation of two variables.The higher the value , more the features are correlated.

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(breast_mean.corr(),annot=True,vmin=0,vmax=1,cmap="YlGnBu")


# perimeter_mean --- radius_mean
# 
# area_mean --- radius_mean
# 
# area_mean --- perimeter_mean
# 
# concavity_mean --- concave points_mean
# 

# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(breast_se.corr(),annot=True,vmin=0,vmax=1,cmap="YlGnBu")


# perimeter_se --- radius_se
# 
# area_se --- radius_se
# 
# area_se --- perimeter_se

# In[ ]:


plt.figure(figsize=(15,10))

sns.heatmap(breast_worst.corr(),annot=True,vmin=0,vmax=1,cmap="YlGnBu")


# perimeter_worst --- radius_worst
# 
# area_worst --- radius_worst
# 
# area_worst --- perimeter_worst
# 
# 

# #### We can have a better look at these features with the help of jointplot

# In[ ]:


sns.jointplot(x='radius_mean',y='perimeter_mean',data=breast_mean,kind='reg')


# In[ ]:


sns.jointplot(x='area_mean',y='radius_mean',data=breast_mean,kind='reg')


# In[ ]:


sns.jointplot(x='area_mean',y='perimeter_mean',data=breast_mean,kind='reg')


# In[ ]:


sns.jointplot(x='concavity_mean',y='concave points_mean',data=breast_mean,kind='reg')


# In[ ]:


sns.jointplot(x='fractal_dimension_mean',y='area_mean',data=breast_mean,kind='reg')


# In[ ]:


y=breast.diagnosis
x=breast.drop(columns="diagnosis",axis=1)
x.head()


# In[ ]:


##Standardize data
breast_dia=y
breast_x=x
breast_2=(breast_x-breast_x.mean())/breast_x.std()
breast_x=pd.concat([y,breast_2.iloc[:,0:10]],axis=1)
breast_x=pd.melt(breast_x,id_vars='diagnosis',
                      var_name='features',
                      value_name='value')
plt.figure(figsize=(12,10))
sns.violinplot(x='features',y='value',hue='diagnosis',split=True,data=breast_x)
plt.xticks(rotation=90)


# In[ ]:


##next 10 features
breast_x=pd.concat([y,breast_2.iloc[:,10:20]],axis=1)
breast_x=pd.melt(breast_x,id_vars='diagnosis',var_name='features',value_name='value')

plt.figure(figsize=(12,10))
sns.violinplot(x='features',y='value',hue='diagnosis',data=breast_x,split=True)
plt.xticks(rotation=90)


# In[ ]:


##next 10 features
breast_x=pd.concat([y,breast_2.iloc[:,20:31]],axis=1)
breast_x=pd.melt(breast_x,id_vars='diagnosis',var_name='features',value_name='value')
plt.figure(figsize=(12,10))
sns.violinplot(x='features',y='value',hue='diagnosis',data=breast_x,split=True)
plt.xticks(rotation=90)


# In[ ]:


##pairplot for 5 features at a time


# In[ ]:


temp=pd.concat([y,x.iloc[:,0:5]],axis=1)
temp.shape
sns.pairplot(data=temp,hue="diagnosis")
plt.figure(figsize=(12,10))


# In[ ]:


temp=pd.concat([y,x.iloc[:,5:10]],axis=1)
temp.shape
sns.pairplot(data=temp,hue="diagnosis")
plt.figure(figsize=(12,10))


# In[ ]:


x


# In[ ]:


y


# In[ ]:


breast_swarm_dia=y
breast_x=x
breast_s_2=(breast_x-breast_x.mean())/(breast_x.std())
breast_x=pd.concat([y,breast_s_2.iloc[:,0:10]],axis=1)
breast_x=pd.melt(breast_x,id_vars="diagnosis",
    var_name="features",
    value_name="value")
breast_x


# In[ ]:


plt.figure(figsize=(12,10))
sns.swarmplot(x="features",y="value",hue="diagnosis",data=breast_x)
plt.xticks(rotation=90)


# In[ ]:


breast=pd.concat([y,breast_s_2.iloc[:,10:21]],axis=1)
breast=pd.melt(breast,id_vars="diagnosis",
    var_name="features",
    value_name="value")
plt.figure(figsize=(12,10))
sns.swarmplot(x="features",y="value",hue="diagnosis",data=breast_x)
plt.xticks(rotation=90)


# In[ ]:


breast=pd.concat([y,breast_s_2.iloc[:,21:30]],axis=1)
breast=pd.melt(breast,id_vars="diagnosis",
    var_name="features",
    value_name="value")
plt.figure(figsize=(12,10))
sns.swarmplot(x="features",y="value",hue="diagnosis",data=breast_x)
plt.xticks(rotation=90)


# In[ ]:


##pairplot
plt.figure(figsize=(18,18))
sns.heatmap(x.corr(),annot=True,linewidths=.5,fmt='.1f')


# In[ ]:


##Feature selection

Features=['smoothness_mean', 'compactness_mean', 'concavity_mean','symmetry_mean', 'fractal_dimension_mean', 'texture_se','smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se','texture_worst',
       'perimeter_worst','smoothness_worst',
       'compactness_worst','symmetry_worst', 'fractal_dimension_worst']

selected_features=x.drop(columns=Features,axis=1)
selected_features=selected_features.drop(columns="texture_mean",axis=1)
selected_features


#                     ##We'll use selected_features for our model and see score

# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(selected_features,y,test_size=0.3,random_state=10)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# # Naive Bayes

# In[ ]:


selected_features.describe()


# In[ ]:


##As we see in describe values are not proper we need to standardize


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train2=scaler.fit_transform(X_train)
X_test2=scaler.transform(X_test)


# In[ ]:


X_train2.shape


# In[ ]:


#but

X_train2


# In[ ]:


##As features are not independent ,its not proper to use naive bayes
##So we'kll just see to our practice


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train2,y_train)
y_pred2=gnb.predict(X_test2)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[ ]:


print("Accuracy using accuracy_score is {}".format(accuracy_score(y_test,y_pred2)))


# In[ ]:


cm=confusion_matrix(y_test,y_pred2)


# In[ ]:


print("COnfusion matrix score is \n{}".format(cm))


# In[ ]:


sns.heatmap(cm,annot=True,
    fmt='d')


# In[ ]:


#This wont work as y is text classification
#from sklearn.metrics import f1_score
#print("Accuracy f1score is {}".format(f1_score(y_test,y_pred)))


# In[ ]:


x


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train1,X_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.3,random_state=10)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler1=StandardScaler()
X_train1=scaler1.fit_transform(X_train1)
X_test1=scaler1.transform(X_test1)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb1=GaussianNB()
gnb1.fit(X_train1,y_train1)
y_pred1=gnb1.predict(X_test1)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy using accuracy_score is {}".format(accuracy_score(y_test,y_pred1)))


# ##As we see there are no changes in the score
# ##therfore the features that contribute to the target value should be used
# ##We'll have to use better features

# # We'll choose best features ......

# In[ ]:


from sklearn.feature_selection import SelectKBest,chi2
selectbest10=SelectKBest(chi2,k=10)
X_best_10=selectbest10.fit(X_train,y_train)


# In[ ]:


X_best_10


# In[ ]:


X_train


# In[ ]:


print("Scores :",X_best_10.scores_)
print("Features:",X_train.columns)


# ##Here we see 10 features we already have we cant find any difference
# ##Now we'll choose only 5 features

# In[ ]:


from sklearn.feature_selection import SelectKBest,chi2
selectbest5=SelectKBest(chi2,5)
X_best_5=selectbest5.fit(x,y)
print("Scores:",X_best_5.scores_)
print("Features:",x.columns)


# In[ ]:


#x5=["area_mean","area_se","texture_mean","concavity_worst","compactness_mean"]
featuresfor5=['radius_mean','perimeter_mean',
       'smoothness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']


# In[ ]:


selected_for5=x.drop(columns=featuresfor5,axis=1)
selected_for5


# In[ ]:


x.columns


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train5,X_test5,y_train5,y_test5=train_test_split(selected_for5,y,test_size=0.3,random_state=10)


 


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler5=StandardScaler()
X_train5=scaler5.fit_transform(X_train5)
X_test5=scaler5.transform(X_test5)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb5=GaussianNB()
gnb5.fit(X_train5,y_train5)
y_pred5=gnb5.predict(X_test5)


# In[ ]:


from sklearn.metrics import accuracy_score
print("Accuracy using 5 selected features is {}".format(accuracy_score(y_test5,y_pred5)))


# ## Random Forest Classifier

# In[ ]:


X_train


# In[ ]:


y_train


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfclf1=RandomForestClassifier()
rfclf1.fit(X_train,y_train)
y_predrfc=rfclf1.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
print("Accuracy using rfclassifier is {}".format(accuracy_score(y_test,y_predrfc)))


# ##USING RANDOM FOREST WE GET GOOD SCORE
