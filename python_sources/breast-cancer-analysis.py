#!/usr/bin/env python
# coding: utf-8

# IIn this **breast cancer analysis** (using Wisconsin Diagnostic data), I have done **visual Exploratory data analysis** to understand the features that are good for classification for Malignant and Benign types of cancer, using **Random Forest Classifier** to predict the outcomes and determine the feature importances. 

# **Importing the necessary packages and reading the CSV File**

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
cancer = pd.read_csv("../input/data.csv")
cancer.head()


# **features overview**

# 1. ID - unique identification
# 2. diagnosis - two values M-'Malignant' B-'Benign
# 3. radius - mean of distances from center to points on the perimeter
# 4. texture - standard deviation of gray-scale values
# 5. perimeter
# 6. area 
# 7. smoothness - local variation in radius lengths
# 8. compactness - perimeter^2 / area - 1.0
# 9. concavity - severity of concave portions of the contour
# 10. concave points - number of concave portions of the contour
# 11. symmetry
# 12. fractal dimension - (coastline approximation) - 1

# **Basic Quantitative EDA** to get the count of the features, mean and other basic values to understand the data in a better way:

# In[ ]:


cancer.describe()


# We can see that **Unnamed:32** and **id** are of no use, and may interfere with our model in an improper manner, so we can drop these columns from our dataframe.

# In[ ]:


cancer.drop(['id','Unnamed: 32'], axis=1, inplace=True)
cancer.columns


# **checking for missing values** 

# In[ ]:


cancer.isnull().sum().sort_values(ascending=False)


# From above. it can be concluded that there are **no missing values** in the dataframe, and we can begin working with the data now:

# In[ ]:


print('counts of Malignant and Benign \n',cancer['diagnosis'].value_counts())
sns.countplot(cancer['diagnosis'],palette="Blues")
plt.title('Distribution of Malignant & Benign')


# Now, we split all the features according to **mean, standard error and worst** into different dataframes

# In[ ]:


features_mean = cancer[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]
features_se = cancer[['diagnosis','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se']]
features_worst = cancer[['diagnosis','radius_worst', 'texture_worst','perimeter_worst', 'area_worst', 
'smoothness_worst','compactness_worst', 'concavity_worst', 'concave points_worst','symmetry_worst',
                   'fractal_dimension_worst']]


# **Visualisation**
# 
# Visualising features for mean, standard error and worst types of features to understand the correlation between them and the distribution for classification of Malignant and Bengin types of cancer.

# **Visualising fetaures for mean values**

# In[ ]:


mean_correlation = features_mean.corr()
plt.figure(figsize=(8,8))
sns.heatmap(mean_correlation,vmax=1,square=True,annot=True,cmap='Greens')


# From the above **Heat map of mean values**  it can be concluded that **area_mean,perimeter_mean and radius_mean have correlation with each other** therefore one best feature can be used from them to predict and similar with **compactness_mean,concave points_mean and concavity_mean.**

# In[ ]:


plt.subplot(221)
sns.violinplot(x='diagnosis',y='texture_mean',data=features_mean,palette="Greens",inner="quartile")
plt.subplot(222)
sns.violinplot(x='diagnosis',y='concavity_mean',data=features_mean,palette="Greens",inner="quartile")
plt.subplot(223)
sns.violinplot(x='diagnosis',y='radius_mean',data=features_mean,palette="Greens",inner="quartile")
plt.subplot(224)
sns.violinplot(x='diagnosis',y='fractal_dimension_mean',data=features_mean,palette="Greens",inner="quartile")
plt.show()


# From above **violin plots**, we find out that **texture_mean and concavity_mean** are good features for classification as they segregate better Malignant and Benign types but **fractal_dimension_mean**, one of the features, has almost a similar mean and a similar distribution, and hence is not a good parameter for classification.

# **Visualising features for standard error values**

# In[ ]:


se_correlation = features_se.corr()
plt.figure(figsize=(8,8))
sns.heatmap(se_correlation,vmax=1,square=True,annot=True,cmap='Oranges')


# From the above **Heatmap** of the **standard error values**  it can be concluded that **area_se,perimeter_se and radius_se have correlation with each other**, which is also the case with **compactness_se,concave points_se and concavity_se.** and hence, one best feature can be used from them for further classification during prediction.  

# In[ ]:


plt.subplot(221)
sns.violinplot(x='diagnosis',y='texture_se',data=features_se,palette="Oranges",inner="quartile")
plt.subplot(222)
sns.violinplot(x='diagnosis',y='concavity_se',data=features_se,palette="Oranges",inner="quartile")
plt.subplot(223)
sns.violinplot(x='diagnosis',y='radius_se',data=features_se,palette="Oranges",inner="quartile")
plt.subplot(224)
sns.violinplot(x='diagnosis',y='fractal_dimension_se',data=features_se,palette="Oranges",inner="quartile")
plt.show()


# From above **violin plots , radius_se and concavity_se** are good features for classification as they segregate better Malignant and Benign types but **fractal_dimension_mean** has same mean and similar pattern for classification, and hence is not a good parameter to do so.

# **Visualising features for worst values**

# In[ ]:


worst_correlation = features_worst.corr()
plt.figure(figsize=(8,8))
sns.heatmap(worst_correlation,vmax=1,square=True,annot=True,cmap='Reds')


# Similarly in the **Heatmap of worst values area, radius and perimeter**, we find out that they are intercorrelated with each other. A similar case arises with **compactness_worst, concavity_worst,concave points_worst**. This means that we can use one from each of the sets.

# In[ ]:


plt.subplot(221)
sns.violinplot(x='diagnosis',y='texture_worst',data=features_worst,palette="Reds",inner="quartile")
plt.subplot(222)
sns.violinplot(x='diagnosis',y='smoothness_worst',data=features_worst,palette="Reds",inner="quartile")
plt.subplot(223)
sns.violinplot(x='diagnosis',y='concavity_worst',data=features_worst,palette="Reds",inner="quartile")
plt.subplot(224)
sns.violinplot(x='diagnosis',y='concave points_worst',data=features_worst,palette="Reds",inner="quartile")
plt.show()


# From the above plots, we find out that **concavity_worst and concave points_worst** have similar plots so we can use one of them. Along with that, we find out that **texture_se and radius_se** can be used for classification.

# Visualising and selecting best from **radius , area and perimeter** for classification using pairplots and swarmplots to see which of them seperates the types of cancer better than the others.

# In[ ]:


pairplot = cancer[['diagnosis','radius_worst','area_worst','perimeter_worst']]
sns.pairplot(pairplot,hue='diagnosis',palette="Blues_d")
plt.show()


# In[ ]:


plt.subplot(221)
sns.swarmplot(x='diagnosis',y='area_worst',data=pairplot,palette="Blues_d")
plt.subplot(222)
sns.swarmplot(x='diagnosis',y='radius_worst',data=pairplot,palette="Blues_d")
plt.subplot(223)
sns.swarmplot(x='diagnosis',y='perimeter_worst',data=pairplot,palette="Blues_d")
plt.show()


# We can use **radius** among them. This is because it it better classifies Malignant & Benign types, and both area aong with the perimeter depend on radius.

# In[ ]:


plt.subplot(221)
sns.swarmplot(x='diagnosis',y='compactness_worst',data=features_worst,palette="Blues_d")
plt.subplot(222)
sns.swarmplot(x='diagnosis',y='concavity_worst',data=features_worst,palette="Blues_d")
plt.subplot(223)
sns.swarmplot(x='diagnosis',y='concave points_worst',data=features_worst,palette="Blues_d")
plt.show()


# And from these plots we can clearly see that **concavity_worst** can be used for classification among these three.

# **Feature selection**

# selecting features based on above visualisation and correlations
# 1. radius_mean 
# 2. texture_mean 
# 3. smoothness_mean
# 4. concavity_mean 
# 5. symmetry_mean
# 6. radius_se 
# 7. texture_se
# 8. smoothness_se
# 9. concavity_se 
# 10. symmetry_se
# 11. radius_worst 
# 12. texture_worst
# 13. smoothness_worst
# 14. concavity_worst
# 15. symmetry_worst

# In[ ]:


features_corr = cancer[['diagnosis','radius_mean', 'texture_mean', 'smoothness_mean', 'concavity_mean','symmetry_mean',
       'radius_se', 'texture_se', 'smoothness_se','concavity_se', 'symmetry_se','radius_worst', 'texture_worst',
       'smoothness_worst','concavity_worst','symmetry_worst']]
features_correlation = features_corr.corr()
plt.figure(figsize=(10,10))
sns.heatmap(features_correlation,vmax=1,square=True,annot=True,cmap='Blues')
plt.show()


# For cross validation of the results we may use the train_test_split. And we are fitting the above selected features using **Random Forest Classifier**.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

features_corr['diagnosis'] = [0 if x == 'B' else 1 for x in features_corr['diagnosis']]
X = features_corr.drop(['diagnosis'],axis = 1 )
y = features_corr.diagnosis

X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.3,random_state=42)
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train,y_train)
print('Accuracy score',rfc.score(X_test,y_test))


# **Performance metrics** to check the model accuracy

# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

y_pred = rfc.predict(X_test)
confusion_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(confusion_matrix,annot=True,fmt='')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
print('Classification Report')
print(classification_report(y_test,y_pred))

from sklearn.model_selection import cross_val_score
import numpy as np
rfc = RandomForestClassifier()
cv_results = cross_val_score(rfc,X,y,cv=5)
print(cv_results)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_results)))


# **ROC Curve** to check the accuracy and detect the performance of the model:

# In[ ]:


from sklearn.metrics import roc_curve
rfc.fit(X_train,y_train)
y_pred_prob  =  rfc.predict_proba(X_test)[:,1]
fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='random forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# 

# 

# Now taking all the features of the dataset and fitting them using the **RandomForestClassifier**.

# In[ ]:


df = pd.read_csv("../input/data.csv")
df['diagnosis'] = [0 if x == 'B' else 1 for x in df['diagnosis']]
y = df.diagnosis        
list = ['Unnamed: 32','id','diagnosis']
X = df.drop(list,axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
forest = RandomForestClassifier()
forest.fit(X_train,y_train)
print('train score',forest.score(X_train,y_train))
print('test score',forest.score(X_test,y_test))


# **Correlation of diagnosis** with other features sorted reverse (descending order), which tells us that the radius parameters relate to the classification better than their area counterparts.

# In[ ]:


corr=df.corr()['diagnosis']
corr[np.argsort(corr,axis=0)[::-1]]


# Detecting **Feature Importances**, from which we conclude that the worst parameters classify better than the others:

# In[ ]:


features = X.columns
for name, importance in zip(features, forest.feature_importances_):
    print(name, "=", importance)

importances = forest.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features) ## removed [indices]
plt.xlabel('Relative Importance')
plt.show()


# **Performance metrics**: classification_report, confusion_matrics and cross_val_score

# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
y_pred = forest.predict(X_test)
confusion_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(confusion_matrix,annot=True,fmt='')
plt.xlabel('Predicted')
plt.ylabel('True')
print('Classification Report')
print(classification_report(y_test,y_pred))

from sklearn.model_selection import cross_val_score
import numpy as np
cv_results = cross_val_score(forest,X,y,cv=5)
print(cv_results)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_results)))


# **ROC Curve** for the model accuracy detection:

# In[ ]:


from sklearn.metrics import roc_curve
rfc.fit(X_train,y_train)
y_pred_prob  =  forest.predict_proba(X_test)[:,1]
fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='random forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# **Conclusion**
#  From the above results we come to a conclusion that the methods employed here are prediction and implementation worthy, and give a better classification of the types of cancers: Benign and Malignant types. Hence,** taking the mean, standard deviation and worst values for radius, texture, smoothness, concavity and symmetry parameters was a good idea.**
