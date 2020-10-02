#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotting libs
import seaborn as sns
import matplotlib.pyplot as plt

#sklearn lib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Load dataset

# Here is some information on the features:
# 1.  **age** in years
# 2.  **sex** (1 = male; 0 = female)
# 3.  **cp** chest pain type (4 values)
# 4.  **trestbps** resting blood pressure (in mm Hg on admission to the hospital)
# 5.  **chol** serum cholestoral in mg/dl
# 6.  **fbs** (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 7.  **restecg** resting electrocardiographic results (3 values; 0,1,2)
# 8.  **thalach** maximum heart rate achieved
# 9.  **exang** exercise induced angina (1 = yes; 0 = no)
# 10. **oldpeak** ST depression induced by exercise relative to rest
# 11. **slope** the slope of the peak exercise ST segment
# 12. **ca** number of major vessels (0-3) colored by flourosopy
# 13. **thal** 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. **target** 1 or 0

# In[ ]:


hd = pd.read_csv("../input/heart-disease-uci/heart.csv")
print(hd.head())
print(hd.shape)


# Check if there is missing data.

# In[ ]:


print("Missing data: \n")
print(str(hd.isnull().sum()))


# Since there are no missing values there is no need to fill up anything.

# In[ ]:


labels=["Male", "Female"] #x-axis label
male = [hd[(hd["target"]==0)&(hd["sex"]==1)]["target"].count(), hd[(hd["target"]==0)&(hd["sex"]==0)]["target"].count()] #bars for males
female = [hd[(hd["target"]==1)&(hd["sex"]==1)]["target"].count(), hd[(hd["target"]==1)&(hd["sex"]==0)]["target"].count()] #bars for females
print(male)
print(female)
x = np.arange(len(labels)) #label locations
width=0.35 #bar widths

sumM = male[0]+female[0]
sumF = male[1]+female[1]

relm = [male[0]/sumM, male[1]/sumF]
relf = [female[0]/sumM, female[1]/sumF]

fig = plt.figure(figsize=(14,8))
ax = fig.subplots()
rects1 = ax.bar(x - width/2, male, width, label='Male')
rects2 = ax.bar(x + width/2, female, width, label='Female')
print(male[0]+female[0])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(["no disease", "disease"])

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()


# This plot shows that the chance that a female has a heart disease is higher than for a male. Plot the samething for the age feature ubt first divide into groups. 

# In[ ]:


hd["age_groups"] = hd["age"].apply(lambda x: 0 if x<6 else (1 if x < 18 else (2 if x < 30 else (3 if x < 50 else (4 if x < 65 else 5)))))
hd


# In[ ]:


fig = plt.figure(figsize=(14,8))
sns.distplot(hd["age"])


# So this dataset contains information mostly for people aged 40+.

# In[ ]:


fig = plt.figure(figsize=(14,8))
sns.countplot(hd["age_groups"], hue=hd["target"])
print(hd["age_groups"].value_counts())


# It is very interesting that there are so many people with a heart disease in group 3 (30 <= x < 50). Since there are just 87 people in this group there is a chance that what we see is completely random. This can be checked by calculating the p-value.

# In[ ]:


print(hd["fbs"].value_counts())


# In[ ]:


fig = plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.countplot(hd[hd["sex"]==1]["fbs"], hue=hd["target"])
plt.title("Male")
plt.gca().set_ylim(0,100)

plt.subplot(1,2,2)
sns.countplot(hd[hd["sex"]==0]["fbs"], hue=hd["target"])
plt.title("Female")
plt.gca().set_ylim(0,100)


# Here we can't draw any conclusion since the fbs feature is heavily imbalanced as this shows:

# In[ ]:


print("fbs==0: {} % \nfbs==1: {} %".format(hd[hd["fbs"]==0]["fbs"].count()/len(hd["fbs"]), hd[hd["fbs"]==1]["fbs"].count()/len(hd["fbs"])))


# It would be interesting to see if there is a connection between exang (excersice induced angina) and the target.

# In[ ]:


fig = plt.figure(figsize=(14,6))
sns.countplot(hd["exang"], hue=hd["target"])


# Interesting. Quite a lot people have a heart disease without having an exercise induced angina.

# In[ ]:


fig = plt.figure(figsize=(14,6))
plt.subplot(2,1,1)
sns.boxplot(hd[hd["target"]==0]["oldpeak"])
plt.xlim(0, 8)
plt.title("no disease")

plt.subplot(2,1,2)
sns.boxplot(hd[hd["target"]==1]["oldpeak"])
plt.xlim(0, 8)
plt.title("has disease")
plt.subplots_adjust(hspace=0.4)


# So people with a heart dieases usually have a lower peak in comparison to people with no disease. There are alos just a few outliers which can be removed.

# In[ ]:


fig = plt.figure(figsize=(14,6))
sns.countplot(hd["cp"], hue=hd["target"])


# From this plot it would be save to assume that if someone experiences chest pain than it is likely that this person also has a heart disease. But since there are a lot of diseases that cause chest pain we can't say for sure that it is a heart disease.

# In[ ]:


fig = plt.figure(figsize=(14,6))
plt.subplot(2,1,1)
sns.boxplot(hd[hd["target"]==0]["chol"])
plt.xlim(0, 600)
plt.title("no disease")

plt.subplot(2,1,2)
sns.boxplot(hd[hd["target"]==1]["chol"])
plt.xlim(0, 600)
plt.title("has disease")
plt.subplots_adjust(hspace=0.4)


# Here we see nothing significant by which we can say wether a person has a disease or not. Maybe we could say anything about it in combination with another feature.
# 
# Now plot the thalach and trestbps features.

# In[ ]:


fig = plt.figure(figsize=(14,6))
plt.subplot(2,1,1)
sns.boxplot(hd[hd["target"]==0]["thalach"])
plt.xlim(60, 210)
plt.title("no disease")

plt.subplot(2,1,2)
sns.boxplot(hd[hd["target"]==1]["thalach"])
plt.xlim(60, 210)
plt.title("has disease")
plt.subplots_adjust(hspace=0.4)


# In[ ]:


fig = plt.figure(figsize=(14,6))
plt.subplot(2,1,1)
sns.boxplot(hd[hd["target"]==0]["trestbps"])
plt.xlim(90, 210)
plt.title("no disease")

plt.subplot(2,1,2)
sns.boxplot(hd[hd["target"]==1]["trestbps"])
plt.xlim(90, 210)
plt.title("has disease")
plt.subplots_adjust(hspace=0.4)


# The trestbps basically tells us nothing but the thalach feature can do this. From this plot we see that a person with a heart disease will also achieve a higher maximum heart rate.
# 
# From this point we could do furter analysis e.g. take a look at the distrbution of the thalach feature for each age group separated by our target or create scatter plots for different combinations of features and see who has a disease and who has no disease. Instead we will now create a few models and compare their predictions. We can select our features based on the correlation with our target feature.

# In[ ]:


fig = plt.figure(figsize=(16,8))
We wouldsns.heatmap(hd.corr().sort_values("target", ascending=False)[["target"]], annot=True) #select only the target feature and sort the correlation


# We would use the features with the highest correlation with our target but instead we will use all features for our model and afterwards take a look at the feature importance. For a selection with a correlation matrix the correlation between the features should be considered too.

# Convert categorical features into dummy variables.

# In[ ]:


pd.get_dummies(hd["cp"], prefix="cp")
pd.get_dummies(hd["thal"], prefix="thal")
pd.get_dummies(hd["slope"], prefix="slope")
hd = pd.concat([hd, pd.get_dummies(hd["cp"], prefix="cp"), pd.get_dummies(hd["thal"], prefix="thal"), pd.get_dummies(hd["slope"], prefix="slope")], axis=1)
hd.drop(["cp", "thal", "slope"], axis=1, inplace=True)
hd


# # Models

# # LogisticRegression

# In[ ]:


x = hd.drop(["target", "age_groups"], axis=1)
y = hd["target"]
scalar = StandardScaler()

lr = LogisticRegression()

pipeline = Pipeline([('transformer', scalar), ('estimator', lr)])

#lr.fit(x_train,y_train)
#acc = lr.score(x_test,y_test)*100
acc = cross_val_score(pipeline, x, y, cv=5)

print(acc)
print("Accuracy: {:.3f} (+/- {:.3f})".format(acc.mean(), acc.std()*2))


# # KNN

# In[ ]:


scoreList = []
for i in range(2,31):
    knn = KNeighborsClassifier(n_neighbors = i)
    cv = StratifiedKFold(n_splits=5)
    pipeline = Pipeline([('transformer', scalar), ('estimator', knn)])

    acc = cross_val_score(pipeline, x, y, cv=cv, scoring="f1")
    scoreList.append(acc.mean())
    #print(acc)
    print("Accuracy: {:.3f} (+/- {:.3f})  Neighbors: {}".format(acc.mean(), acc.std()*2,i))

print("\nMax accuracy achieved: {:.3f}".format(max(scoreList)))
plt.figure(figsize=(16,8))
plt.plot(scoreList)


# # Random Forests

# In[ ]:


rf = RandomForestClassifier(n_estimators=100, max_depth=5, n_jobs=-1)
cv = StratifiedKFold(n_splits=7)
pipeline = Pipeline([('transformer', scalar), ('estimator', rf)])

acc = cross_val_score(pipeline, x, y, cv=cv, scoring="f1")
scoreList.append(acc.mean())
#print(acc)
print("Accuracy: {:.3f} (+/- {:.3f})".format(acc.mean(), acc.std()*2))


# Explore the feature importance for the random forest.

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
rf.fit(x_train, y_train)

fi = pd.DataFrame(rf.feature_importances_, index = x.columns, columns=['importance']).sort_values('importance', ascending=False)
plt.figure(figsize=(20,8))
sns.heatmap(fi, annot=True, cmap=sns.diverging_palette(10, 140, s=90, l=60, as_cmap=True))


# As we could already say in the beginning thalach and oldpeak are among the most important features while fbs or slope is among the least important ones. So for this task a random forest or a knn should be used in order to achieve a high accuracy.
