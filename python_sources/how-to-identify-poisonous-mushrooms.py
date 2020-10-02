#!/usr/bin/env python
# coding: utf-8

# ## Objective 
# In this kernel we will explore the world of mushrooms using tools of Exploratory Data Analysis, will try to build a predictive modeling tool and in the end, we will make a guideline that can aid in distinguishing edible from poisonous variety.  

# ### Importing modules
# We begin with importing the necessary modules.

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('../input/mushrooms.csv')


# Next, using the `head`, `shape`, `column` and `info` methods, I found that there are 8124 samples and 23 features. These categorical features mostly include the various aspects of a mushroom's morphology/physical appearance. Moreover, there is no missing data!

# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.info()


# Further, probing shows that except 'veil-type', all other features have atleast 2 or more categories.

# In[ ]:


lst=[]
for col in data.columns:
    lst.append(data[col].nunique())
x=dict(zip(data.columns,lst) )
x


# Next, checking the target variable, 'class'  it is evident hat there are data for 4208 edible (e) mushrooms and for 3916 poisonous (p) mushrooms.

# In[ ]:


data['class'].value_counts()


# In[ ]:


sns.countplot(x='class', palette='RdBu', data=data)
plt.title('Number of p and e mushrooms')


# For further data analysis, I made 2 dataframes for poisonous and edible mushrooms.

# In[ ]:


p_data=data.loc[data['class']=='p']
e_data=data.loc[data['class']=='e']
e_data.head(2)


# ### Plots
# To get a better understanding of the characteristics of edible and poisonous mushrooms and which characters are distinct, I made plots for each feature. Instead of plugging the numbers of mushrooms directly in the characteres/categories of each feature/column, I calculated first the percentage of edible and poisonous mushrooms falling in the categories in each feature or column. This was done because we have unequal amount of data for edible and poisonous mushrooms (4208 edible (e) mushrooms and 3916 poisonous (p) ). For a better understanding, I have kept tables showing the percentage distribution of edible and poisonous mushrooms in different categories here.

# Some of the features that show vast differences between the two classes are - Odor, Bruises, Stalk surface above the ring, Spore print color. Within the features there are categories where either of the classes having complete dominance such as in the case of buff gill color shown only by poisonous mushrooms.

# In[ ]:


lst=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']
fig, axes = plt.subplots(11, 2, figsize=(15,45))
sns.set_style('white')
plt.subplots_adjust (hspace=0.4, wspace=0.2)
n=0
for i in lst:
    data_p_1=p_data.groupby(['class',i]).agg({i:'count'})
    data_p_1.columns=['number']
    data_p_1['perc']= (data_p_1['number']*100)/3916
    data_e_1=e_data.groupby(['class',i]).agg({i:'count'})
    data_e_1.columns=['number']
    data_e_1['perc']= (data_e_1['number']*100)/4208
    data_new_i=pd.concat([data_p_1,data_e_1],axis=0)
    data_new_i.drop(['number'],axis=1,inplace=True)
    data_new1=data_new_i.unstack(level=0).fillna(0)
    print(data_new1)
    data_new1.plot(kind='bar',cmap='rainbow', ax=axes[n//2, n%2])
    axes[n//2, n%2].set_ylabel('percentage')
    n+=1


# ### Building models
# Before building our machine learning algorithms, I dropped 'veil-type' as this feature has only one category. This is followed by label encoding and one hot encoding to convert the categorical data into a machine learning algorithm-readable form. For our target variables - 0 corresponds to edible mushrooms and 1 to poisonous mushrooms.

# In[ ]:


data=data.drop(['veil-type'],axis=1)
data.shape


# In[ ]:


encoder=LabelEncoder()
lst=[]
for col in data.columns:
    if data[col].nunique()<=2:
        lst.append(str(col))
print(lst)
for i in lst:
    encoder.fit(data[i].drop_duplicates())
    data[i]=encoder.transform(data[i])

print(data.head(2))
#p=1
#e=0


# In[ ]:


data=pd.get_dummies(data)
data.shape


# Next, I obtained the train and test sets.

# In[ ]:


x=data.drop(['class'],axis=1)
X=x.values
y=data['class']
X.shape


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=21)


# Principal Component Analysis (PCA) was performed for dimensionality reduction.

# In[ ]:


pca=PCA().fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.xlim(0,115,5)


# In[ ]:


pca=PCA(n_components=40)
pca.fit(X_train)
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)
X_train_pca.shape


# I started with a simple classifier- Logistic Regression. This gave a high accuracy of 99%  and over 98% scores for both recall and precision.

# In[ ]:


logreg=LogisticRegression(random_state=1)
score = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='accuracy'))
p_scores = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='precision'))
r_scores = np.mean(cross_val_score(logreg,  X_train_pca, y_train, scoring='recall'))
print("Accuracy: %s" % '{:.2%}'.format(score))
print ('Precision : %s' %'{:.2%}' .format(p_scores))
print ('Recall score: %s' % '{:.2%}'.format(r_scores))


# After hyperparameteres tuning, I obtained the best parameteres for logistic regression- penalty =l1 and C=100. This gave a 100% accuracy score. 

# In[ ]:


logreg=LogisticRegression(random_state=1)
param_grid = {'penalty': ['l1','l2'], 'C': [10,100,1000]}
logreg_cv = GridSearchCV(estimator = logreg, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 0)
logreg_cv.fit( X_train_pca, y_train)
print(logreg_cv.best_params_)
score=logreg_cv.best_score_
print("Accuracy: %s" % '{:.2%}'.format(score))


# Plugging the hyperparameters in logistic regression, gave 100% accuracy, precision and recall score for the test set. 

# In[ ]:


logreg2=LogisticRegression(random_state=1,penalty= 'l1',C=100)
logreg2.fit(X_train_pca, y_train)
y_pred=logreg2.predict(X_test_pca)
ascore=accuracy_score(y_test,y_pred)
pscore=precision_score(y_test,y_pred)
rscore=recall_score(y_test,y_pred)
matrix=confusion_matrix(y_test,y_pred)
print("Accuracy: %s" % '{:.2%}'.format(ascore))
print ('Precision : %s' %'{:.2%}' .format(pscore))
print ('Recall score: %s' % '{:.2%}'.format(rscore))

sns.heatmap(matrix,annot=True,fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')


# Excited with the 100% accuracy obtained using LogisticRegression and a bit skeptical, I tried Random Forest Classifier. Another reason for using RandomForest classfier was to obtain important features that can aid in distinguishing edible and poisonous mushrooms. I used the data that was obtained before PCA as in my exerience I have seen that RandomForest Classfier works better (gives higher accuracy) in non-PCA data than when the data is obtained after PCA. RandomForest classfier also gave a 100% accuracy score in that case. (Accuracy was 99.86% in data obtained after PCA. If you have an explaination, why this hapeens, please let me know.)

# In[ ]:


rf=RandomForestClassifier(random_state=21)
score_rf = np.mean(cross_val_score(rf,  X_train, y_train, scoring='accuracy'))
p_score_rf = np.mean(cross_val_score(rf,  X_train, y_train, scoring='precision'))
r_score_rf = np.mean(cross_val_score(rf,  X_train, y_train, scoring='recall'))
print("Accuracy for RandomForest: %s" % '{:.2%}'.format(score_rf))
print ('Precision RandomForest:: %s' %'{:.2%}' .format(p_score_rf))
print ('Recall score RandomForest:: %s' % '{:.2%}'.format(r_score_rf))


# The same results were echoed in the test set.

# In[ ]:


rf=RandomForestClassifier(random_state=21)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
ascore=accuracy_score(y_test,y_pred)
pscore=precision_score(y_test,y_pred)
rscore=recall_score(y_test,y_pred)
matrix=confusion_matrix(y_test,y_pred)
print("Accuracy: %s" % '{:.2%}'.format(ascore))
print ('Precision : %s' %'{:.2%}' .format(pscore))
print ('Recall score: %s' % '{:.2%}'.format(rscore))
print('Confusion matrix: ')
print(matrix)

sns.heatmap(matrix,annot=True,fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')


# RandomForest Classfier was then used to get the important features and their degree of  importance.

# In[ ]:


fig, ax=plt.subplots(figsize=(15,25))
features = x.columns
importances = rf.feature_importances_
indices = np.argsort(importances)

plt.title('Features Importance')
plt.barh(range(len(indices)), importances[indices], color='purple', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show() 


# The top 10 features for distinguishing edible from poisonous mushrooms are as follows:
# 1. odor_n= This correspond to odor-none i.e. odorless, which is a characteristic shown by over 81% of the edible mushrooms compared to only 3% poisonous mushrooms. Odor in general is one of the features that shows huge difference between edible and posionous mushrooms. This was already quite evident from the EDA performed above.
# 2. gill_color_b= This corresponds to buff coloured gill shown only by poisonous mushrooms. It is a characteristic shown by 44% of the poisonous variety.
# 3. gill spacing= Closed spacing is shown by 97% of poisonous mushrooms and 71% edible ones. The difference is more evident when 28% of edible ones show crowded gill spacing compared to only 3% of poisonous ones.
# 4. ring type p= Pendant type ring is shown by over 75% edible mushrooms compared to 21% poisonous ones.
# 5. population_v= 'several' population, with 21% edible mushrooms compared to 9% poisonous ones.
# 6. stalk surface above ring s= 'Smooth' stalk surface for nearly 86% edible mushrooms compared to 40% poisonous ones.
# 7. ring type i= If the ring type is large, then the mushroom is definitely poisonous. 
# 8. bruises= Majority of poisonous mushroom have no bruises while majority of edible ones have.
# 9. spore print color h- 40% of poisonous mushrooms show chocolate coloured spores, while only 1% of edible ones have chocolate colored spores.
# 10. stalk surface above ring k=Majority of poisonous mushrooms have silky stalk surface above ring.

# ###  Conclusion
# So, a **general guideline** (obtained after EDA and from RandomForestClassfier's features importantance) while shrooming is -
# A mushroom is **poisononous**, if it has-
# 1.  an odor (such as creosote ,fishy, foul, pungent, spicy , a character of 95% poisonous mushrooms), or 
# 2. has a large ring (a character shown only by poisonous mushrooms),  or 
# 3. a combination of any of the above and lower mentioned points-
# i).   buff colored gills, 
# ii).  closed gill spacing, 
# iii). no bruises, 
# iv). chocolate colored spore print (or even white colored), 
# v). has a silky stalk surface  above the ring
# 
# ### Thanks for reading. 
