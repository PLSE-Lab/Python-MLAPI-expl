#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.linear_model import LogisticRegression
import scipy
from scipy.spatial.distance import pdist,cdist
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
import re
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, log_loss, classification_report,f1_score,confusion_matrix)
import xgboost
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder,OneHotEncoder
import xgboost as xgb
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


df = pd.read_csv('/kaggle/input/onlinenewspop/OnlineNewsPopularity.csv')


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# **Dropping URL as its not making any sense**

# In[ ]:


#droping url
df.drop(columns=['url'],inplace=True)


# In[ ]:


#correlation plot
plt.figure(figsize=(40,40))
sns.heatmap(data=df.corr(),annot=True,cmap='BuPu')


# From correlation plot we can see most of the features are important

# In[ ]:


df.columns


# **Box plots**

# In[ ]:


#box plot to check outliers
for i in df.columns:
    sns.boxplot(df[i])
    plt.show()


# **Too many outliers.So removing them**

# In[ ]:


#removing outliers
Q1 = df.quantile(q=0.25) 

Q3 = df.quantile(q=0.75)

IQR = Q3-Q1
print('IQR for each column:- ')
print(IQR)


# In[ ]:



sorted_shares = df.sort_values(' shares') 

median = sorted_shares[' shares'].median() 

q1 = sorted_shares[' shares'].quantile(q=0.25) 

q3 = sorted_shares[' shares'].quantile(q=0.75) 

iqr = q3-q1


# In[ ]:


Inner_bound1 = q1-(iqr*1.5) 
print(f'Inner Boundary 1 = {Inner_bound1}')
Inner_bound2 = q3+(iqr*1.5)  
print(f'Inner Boundary 2 = {Inner_bound2}')
Outer_bound1 = q1-(iqr*3)    
print(f'Outer Boundary 1 = {Outer_bound1}')
Outer_bound2 = q3+(iqr*3)   
print(f'Outer Boundary 2 = {Outer_bound2}')


# In[ ]:


Df = df[df[' shares']<=Outer_bound2]


# In[ ]:


print(f'Data before Removing Outliers = {df.shape}')
print(f'Data after Removing Outliers = {Df.shape}')
print(f'Number of Outliers = {df.shape[0] - Df.shape[0]}')


# **Hist plot**

# In[ ]:


Df.hist(figsize=(30,30))
plt.show()


# We can clearly see data is not properly distributed so we need to do scaling and smote 

# In[ ]:


#EDA
a,b = Df[' shares'].mean(),Df[' shares'].median()
print(f'Mean article shares = {a}')
print(f'Median article share = {b}')


# In[ ]:


Wd = Df.columns.values[30:37]
Wd


# In[ ]:


Unpop=Df[Df[' shares']<a]
Pop=Df[Df[' shares']>=a]
Unpop_day = Unpop[Wd].sum().values
Pop_day = Pop[Wd].sum().values

fig = plt.figure(figsize = (13,5))
plt.title("Count of popular/unpopular news over different day of week (Mean)", fontsize = 16)
plt.bar(np.arange(len(Wd)), Pop_day, width = 0.3, align="center", color = 'r',           label = "popular")
plt.bar(np.arange(len(Wd)) - 0.3, Unpop_day, width = 0.3, align = "center", color = 'b',           label = "unpopular")
plt.xticks(np.arange(len(Wd)), Wd)
plt.ylabel("Count", fontsize = 12)
plt.xlabel("Days of week", fontsize = 12)
    
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()


# In[ ]:


Unpop2=Df[Df[' shares']<b]
Pop2=Df[Df[' shares']>=b]
Unpop_day2 = Unpop2[Wd].sum().values
Pop_day2 = Pop2[Wd].sum().values
fig = plt.figure(figsize = (13,5))
plt.title("Count of popular/unpopular news over different day of week (Median)", fontsize = 16)
plt.bar(np.arange(len(Wd)), Pop_day2, width = 0.3, align="center", color = 'r',           label = "popular")
plt.bar(np.arange(len(Wd)) - 0.3, Unpop_day2, width = 0.3, align = "center", color = 'b',           label = "unpopular")
plt.xticks(np.arange(len(Wd)), Wd)
plt.ylabel("Count", fontsize = 12)
plt.xlabel("Days of week", fontsize = 12)
    
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()


# In[ ]:


Dc = Df.columns.values[12:18]


# In[ ]:


Unpop3=Df[Df[' shares']<a]
Pop3=Df[Df[' shares']>=a]
Unpop_day3 = Unpop3[Dc].sum().values
Pop_day3 = Pop3[Dc].sum().values
fig = plt.figure(figsize = (13,5))
plt.title("Count of popular/unpopular news over different data channel (Mean)", fontsize = 16)
plt.bar(np.arange(len(Dc)), Pop_day3, width = 0.3, align="center", color = 'r',           label = "popular")
plt.bar(np.arange(len(Dc)) - 0.3, Unpop_day3, width = 0.3, align = "center", color = 'b',           label = "unpopular")
plt.xticks(np.arange(len(Dc)), Dc)
plt.ylabel("Count", fontsize = 12)
plt.xlabel("Days of week", fontsize = 12)
    
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()


# In[ ]:


Unpop4=Df[Df[' shares']<b]
Pop4=Df[Df[' shares']>=b]
Unpop_day4 = Unpop4[Dc].sum().values
Pop_day4 = Pop4[Dc].sum().values
fig = plt.figure(figsize = (13,5))
plt.title("Count of popular/unpopular news over different data channel (Median)", fontsize = 16)
plt.bar(np.arange(len(Dc)), Pop_day4, width = 0.3, align="center", color = 'r',           label = "popular")
plt.bar(np.arange(len(Dc)) - 0.3, Unpop_day4, width = 0.3, align = "center", color = 'b',           label = "unpopular")
plt.xticks(np.arange(len(Dc)), Dc)
plt.ylabel("Count", fontsize = 12)
plt.xlabel("Days of week", fontsize = 12)
    
plt.legend(loc = 'upper right')
plt.tight_layout()
plt.show()


# In[ ]:


Df.head()


# In[ ]:


mean = Df[' shares'].mean()


# In[ ]:


#Converting output columns to 0 and 1
Df[' shares'] = Df[' shares'].apply(lambda x: 0 if x <mean  else 1)


# In[ ]:


Df[' shares'].value_counts()


# **Scaling and SMOTE**

# Scaling and SMOTE is needed because data is not properly distributed

# In[ ]:


#Scaling and Doing SMOTE 
X = Df.drop(' shares',axis=1)
y = Df[' shares']

scaler=StandardScaler()
X=scaler.fit_transform(X)
from imblearn.over_sampling import SMOTE
SMOTE().fit_resample(X, y)
X,y = SMOTE().fit_resample(X, y)


# In[ ]:


print(y)


# In[ ]:


def calculateScore(confMat):
    TP = confMat[0][0]
    TN = confMat[1][1]
    FP = confMat[0][1]
    FN = confMat[1][0]
    Sen.append(TP / (TP + FN))
    Spe.append(TN / (FP + TN))
    FPR.append(FP / (FP + TN))
    FNR.append(FN / (FN + TP))


# **Splitting train and test**

# In[ ]:


train, test, target_train, target_val = train_test_split(X, 
                                                         y, 
                                                         train_size= 0.80,
                                                         random_state=0);


# In[ ]:


#Using multiple classifiers
Model = []
Accuracy= []
F1Score = []
Sen = []
Spe = []
FPR = []
FNR = []


# **Logistic Regression**

# In[ ]:


LR = LogisticRegression(multi_class='auto')
LR.fit(train,target_train)
lr_pred = LR.predict(test)
Model.append('Logistic Regression')
Accuracy.append(accuracy_score(target_val,lr_pred))
F1Score.append(f1_score(target_val,lr_pred,average=None))


# In[ ]:


data = confusion_matrix(target_val,lr_pred)
calculateScore(data)
df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})


# **Random Forrest**

# In[ ]:


seed = 0
params = {
    'n_estimators':range(10,100,10),
    'criterion':['gini','entropy'],
}
rf = RandomForestClassifier()
rs = RandomizedSearchCV(rf, param_distributions=params, scoring='accuracy', n_jobs=-1, cv=5, random_state=42)
rs.fit(X,y)


# In[ ]:


rs.best_params_


# In[ ]:


rf = RandomForestClassifier(**rs.best_params_)
rf.fit(train, target_train)
rf_pred = rf.predict(test)


# **Plot feature importance**

# In[ ]:


features = Df.columns
importance = rf.feature_importances_
indices = np.argsort(importance)
plt.figure(1,figsize=(10,20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='lightblue', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')


# We can clearly see almost all features are important

# In[ ]:


Model.append('Random Forrest')
Accuracy.append(accuracy_score(target_val,rf_pred))
F1Score.append(f1_score(target_val,rf_pred,average=None))


# In[ ]:


data = confusion_matrix(target_val,rf_pred)
calculateScore(data)
df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})


# From confusion matrix we can see there are false positive as well

# **Decision Tree****

# In[ ]:


params = {
    
    'criterion':['gini','entropy'],
    'splitter':['best','random'],
    'max_depth':range(1,10,1),
    'max_leaf_nodes':range(2,10,1),
}
dt = DecisionTreeClassifier()
rs = RandomizedSearchCV(dt, param_distributions=params, scoring='accuracy', n_jobs=-1, cv=5, random_state=42)
rs.fit(X,y)


# In[ ]:


rs.best_params_


# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(train, target_train)
dt_pred = dt.predict(test)
features = Df.columns
importance = dt.feature_importances_
indices = np.argsort(importance)
plt.figure(1,figsize=(10,20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importance[indices], color='lightblue', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')


# Again we can see almost all features have importance. few features like n_stop_words has very less importance when compared to others

# In[ ]:


Model.append('Decision Tree')
Accuracy.append(accuracy_score(target_val,dt_pred))
F1Score.append(f1_score(target_val,dt_pred,average=None))


# In[ ]:


data = confusion_matrix(target_val,dt_pred)
calculateScore(data)
df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})


# **Gradient Boosting**

# In[ ]:


gb_Boost = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01)
gb_Boost.fit(train, target_train)
y_pred = rf.predict(test)


# In[ ]:


Model.append('Gradient Boosting')
Accuracy.append(accuracy_score(target_val,y_pred))
F1Score.append(f1_score(target_val,dt_pred,average=None))


# In[ ]:


data = confusion_matrix(target_val,y_pred)
calculateScore(data)
df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})


# **Naive**

# In[ ]:


naiveClassifier=GaussianNB()
naiveClassifier.fit(train, target_train)
naiveClassifier_pred = naiveClassifier.predict(test)


# In[ ]:


Model.append('Naive')
Accuracy.append(accuracy_score(target_val,naiveClassifier_pred))
F1Score.append(f1_score(target_val,naiveClassifier_pred,average=None))


# In[ ]:


data = confusion_matrix(target_val,naiveClassifier_pred)
calculateScore(data)
df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})


# **KNN**

# In[ ]:


# knn = KNeighborsClassifier(n_neighbors=8)
# knn.fit(train, target_train)
# knn_pred = knn.predict(test)


# In[ ]:


# Model.append('KNN')
# Accuracy.append(accuracy_score(target_val,knn_pred))
# F1Score.append(f1_score(target_val,knn_pred,average=None))


# In[ ]:


# data = confusion_matrix(target_val,knn_pred)
# calculateScore(data)
# df_cm = pd.DataFrame(data, columns=np.unique(target_val), index = np.unique(target_val))
# df_cm.index.name = 'Actual'
# df_cm.columns.name = 'Predicted'
# plt.figure(figsize = (10,7))
# sns.set(font_scale=1.4)
# sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})


# **Comparing models**

# In[ ]:


result = pd.DataFrame({'Model':Model,'Accuracy':Accuracy,'F1Score':F1Score,'Sensitivity':Sen,'Specificity':Spe,'FPR':FPR,'FNR':FNR})
result


# **KNN model is taking long time for differnt values of k but optimal value got was 8. We can see that Random forrest ensemble technique gave maximum accuracy and also F1 score is higher for the same. Also we can see gradient boosting gave almost same accuracy but since F1 score of random forrest is more. Hence ensemble technique is most accurate.**

# In[ ]:




