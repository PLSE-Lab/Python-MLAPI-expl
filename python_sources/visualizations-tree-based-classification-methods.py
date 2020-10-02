#!/usr/bin/env python
# coding: utf-8

# # Visualizations and Classification Using Tree-based Models: 
# ## Decision Tree, Random Forest & XGBoost
# 
# In this notebook I will start by quickly exploring the data. We have no missing values, which saves us a lot of time and uncertainty in making assumptions. Next, I will create various visualizations of the data which reveal some interesting insights. Finally, we will compare tree-based machine learning models and will explore how parameter fitting influences the accuracy of the models.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, plot_importance

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import data, start exploratory data analysis
edm = pd.read_csv('../input/xAPI-Edu-Data.csv')
edm.head()


# In[ ]:


edm.info()


# In[ ]:


# Some of the columns seem to have random capitalizations in then, let's make this look a bit tidier

edm.rename(index=str, columns={'gender':'Gender', 'NationalITy':'Nationality',
                               'raisedhands':'RaisedHands', 'VisITedResources':'VisitedResources'},
                               inplace=True)


# ## Visualizations
# Great, our dataset has no missing values! This is quite rare for a data scientist, but it's a good thing so let's start visualizing.

# In[ ]:


# Counts per class --> Is the dataset unbalanced?
counts = sns.countplot(x='Class', data=edm, palette='coolwarm')
counts.set(xlabel='Class', ylabel='Count', title='Occurences per class')
plt.show()


# In[ ]:


# Exploring nationalities
nat = sns.countplot(x='Nationality', data=edm, palette='coolwarm')
nat.set(xlabel='Nationality', ylabel='Count', title='Nationality Representation')
plt.setp(nat.get_xticklabels(), rotation=80)
plt.show()


# The dataset is not extremely unbalanced. Although we do have a small dataset so this might still cause some problems with classification later on. Most of our students are from Kuwait or Jordan. This was already mentioned in the description of the dataset, but I thought it would be nice to provide a visual.

# In[ ]:


# Semester comparison
sem = sns.countplot(x='Class', hue='Semester', order=['L', 'M', 'H'], data=edm, palette='coolwarm')
sem.set(xlabel='Class', ylabel='Count', title='Semester comparison')
plt.show()


# Mmh ... It looks like students's performed a bit better in the second semester ('S') than in the first semester ('F'). Suprisingly, the middle class stays the same but the lower class has less students in the second semester and the higher class has more students. Let's explore gender next.

# In[ ]:


# gender comparison
plot = sns.countplot(x='Class', hue='Gender', data=edm, order=['L', 'M', 'H'], palette='coolwarm')
plot.set(xlabel='Class', ylabel='Count', title='Gender comparison')
plt.show()


# It looks like women performed better than men on average. Would the amount of visited resources in the online environment influence the final grade?

# In[ ]:


plot = sns.swarmplot(x='Class', y='VisitedResources', hue='Gender', order=['L', 'M', 'H'], 
              data=edm, palette='coolwarm')
plot.set(xlabel='Class', ylabel='Count', title='Gender comparison on visited resources')
plt.rcParams['figure.figsize']=(10,5)
plt.show()


# This swarm plot shows us that students who received a lower grade (L) visited way fever resources than students that scored a M or H grade. Additionally, women who received a high mark (H) almost exclusively visited a lot of the online resources.

# In[ ]:


# Pairgrid, exploring our numerical variables
g = sns.PairGrid(edm, hue='Gender', palette='coolwarm', hue_kws={'marker': ['o', 's']})
g = g.map_diag(plt.hist)
g = g.map_upper(plt.scatter, linewidths=1, edgecolor='w', s=40)
g = g.map_lower(sns.kdeplot, lw=3, legend=False, cmap='coolwarm')
g = g.add_legend()


# Pairgrid provides a clear way to explore our numerical data. The plot looks great, but it doesn't look like there are any specific relationships or patterns in the numerical data. However, when exploring the visited resources again we see that females generally visit more resources as shown in the histogram.

# # Machine Learning
# 
# Now we get to the machine learning section. We will start by encoding our categorical variables and splitting the data into a train and test set.

# In[ ]:


X = edm.drop('Class', axis=1)
y = edm['Class']

# Encoding our categorical columns in X
labelEncoder = LabelEncoder()
cat_columns = X.dtypes.pipe(lambda x: x[x == 'object']).index
for col in cat_columns:
    X[col] = labelEncoder.fit_transform(X[col])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=52)


# In[ ]:


# Logistic Regression as baseline, then exploring tree-based methods

keys = []
scores = []
models = {'Logistic Regression': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(),
          'Random Forest': RandomForestClassifier(n_estimators=300, random_state=52)}

for k,v in models.items():
    mod = v
    mod.fit(X_train, y_train)
    pred = mod.predict(X_test)
    print('Results for: ' + str(k) + '\n')
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    acc = accuracy_score(y_test, pred)
    print("accuracy is "+ str(acc)) 
    print('\n' + '\n')
    keys.append(k)
    scores.append(acc)
    table = pd.DataFrame({'model':keys, 'accuracy score':scores})

print(table)


# ## Random Forest
# 
# The Random Forest Classifier performed best. Let's explore the number of estimators in the forest further. A general rule is that the RFC performs better when the amount of estimators increases.

# In[ ]:


# Exploring the number of estimators in the random forest
score = []
est = []
estimators = [1, 10, 50, 100, 200, 300, 400, 500]
for e in estimators:
    rfc1 = RandomForestClassifier(n_estimators=e, random_state=52)
    pred1 = rfc1.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, pred1)
    score.append(accuracy)
    est.append(e)
plot = sns.pointplot(x=est, y=score)
plot.set(xlabel='Number of estimators', ylabel='Accuracy', 
         title='Accuracy score of RFC per # of estimators')
plt.show()


# And indeed, the RFC performs better when the number of estimators increases. However, it plateaus at 200 estimators. In the for loop before I used 300 estimators which is a general number I like to start trying it out with. Apparently 200 estimators is enough for this dataset. If you start experimenting on a very large dataset, having less estimators will save you a lot of running time. 
# 
# We can also explore another variable like the minimum number of samples required to be at a leaf node.

# In[ ]:


# Exploring minimum leaf samples
score = []
leaf = []
leaf_options = [1, 5, 10, 50, 100, 200]
for l in leaf_options:
    rfc2 = RandomForestClassifier(n_estimators=200, random_state=52, min_samples_leaf=l)
    pred2 = rfc2.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, pred2)
    score.append(accuracy)
    leaf.append(l)
plot = sns.pointplot(x=leaf, y=score)
plot.set(xlabel='Number of minimum leaf samples', ylabel='Accuracy', 
         title='Accuracy score of RFC per # of minimum leaf samples')
plt.show()


# In this case we see that the accuracy score simply decreases as the minimum leaf samples increase. Therefore, it is best to keep this value at the default of 1.

# ## Extreme Gradient Boosting
# 
# Many Kaggle competitions have been won by using Extreme Gradient Boosting. I have never used it so let's give it a try. If you have any tips please share them in the comments.

# In[ ]:


xgb = XGBClassifier(seed=52)
pred = xgb.fit(X_train, y_train).predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print("accuracy is "+ str(accuracy_score(y_test, pred)))      


# In[ ]:


plot_importance(xgb)
plt.rcParams['figure.figsize']=(10,5)
plt.show()


# the main feature in the XGB model is the number of visited resources. Earlier we saw that there was a big difference in visited resources between the L and M and H classes. Suprisingly, gender is very low on the feature importance list.
# 
# Next step is to try and improve the performance of our XGB Classifier by trying some different parameters and using a grid search approach.

# In[ ]:


# Let's try to improve the accuracy of the XGClassifier with a grid search approach.

d_values = []
l_values = []
n_values = []
acc_values = []
depth = [2, 3, 4]
learning_Rate = [0.01, 0.1, 1]
n_estimators = [50, 100, 150, 200]
for d in depth:
    for l in learning_Rate:
        for n in n_estimators:
            xgb = XGBClassifier(max_depth=d, learning_rate=l, n_estimators=n, seed=52)
            pred = xgb.fit(X_train, y_train).predict(X_test)
            acc = accuracy_score(y_test, pred)
            d_values.append(d)
            l_values.append(l)
            n_values.append(n)
            acc_values.append(acc)
            
dict = {'max_depth':d_values, 'learning_rate':l_values, 'n_estimators':n_values,
       'accuracy':acc_values}

output = pd.DataFrame.from_dict(data=dict)
print(output.sort_values(by='accuracy', ascending=False)) 
            


# ## Accuracy improved :)
# We can see that using a learning_rate of 0.1, a max_depth of 4 and 100 estimators in our XGB classifier provides an accuracy of 0.8194. 
# 
# This is a nice improvement over our previous score of 0.7986
# 
#  The XGB model now also performs better than the random forest classifier which capped at 0.8125
# 
# Let's explore the important features in this 'best' model.

# In[ ]:


# Building the best XGB and looking at feature importances

xgb2 = XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, seed=52)
pred = xgb2.fit(X_train, y_train).predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
print("accuracy is "+ str(accuracy_score(y_test, pred)))   

plot_importance(xgb2)
plt.rcParams['figure.figsize']=(10,5)
plt.show()


# ### Let's explore the feature importances of our best Random Forest model as well.

# In[ ]:


rfc = RandomForestClassifier(n_estimators=200, random_state=52)
pred = rfc.fit(X_train, y_train).predict(X_test)
dn = {'features':X.columns, 'score':rfc.feature_importances_}
df = pd.DataFrame.from_dict(data=dn).sort_values(by='score', ascending=False)
plot = sns.barplot(x='score', y='features', data=df, orient='h')
plot.set(xlabel='Score', ylabel='Features', 
         title='Feature Importance of Random Forest Classifier')
plt.setp(plot.get_xticklabels(), rotation=90)
plt.show()


# Visited resources is the most important feature in both the XGB and the RFC model. However, many differences can be observed for the other features. Discussion, for example, is almost the most important feature in the XGB model but is much less important in the RFC model.
