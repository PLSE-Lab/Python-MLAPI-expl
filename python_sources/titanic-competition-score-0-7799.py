#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


file = '../input/train.csv'
df = pd.read_csv(file)


# ## EXPLORATORY DATA ANALYSIS

# In[ ]:


df.head()


# In[ ]:


df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df['Survived'].value_counts()


# In[ ]:


#cheking for outliers - none.
#sns.set(style="whitegrid")
#g = sns.boxplot(data = df, orient = 'h')
#g.figure.set_size_inches(8,12)
#plt.tight_layout()
#plt.show()


# In[ ]:


bins = np.linspace(df.Fare.min(), df.Fare.max(), 10)
g = sns.FacetGrid(df, col="Sex", hue="Survived", palette="muted", col_wrap=2)
g.map(plt.hist, 'Fare', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


bins = np.linspace(df.Age.min(), df.Age.max(), 10)
g = sns.FacetGrid(df, col="Sex", hue="Survived", palette="muted", col_wrap=2)
g.map(plt.hist, 'Age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


bins = np.linspace(df.Age.min(), df.Age.max(), 10)
g = sns.FacetGrid(df, col="Pclass", hue="Survived", palette="muted", col_wrap=3)
g.map(plt.hist, 'Age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


bins = np.linspace(df.Age.min(), df.Age.max(), 10)
g = sns.FacetGrid(df, col="SibSp", hue="Survived", palette="muted", col_wrap=4)
g.map(plt.hist, 'Age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


df['SibSp'].value_counts()


# In[ ]:


#df['Cabin'].value_counts()


# In[ ]:


df.isnull().sum(axis = 0)


# ## FEATURE ENGINEERING

# I'll make two data sets for further model training: `df_hot` - with One Hot Encoding and <br> `df1` - with  Numeric Encoding only.

# In[ ]:


df_hot = pd.concat([df, pd.get_dummies(df['Sex'])], axis=1)
df_hot = df_hot.drop(['Sex'], axis=1)


# In[ ]:


df_hot = pd.concat([df_hot, pd.get_dummies(df['Pclass'])], axis=1)
df_hot = df_hot.drop(['Pclass'], axis=1)


# In[ ]:


df_hot = df_hot.rename(columns={df_hot.columns[9]:'Pclass1', df_hot.columns[10]:'Pclass2', df_hot.columns[11]:'Pclass3'})


# In[ ]:


df_hot['Embarked'].value_counts()


# In[ ]:


df_hot['Embarked'].fillna("S", inplace = True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
pd.options.display.float_format = '{:.2f}'.format # to make legible
le = LabelEncoder()
T = df_hot['Embarked']
encoded = le.fit_transform(np.ravel(T))


# In[ ]:


df_hot = pd.concat([df_hot, pd.DataFrame(encoded, columns = ['Emb'])], axis=1)
df_hot = df_hot.drop(['Embarked'], axis=1)


# In[ ]:


df_hot = df_hot[['Survived', 'SibSp', 'Parch', 'Fare', 'Emb', 'Cabin', 'Age',
       'female', 'male', 'Pclass1', 'Pclass2', 'Pclass3']]


# In[ ]:


df_hot = df_hot.drop(['Cabin'], axis=1)


# In[ ]:


df_hot.head(10)


# In[ ]:


# no significant correlations found
#sns.set(style="ticks", color_codes=True)
#sns.pairplot(df_hot)


# #### Imputing NaNs in Age column with KNN

# In[ ]:


from fancyimpute import KNN 
train_cols = list(df_hot)


# In[ ]:


# Use 5 nearest rows which have a feature to fill in each row's
# missing features
dfi = pd.DataFrame(KNN(k=5).fit_transform(df_hot))
dfi.columns = train_cols


# In[ ]:


dfi.head(10)


# In[ ]:


dfi.isnull().sum()


# In[ ]:


df_cor = dfi.corr()

plt.figure(figsize=(15,8))
colormap = plt.cm.cubehelix_r
ax = sns.heatmap(df_cor, annot = True, cmap=colormap)


# I'll use `Age` data in the `df1` set as well. In addition to label encoding and other <br> data set transformations identical to `df_hot` set.

# In[ ]:


df1 = df.drop(['Age'], axis=1)
df1 = pd.concat([df1, dfi['Age']], axis=1)


# In[ ]:


df1['Embarked'].fillna("S", inplace = True)


# In[ ]:


df1 = df1.drop(['Cabin'], axis=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
T = df1['Sex']
encoded = le.fit_transform(np.ravel(T))


# In[ ]:


df1 = pd.concat([df1, pd.DataFrame(encoded, columns = ['Gender'])], axis=1)
df1 = df1.drop(['Sex'], axis=1)


# In[ ]:


le = LabelEncoder()
T = df1['Embarked']
encoded = le.fit_transform(np.ravel(T))


# In[ ]:


df1 = pd.concat([df1, pd.DataFrame(encoded, columns = ['Emb'])], axis=1)
df1 = df1.drop(['Embarked'], axis=1)


# In[ ]:


df1.isnull().sum()


# In[ ]:


#df1.to_csv('tit2.csv')
#dfi.to_csv('tit1.csv')


# ## Fitting models

# For this classification problem I'm going to apply the following algorithms:<br>
# * Naive Bayes
# * Logistic Regression
# * Random Forest
# * SVC
# * K-Nearest Neighbours
# * XGBoost

# To minimise possible overfitting (without y_test data) I'll use cross-validation with 100 folds <br> (except for XGBoost to save computing resources) <br> I'll test both data sets (`X` and `X1`) but leave rusolts only for the best performing.

# In[ ]:


from sklearn import preprocessing
X = np.asarray(dfi.drop(['Survived'], axis = 1))
y = np.asarray(dfi[['Survived']])


# In[ ]:


X1 = np.asarray(df1.drop(['Survived'], axis = 1))
y1 = np.asarray(df1[['Survived']])


# In[ ]:


X = preprocessing.StandardScaler().fit(X).transform(X)


# In[ ]:


X1 = preprocessing.StandardScaler().fit(X1).transform(X1)


# In[ ]:


print(X.size/10)
print(y.size)
print(X1.size/7)
print(y1.size)


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics


# In[ ]:


from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
scores_GNB = cross_val_score(GNB, X, y.ravel(), cv=100, scoring = "accuracy")


# In[ ]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(solver = 'lbfgs', max_iter=1000)
scores_LR = cross_val_score(LR, X, y.ravel(), cv=100, scoring = "accuracy")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100, random_state=0)
scores_RF = cross_val_score(RF, X1, y1.ravel(), cv=100, scoring = "accuracy")


# In[ ]:


from sklearn.svm import SVC 
VM = SVC(C = 0.01, kernel = 'rbf', gamma = 'scale', class_weight = 'balanced')
scores_SVC = cross_val_score(VM, X1, y1.ravel(), cv=100, scoring = "accuracy")


# In[ ]:


# in this case the optimal N corresponds to the target variable ('Survived') outcomes.
# No need to run the elbow test.
from sklearn.neighbors import KNeighborsClassifier
NGH = KNeighborsClassifier(n_neighbors = 2)
scores_KNN = cross_val_score(NGH, X, y.ravel(), cv=100, scoring = "accuracy")


# In[ ]:


from xgboost import XGBClassifier
BST = XGBClassifier()
scores_XGB = cross_val_score(BST, X, y.ravel(), cv=10, scoring = "accuracy")


# ### Summary of accuracy scores on the train set.

# In[ ]:


result_df = pd.DataFrame(columns=['Accuracy','Variance'], index=['Naive Bayes', 'Logistic Regression', 'Random Forest', 'SVC', 'KNN', 'XGBoost'])


# In[ ]:


result_df.iloc[0] = pd.Series({'Accuracy':scores_GNB.mean(),
                               'Variance':np.std(scores_GNB)})
result_df.iloc[1] = pd.Series({'Accuracy':scores_LR.mean(),
                               'Variance':np.std(scores_LR)})
result_df.iloc[2] = pd.Series({'Accuracy':scores_RF.mean(),
                               'Variance':np.std(scores_RF)})
result_df.iloc[3] = pd.Series({'Accuracy':scores_SVC.mean(),
                               'Variance':np.std(scores_SVC)})
result_df.iloc[4] = pd.Series({'Accuracy':scores_KNN.mean(),
                               'Variance':np.std(scores_KNN)})
result_df.iloc[5] = pd.Series({'Accuracy':scores_XGB.mean(),
                               'Variance':np.std(scores_XGB)})


# In[ ]:


result_df = result_df.sort_values('Accuracy', ascending = False,  axis=0)
result_df.head(6)


# The best accuracy score (0.840) is provided by Random Trees algorithm with 0.13 variance.<br>
# XGBoost results are slightly lower (0.825) but with variance of only 0.03.

# ### Conclusion: Select XGBoost as the winning model

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y.ravel(), test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[ ]:


# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
cls = XGBClassifier()
cls.fit(X_train, y_train)


# ## Predicting outcomes from the test set

# ### Feature engineering

# In[ ]:


file3 = '../input/test.csv'
dft = pd.read_csv(file3)


# In[ ]:


dft = dft.drop(['PassengerId', 'Name', 'Ticket'], axis=1)


# In[ ]:


dft.head()


# In[ ]:


dft_hot = pd.concat([dft, pd.get_dummies(dft['Sex'])], axis=1)
dft_hot = dft_hot.drop(['Sex'], axis=1)


# In[ ]:


dft_hot = pd.concat([dft_hot, pd.get_dummies(dft['Pclass'])], axis=1)
dft_hot = dft_hot.drop(['Pclass'], axis=1)


# In[ ]:


dft_hot = dft_hot.rename(columns={dft_hot.columns[8]:'Pclass1', dft_hot.columns[9]:'Pclass2', dft_hot.columns[10]:'Pclass3'})


# In[ ]:


dft_hot.head()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
T = dft_hot['Embarked']
encoded = le.fit_transform(np.ravel(T))


# In[ ]:


dft_hot = pd.concat([dft_hot, pd.DataFrame(encoded, columns = ['Emb'])], axis=1)
dft_hot = dft_hot.drop(['Embarked'], axis=1)


# In[ ]:


dft_hot = dft_hot[['SibSp', 'Parch', 'Fare', 'Emb', 'Cabin', 'Age',
       'female', 'male', 'Pclass1', 'Pclass2', 'Pclass3']]


# In[ ]:


dft_hot = dft_hot.drop(['Cabin'], axis=1)


# In[ ]:


dft_hot['Fare'].fillna((dft_hot['Fare'].mean()), inplace=True)


# In[ ]:


from fancyimpute import KNN 
#We use the train dataframe from Titanic dataset
#fancy impute removes column names.

train_cols = list(dft_hot)


# In[ ]:


# Use 5 nearest rows which have a feature to fill in each row's
# missing features
dft_i = pd.DataFrame(KNN(k=5).fit_transform(dft_hot))
dft_i.columns = train_cols


# ### Predicting outcomes 

# In[ ]:


from sklearn import preprocessing
X_test = np.asarray(dft_i)


# In[ ]:


X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)


# In[ ]:


print(X_test.size/10)


# In[ ]:


# Predicting the Test set results with XGBoost model
y_pred = cls.predict(X_test)


# In[ ]:


dft2 = pd.read_csv(file3)


# In[ ]:


dft2 = pd.concat([dft2['PassengerId'], pd.DataFrame(y_pred, columns = ['Survived']).astype(int)], axis=1)


# In[ ]:


dft2.head()


# In[ ]:


dft2.to_csv('titanic_results.csv', index=False)

