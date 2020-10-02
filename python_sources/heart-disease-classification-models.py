#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


heart = pd.read_csv('../input/heart-disease-uci/heart.csv')


# Attribute Information:
# > 1. age
# > 2. sex (1 = male; 0 = female)
# > 3. chest pain type (4 values)
# > 4. resting blood pressure
# > 5. serum cholestoral in mg/dl
# > 6. fasting blood sugar > 120 mg/dl
# > 7. resting electrocardiographic results (values 0,1,2)
# > 8. maximum heart rate achieved
# > 9. exercise induced angina
# > 10. oldpeak = ST depression induced by exercise relative to rest
# > 11. the slope of the peak exercise ST segment
# > 12. number of major vessels (0-3) colored by flourosopy
# > 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# > 14. target (1=yes, 0=no)

# # Data exploration

# In[ ]:


heart.info()


# In[ ]:


heart.describe()


# In[ ]:


heart.sample(5)


# Check for any null values

# In[ ]:


heart.isnull().sum()


# In[ ]:


heart.isnull().any(axis=1).sum()


# In[ ]:


heart.groupby('target').mean()


# In[ ]:


plt.figure(figsize=(15,10))
sns.heatmap(heart.corr(), cmap='Spectral', annot=True)
plt.show()


# Difference of correlation coefficients in Male and Female

# In[ ]:


abs(heart[heart['sex']==0].corr()['target'].drop(labels=['sex'])) - abs(heart[heart['sex']==1].corr()['target'].drop(labels=['sex']))


# In[ ]:


fig, ax = plt.subplots(figsize=(15,7))
ax.plot(heart[heart['sex']==0].corr()['target'].drop(labels=['sex', 'target']), label='Female')
ax.plot(heart[heart['sex']==1].corr()['target'].drop(labels=['sex', 'target']), label='Male')
ax.plot(heart.corr()['target'].drop(labels=['sex', 'target']), label='Both')
plt.title('Correlation of heart disease with various parameters')
plt.legend()


# ## Observations:
# 1. Age is negatively correlated with heart disease. As older people are more likely to get heart disease, they are likely to go for health check-up even when they have mild or no symptoms. Young people only go for health check-up when they have clear symptoms, so they are more likely to be diagnosed with having heart disease.
# 2. cholesterol level, fasting blood glucose have negligible correlation with heart disease.
# 3. Chest pain(cp), maximum heart rate(thalach), slope of ST segment in ECG are positively correlated with heart disease.
# 4. exercise induced angina(exang), oldpeak(ST depression induced by exercise), number of major vessels (0-3) colored by flourosopy(ca), thal are negatively correlated with heart disease. In all of these correlation is less for Males than Females.
# 5. trestbps and fbs are negatively correlated for females compared to Males.

# In[ ]:


heart.groupby(['sex','target']).count()


# In[ ]:


heart['age range'] = pd.cut(heart['age'], bins=[0, 40, 50, 60, 70, 100])


# In[ ]:


sns.countplot(heart['age range'], hue='target', data=heart)


# In[ ]:


heart.groupby(['age range', 'sex', 'target'])['age'].count()


# In[ ]:


heart.groupby(['age range', 'target', 'sex'])['age'].count()


# Females are more likely to be diagnosed with heart disease in all age groups.

# In[ ]:


categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']


# In[ ]:


numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


# In[ ]:


for i in numerical_features:
    g = sns.FacetGrid(heart, col='sex', hue='target', height=5)
    g.map(sns.distplot, i)


# In[ ]:


fig, axes = plt.subplots(2, 4, figsize=(16,8))
for i, ax in enumerate(axes.ravel()):
    sns.countplot(heart[categorical_features[i]], ax=ax, hue=heart['target'])
    ax.set_xlabel(categorical_features[i])
plt.tight_layout()


# In[ ]:


pp = numerical_features
pp.append('target')
pp


# In[ ]:


sns.pairplot(heart.loc[:, pp], hue='target')


# # Logistic Regression

# In[ ]:


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score


# In[ ]:


heart.head(3)


# In[ ]:


score_mean = {}
score_max = {}


# In[ ]:


def process(dataframe, rand):
    y = dataframe['target']
    X = dataframe.drop(['target', 'age range'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rand)
    mct = make_column_transformer(
            (OneHotEncoder(categories='auto', handle_unknown='ignore',sparse=False), ['cp', 'slope', 'thal']), 
            remainder=MinMaxScaler())
    X_train = mct.fit_transform(X_train)
    X_test = mct.transform(X_test)
    return X_train, X_test, y_train, y_test


# In[ ]:


def regression(dataframe, rand):
    y = dataframe['target']
    X = dataframe.drop(['target', 'age range'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = rand)
    mct = make_column_transformer(
            (OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'), ['cp', 'slope', 'thal']), 
            remainder=MinMaxScaler())
    X_train = mct.fit_transform(X_train)
    logreg = LogisticRegression(solver='liblinear')
    logreg.fit(X_train, y_train)
    X_test = mct.transform(X_test)
    return logreg, X_test, y_test


# Regression score is varying with random state chosen for splitting data.

# In[ ]:


scores = []
for i in range(0, 200):
    logreg, X_test, y_test = regression(heart, i)
    scores.append(logreg.score(X_test, y_test))


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(scores)
plt.xlabel('random state')
plt.ylabel('regression score')


# In[ ]:


np.array(scores).mean()


# In[ ]:


score_mean['Logistic Regression'] = np.round(np.array(scores).mean(), 2)


# Average score for random states between 0 and 200 is 0.84

# In[ ]:


logreg, X_test, y_test = regression(heart, 153)


# In[ ]:


logreg.score(X_test, y_test)


# In[ ]:


score_max['Logistic Regression'] = np.round(logreg.score(X_test, y_test), 2)


# In[ ]:


predictions = logreg.predict(X_test)


# In[ ]:


confusion_matrix(y_test, predictions)


# In[ ]:


print(classification_report(y_test, predictions))


# In[ ]:


prob = logreg.predict_proba(X_test)


# In[ ]:


roc_score = roc_auc_score(y_test, predictions)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test, prob[:, 1])


# In[ ]:


plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % roc_score)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")


# Heat map of probabilities for each case.

# In[ ]:


sns.heatmap(prob[np.argsort(prob[:, 0])])


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(prob[np.argsort(prob[:, 0])])
plt.xlabel('test case number')
plt.ylabel('probability')
plt.legend(['disease', 'no disease'])


# ## Running the model separately for females and males

# In[ ]:


heart_f = heart[heart['sex']==0].drop(['sex'], axis=1)


# In[ ]:


logreg_f, X_test, y_test = regression(heart_f, 0)


# In[ ]:


logreg_f.score(X_test, y_test)


# In[ ]:


heart_m = heart[heart['sex']==1].drop(['sex'], axis=1)


# In[ ]:


logreg_m, X_test, y_test = regression(heart_m, 33)


# In[ ]:


logreg_m.score(X_test, y_test)


# Regression model score is higher for females than males probably because correlation is stronger.

# In[ ]:


scores = []
for i in range(0, 200):
    logreg, X_test, y_test = regression(heart_f, i)
    scores.append(logreg.score(X_test, y_test))


# In[ ]:


np.array(scores).mean()


# Average score for random states between 0 and 200 is 0.89 for females.

# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(scores)
plt.xlabel('random state')
plt.ylabel('regression score')


# In[ ]:


scores = []
for i in range(0, 200):
    logreg, X_test, y_test = regression(heart_m, i)
    scores.append(logreg.score(X_test, y_test))


# In[ ]:


np.array(scores).mean()


# Average score for random states between 0 and 200 is 0.79 for females.

# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(scores)
plt.xlabel('random state')
plt.ylabel('regression score')


# # K Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


max_score = []
for i in range(200):
    X_train, X_test, y_train, y_test = process(heart, i)
    score_list = []
    for i in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_train, y_train)
        score_list.append(knn.score(X_test, y_test))
    max_score.append(max(score_list))


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(max_score)


# In[ ]:


np.array(max_score).mean()


# In[ ]:


score_mean['K Nearest Neighbors'] = np.round(np.array(max_score).mean(), 2)


# Average score for random states between 0 and 200 is 0.85

# In[ ]:


X_train, X_test, y_train, y_test = process(heart, 153)


# In[ ]:


score_list = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train, y_train)
    score_list.append(knn.score(X_test, y_test))


# In[ ]:


max(score_list)


# In[ ]:


plt.plot(range(1,20), score_list)
plt.xticks(range(1,20))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()


# Highest score is achieved with neighbors = 12

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 12)


# In[ ]:


knn.fit(X_train, y_train)


# In[ ]:


predictions = knn.predict(X_test)


# In[ ]:


knn.score(X_test, y_test)


# In[ ]:


score_max['K Nearest Neighbors'] = np.round(knn.score(X_test, y_test), 2)


# # Support Vector Machines

# In[ ]:


score_mean


# In[ ]:


from sklearn.svm import SVC


# In[ ]:


score_list = []
for i in range(200):
    X_train, X_test, y_train, y_test = process(heart, i)
    svm = SVC(1, gamma='scale')
    svm.fit(X_train, y_train)
    score_list.append(svm.score(X_test, y_test))


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(score_list)


# In[ ]:


np.array(score_list).mean()


# In[ ]:


score_mean['Support Vector Machines'] = np.round(np.array(score_list).mean(), 2)


# Average score for random states between 0 and 200 is 0.83

# In[ ]:


X_train, X_test, y_train, y_test = process(heart, 153)


# In[ ]:


svm = SVC(1, gamma='scale')


# In[ ]:


svm.fit(X_train, y_train)


# In[ ]:


svm.score(X_test, y_test)


# In[ ]:


score_max['Support Vector Machines'] = np.round(svm.score(X_test, y_test), 2)


# # Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


score_list = []
for i in range(200):
    X_train, X_test, y_train, y_test = process(heart, i)
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    score_list.append(dtc.score(X_test, y_test))


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(score_list)


# In[ ]:


np.array(score_list).mean()


# In[ ]:


score_mean['Decision Tree Classifier'] = np.round(np.array(score_list).mean(), 2)


# Average score for random states between 0 and 200 is 0.74

# In[ ]:


X_train, X_test, y_train, y_test = process(heart, 5)


# In[ ]:


dtc = DecisionTreeClassifier()


# In[ ]:


dtc.fit(X_train, y_train)


# In[ ]:


dtc.score(X_test, y_test)


# In[ ]:


score_max['Decision Tree Classifier'] = np.round(dtc.score(X_test, y_test), 2)


# # Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


score_list = []
for i in range(200):
    X_train, X_test, y_train, y_test = process(heart, i)
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train)
    score_list.append(rfc.score(X_test, y_test))


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(score_list)


# In[ ]:


np.array(score_list).mean()


# In[ ]:


score_mean['Random Forest Classifier'] = np.round(np.array(score_list).mean(), 2)


# Average score for random states between 0 and 200 is 0.82

# In[ ]:


X_train, X_test, y_train, y_test = process(heart, 153)


# In[ ]:


rfc = RandomForestClassifier(n_estimators=100)


# In[ ]:


rfc.fit(X_train, y_train)


# In[ ]:


rfc.score(X_test, y_test)


# In[ ]:


score_max['Random Forest Classifier'] = np.round(rfc.score(X_test, y_test), 2)


# # Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


score_list = []
for i in range(200):
    X_train, X_test, y_train, y_test = process(heart, i)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    score_list.append(gnb.score(X_test, y_test))


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(score_list)


# In[ ]:


np.array(score_list).mean()


# In[ ]:


score_mean['Naive Bayes'] = np.round(np.array(score_list).mean(), 2)


# Average score for random states between 0 and 200 is 0.79

# In[ ]:


X_train, X_test, y_train, y_test = process(heart, 153)


# In[ ]:


gnb = GaussianNB()


# In[ ]:


gnb.fit(X_train, y_train)


# In[ ]:


gnb.score(X_test, y_test)


# In[ ]:


score_max['Naive Bayes'] = np.round(gnb.score(X_test, y_test), 2)


# # Neural Network

# In[ ]:


from sklearn.neural_network import MLPClassifier


# In[ ]:


# To filter ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
import warnings
warnings.filterwarnings('ignore') 


# In[ ]:


score_list = []
for i in range(200):
    X_train, X_test, y_train, y_test = process(heart, i)
    mlp = MLPClassifier(10, max_iter=200)
    mlp.fit(X_train, y_train)
    score_list.append(mlp.score(X_test, y_test))


# In[ ]:


plt.figure(figsize=(15,5))
plt.plot(score_list)


# In[ ]:


np.array(score_list).mean()


# In[ ]:


score_mean['Neural Network'] = np.round(np.array(score_list).mean(), 2)


# Average score for random states between 0 and 200 is 0.83

# In[ ]:


X_train, X_test, y_train, y_test = process(heart, 153)


# In[ ]:


mlp = MLPClassifier(10, max_iter=200)


# In[ ]:


mlp.fit(X_train, y_train)


# In[ ]:


mlp.score(X_test, y_test)


# In[ ]:


score_max['Neural Network'] = np.round(mlp.score(X_test, y_test), 2)


# # Comparison of Classifiers

# In[ ]:


score_max


# In[ ]:


score_mean


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(list(score_mean.keys()), list(score_mean.values()), 'b-o', label = 'mean score')
plt.plot(list(score_max.keys()), list(score_max.values()), 'r-*', label = 'max score')
for i, v in enumerate(score_mean.values()):
    plt.text(i, v+.01, v)
for i, v in enumerate(score_max.values()):
    plt.text(i, v+.01, v)
plt.xticks(rotation=45)
plt.ylim(0.7, 1)
plt.legend()

