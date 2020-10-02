#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


print(os.listdir('../input'))


# In[ ]:


df = pd.read_csv('../input/heart.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# UCI dataset archives has additional information on each of the attributes. Some other information could be gathered through some google searches.
# Here's a quick run through of what each column mean:
# 1.  age - how old the person is.
# 2.  sex - observation's gender. 1 = male, 0 = female.
# 3.  cp - chest pain type. 1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 - asymptomatic .
#     - angina - chest pain that happens when the heart doesn't get enough blood or O2. Provoked by extertion or stress, and can be relieved with rest.
#     - atypical angina - Basically angina that doesn't completely fit the bill of typical angina.
#     - non-anginal - chest pain that is completely unrelated to angina symptoms and causes.
#     - asymptomatic- pain that shows zero symptoms.
# 4. trestbps - resting blood pressure. Info was taken during admission to hospital. Too high could be dangerous, too low as well.
# 5. chol - serum cholestoral. Higher cholesterol, higher chance of heart disease.
# 6. fbs - fasting blood sugar > 1 120 mg/dl. 1 = True, 0 = false. True is a sign of prediabetes, which could lead to HD.
# 7. restecg - resting electrocardiographic results
#     - 0 - normal.
#     - 1 - Wave abnormalities - Can be considered a safe variation of a heart's rhythm or a sign of heart disease.
#     - 2 - Probable or definite left ventricular hypertrophy. A result of high blood pressure or some heart condition so we can expect this to mean heart disease.
# 8. thalach - maximum heart rate achieved.
# 9. exang - excercise induced angina. Binary values.
# 10. oldpeak - ST depression induced by exercise relative to rest. ST depression is  sign of myocardial ischemia.
# 11. Slope - slope of peak exercise ST segment.
#     - 1 - up
#     - 2 - flat
#     - 3 - down
# 12. ca - # of major vessels (0 - 3) colored by flourscopy. If a vessel is not colored, that means there's blockage in the vessel.
# 13. thal - 3 = normal, 6 = fixed defect, 7 = reversable defect. lower is better presumably. NOTE: This information might be a typo. I only found 4 unique variables in this column (0,1,2,3)
# 
# We're going to do some EDA to get another perspective on the data.

# In[ ]:


plt.figure(figsize=(15,7))

corr = df.corr()
sns.heatmap(corr, annot=True )


# A correlation map of all the variables paired with each other is a good way to start making assumptions.

# In[ ]:


sns.set_style('whitegrid')

hd = df[df['target'] == 1]['age']
no_hd = df[df['target'] == 0]['age']

sns.distplot(hd, kde=False, label='Heart Disease')
sns.distplot(no_hd, kde=False, label='None')

plt.title('Age: Heart Disease VS None')
plt.legend()


# It seems like heart disease claims the patients who were young and do not take care of themselves well.

# In[ ]:


g = sns.countplot(x='sex', hue='target', data=df)

g.set(xticklabels=['Female', 'Male'])

plt.legend(['None', 'Heart Disease'])
plt.title('Sex: Heart Disease VS None')


# It is commonly believed that men are more likely to develop heart disease. Here, the data shows that women are more likely to have heart disease compared to men relatively speaking. This may be due to the assumption of most men admitted to care we're believed to have heart disease compared to women.

# In[ ]:


g = sns.violinplot(x='exang', y='slope', data=df)
g.set(xticklabels=['No', 'Yes'])
plt.show()


# For those who do not have exercise triggered angina, they will usually have a flat st segment as opposed to those who do have exercise triggered angina (up sloping st segment).

# In[ ]:


g = sns.countplot(x='fbs', hue='target', data=df)
plt.title('Heart Disease: Low FBS vs High FBS')
plt.legend(labels=['None', 'Heart Disease'])
g.set(xticklabels=['< 120', '> 120'])
plt.show()


# It appears that being prediabetic doesn't affect the rate of heart disease. That being said, a high fasting blood sugar still comes with tons of health problems.

# In[ ]:


plt.figure(figsize=(16, 4))

sns.lineplot(x='age', y='thalach', hue='target', data=df)


# In[ ]:


plt.figure(figsize=(16, 4))

sns.lineplot(x='age', y='trestbps', hue='target', data=df)


# In[ ]:


plt.figure(figsize=(16, 4))

sns.lineplot(x='age', y='chol', hue='target', data=df)


# In[ ]:


plt.figure(figsize=(16, 4))


g = sns.barplot(x='cp', y='thalach', hue='target', data=df)

g.set(xticklabels=['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])

plt.show()


# In[ ]:


sns.barplot(x='exang', y='thalach', hue='target', data=df)


# In[ ]:


plt.figure(figsize=(16, 4))

sns.lineplot(x='oldpeak', y='thalach', hue='target', data=df)


# In[ ]:


g = sns.countplot(x='slope', hue='target',data=df)


g.set(xticklabels=['Up', 'Flat', 'Down',])


# In[ ]:


sns.barplot(x='target', y='thalach', data=df)


# In[ ]:


plt.figure(figsize=(16, 4))

g = sns.countplot(x='cp', hue='exang', data=df)

g.set(xticklabels=['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])

plt.show()


# In[ ]:


plt.figure(figsize=(16, 4))

g = sns.countplot(x='cp', hue='target', data=df)

g.set(xticklabels=['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])

plt.show()


# In[ ]:


sns.boxplot(x='exang', y='oldpeak', hue='target', data=df)


# In[ ]:


g = sns.countplot(x='slope', hue='exang',data=df)


g.set(xticklabels=['Up', 'Flat', 'Down',])


# In[ ]:


sns.countplot(x='thal', hue='target', data=df)


# In[ ]:


sns.countplot(x='ca', hue='target', data=df)


# In[ ]:


sns.violinplot(x='ca', y='age', data=df)


# In[ ]:


cp = pd.get_dummies(df['cp'], prefix='cp')
restecg = pd.get_dummies(df['restecg'], prefix='restecg')
slope = pd.get_dummies(df['slope'], prefix='slope')


# In[ ]:


df = df.drop(['cp','restecg', 'slope'], axis=1)


# In[ ]:


df = pd.concat([df, cp, restecg, slope], axis=1)


# In[ ]:


df.columns


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis=1), 
                                                    df['target'], test_size=0.30, 
                                                    random_state=101)


# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)


# In[ ]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# In[ ]:


kn = KNeighborsClassifier()


# In[ ]:


error_rate = []

for i in range(1,30):
    
    kn = KNeighborsClassifier(n_neighbors=i)
    kn.fit(X_train,y_train)
    pred_i = kn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(range(0, len(error_rate)),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[ ]:


kn = KNeighborsClassifier(n_neighbors=23)
kn.fit(X_test, y_test)
kn_pred = kn.predict(X_test)


# In[ ]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)


# In[ ]:


sv = SVC()
sv.fit(X_train, y_train)
sv_pred = sv.predict(X_test)


# In[ ]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[ ]:


grid.fit(X_train, y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# In[ ]:


grid_pred = grid.predict(X_test)


# In[ ]:


models = {'Logistic Regression': lr_pred, 'Random Forest': rf_pred, 'K-Nearest': kn_pred, 'Decision Tree': dt_pred, 'Support vector classifier': grid_pred}


# In[ ]:


for pred in models:
    print('\n', '-'*50,'\n',pred, '\n',  '-'*50,'\n')
    print(classification_report(y_test, models[pred]))


# In[ ]:


models_avg = {}

for model in models:
    models_avg[model] = np.mean(models[model] == y_test)
    
models_avg = pd.Series(models_avg)


# In[ ]:


plt.figure(figsize=(8, 5))

models_avg.plot(kind='bar', yticks=np.arange(0, 1, .05))


# In[ ]:




