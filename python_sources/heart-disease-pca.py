#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report


# # Dataset

# **Columns**
# 
# 1. **Age** - Age of patient
# 2. **Sex** - Sex of patient
# 3. **Chest Pain type** - chest pain type (4 values)
# 4. **Resting Blood pressure** - resting blood pressure
# 5. **Serum Cholestoral in mg/dl** - serum cholestoral in mg/dl
# 6. **Fasting Blood sugar > 120 mg/dl** - fasting blood sugar > 120 mg/dl
# 7. **Resting Electrocardiographic results** - resting electrocardiographic results (values 0,1,2)
# 8. **Max heart rate achieved** - maximum heart rate achieved
# 9. **Exercise induced angina** - exercise induced angina
# 10. **Oldpeak** - oldpeak = ST depression induced by exercise relative to rest
# 11. **Slope of the peak exercise ST segment** - the slope of the peak exercise ST segment
# 12. **No of major vessels** - number of major vessels (0-3) colored by flourosopy
# 13. **Thal** - thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
# 14. **Presence** - Target :Absence (1) or presence (2) of heart disease

# In[ ]:


data_path = '/kaggle/input/heart-disease-uci-dataset/heart.csv'

df = pd.read_csv(data_path)
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.columns


# In[ ]:


new_columns = [col.replace(' ', '_') for col in df.columns]
df.columns = new_columns
new_columns


# # Visualization

# ## Correlation

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), cmap="YlGnBu");


# ## Heart disease

# In[ ]:


df_presence = df[df.Presence == 2]
df_absence = df[df.Presence == 1]

df_sex_0 = df[df.Sex == 0]
df_sex_1 = df[df.Sex == 1]


# ## Age distribution of the presence of heart disease by gender

# In[ ]:


sex_0 = df_presence[df_presence.Sex == 0]
sex_1 = df_presence[df_presence.Sex == 1]
plt.figure(figsize=(15,5))
plt.title("Age distribution of the presence of heart disease by gender")
sns.kdeplot(data=sex_0['Age'], shade=True, label='Sex_0');
sns.kdeplot(data=sex_1['Age'], shade=True, label='Sex_1');


# ## Sex

# In[ ]:


plt.subplots(figsize=(15, 5))
plt.title('Female and man with heart disease')
sns.countplot(y="Sex", data=df_presence, color="c");


# ## Chest pain type

# In[ ]:


plt.subplots(figsize=(15, 5))
sns.countplot(y="Chest_Pain_type", data=df_presence, color="c");


# In[ ]:


pain_1 = df_presence[df_presence.Chest_Pain_type == 1]
pain_2 = df_presence[df_presence.Chest_Pain_type == 2]
pain_3 = df_presence[df_presence.Chest_Pain_type == 3]
pain_4 = df_presence[df_presence.Chest_Pain_type == 4]

plt.figure(figsize=(15,5))
plt.title("Chest pain distribution of the presence of heart disease by gender")
sns.kdeplot(data=pain_1['Age'], shade=True, label='Pain 1');
sns.kdeplot(data=pain_2['Age'], shade=True, label='Pain 2');
sns.kdeplot(data=pain_3['Age'], shade=True, label='Pain 3');
sns.kdeplot(data=pain_4['Age'], shade=True, label='Pain 3');


# ## Mean resting blood pressure for sex 0

# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Mean resting blood pressure for sex 0")
plt.xlabel('Age')
plt.ylabel('Resting blood pressure')
data = df[df.Sex == 0]

sns.lineplot(
    data=data[data.Presence == 2].groupby(['Age'])['Resting_Blood_pressure'].mean(), 
    label='Presence'
);

sns.lineplot(
    data=data[data.Presence == 1].groupby(['Age'])['Resting_Blood_pressure'].mean(), 
    label='Absence'
);


# ## Mean resting blood pressure for sex 1

# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Mean resting blood pressure for sex 1")
plt.xlabel('Age')
plt.ylabel('Resting blood pressure')
data = df[df.Sex == 1]

sns.lineplot(
    data=data[data.Presence == 2].groupby(['Age'])['Resting_Blood_pressure'].mean(), 
    label='Presence'
);

sns.lineplot(
    data=data[data.Presence == 1].groupby(['Age'])['Resting_Blood_pressure'].mean(), 
    label='Absence'
);


# ## Serum cholestoral in mg/dl

# In[ ]:


plt.figure(figsize=(15, 10))
sns.scatterplot(x="Age", y="Serum_Cholestoral_in_mg/dl", hue="Presence", palette="Set1", data=df);


# # PCA - Principal component analysis

# In[ ]:


num_features = len(df.columns) - 1
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


# In[ ]:


cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('Eigen vals: {}'.format(eigen_vals))


# In[ ]:


tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.figure(figsize=(15,5))
plt.bar(range(num_features), var_exp, alpha=.5, align='center', label='Single variance')
plt.step(range(num_features), cum_var_exp, where='mid', label='Cumsum variance')
plt.ylabel('Factor of variance')
plt.xlabel('Main components')
plt.legend(loc='best')
plt.show()


# In[ ]:


eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key = lambda k: k[0], reverse=True)
W = np.hstack((eigen_pairs[0][1][:, np.newaxis], 
               eigen_pairs[1][1][:, np.newaxis]))
print('Matrix W: \n{}'.format(W))


# In[ ]:


X_train_pca = X_train_std.dot(W)
X_test_pca = X_test_std.dot(W)


# # Models

# In[ ]:


def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print('Report:\n{}'.format(classification_report(y_test, y_pred)))
    print('Score: {}'.format(model.score(X_test, y_test)))


# ## Logistic regression

# In[ ]:


lr = LogisticRegression()
run_model(lr, X_train=X_train_pca, y_train=y_train, X_test=X_test_pca, y_test=y_test)


# ## SVC

# In[ ]:


svc = SVC(kernel='linear', C=1.0, random_state=0)
run_model(svc, X_train=X_train_pca, y_train=y_train, X_test=X_test_pca, y_test=y_test)


# ### K-neighbors

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=20, p=2, metric='minkowski')
run_model(knn, X_train=X_train_pca, y_train=y_train, X_test=X_test_pca, y_test=y_test)


# # tbc ...

# In[ ]:




