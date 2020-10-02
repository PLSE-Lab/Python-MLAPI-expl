#!/usr/bin/env python
# coding: utf-8

# <h1>Item Purchased or Not</h1>

# In[3]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "whitegrid", color_codes=True)


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


dataset = pd.read_csv('../input/Social_Network_Ads.csv')
col = ['User ID']
dataset = dataset.drop(col, axis = 1)
dataset.describe()


# In[7]:


dataset.head()


# <h2>Data Visualizations</h2>

# <h4>Relation b/w Salary and Purchased</h4>

# In[8]:


plot1 = sns.swarmplot(x ='Purchased', y='EstimatedSalary', hue='Gender', data = dataset)


# In[9]:


sns.violinplot(x="Purchased", y="EstimatedSalary", data=dataset, inner=None)
plot5 = sns.swarmplot(x="Purchased",y="EstimatedSalary", data=dataset, color="w", alpha=0.5)


# <h4>Relation b/w Age and Purchased</h4>

# In[10]:


plot2 = sns.swarmplot(x ='Purchased', y='Age', hue='Gender', data = dataset)


# <h4>Using a boxplot</h4>

# In[11]:


plot4 = sns.violinplot(x="Purchased", y="Age", hue="Gender", data = dataset, split = True, inner = "stick", palette="Set3")


# <h2>Data Preprocessing</h2>

# <h4>Splitting the dataset</h4>

# In[12]:


x = dataset.iloc[:, [1,2]].values
y = dataset.iloc[:,3].values


# In[13]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# <h4>Feature Scaling</h4>

# In[14]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# <h2>Fitting K-NN to the Training Set</h2>

# In[16]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)


# <h4>Predicting test set results</h4>

# In[18]:


y_pred = classifier.predict(x_test)


# <h4>Confusion Matrix to check number of correct/incorrect predictions</h4>

# In[19]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# <h2>Visualizing Results</h2>

# <h4>Visualizing Training Set Results</h4>

# In[20]:


from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# <h4>Visualizing Test Set Results</h4>

# In[22]:


from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('pink', 'lightgreen')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# <h3>That's it!</h3>
