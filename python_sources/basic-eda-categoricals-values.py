#!/usr/bin/env python
# coding: utf-8

# <a id='top'></a>
# <h1 style="text-align:center;font-size:200%;;">Categorical Feature Encoding Challenge II</h1>
# <img src="https://miro.medium.com/max/1200/1*TBSV23ud8tae3E4szI5EDA.jpeg">

# ## Competition Description
# This follow-up competition offers an even more challenging dataset so that you can continue to build your skills with the common machine learning task of encoding categorical variables. This challenge adds the additional complexity of feature interactions, as well as missing data.
# 
# This Playground competition will give you the opportunity to try different encoding schemes for different algorithms to compare how they perform. We encourage you to share what you find with the community.

# <div class="list-group" id="list-tab" role="tablist">
#   <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Notebook Content!</h3>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#libraries" role="tab" aria-controls="profile">Import Libraries<span class="badge badge-primary badge-pill">1</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#load" role="tab" aria-controls="messages">Load Data<span class="badge badge-primary badge-pill">2</span></a>
#   <a class="list-group-item list-group-item-action"  data-toggle="list" href="#visual" role="tab" aria-controls="settings">Visualization of data<span class="badge badge-primary badge-pill">3</span></a>
#   <a class="list-group-item list-group-item-action" data-toggle="list" href="#separate" role="tab" aria-controls="settings">Separate continuous, categorical and label column names<span class="badge badge-primary badge-pill">4</span></a> 
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#split" role="tab" aria-controls="settings">Train and test Split<span class="badge badge-primary badge-pill">5</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#model" role="tab" aria-controls="settings"> Creating the Model<span class="badge badge-primary badge-pill">6</span></a>
#     <a class="list-group-item list-group-item-action" data-toggle="list" href="#eval" role="tab" aria-controls="settings">Model Evaluation<span class="badge badge-primary badge-pill">7</span></a>
#      <a class="list-group-item list-group-item-action" data-toggle="list" href="#reference" role="tab" aria-controls="settings">References<span class="badge badge-primary badge-pill">8</span></a>  

# <a id='libraries'></a>
# ## 1. Import Libraries

# In[ ]:


# Operating system dependent
import os

# linear algebra
import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Display HTML
from IPython.core.display import display, HTML

#collection of machine learning algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#Common Model Helpers
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import model_selection
import pylab as pl
from sklearn.metrics import roc_curve
from sklearn.preprocessing import Imputer

import plotly.graph_objects as go
#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import cufflinks as cf
cf.go_offline()

#%matplotlib inline = show plots in Jupyter Notebook browser
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='load'></a>
# # 2. Load Data

# In[ ]:


train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")
test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")
submission =  pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")


# In[ ]:


train_size = str(round(os.path.getsize('/kaggle/input/cat-in-the-dat-ii/train.csv') / 1000000, 2)) + 'MB'
test_size = str(round(os.path.getsize('/kaggle/input/cat-in-the-dat-ii/test.csv') / 1000000, 2)) + 'MB'
sample_submission_size = str(round(os.path.getsize('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv') / 1000000, 2)) + 'MB'


# In[ ]:


display(HTML(f"""

    
    <table style="width:40%;font-size:100%;">
      <tr>
        <th>Filename</th>
        <th>size</th>
      </tr>
      <tr>
        <td>Train</td>
        <td>{train_size}</td>
      </tr>
      <tr>
        <td>Test</td>
        <td>{test_size}</td>
      </tr>
       <tr>
        <td>Sample_Submission</td>
        <td>{sample_submission_size}</td>
      </tr>
    </table>
    
  
"""))


# In[ ]:


train.head()


# In[ ]:


train.columns


# <ul style="list-style-type:square;">
#    <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Data Colums Description!</h3>
#   <li><span class="label label-default">id</span> a unique identifier for each tweet</li>
#   <li><span class="label label-default">bin_* </span> The data contains binary features </li>
#   <li><span class="label label-default">nom_*</span>  Nominal features </li>
#     <li><span class="label label-default">ord_*</span>  Ordinal features</li>
#     <li><span class="label label-default">ord_{3-5}</span>  String ordinal features,  are lexically ordered according to string.ascii_letters.</li>
#     <li><span class="label label-default">day</span>  Day of the week features</li>
#     <li><span class="label label-default">month</span>  Month Feachures</li>
#     <li><span class="label label-default">target</span> you will be predicting the probability [0, 1] of a binary target column.</li>
# </ul>
# 

# In[ ]:


display(HTML(f"""
   
        <ul class="list-group">
          <li class="list-group-item disabled" aria-disabled="true"><h4>Shape of Train and Test Dataset</h4></li>
          <li class="list-group-item"><h4>Number of rows in Train dataset is: <span class="label label-primary">{ train.shape[0]:,}</span></h4></li>
          <li class="list-group-item"> <h4>Number of columns Train dataset is <span class="label label-primary">{train.shape[1]}</span></h4></li>
          <li class="list-group-item"><h4>Number of rows in Test dataset is: <span class="label label-success">{ test.shape[0]:,}</span></h4></li>
          <li class="list-group-item"><h4>Number of columns Test dataset is <span class="label label-success">{test.shape[1]}</span></h4></li>
        </ul>
  
    """))


# In[ ]:


train.info()


# <a id='visual'></a>
# # 3. Visualization of data

# In[ ]:


train.target.value_counts().iplot(kind='bar',text=['0', '1'], title='Distribution Binary target column',color=['blue'])


# In[ ]:


counts_train = train.target.value_counts(sort=False)
labels = counts_train.index
values_train = counts_train.values

data = go.Pie(labels=labels, values=values_train ,pull=[0.03, 0])
layout = go.Layout(title='Comparing Target is binary (1) or not (0) in %')

fig = go.Figure(data=[data], layout=layout)
fig.update_traces(hole=.3, hoverinfo="label+percent+value")
fig.update_layout(
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Train', x=0.5, y=0.5, font_size=20, showarrow=False)])
fig.show()


# In[ ]:


missing = train.isnull().sum()  
missing[missing>0].sort_values().iplot(kind='bar',title='Null values present in train Dataset', color=['red'])


# In[ ]:


bin_ = [col for col in train.columns if 'bin_' in col]
print(bin_ )
nom_ = [col for col in train.columns if 'nom_' in col]
print(nom_)
ord_ = [col for col in train.columns if 'ord_' in col]
print(ord_)


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,10))

for ax, column in zip(axes.flatten(), bin_):
    sns.countplot(x = column, ax = ax, data = train)

plt.show()


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,10))

for ax, column in zip(axes.flatten(), ord_):
    sns.countplot(x = column, ax = ax, data = train)

plt.show()


# In[ ]:


fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14,12))

for ax, column in zip(axes.flatten(), nom_):
    sns.countplot(x = column, ax = ax, data = train)

plt.show()


#  <a id='separate'></a>
# #  4. Separate continuous, categorical and label column names

# In[ ]:


cat_cols = [ col  for col, dt in train.dtypes.items() if dt == object]
y_col = ['target']
cont_cols = [col for col in train.columns if col not in cat_cols + y_col]


print(f'cat_cols  has {len(cat_cols)} columns')  
print(f'cont_cols has {len(cont_cols)} columns')   


# In[ ]:


# Taking care of missing data in continous
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy = 'most_frequent')
imputer = imputer.fit(train[cont_cols])
train[cont_cols] = imputer.transform(train[cont_cols])


# In[ ]:


#now for test
imputer = imputer.fit(test[cont_cols])
test[cont_cols] = imputer.transform(test[cont_cols])


# In[ ]:


for cat in cat_cols:
    if train[cat].isnull().sum() > 0:
        train[cat] = train[cat].fillna(train[cat].mode()[0])
    if test[cat].isnull().sum() > 0:
        test[cat] = test[cat].fillna(test[cat].mode()[0])


# In[ ]:


# check again for Null values
missing = train.isnull().sum()  
missing[missing>0].sort_values()


# 
# ## Categorify
# Remember that Pandas offers a <a href='https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html'><strong>category dtype</strong></a> for converting categorical values to numerical codes.  Pandas replaces the column values with codes, and retains an index list of category values. In the steps ahead we'll call the categorical values "names" and the encodings "codes".

# In[ ]:


# Convert our categorical columns to category dtypes.
for cat in cat_cols:
    train[cat] = train[cat].astype('category')


# In[ ]:


train.info()


# In[ ]:


for cat in cat_cols:
    test[cat] = test[cat].astype('category')


# In[ ]:


train.nom_3.unique()


# In[ ]:


# this is how cat.codes work, assign a numeric value to each categorie.
train.nom_3.cat.codes.unique()


# In[ ]:


pd.DataFrame(train.nom_3.cat.codes.unique(), train.nom_3.unique())
# Check here Russia is change for 5 values and so on


# In[ ]:


# lets make as encoder this train columns
for col in cat_cols:
    train[col] = train[col].cat.codes
    


# In[ ]:


for col in cat_cols:
    test[col] = test[col].cat.codes


# In[ ]:


train.head()


# In[ ]:


# Number of unique itemns by colum
train.nunique()


# In[ ]:


#check correlation in train dataset
train.corr()['target'][:-1].sort_values().plot.bar(figsize=(10,10))


# <a id='split'></a>
# # 5. Train Test Split

# In[ ]:



X = train.drop(['id', 'target'],axis=1).values
y = train['target'].values


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)


# 
# # Scaling Data

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[ ]:


scaler.fit(X_train)


# In[ ]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


X_train.max()


# <a id='model'></a>
# # 6. Creating the Model    

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


# In[ ]:


#Grid Search
logreg = LogisticRegression(class_weight='balanced')
parameters = {'C':[0.1,5,10]}
clf_grid = GridSearchCV(logreg,parameters,scoring='roc_auc',refit=True,cv=5, n_jobs=-1)
clf_grid.fit(X_train, y_train)
print('Best roc_auc: {:.4}, with best C: {}'.format(clf_grid.best_score_, clf_grid.best_params_))


# In[ ]:


# Fitting LogisticRegression Classification to the Training set with paramns

classifier = LogisticRegression(C=5, solver='lbfgs', class_weight='balanced')
classifier.fit(X_train, y_train)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)


# <a id='eval'></a>
# # 7. Model evaluation

# In[ ]:


from sklearn.metrics import confusion_matrix
print( confusion_matrix(y_test, y_pred))


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))


# In[ ]:


# Showing Confusion Matrix
def plot_cm(y_true, y_pred, title, figsize=(5,4)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)


# In[ ]:


# Showing Confusion Matrix
plot_cm(y_test,y_pred, 'Confution matrix of Tweets', figsize=(5,5))


# In[ ]:


test = test.drop(['id'],axis=1).values
scalert = MinMaxScaler()
scalert.fit(test)
r_test = scalert.transform(test)


# In[ ]:


predictions = classifier.predict(r_test)
predictions


# In[ ]:


# sample of submission
submission.head()


# In[ ]:


submission['target'] = predictions 


# In[ ]:


submission.to_csv("submission.csv", index=False, header=True)


# <a id='reference'></a>
# # 8. References

# [Competition](https://www.kaggle.com/c/cat-in-the-dat-ii/overview)

# <h2>I hope this notebook <span style="color:red">Usefull</span> for you! </h3>

# 
# <a href="#top" class="btn btn-primary btn-lg active" role="button" aria-pressed="true">Go to TOP</a>
