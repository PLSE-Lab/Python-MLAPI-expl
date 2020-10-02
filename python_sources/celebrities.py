#!/usr/bin/env python
# coding: utf-8

# ## Install libraries/packages

# In[ ]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install plotly')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install graphviz')
get_ipython().system('pip install pydot')


# ## Install libraries/packages

# In[ ]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load data

# In[ ]:


# Import data set.
data = pd.read_json('../input/master.json')

# Also reset the index to avoid any problems later.
data = data.reset_index(drop=True)


# In[ ]:


# Print the first 5 rows of the data set.
data.head()


# ## Data cleaning/Preprocessing

# In[ ]:


# Filter the data to the columns that we will use in our models.
ds=data[['name','birth_date','Age','Weight','Eyes color','Hair color','Height']]

# Copy of the data set in case we need it.
cp=ds.copy()


# In[ ]:


# Check the number of null values for each column.
ds.isna().sum()


# In[ ]:


# Remove all the null values.
ds=ds.dropna()


# In[ ]:


# Check if there are any duplicate rows.
ds.duplicated().any()


# In[ ]:


# Remove any duplicates from the new data set.
ds=ds.drop_duplicates()


# In[ ]:


# Examine the data types of the clean data set.
ds.info()


# ### Column birth_date

# In[ ]:


# Convert birth_date to datetime type.
ds['birth_date']=pd.to_datetime(ds['birth_date'])


# ### Column Age

# In[ ]:


# Convert Age to integer type.
ds['Age']=ds['Age'].astype('int')


# In[ ]:


# Age Violinplot
fig = px.violin(ds, y="Age",points="all",box=True)
fig.update_layout(title_text='Age Boxplot')
fig.show()


# In[ ]:


# Age Distplot
fig = px.histogram(ds, x="Age",marginal="violin", # or violin, rug
                   hover_data=ds.columns)
fig.update_layout(title_text='Age Distplot')
fig.show()


# ### Column Weight

# In[ ]:


# Split the text and keep the weight in kilos (kg).
lst=[]
for e in ds.Weight.str.split():
    lst.append(int(e[0]))
values= np.asarray(lst)
ds=ds.reset_index(drop=True)
ds['Weight']=pd.Series(lst)


# In[ ]:


# Weight Boxplot
fig = px.box(ds, y="Weight",points="all")
fig.update_layout(title_text='Weight Boxplot')
fig.show()


# In[ ]:


# Weight Distplot
fig = px.histogram(ds, x="Weight",marginal="violin", # or violin, rug
                   hover_data=ds.columns)
fig.update_layout(title_text='Weight Distplot')
fig.show()


# ### Column Eyes color

# In[ ]:


# Values of column "Eyes color"
ds['Eyes color'].value_counts()


# In[ ]:


# Plot barplot for 'Eyes color' values.
colors=ds['Eyes color'].unique()
fig = go.Figure([go.Bar(x=colors, y=ds['Eyes color'].value_counts())])
fig.update_layout(title_text='Values of "Eyes color"')
fig.show()


# In[ ]:


# Plot a pie chart for 'Eyes color' values.
fig = go.Figure(data=[go.Pie(labels=colors, values=ds['Eyes color'].value_counts())])
fig.update_layout(title=go.layout.Title(text='Values for column "Eyes color"',xref="paper",x=.5))
fig.show()


# In[ ]:


# Import Label encoder from sklearn.preprocessing.
from sklearn.preprocessing import LabelEncoder

# Instantiate the Label encoder
le=LabelEncoder()

# Fit and transform
ds['Eyes color']=le.fit_transform(ds['Eyes color'])


# ### Column Hair color

# In[ ]:


# Values of column "Hair color"
ds['Hair color'].value_counts()


# In[ ]:


# Plot barplot for 'Hair color' values.
colors=ds['Hair color'].unique()
fig = go.Figure([go.Bar(x=colors, y=ds['Hair color'].value_counts())])
fig.update_layout(title_text='Values of "Hair color"')
fig.show()


# In[ ]:


# Plot a pie chart for 'Hair color' values.
fig = go.Figure(data=[go.Pie(labels=colors, values=ds['Hair color'].value_counts())])
fig.update_layout(title=go.layout.Title(text='Values for column "Hair color"',xref="paper",x=.5))
fig.show()


# In[ ]:


# Instantiate the Label encoder
le=LabelEncoder()

#  Fit and transform
ds['Hair color']=le.fit_transform(ds['Hair color'])


# In[ ]:


# In case we want to transform back to the original values.

# le.inverse_transform(ds['Hair color'])
# le.classes_


# ### Column Height

# In[ ]:


# Split the text and keep the height in centimeters (cm).
lst=[]
for e in ds.Height.str.split():
    lst.append(e[0])
values= np.asarray(lst)
ds['Height']=pd.Series(values).astype('float')


# In[ ]:


fig = px.violin(ds, y="Height", box=True, # draw box plot inside the violin
                points='all') # can be 'outliers', or False
               
fig.show()


# In[ ]:


# Number of cuts.
# Create 5 classes for the 'Height' column.
ds['Height']= pd.qcut(ds['Height'],5)


# In[ ]:


# Instantiate the Label encoder
le=LabelEncoder()

#  Fit and transform
ds['Height']=le.fit_transform(ds['Height'])

# See the classes.
# le.classes_


# In[ ]:


# Filter the columns in order to use them in modeling.
X=ds[ds.columns[2:-1]]
y=ds['Height']


# In[ ]:


# Import 'train_test_split' module.
from sklearn.model_selection import train_test_split

# Set training and test data and select ratio of training to test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# ## Modeling

# - Decision trees
# - KNN

# ## Decision Trees

# In[ ]:


# Import decision tree classifier module.
from sklearn import tree

# Instantiate decision tree classifier.
clf = tree.DecisionTreeClassifier(random_state=0,max_depth=3,criterion='entropy') # can be 'gini' or 'entropy'

# Train the classifier.
clf = clf.fit(X_train, y_train)


# In[ ]:


# Plot the tree.
tree.plot_tree(clf);


# In[ ]:


# Create an image with the rules of the tree.
from graphviz import Source
from IPython.display import SVG

# Simpler black & white.
# graph=Source(tree.export_graphviz(clf, out_file=None, feature_names=X_train.columns))

# More info with colors.
graph=Source(tree.export_graphviz(clf,feature_names = X_train.columns,
                                  class_names=['(156.2, 170.72]','(170.72, 175.3]','(175.3, 180.44]','(180.44, 185.16]','(185.16, 193.0]'],
                                  out_file=None,filled=True, rounded=True, special_characters=True)) #remove 'class_names' if you've changed the number of cuts previously.
SVG(graph.pipe(format='svg'))


# In[ ]:


# Print the tree in a simplified version.
from sklearn.tree.export import export_text
r = export_text(clf, feature_names=X.columns.tolist())
print(r)


# In[ ]:


# Accuracy score.
clf.score(X_test,y_test)


# ## KNN

# In[ ]:


# Import KNN module.
from sklearn import neighbors


# In[ ]:


# Invoke KNearestNeighbors classification approach.
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Report accuracy score.
accuracy = knn.score(X_test, y_test)
print(accuracy)


# In[ ]:


# Plot accuracy graph
scores=[]
for i in range(1,21):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    scores.append(accuracy)

fig = go.Figure(data=go.Scatter(x=[i for i in range(1,21)] , y=scores, mode='lines+markers'))
fig.update_layout(title='Accuracy per number of neighbors',xaxis_title='Number of neighbors',yaxis_title='Accuracy')
fig.show()


# In[ ]:


# Calculate the accuracy of the model 
train_acc=[]
test_acc=[]
max_neighbors=21
for k in range(1,max_neighbors):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_acc.append(knn.score(X_train, y_train))
    test_acc.append(knn.score(X_test, y_test))

# Plot accuracy graph
fig = go.Figure()
fig.add_trace(go.Scatter(x=[k for k in range(1,max_neighbors)], y=train_acc, mode='lines+markers',name='train_acc'))
fig.add_trace(go.Scatter(x=[k for k in range(1,max_neighbors)], y=test_acc, mode='lines+markers',name='test_acc'))
fig.update_layout(title='Train accuracy vs test accuracy',xaxis_title='Number of neighbors',yaxis_title='Accuracy')
fig.show()


# In[ ]:


# Stats for k with best score (k=14).
knn = neighbors.KNeighborsClassifier(n_neighbors=14)
knn.fit(X_train, y_train)


# In[ ]:


pred=knn.predict(X_test)


# In[ ]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

# The mean squared error (relative error).
print("Mean squared error: %.2f" % mean_squared_error(y_test, pred))

# Explained average absolute error (average error).
print("Average absolute error: %.2f" % mean_absolute_error(y_test, pred))

# Explained variance score: 1 is perfect prediction.
print('Variance score: %.2f' % knn.score(X_test, y_test))


# In[ ]:




