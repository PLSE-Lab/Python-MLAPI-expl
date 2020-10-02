#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
from sklearn import linear_model


#  #   Selection of data in KDD process

# In[ ]:


path_to_file="../input/googleplaystore.csv"


# In[ ]:


data=pd.read_csv(path_to_file,encoding='utf-8')


# # Preprossesing  data in KDD PROCESS

# In observing the dataset we realied that some of the apps listed were not even released as yet.
# We also spotted the pattern that these apps all contained some NULL values and would not be important 
# for our analysis so we decided to get rid of them.

# In[ ]:


##checking for all null values in dataset
missing_data_results =data.isnull().sum()
print(missing_data_results)


# Complete case analysis, Complete case analysis followed by nearest-neighbor assignment for partial data, Partial data cluster analysis, Replacing missing values or incomplete data with means Imputation are all ways to deals with missing data. However we decided to delete all rows where  column values is null

# In[ ]:


#loops through dataset and delete rows where column values is null
data =data.dropna()
data.isnull().sum()


# Observing the install columns we realized that it contains string characters and as such we remove those charaters in order to work with integer values.
#                    BEFORE:                                    
# ![image.png](attachment:image.png)
# 

# In[ ]:


#Below we are using the regex \D to remove any non-digit characters
data['Installs']=data['Installs'].replace(regex=True,inplace=False,to_replace=r'\D',value=r'')
#data['Installs']=data['Installs'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
#data.Installs


# AFTER:
# ![image.png](attachment:image.png)

# In[ ]:


data['Installs']


# In[ ]:


np.sort(data.Installs)
#for col in data.columns:
  #  data[col]=np.sort(data[col].values)


#         **Check for duplicate records**

# In[ ]:


data.shape


# In[ ]:


dupes=data.duplicated()
sum(dupes)


# In[ ]:


data=data.drop_duplicates()


# In[ ]:


data.shape


# Checking to see if we are working with ligit data types

# In[ ]:


data.dtypes
    


# In[ ]:


data['Price']=data['Price'].replace(regex=True,inplace=False,to_replace=r'\D',value=r'')


# In[ ]:


#converting installs and Price to appropriate data types
data["Installs"] = pd.to_numeric(data["Installs"])
data["Reviews"] = pd.to_numeric(data["Reviews"])
data["Price"] = pd.Float64Index(data["Price"])
data.dtypes


# **VISUALIZATIONS**

# 1. Number of Apps available basedon content ratings

# In[ ]:


data.columns =data.columns.str.replace(' ', '_')


# In[ ]:


#Apps available based on Content rating
plt.figure(figsize=(10,10))
sns.countplot(x='Content_Rating',data=data,)
plt.xticks(rotation=45)
plt.title("Number of Apps available based on Content rating")


# 2. Plot to show the distribution of apps from each category in the data set. 

# In[ ]:



#data['Category'].value_counts()
plt.figure(figsize=(12,12))
data['Category'].value_counts().plot(kind='bar',title='Distribution of Categories')
plt.xlabel('Categories')
plt.ylabel('Number of Apps')


#         *Most installed apps based on Category*

# In[ ]:



plt.figure(figsize=(12,12))
sns.barplot(x='Installs',y='Category',data=data,ci=None)
plt.title("Number of Apps installed based on Category")


# 3. In the series of steps to determine what appp to develop we would like to identify whether there are more downloads for "Paid" or "Free" apps

# In[ ]:


data['Type'].value_counts().plot(kind='bar',title='Distribution App Types')
plt.xlabel('Type of Apps')
plt.ylabel('Count')


# Application of knowledge from dataset.....
# Select all apps where there downloads are between 10000 and 10000000
# ________________________________________________________________________________

# In[ ]:



find=((data.Installs.values >=10000)& (data.Installs.values <=10000000))
data1 = data[find]
data1


# In[ ]:


#data.hist(column= 'Installs')
data1.hist(column= 'Installs')


# In[ ]:


len(data1.Installs.values)


# # Transformation step in KDD process

# In[ ]:


data1.Reviews


# In[ ]:


data1.Reviews=pd.qcut(data1.Reviews,20)


# In[ ]:


tree_data = data1[['Installs','Category','Type','Reviews']]
tree_data


# In[ ]:


data1.Installs.value_counts()


# In[ ]:


tree_data['Installs'] = pd.cut(tree_data['Installs'], [9999
,50000
,100000
,500000
,1000000
,5000000
,10000000
                                              ])
tree_data.Installs.value_counts()


# In[ ]:


tree_data.Installs.value_counts()


# In[ ]:


# Encoder function.....transforming


# In[ ]:


def encoder(dataset):
    from sklearn.preprocessing import LabelEncoder
    #dictionary to store values
    encoder = {}
    for column in dataset.columns:
        # Only creating encoder for categorical data types
      #  if not np.issubdtype(dataset[column].dtype, np.number) and column != 'Installs':
            encoder[column]= LabelEncoder().fit(dataset[column])
            #returning the dictionary with values
    return encoder


# In[ ]:


tree_data


# In[ ]:


#transforming tree data
encoded_labels = encoder(tree_data)
print("Encoded Values for each Label")
print("="*32)
for column in encoded_labels:
    print("="*32)
    print('Encoder(%s) = %s' % (column, encoded_labels[column].classes_ ))
    print(pd.DataFrame([range(0,len(encoded_labels[column].classes_))], columns=encoded_labels[column].classes_, index=['Encoded Values']  ).T)


# In[ ]:


data1.Installs.value_counts()


# In[ ]:


transformed_data= tree_data.copy()
for col in transformed_data.columns:
    if col in encoded_labels:
       transformed_data[col] = encoded_labels[col].transform(transformed_data[col])
print("Transformed data set with category and type encoded")
print("="*32)
transformed_data


# # Data Mining  in KDD process

# *************************************Multinomial LogisticRegression************************
# # Aim: Does the numbr of Installs for an app incrof installs go up with increase in reviews?
# Type of algorithm ?
#     -supervised machine learning algorithm
# Type ofsupervised machine learning?
#     - Classification since dependent variable is categorical and dealing with current behavior
#         
#     

# In[ ]:


from sklearn.model_selection import train_test_split
#Seperate our data into independent X and dependent Y 
X_data = transformed_data[['Category','Type']]
Y_data= transformed_data['Installs']
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.30)


# In[ ]:


from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# creating multinomial model since we have more than one predictor then fit training data.
regr = linear_model.LogisticRegression(solver='newton-cg')
#regr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(pd.DataFrame(X_train),Y_train)
#regr = GaussianNB()
regr.fit(pd.DataFrame(X_train),Y_train)


# In[ ]:


#given a trained model, we are predicting the label of a new set of X test data.
Prediction = regr.predict(pd.DataFrame(X_test))


# In[ ]:


transformed_data['Installs'].value_counts()


# In[ ]:


print(Prediction)


# In[ ]:


# The coefficient of our determinant(x)
print('Coefficients: \n', regr.coef_)


# In[ ]:


regr.intercept_


# In[ ]:


from sklearn.metrics import r2_score
# Use score method to get accuracy of model
print('Variance score:%2f'% r2_score(Y_test,Prediction)) 


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
cm = metrics.confusion_matrix(Y_test,Prediction)
print(cm)
cm.shape


# In[ ]:


plt.figure(figsize=(5,5))
plt.imshow(cm, interpolation='nearest', cmap='Pastel1')
plt.title('Confusion matrix', size = 15)
plt.colorbar()
plt.xticks(Prediction)
plt.yticks(Y_test)
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
width,height = cm.shape
for x in range(width):
 for y in range(height):
  plt.annotate(str(cm[x][y]), xy=(y, x), 
  horizontalalignment='center',
  verticalalignment='center')


# ### check correlation between Category and Installs variables
# np.corrcoef(transformed_data.Category,transformed_data.Installs)
# 

# In[ ]:


data1


# In[ ]:


plt.scatter(transformed_data.Category,transformed_data.Installs)
plt.show()


# In[ ]:





# In[ ]:


np.corrcoef(transformed_data.Type,transformed_data.Installs)


# In[ ]:


plt.scatter(transformed_data.Type,transformed_data.Installs)
plt.show()


# In[ ]:


np.corrcoef(transformed_data.Type,transformed_data.Installs)


# # Interpretation/ Evaluation of our Regression Model

# **Regression Explanation**
# The score above indicates that our model is extreamly bad!! We considered many solutions such as using different models and allocating  a larger training sample, none of which worked. We then observed the relationship between our two independent variables and dependent variable. There are of no relation and as such contributing to our bad model. So in conclusion, we went wrong in selecting our independent variables!! :(

# # We attempted to use decision tree to give a better visual of the question above but had some problem......*not one of our algorithm!* 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

from sklearn import tree


# In[ ]:


# Create the classifier with a maximum depth of 2 using entropy as the criterion for choosing most significant nodes
# to build the tree
clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=2)
# Hint : Change the max_depth to 10 or another number to see how this affects the tree


# In[ ]:


clf.fit(X_train, Y_train)


# In[ ]:


pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100)
], index = X_data.columns, columns = ['Feature Significance in Decision Tree'])


# In[ ]:


import graphviz


# In[ ]:


Y_data


# In[ ]:


dot_data = tree.export_graphviz(clf,out_file=None,

feature_names=X_data.columns,
class_names= None,
filled=True, rounded=True, proportion=True,
node_ids=True, #impurity=False,
special_characters=True)


# In[ ]:


graph = graphviz.Source(dot_data)

graph
tree.export_graphviz(clf,out_file='tree.dot') 


# In[ ]:


corrmat = transformed_data.corr()
#f, ax = plt.subplots()
p =sns.heatmap(corrmat, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# In[ ]:


transformed_data['Reviews'].corr(transformed_data['Installs'])


# # **********************Linear Regression************************
# Type of algorithm ?
#     -supervised machine learning algorithm
# Type ofsupervised machine learning?
#     - Classification since dependent variable is categorical and dealing with current behavior
#         

# # Aim: Does the amount of installs go up with increase in reviews?

# In[ ]:


X_data1 = transformed_data['Reviews']
Y_data1 = transformed_data['Installs']


# In[ ]:


from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_data1, Y_data1, test_size=0.30)


# In[ ]:


reg1 = linear_model.LinearRegression()


# In[ ]:


reg1.fit(pd.DataFrame(X_train1),y_train1)


# In[ ]:


Prediction1 = reg1.predict(pd.DataFrame(X_test1))


# In[ ]:


y_test1.index


# In[ ]:


Prediction1[:12]


# In[ ]:


reg1.coef_


# In[ ]:


reg1.intercept_


# In[ ]:


reg1.score(pd.DataFrame(X_test1),y_test1)


# In[ ]:


plt.scatter(X_test1,y_test1,  color='black')
plt.plot(X_test1,Prediction1,color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]:


import seaborn as sns
sns.set(style="whitegrid")
# Plot the residuals after fitting a linear model
sns.residplot(X_train1, y_train1, lowess=True, color="b")


# In[ ]:


install = 0.2*4 -0.354
install


# # Output of linear regression
# **install = 0.2review -0.354**

# # Interpretation/ Evaluation of Linear model
# The linear regression module was considered better than the multinomial regression as the coefficient of determination had a higher value. The R^2 value for this module was 0.85 which means that Installs have an 85 percent chance of being predicted from Reviews.
# 

# # K-MEANS

#   **CLUSTERING******
#   Aim: Can we identify groups based on Review and Installs?
# If that is the case, developers could develop Apps of a certain category based on the reviews

# *************************************k-Nearest Neighbors Method************************
# Type of algorithm ?
#     -supervised machine learning algorithm
# Type ofsupervised machine learning?
#     - Classification since dependent variable is categorical and dealing with current behavior

# In[ ]:


cluster_data = transformed_data[['Reviews','Installs']]
cluster_data.head(50)


# In[ ]:


cluster_data.plot(kind='scatter',x='Reviews',y='Installs')


# In[ ]:


# Is there any missing data
missing_data_results = cluster_data.isnull().sum()
print(missing_data_results)


# In[ ]:


data_values = cluster_data.iloc[ :, :].values
data_values


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# In[ ]:


mms = MinMaxScaler()
mms.fit(data_values)
data_transformed = mms.transform(data_values)


# In[ ]:


Sum_of_squared_distances = []
K = range(1,15)
for i in K:
    km = KMeans(n_clusters=i)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)


# In[ ]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.title('Computing WCSS for KMeans++')
plt.xlabel("Number of clusters")
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300)
cluster_data["cluster"] = kmeans.fit_predict( data_values )
cluster_data


# In[ ]:


#viewing amount of elements in clusters
cluster_data['cluster'].value_counts()


# In[ ]:


import seaborn as sns
sns.set(color_codes=True)


# In[ ]:


cluster_data['cluster'].value_counts().plot(kind='bar',title='Distribution of Apps')


# In[ ]:


grouped_cluster_data = cluster_data.groupby('cluster')
grouped_cluster_data


# In[ ]:


grouped_cluster_data.plot(subplots=True)


# In[ ]:


sns.pairplot(cluster_data,hue="cluster")


# # Interpretation/ Evaluation of K-Mean

# With the clustering algorithm, we used the elbow method to deduce the number of groups we could possible obtain. After trying the various K values, we decided that K =3 give the most suitable results. With K= 3 we deduce that apps each group has at least 2000 apps which are group together based on their Installs and Reviews. The values for installs show the different bins in which apps bring for the 3 groups. It can be deduced that apps in group 0 are apps received more installs than review, while group 1 on shows that the apps in that group received.
