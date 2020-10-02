#!/usr/bin/env python
# coding: utf-8

# ## Cluster and Random Forest Analysis of Student Performance ##
# 
# In this Kernel I plan on implementing two types of analysis on the student performance dataset. First, I will use cluster analysis to see if the grades can be grouped into three categories. Second, I will then use a random forest to set a model up to predict student performance. I will judge my random forest based on the root mean squared error metric.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #For inital graphs. Super impressed with this
import matplotlib as mpl #Not sure if I need any matplotlib, possibly a prereq for seaborn.
import matplotlib.pyplot as plt
########################################################################################
#I mainly used sklearn libraries for all of my data analysis, which can be seen below. 
########################################################################################
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA #Principle component analysis. 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.cross_validation import train_test_split #deprecated, should find another method
from sklearn.metrics import mean_squared_error #To judge how my model did. 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/xAPI-Edu-Data.csv')

# Any results you write to the current directory are saved as output.


# In[ ]:


df.head(10)


# Here I'll graph some of the variables to see what the data looks like. 

# In[ ]:


sns.stripplot(x="Class", y="Discussion", data=df, jitter=True);


# Notice that those with a low overall grade visited the discussion tab the least

# In[ ]:


sns.stripplot(x="Class", y="raisedhands", data=df, jitter=True);


# Similarly, those that achieved a high grade raised their hands more then those that did not. For my last bit of data exploration, I want to see how many people are in each section and where all of the students are coming from. 

# In[ ]:


print(df['SectionID'].value_counts())
print(df['PlaceofBirth'].value_counts())


# Here, we can see that a majority of the students are in the first section, with hardly any in the last section. Similarly, most of the students come from Kuwait and Jordan. 

# ## Data Preparation ##
# 
# Now I'm going to want to convert some of the categorical variables (Topic, Gender, Student Absence Days) to be dummy variables. This will make them numerical to help with the analysis of both the cluster and the regression below. 

# In[ ]:


df1 = pd.get_dummies(df['gender'])
df2 = pd.get_dummies(df['StudentAbsenceDays'])
df3 = pd.get_dummies(df['Topic'])
df = pd.concat([df, df1, df2, df3], axis=1)
df.head()


# In a possibly wasteful way, I created three data frames that split the categorical variables gender, StudentAbsenceDays and Topic into dummy variables, and then concatenated those with the original data frame df to get everything ready for the regression. I'm sure there's a better way to do this, but this is what made the most sense to me. 

# ## Cluster Plotting Time ##
# 
# Now that we have some Descriptive Graphs, I now want to plot some clusters to see what the data looks like.

# In[ ]:


kmeans_model = KMeans(n_clusters=3, random_state=1)
# Get only the numeric columns from games.
numericalColumns = df._get_numeric_data()
# Fit the model using the good columns.
kmeans_model.fit(numericalColumns)
# Get the cluster assignments.
labels = kmeans_model.labels_


# Here, I've initialized the model so it starts off with three clusters and has a reproducible random state. From there, this model only works with numerical columns, so I throw those values into numericalColumns. From there, I run the model and put the result into labels. 
# 
# ## Principle Component Analysis ##
# Now, I'm going to use Principle Component Analysis to compress the multiple dimensions we have currently down to two. This will let us graph the clusters and we can see how we did. 

# In[ ]:


# Create a PCA model.
pca_2 = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(numericalColumns)
# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
# Show the plot.
plt.show()


# As we can see, there are three pretty distinct clusters that show up, pointing us to the fact that the authors of the original study did a good job grouping the scores. 

# ## Decision Tree Analysis ##
# 
# For the other part of my analysis, I thought it would be fun to build a decision tree and see if I can predict what grade the student would achieve with my dependent variables. In order to do that, I first have to make the prediction variable Class, which is a classification variable, into a numerical variable. I'll call this new variable Grade.

# In[ ]:


def map_values(row, values_dict):
    return values_dict[row]

values_dict = {'L': 1, 'M': 2, 'H': 3}

df['Grade'] = df['Class'].apply(map_values, args = (values_dict,))


# So this handy function changes the Class variable (which is the grade the student received) into a numerical grade value. I learned of this method from [this][1] StackOverflow Post.
# 
# 
#   [1]: http://stackoverflow.com/questions/19913659/pandas-conditional-creation-of-a-series-dataframe-column

# ## Breaking the Data into a Training and Testing Section ##
# 
# In order to get started, I first have to break the data up into a training portion and a testing portion. I set 80% of my sample to be training, and set a reproducible random state so my analysis can be replicated. 

# In[ ]:


train = df.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = df.loc[~df.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)


# And now I'm going to remove any of the columns that I don't want in my regression, as well as note down my target column. Grade is in the unwanted columns because I don't want to include it in my list of independent variables. I want to try and predict what Grade will be!

# In[ ]:


unwantedCols = ["gender", "NationalITy", 'Grade', "PlaceofBirth", "StageID", "GradeID", "SectionID", "Topic", "Semester", "Relation", "ParentAnsweringSurvey", "ParentschoolSatisfaction", "StudentAbsenceDays", "Class", "Grade"]
columns = df.columns.tolist()
columns = [c for c in columns if c not in unwantedCols]
target = "Grade"


# From here, I'm going to initialize my model, fit it to my columns and training target, make my predictions and see how well I did.

# In[ ]:


# Initialize the model with some parameters.
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Fit the model to the data.
model.fit(train[columns], train[target])
# Make predictions.
predictions = model.predict(test[columns])
# Compute the error.
mean_squared_error(predictions, test[target])


# ## Final Results ##
# 
# So there we have it. Above I calculated both a cluster analysis as well as a random forest, and got a pretty good mean squared error! This was exciting. As my first ever post on Kaggle, I would appreciate any comments on how I could improve.
# 
# One article that I thought was helpful was [this][1] one. It was incredible well written, and I based most of my analysis off of it. 
# 
# Next steps in this analysis could be incorporating the grade that the student was in, the place of birth as well as seeing if a different section affected the final result. 
# 
# 
#   [1]: https://www.dataquest.io/blog/machine-learning-python/
