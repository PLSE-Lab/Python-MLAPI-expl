#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **The Data**

# Let's start by reading in the Iris.csv file into a pandas dataframe.

# In[ ]:


df=pd.read_csv('../input/Iris.csv')


# In[ ]:


df


# **Top five values**

# In[ ]:


df.head()


# To pop up 5 random rows from the data set, we can use sample(5) function

# In[ ]:


df.sample(5)


# **Find Columns**

# To print dataset columns, we can use columns atribute

# In[ ]:


df.columns


# Columns Id:-SPL-SPW-PTL-PTW(CM) SepalLengthCm:-Length of the sepal (in cm) SepalWidthCm:-Width of the sepal (in cm) PetalLengthCm:-Length of the petal (in cm) PetalWidthCm:-Width of the petal (in cm) Species:-Species name

# **Shape of dataset**

# In[ ]:


df.shape


# **Information about dataset **

# In[ ]:


df.info()


# **Describe of the dataset**

# To give a statistical summary about the dataset, we can use **describe()

# In[ ]:


df.describe()


# ***To check out how many null info are on the dataset, we can use **isnull().sum().***

# In[ ]:


df.isnull().sum()


# In[ ]:


df.groupby('Species').count()


# **Scatter plot**

# Scatter plot Purpose To identify the type of relationship (if any) between two quantitative variables

# In[ ]:


# Modify the graph above by assigning each species an individual color.
sns.FacetGrid(df, hue="Species", size=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()
plt.show()


# **PairPlot**

# Plot pairwise relationships in a dataset

# In[ ]:


sns.pairplot(df)


# **BarPlot**

# In descriptive statistics, a box plot or boxplot is a method for graphically depicting groups of numerical data through their quartiles. Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability outside the upper and lower quartiles, hence the terms box-and-whisker plot and box-and-whisker diagram.[wikipedia]

# In[ ]:


df.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False)
plt.figure()
#This gives us a much clearer idea of the distribution of the input attributes:


# In[ ]:


# To plot the species data using a box plot:

sns.boxplot(x="Species", y="PetalLengthCm", data=df )
plt.show()


# In[ ]:


ax= sns.boxplot(x="Species", y="PetalLengthCm", data=df)
ax= sns.stripplot(x="Species", y="PetalLengthCm", data=df, jitter=True, edgecolor="gray")
plt.show()


# In[ ]:


# Tweek the plot above to change fill and border color color using ax.artists.
# Assing ax.artists a variable name, and insert the box number into the corresponding brackets

ax= sns.boxplot(x="Species", y="PetalLengthCm", data=df)
ax= sns.stripplot(x="Species", y="PetalLengthCm", data=df, jitter=True, edgecolor="gray")

boxtwo = ax.artists[2]
boxtwo.set_facecolor('red')
boxtwo.set_edgecolor('black')
boxthree=ax.artists[1]
boxthree.set_facecolor('yellow')
boxthree.set_edgecolor('black')

plt.show()


# **Histogram**

# We can also create a histogram of each input variable to get an idea of the distribution.

# In[ ]:


# histograms
df.hist(figsize=(15,20))
plt.figure()


# It looks like perhaps two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can exploit this assumption.

# In[ ]:


df["PetalLengthCm"].hist();


# **Volinplots**

# In[ ]:


# violinplots on petal-length for each species
sns.violinplot(data=df,x="Species", y="PetalLengthCm")


# **Jointplot**

# In[ ]:


# Use seaborn's jointplot to make a hexagonal bin plot
#Set desired size and ratio and choose a color.
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=df, size=10,ratio=10, kind='hex',color='green')
plt.show()


# **Heatmap**

# Plot rectangular data as a color-encoded matrix.

# In[ ]:


sns.heatmap(df.corr())


# **Distplot**

# ![](http://)Flexibly plot a univariate distribution of observations.

# In[ ]:


sns.distplot(df['SepalLengthCm'])


# In[ ]:


sns.distplot(df['PetalLengthCm'])


# **Countplot**

# Show the counts of observations in each categorical bin using bars.

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='Species',data=df,palette='RdBu_r')


# In[ ]:


sns.distplot(df['SepalLengthCm'].dropna(),kde=False,color='darkred',bins=30)


# In[ ]:


sns.distplot(df['SepalWidthCm'].dropna(),kde=False,color='green',bins=30)


# **Data Cleaning**

# When dealing with real-world data, dirty data is the norm rather than the exception. We continuously need to predict correct values, impute missing ones, and find links between various data artefacts such as schemas and records. We need to stop treating data cleaning as a piecemeal exercise (resolving different types of errors in isolation), and instead leverage all signals and resources (such as constraints, available statistics, and dictionaries) to accurately predict corrective actions.[12]

# In[ ]:


cols = df.columns
features = cols[0:4]
labels = cols[4]
print(features)
print(labels)


# In[ ]:


#Well conditioned data will have zero mean and equal variance
#We get this automattically when we calculate the Z Scores for the data

data_norm = pd.DataFrame(df)

for feature in features:
    df[feature] = (df[feature] - df[feature].mean())/df[feature].std()

#Show that should now have zero mean
print("Averages")
print(df.mean())

print("\n Deviations")
#Show that we have equal variance
print(pow(df.std(),2))


# **Building a Logistic Regression model**

# **Logistic regression**
# Logistic regression is the appropriate regression analysis to conduct when the dependent variable is dichotomous (binary). Like all regression analyses, the logistic regression is a predictive analysis statisticssolutions.
# 
# In statistics, the logistic model (or logit model) is a widely used statistical model that, in its basic form, uses a logistic function to model a binary dependent variable; many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model; it is a form of binomial regression. Mathematically, a binary logistic model has a dependent variable with two possible values, such as pass/fail, win/lose, alive/dead or healthy/sick; these are represented by an indicator variable, where the two values are labeled "0" and "1"

# Let's start by splitting our data into a training set and test set (there is another test.csv file that you can play around with in case you want to use all this data for training).

# **Train Test Split**

# In[ ]:


X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# **Training and Predicting**

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[ ]:


predictions = logmodel.predict(X_test)


# Let's move on to evaluate our model!

# **Evaluation**

# We can check precision,recall,f1-score using classification report!

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))


# **Using SVM**

# In[ ]:


#Using SVM
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


cm


# In[ ]:


print(classification_report(y_test,predictions))


# **Conclusion**

# **In this kernel, I have tried to cover all the parts related to the process of Machine Learning algorithm logistic regression and Support vector machine with a variety of Python packages . I hope to get your feedback to improve it. **
