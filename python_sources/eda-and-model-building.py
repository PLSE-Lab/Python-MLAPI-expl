#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing the path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# importing the preprocessing libraries
import numpy as np
import pandas as pd

# importing the visualization
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 14,7
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# importing the Ml libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix , classification_report


# In[ ]:


# importing the dataset
dataset = pd.read_csv("../input/carsdata/cars.csv" , na_values=" ")
dataset.head()


# In[ ]:


# checking the shape of the dataset
dataset.shape


# In[ ]:


# checking for any missing values
dataset.isnull().sum()


# In[ ]:


# replacing the names of the columns
dataset.columns = ["mpg" , "cylinders" , "cubicinches" , "hp" , "weightlbs" , "time-to-60", "year" , "brand"]
dataset.head()


# In[ ]:


# Descriptive Stats
dataset.describe()


# In[ ]:


# descriptive Stats
dataset.info()


# #### Here brand is our target Variable

# In[ ]:


# replacing and filling the missing values
dataset["cubicinches"] = dataset["cubicinches"].replace(np.nan)
dataset["weightlbs"] = dataset["weightlbs"].replace(np.nan)

mean_cubicinches = dataset["cubicinches"].mean()
mean_weightlbs = dataset["weightlbs"].mean()

dataset["cubicinches"] = dataset["cubicinches"].replace(np.nan , mean_cubicinches)
dataset["weightlbs"] = dataset["weightlbs"].replace(np.nan , mean_weightlbs)


# In[ ]:


# looking at the data
dataset.info()

"""thus missing values is filled with mean values"""


# In[ ]:


# Changing int64 to float64
dataset["cylinders"] = dataset["cylinders"].astype("float64")
dataset["hp"] = dataset["hp"].astype("float64")
dataset["time-to-60"] = dataset["time-to-60"].astype("float64")


# In[ ]:


dataset.info()


# In[ ]:


dataset.head()


# In[ ]:


# Checking for any outliers
sns.boxplot(dataset[["mpg"]])


# In[ ]:


sns.boxplot(dataset[["cylinders"]])


# In[ ]:


sns.boxplot(dataset[["cubicinches"]])


# In[ ]:


sns.boxplot(dataset[["hp"]])


# In[ ]:


sns.boxplot(dataset["weightlbs"])


# In[ ]:


sns.boxplot(dataset["time-to-60"])


# In[ ]:


# Removing outliers in time-to-60 column by IQR method
Q1 = dataset["time-to-60"].quantile(0.25)
Q3 = dataset["time-to-60"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = IQR - (1.5 * Q1)
upper_bound = IQR + (1.5 * Q3)
print("Lower Bound of Boxplot: " , lower_bound)
print("Upper Bound of Boxplot: " , upper_bound)

# Removing the outliers
dataset["time-to-60"] = dataset[(dataset["time-to-60"] > lower_bound) & (dataset["time-to-60"] < upper_bound)]


# In[ ]:



sns.boxplot(dataset["time-to-60"])


# In[ ]:


# making the count plot to understand the target variables
sns.countplot(dataset["brand"])


# In[ ]:


# understanding the correration between different features
corr_data = dataset.corr()
sns.heatmap(data = corr_data , annot = True)


# #### Thus we can see that there is a very high relationship between different features

# In[ ]:


# changing year column
print(dataset["year"].max())
dataset["year"].min()


# In[ ]:


# making a new column in a dataset
dataset["old"] = dataset["year"].max() - dataset["year"]

# Dropping the year column
dataset.drop(["year"] , inplace = True , axis = 1)


# In[ ]:


dataset.head()


# In[ ]:


# checking the correration of the data
new_corr = dataset.corr()
sns.heatmap(data = new_corr , annot = True)


# In[ ]:


Target = dataset["brand"]
dataset.drop(["brand"] , axis = 1 , inplace = True)
dataset["Target"] = Target
dataset.head()


# In[ ]:


dataset["old"] = dataset["old"].astype("float64")
dataset.head()


# In[ ]:


# looking at the dataset
dataset.head()


# In[ ]:





# ## Now the dataset is perfect for Model Building

# In[ ]:


# splitting the dataset into independent(x) and dependent(y) variables
x = dataset.iloc[: ,: -1].values
y = dataset.iloc[: , -1].values
print(x)
print(y)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)


# In[ ]:


# Normalizing the x
x_data = x /x.max()  
x_data


# In[ ]:


# Now Splitting the dataset into training and testing dataset
x_train , x_test , y_train , y_test = train_test_split(x_data , y , test_size = .20 , random_state = None)


# In[ ]:


# applying knearest classifier
from sklearn.ensemble import RandomForestClassifier
forest_classifier = RandomForestClassifier(random_state = None)
forest_classifier.fit(x_train , y_train)

y_preg = forest_classifier.predict(x_test) 


# In[ ]:


# Making the Confusion Matrix and Classification Report
cm = confusion_matrix(y_test, y_preg)
sns.heatmap(data =  cm , annot = True , cmap = "Blues" , linewidths= 1 , linecolor= "black")

report = classification_report(y_test , y_preg)
print(report)

