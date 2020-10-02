#!/usr/bin/env python
# coding: utf-8

# ## TABLE OF CONTENTS
# 
# * [1. INTRODUCTION](#1)
# * [2. USED LIBRARIES](#2)
# * [3. DATA EXPLORATION](#3)
#     * [3.1. Detailed Information of the Dataset](#31)
#     * [3.2. Label Encoding](#32)
#     * [3.3. Correlation of Columns(Attributes)](#33)
#     * [3.4. Various Visualizations from Dataset](#34)
# * [4. DATA PREPARATION AND CLEANING](#4)
#     * [4.1. Drop Irrelevant Columns](#41)
#     * [4.2. Split Data and Target](#42)
#     * [4.3. Scaling of Data](#43)
#     * [4.4. Principal Component Analysis (PCA)](#44)
#     * [4.5. Preparation of Test and Train Data](#45)
# * [5. BUILDING MODELS](#5)
#     * [5.1. Building Models with Default Hyperparameters](#51)
#         * [5.1.1. Logistic Regression](#511)
#         * [5.1.2. Gaussian Naive Bayes](#512)
#         * [5.1.3. Decision Tree](#513)
#         * [5.1.4. Random Forest](#514)
#         * [5.1.5. SVC](#515)
#         * [5.1.6. K-NN](#516)
#     * [5.2. Hyperparameter Tuning with Grid Search](#52)
#         * [5.2.1. Logistic Regression](#521)
#         * [5.2.2. Decision Tree](#522)
#         * [5.2.3. Random Forest](#523)
#         * [5.2.4. SVC (with RBF)](#524)
#         * [5.2.5. K-NN](#525)
#     * [5.3. Rebuild Models with Tuned Hyperparameters](#53)
# * [6. EVALUATING MODELS](#6)
#     * [6.1. Evaluating Logistic Regression Model](#61)
#     * [6.2. Evaluating Gaussian Naive Bayes Model](#62)
#     * [6.3. Evaluating Decision Tree Classifier Model](#63)
#     * [6.4. Evaluating Random Forest Classifier Model](#64)
#     * [6.5. Evaluating Support Vector Classifier Model](#65)
#     * [6.6. Evaluating K-Nearest Neighbors Classifier Model](#66)
# * [7. EXPLORATION OF RESULTS](#7)
# * [8. CONCLUSION](#8)
# * [REFERENCES](#9)

# <a id="1"></a>
# # 1. INTRODUCTION
# 
# In this study, EDA (Exploratory Data Analysis) related to "Mushroom Classification" dataset will be made and various classification models such as Logistic Regression, Gaussian Naive Bayes, Decision Tree, Random Forest, SVM and K-NN will be created. Then the dataset will be fit with these models and the results will be observed. In addition, in this study, a simple level of dimensionality reduction will be performed by PCA (Principal Component Analysis) method which plays an important role in reducing the number of features. In addition to this, the hyperparameter will be tuned with the Grid Search technique for each model. We will plot Confusion Matrix and ROC Curve for each model. Dataset has 8124 mushrooms with 23 different attributes. We'll create models that find poisonous or edible mushrooms given feature with classification algorithms.

# <a id="2"></a>
# # 2. USED LIBRARIES
# 
# This section will give information about Python libraries to be used in the study and these libraries will be imported into the project. Here are the libraries and explanations we will use:
# 
# * **NumPy :** This library is actually a dependency for other libraries. The main purpose of this library is to provide a variety of mathematical operations on matrices and vectors in Python. Our project will be used this library to provide support to other libraries.
# * **Pandas :** This library performs import and processing of dataset in Python. In our project, it will be used to include the CSV extension dataset in the project and to perform various operations on it.
# * **Matplotlib :** This library, which is usually used to visualize data. It will perform the same task in our project.
# * **Seaborn :** This library which has similar features to Matplotlib is another library used for data visualization in Python. In our project, it will be used for the implementation of various features not included in the Matplotlib library.
# * **Sckit-Learn :** This library includes the implementation of various machine larning algorithms. With this library, we will perform all operations from building to evaluation of regression models using functions and classes in this library.
# 
# Now let's import NumPy, Pandas, Matplotlib and Seaborn libraries into our project and get them ready for use:

# In[ ]:


import numpy as np  # Importing NumPy library
import pandas as pd  # Importing Pandas library
import matplotlib.pyplot as plt  # Importing Matplotlib library's "pyplot" module
import seaborn as sns  # Imorting Seaborn library

# Ignore all warnings:
import warnings
warnings.filterwarnings("ignore")

# This lines for Kaggle:
import os
print(os.listdir("../input"))


# <a id="3"></a>
# # 3. DATA EXPLORATION
# 
# In this section, detailed information about dataset will be given and various visualizations will be made about features in dataset. In addition, during the "Data Preprocessing" step, the Label Encoding process will be performed. Because we need to convert them from "_object_" to numeric types so we can find the correlation between features. As you can see from this section, there will also be a correlation between features and will be plotted with the Heatmap.

# <a id="31"></a>
# ## 3.1. Detailed Information of the Dataset
# 
# First of all we import our dataset and look at the quantities related to the columns with the "_info_" method:

# In[ ]:


data = pd.read_csv("../input/mushrooms.csv")  # Read CSV file and load into "data" variable
data.info()  # Show detailed information for dataset columns(attributes)


# There are 8124 records in this dataset. Each record is indexed from 0 to 8123. In addition, there is no missing value in this dataset. In the "_stalk-root_" column, the data indicated by "_?_" will be taken as "_None_". Therefore, there is no need to work for missing values in data cleaning. There are 23 columns in total and all of them are "_object_" type. Now, let's explain what the columns mean:
# 
# * **class :** It is the column that we will use as the target, indicating whether the mushroom is edible or poisonous. The values and the meaning of values of this column are as follows:
# 	* _e :_ Edible
# 	* _p :_ Poisonous
# * **cap-shape :** It specifies the shape of the mushroom cap. The values that can be taken are:
# 	* _b :_ Bell
# 	* _c :_ Conical
# 	* _x :_ Convex
# 	* _f :_ Flat
# 	* _k :_ Knobbed
# 	* _s :_ Sunken or Depressed
# ![](https://datascienceplus.com/wp-content/uploads/2018/02/mushroom-cap-shape.jpg)
# <br>
# * **cap-surface :** It specifies the shape on the surface of the cap of the mushroom. The values that can be taken are:
# 	* _f :_ Fibrous
# 	* _g :_ Grooves
# 	* _y :_ Scaly
# 	* _s :_ Smooth
# ![](https://datascienceplus.com/wp-content/uploads/2018/02/mushroom-cap-surface.jpg)
# <br>
# * **cap-color :** It specifies the color of the cap of the mushroom. The values that can be taken are:
# 	* _n :_ Brown
# 	* _b :_ Buff
# 	* _c :_ Cinnamon
# 	* _g :_ Gray
# 	* _r :_ Green
# 	* _p :_ Pink
# 	* _u :_ Purple
# 	* _e :_ Red
# 	* _w :_ White
# 	* _y :_ Yellow
# * **bruises :** Specifies whether the mushroom has bruises. The values that can be taken are:
# 	* _t :_ Exists
# 	* _f :_ Not Exists
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Gyroporus_cyanescens_37226.jpg/640px-Gyroporus_cyanescens_37226.jpg)
# <br>
# * **odor :** Indicates the smell of mushrooms. The values that can be taken are:
# 	* _a :_ Almond
# 	* _l :_ Anise
# 	* _c :_ Creosote
# 	* _y :_ Fishy
# 	* _f :_ Foul
# 	* _m :_ Musty
# 	* _n :_ None
# 	* _p :_ Pungent
# 	* _s :_ Spicy
# * **gill-attachment :** It specifies the connection between mushroom gill and its stem. The values that can be taken are:
# 	* _a :_ Attached
# 	* _d :_ Descending
# 	* _f :_ Free
# 	* _n :_ Notched
# ![](https://datascienceplus.com/wp-content/uploads/2018/02/mushroom-gill-attachment.jpg)
# <br>
# * **gill-spacing :** It specifies the spaces among mushroom gills. The values that can be taken are:
# 	* _c :_ Close
# 	* _w :_ Crowded
# 	* _d :_ Distant
# ![](https://datascienceplus.com/wp-content/uploads/2018/02/mushroom-gill-spacing.jpg)
# <br>
# * **gill-size :** It specifies the size of mushroom gills. The values that can be taken are:
# 	* _b :_ Broad
# 	* _n :_ Narrow
# * **gill-color :** It specifies the color of mushroom gills. The values that can be taken are:
# 	* _k :_ Black
# 	* _n :_ Brown
# 	* _b :_ Buff
# 	* _h :_ Chocolate
# 	* _g :_ Gray
# 	* _r :_ Green
# 	* _o :_ Orange
# 	* _p :_ Pink
# 	* _u :_ Purple
# 	* _e :_ Red
# 	* _w :_ White
# 	* _y :_ Yellow
# * **stalk-shape :** It specifies the shape of the mushroom stem. The values that can be taken are:
# 	* _e :_ Enlarging
# 	* _t :_ Tapering
# * **stalk-root :** It specifies the shape of the mushroom root. The values that can be taken are:
# 	* _b :_ Bulbous
# 	* _c :_ Club Shaped
# 	* _u :_ Cup or Volva
# 	* _e :_ Equal
# 	* _z :_ Rhizomorphs
# 	* _r :_ Rooted
# 	* _? :_ Missing or None
# ![](https://datascienceplus.com/wp-content/uploads/2018/02/mushroom-stalk.jpg)
# <br>
# * **stalk-surface-above-ring :** It specifies the surface of the body piece above the ring. The values that can be taken are:
# 	* _f :_ Fibrous
# 	* _y :_ Scaly
# 	* _k :_ Silky
# 	* _s :_ Smooth
# * **stalk-surface-below-ring :** It specifies the surface of the body piece below the ring. The values that can be taken are:
# 	* _f :_ Fibrous
# 	* _y :_ Scaly
# 	* _k :_ Silky
# 	* _s :_ Smooth
# * **stalk-color-above-ring :** It specifies the color of the body piece above the ring. The values that can be taken are:
# 	* _n :_ Brown
# 	* _b :_ Buff
# 	* _c :_ Cinnamon
# 	* _g :_ Gray
# 	* _o :_ Orange
# 	* _p :_ Pink
# 	* _e :_ Red
# 	* _w :_ White
# 	* _y :_ Yellow
# * **stalk-color-below-ring :** It specifies the color of the body piece below the ring. The values that can be taken are:
# 	* _n :_ Brown
# 	* _b :_ Buff
# 	* _c :_ Cinnamon
# 	* _g :_ Gray
# 	* _o :_ Orange
# 	* _p :_ Pink
# 	* _e :_ Red
# 	* _w :_ White
# 	* _y :_ Yellow
# * **veil-type :** It specifies the veil type. The values that can be taken are:
# 	* _p :_ Partial
# 	* _u :_ Universal
# * **veil-color :** It specifies the veil color. The values that can be taken are:
# 	* _n :_ Brown
# 	* _o :_ Orange
# 	* _w :_ White
# 	* _y :_ Yellow
# * **ring-number :** It specifies the ring numbers in stem of the mushroom. The values that can be taken are:
# 	* _n :_ None
# 	* _o :_ One
# 	* _t :_ Two
# * **ring-type :** It specifies the shape of the ring in stem of the mushroom. The values that can be taken are:
# 	* _c :_ Cobwebby
# 	* _e :_ Evanescent
# 	* _f :_ Flaring
# 	* _l :_ Large
# 	* _n :_ None
# 	* _p :_ Pendant
# 	* _s :_ Sheathing
# 	* _z :_ Zone
# ![](https://datascienceplus.com/wp-content/uploads/2018/02/mushroom-ring-type.jpg)
# <br>
# * **spore-print-color :** It specifies the color of mushroom spores. The values that can be taken are:
# 	* _k :_ Black
# 	* _n :_ Brown
# 	* _b :_ Buff
# 	* _h :_ Chocolate
# 	* _r :_ Green
# 	* _o :_ Orange
# 	* _u :_ Purple
# 	* _w :_ White
# 	* _y :_ Yellow
# * **population :** It specifies the rate at which the mushroom is present in the population. The values that can be taken are:
# 	* _n :_ Numerous	
# 	* _a :_ Abundant
# 	* _v :_ Several
# 	* _c :_ Clustered
# 	* _s :_ Scattered
# 	* _y :_ Solitary
# * **habitat :** It specifies the habitat the mushroom is in. The values that can be taken are:
# 	* _g :_ Grasses
# 	* _l :_ Leaves
# 	* _m :_ Meadows
# 	* _p :_ Paths
# 	* _u :_ Urban
# 	* _w :_ Waste
# 	* _d :_ Woods
# 
# We gave the necessary information about dataset. Now, looking at the first 5 and last 5 entries of dataset, what are the values that are being held:

# In[ ]:


data.head()  # Prints first 5 entries of the dataset


# In[ ]:


data.tail()  # Prints last 5 entries of the dataset


# For correlation and many statistical analyzes, data must be of a numerical type. So before going on to the next step, we'll do the Label Encoding event here. Label Encoding is the conversion of categorical data to the corresponding integers.

# <a id="32"></a>
# ## 3.2. Label Encoding
# 
# For Label Encoding we will use the "_LabelEncoder_" class of the Scikit-Learn library. With the "*fit_transform*" method of this class, we will be able to cycle and transform values through all columns. Let's write our code as follows:

# In[ ]:


from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder class

label_encoder = LabelEncoder()  # Create a instance for the label encoder
encoded_data = pd.DataFrame()  # Create empty DataFrame

for column in data.columns:
    encoded_data[column] = label_encoder.fit_transform(data[column])  # Iterate all columns and transform its values


# Now let's take a look at the data we encode:

# In[ ]:


encoded_data.head()  # Print first 5 record for the encoded data


# We kept them in separate variables. Because sometimes categorical data will be useful in some statistics. Finally, let's examine the output of our "_describe_" function and go to the next section:

# In[ ]:


encoded_data.describe()  # Print some statistics for data


# <a id="33"></a>
# ## 3.3. Correlation of Columns(Attributes)
# 
# In this section, we'll find the correlation matrix between the columns and we'll visualize it into a Heatmap. In this way, we will be able to see the relationship between the attributes more clearly and visualize them in the future.

# In[ ]:


encoded_data.corr()  # Prints correlation matrix


# Now we visualize this correlation matrix with Heatmap:

# In[ ]:


fig, axes = plt.subplots(figsize=(18, 18))  # This method creates a figure and a set of subplots
sns.heatmap(data=encoded_data.corr(), annot=True, linewidths=.5, cmap="coolwarm", ax=axes)  # Figure out heatmap
plt.show()  # Shows only plot and remove other informations


# We cannot see any statistical data or any correlation in the "*veil-type*" column. This is because this attribute has a single value for all records. This means that it is a useless column for us. In the future we will drop this column from dataset.

# <a id="34"></a>
# ## 3.4. Various Visualizations from Dataset
# 
# In this section we will perform various visualization operations related to dataset. First, let's take a look at the statistics of the columns with a correlation greater than 0.3 and less than -0.3 and the "_class_" column:

# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 13))  # Adjust subplots

# Draw frequency of the "bruises" values according to "class":
bruises_bar = sns.countplot(x="bruises", hue="class", data=data, ax=axes[0][0]);
bruises_bar.set_xticklabels(["True", "False"])

# Draw frequency of the "gill-spacing" values according to "class":
gill_spacing_bar = sns.countplot(x="gill-spacing", hue="class", data=data, ax=axes[0][1]);
gill_spacing_bar.set_xticklabels(["Close", "Crowded", "Distant"])

# Draw frequency of the "gill-size" values according to "class":
gill_size_bar = sns.countplot(x="gill-size", hue="class", data=data, ax=axes[0][2]);
gill_size_bar.set_xticklabels(["Narrow", "Broad"])

# Draw frequency of the "gill-color" values according to "class":
gill_color_bar = sns.countplot(x="gill-color", hue="class", data=data, ax=axes[1][0]);
gill_color_bar.set_xticklabels(["Black", "Brown", "Gray", "Pink", "White", "Choco", "Purple", "Red", "Buff", "Green", "Yellow", "Orange"], rotation=60)

# Draw frequency of the "stalk-root" values according to "class":
stalk_root_bar = sns.countplot(x="stalk-root", hue="class", data=data, ax=axes[1][1]);
stalk_root_bar.set_xticklabels(["Equal", "Club", "Bulbous", "Rooted", "None"], rotation=60)

# Draw frequency of the "stalk-surface-above-ring" values according to "class":
stalk_sar_bar = sns.countplot(x="stalk-surface-above-ring", hue="class", data=data, ax=axes[1][2]);
stalk_sar_bar.set_xticklabels(["Smooth", "Fibrous", "Silky", "Scaly"], rotation=60)

# Draw frequency of the "stalk-surface-below-ring" values according to "class":
stalk_sbr_bar = sns.countplot(x="stalk-surface-below-ring", hue="class", data=data, ax=axes[2][0]);
stalk_sbr_bar.set_xticklabels(["Smooth", "Fibrous", "Silky", "Scaly"], rotation=60)

# Draw frequency of the "ring-type" values according to "class":
ring_type_bar = sns.countplot(x="ring-type", hue="class", data=data, ax=axes[2][1]);
ring_type_bar.set_xticklabels(["Pendant", "Evanescent", "Large", "Flaring", "None"], rotation=60)

# Draw frequency of the "population" values according to "class":
population_bar = sns.countplot(x="population", hue="class", data=data, ax=axes[2][2]);
population_bar.set_xticklabels(["Scattered", "Numerous", "Abundant", "Several", "Solitary", "Clustered"], rotation=60)

fig.tight_layout()  # Slightly spacing between axis labels and values
plt.show()


# Now let's show the distribution of the edible and poisonous classes with the Pie Chart:

# In[ ]:


poisonous_count = len(data[data["class"] == "p"].index)  # Get poisonous count
edible_count = len(data[data["class"] == "e"].index)  # Get edible count

# Draw Pie Chart:
plt.pie([poisonous_count, edible_count], labels=["Poisonous", "Edible"], autopct='%1.1f%%', radius=2.0, shadow=True, colors=["r", "g"])
plt.show()


# Finally, let's show with the Stacked Bar Chart how much poisonous mushrooms are found in the habitats:

# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))  # For specify figure size
data.groupby(['habitat', 'class']).size().unstack().plot.bar(stacked=True, ax=ax)  # Draw Stacked Bar Chart
plt.show()


# You can find out what letter in the above description. Let's move on to the next topic without much visualization. Because this is the main subject of the kernel classification algorithms and their comparison to the foreground should be more important.

# <a id="4"></a>
# # 4. DATA PREPARATION AND CLEANING
# 
# In this section, we will perform data cleaning and preprocessing. Since I'm going to use Column names as a string, I won't make any edits to them. Also, because missing values are not available in this dataset, I also skip "Handling Missing Values". In this section we will see the PCA (Principal Component Analysis) topic.

# <a id="41"></a>
# ## 4.1. Drop Irrelevant Columns
# 
# We'll delete columns from dataset, which doesn't mean much to us here. In this section and later sections, we will go through "encoded_data" which has Label Encoding process and fit to the models accordingly. Now, delete the "_cap-shape_" and "_veil-type_" columns that don't have much to do with other attributes from the dataset:

# In[ ]:


encoded_data.drop(["cap-shape", "veil-type"], axis=1, inplace=True)  # Drop "cap-shape" and "veil-type" columns from dataset
encoded_data.columns


# <a id="42"></a>
# ## 4.2. Split Data and Target
# 
# Here we separate the data and target parts of the dataset and assign them to variables:

# In[ ]:


X = encoded_data.drop(["class"], axis=1)  # Put all data (except "class" column) to the X variable
y = encoded_data["class"] # Put only "class" column to the Y variable


# In[ ]:


X.head()


# In[ ]:


y.head()


# <a id="43"></a>
# ## 4.3. Scaling of Data
# 
# Here we need to standardize them so that we can use the data properly. To do this, we need to apply the Scaling process to the data. Scikit-Learn's "_StandardScaler_" class can be used for this process:

# In[ ]:


from sklearn.preprocessing import StandardScaler  # Import StandartScaler class
std_scaler = StandardScaler()  # Create instance for scaler

X = std_scaler.fit_transform(X)  # Fit and transform data with scaler


# <a id="44"></a>
# ## 4.4 Principal Component Analysis (PCA)
# 
# Although the PCA method is used for many different purposes, it is usually used to reduce the number of features. We will also use it here for this purpose. I would like to remind you that after the last steps we have 20 features. Now we'll reduce the number of these features, but we will try to keep the variance high while doing so. Because if the variance value of our data falls, information loss occurs and this is undesirable. Then let's first see how much variance you have with PCA on how many features:

# In[ ]:


from sklearn.decomposition import PCA  # Import class for PCA

for i in range(2, 20):
    pca = PCA(n_components=i)  # n_components = Specify the number of features you want to reduce.
    pca.fit_transform(X)
    print("Sum of Variance Ratio in " + str(i) + " Components: ", sum(pca.explained_variance_ratio_))


# As seen above, we can maintain approximately 93% of our data by using 13 features. For this reason, we will give the number of components as 13:

# In[ ]:


pca = PCA(n_components=13)  # We will reduce the feature count to the 13
X = pca.fit_transform(X)  # Fit and transform with data

print(sum(pca.explained_variance_ratio_))  # Print variance ratio


# <a id="45"></a>
# ## 4.5. Preparation of Test and Train Data
# 
# The final process here is the smooth and random separation of test and train data. For this, we will benefit from the method named "train_testsplit" from the Scikit-Learn library. I would like to use 20% of our data for testing and 80% for training purposes. The process is very simple:

# In[ ]:


from sklearn.model_selection import train_test_split  # Import "train_test_split" method

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Parameters:
# test_size : It decides how many test data in percentage.
# random_state : This parameter can take any value. This value decides randomness seed.


# <a id="5"></a>
# # 5. BUILDING MODELS
# 
# In this section we will fit various classification models with our data. We will also perform the process of tunning the hyperparameter of the models. In doing so, we will use the Grid Search method. We will create new models that are tune with the parameters we found and we will fit our data to them. The classification algorithms that we will use in this kernel are:
# 
# 1. Logistic Regression
# 1. Gaussian Naive Bayes Classification
# 1. Decision Tree Classification
# 1. Random Forest Classification
# 1. Support Vector Classification
# 1. K-Nearest Neighbour Classification
# 
# Now let's do our import operations for these models and for Grid Search Cross Validation:

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# <a id="51"></a>
# ## 5.1. Building Models with Default Hyperparameters
# 
# In this section, we will fit the models by creating them with the default hyperparameters. After creating these models and fit with our data, we will calculate train and test scores. Last but not least, we will determine the hyperparameters that we will tune.

# <a id="511"></a>
# ### 5.1.1. Logistic Regression
# 
# First, we create the default model and fit it with our data. Then we calculate train and test scores:

# In[ ]:


logistic_regression_cls = LogisticRegression(random_state=13)  # Create instance for model
logistic_regression_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for Logistic Regression: ", logistic_regression_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for Logistic Regression: ", logistic_regression_cls.score(x_test, y_test))  # Print Test Score


# Now let's list the parameters and current values of the Logistic Regression model:

# In[ ]:


logistic_regression_cls.get_params()  # Print hyperparameters and their values for the model


# We will do the operations related to "*C*" and "_penalty_" parameters.

# <a id="512"></a>
# ### 5.1.2. Gaussian Naive Bayes
# 
# First, we create the default model and fit it with our data. Then we calculate train and test scores:

# In[ ]:


naive_bayes_cls = GaussianNB()  # Create instance for model
naive_bayes_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for Gaussian Naive Bayes: ", naive_bayes_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for Gaussian Naive Bayes: ", naive_bayes_cls.score(x_test, y_test))  # Print Test Score


# Now let's list the parameters and current values of Gaussian Naive Bayes Classifier model:

# In[ ]:


naive_bayes_cls.get_params()  # Print hyperparameters and their values for the model


# As you can see, the Gaussian Naive Bayes model does not have any hyperparameter to tune, so we don't need to take any action on it.

# <a id="513"></a>
# ### 5.1.3. Decision Tree
# 
# First, we create the default model and fit it with our data. Then we calculate train and test scores:

# In[ ]:


decision_tree_cls = DecisionTreeClassifier(random_state=13)  # Create instance for model
decision_tree_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for Decision Tree: ", decision_tree_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for Decision Tree: ", decision_tree_cls.score(x_test, y_test))  # Print Test Score


# Now let's list the parameters and current values of Decision Tree Classifier model:

# In[ ]:


decision_tree_cls.get_params()  # Print hyperparameters and their values for the model


# We will only perform operations related to the "*max_depth*" parameter.

# <a id="514"></a>
# ### 5.1.4. Random Forest
# 
# First, we create the default model and fit it with our data. Then we calculate train and test scores:

# In[ ]:


random_forest_cls = RandomForestClassifier(random_state=13)  # Create instance for model
random_forest_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for Random Forest: ", random_forest_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for Random Forest: ", random_forest_cls.score(x_test, y_test))  # Print Test Score


# Now let's list the parameters and current values of Random Forest Classifier model:

# In[ ]:


random_forest_cls.get_params()  # Print hyperparameters and their values for the model


# We will do the operations with parameters "*n_estimators*" and "*max_depth*".

# <a id="515"></a>
# ### 5.1.5. SVC
# 
# First, we create the default model and fit it with our data. Then we calculate train and test scores:

# In[ ]:


support_vector_cls = SVC(random_state=13)  # Create instance for model
support_vector_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for SVC: ", support_vector_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for SVC: ", support_vector_cls.score(x_test, y_test))  # Print Test Score


# Now let's list the parameters and current values of the SVC model:

# In[ ]:


support_vector_cls.get_params()  # Print hyperparameters and their values for the model


# We will do the operations related to "*C*" and "*gamma*" parameters. The "kernel" parameter will remain "*rbf*" by default.

# <a id="516"></a>
# ### 5.1.6. K-NN
# 
# First, we create the default model and fit it with our data. Then we calculate train and test scores:

# In[ ]:


knn_cls = KNeighborsClassifier()  # Create instance for model
knn_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for K-NN: ", knn_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for K-NN: ", knn_cls.score(x_test, y_test))  # Print Test Score


# Now let's list the parameters and current values of K-NN Classifier model:

# In[ ]:


knn_cls.get_params()  # Print hyperparameters and their values for the model


# We will only perform the "*n_neighbors*" parameter.

# <a id="52"></a>
# ## 5.2. Hyperparameter Tuning with Grid Search
# 
# In this section, we will use the Grid Search technique to tune the hyperparameter of the models. In addition, 6-Fold Cross Validation process will find the scores of the parameter combinations. Finally, we will decide what the best parameters are for the model.

# <a id="521"></a>
# ### 5.2.1. Logistic Regression

# In[ ]:


# Specifying hyperparameters' range for the model:
parameters_LR = {"C" : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                 "penalty" : ["l1", "l2"]}

# Create a Grid Search Cross Validation instance ("n_jobs=-1" means use all cores of the processor):
grid_search_LR = GridSearchCV(logistic_regression_cls, parameters_LR, cv=6, n_jobs=-1, return_train_score=True)

# Fit Grid Search model with data:
grid_search_LR.fit(x_train, y_train)

# Show results for all combinations:
pd.DataFrame(grid_search_LR.cv_results_)[["params", "mean_fit_time", "mean_train_score", "mean_test_score"]]


# In[ ]:


grid_search_LR.best_params_  # Print best hyperparameters for the model


# <a id="522"></a>
# ### 5.2.2. Decision Tree

# In[ ]:


# Specifying hyperparameters' range for the model:
parameters_DT = {"max_depth" : np.arange(3, 20)}

# Create a Grid Search Cross Validation instance ("n_jobs=-1" means use all cores of the processor):
grid_search_DT = GridSearchCV(decision_tree_cls, parameters_DT, cv=6, n_jobs=-1, return_train_score=True)

# Fit Grid Search model with data:
grid_search_DT.fit(x_train, y_train)

# Show results for all combinations:
pd.DataFrame(grid_search_DT.cv_results_)[["params", "mean_fit_time", "mean_train_score", "mean_test_score"]]


# In[ ]:


grid_search_DT.best_params_  # Print best hyperparameters for the model


# <a id="523"></a>
# ### 5.2.3. Random Forest

# In[ ]:


# Specifying hyperparameters' range for the model:
parameters_RF = {"max_depth" : np.arange(5, 15),
                 "n_estimators" : [100, 200, 300]}

# Create a Grid Search Cross Validation instance ("n_jobs=-1" means use all cores of the processor):
grid_search_RF = GridSearchCV(random_forest_cls, parameters_RF, cv=6, n_jobs=-1, return_train_score=True)

# Fit Grid Search model with data:
grid_search_RF.fit(x_train, y_train)

# Show results for all combinations:
pd.DataFrame(grid_search_RF.cv_results_)[["params", "mean_fit_time", "mean_train_score", "mean_test_score"]]


# In[ ]:


grid_search_RF.best_params_  # Print best hyperparameters for the model


# <a id="524"></a>
# ### 5.2.4. SVC (with RBF)

# In[ ]:


# Specifying hyperparameters' range for the model:
parameters_SVC = {"C" : [0.1, 1, 10, 100],
                 "gamma" : [0.001, 0.01, 0.1, 1]}

# Create a Grid Search Cross Validation instance ("n_jobs=-1" means use all cores of the processor):
grid_search_SVC = GridSearchCV(support_vector_cls, parameters_SVC, cv=6, n_jobs=-1, return_train_score=True)

# Fit Grid Search model with data:
grid_search_SVC.fit(x_train, y_train)

# Show results for all combinations:
pd.DataFrame(grid_search_SVC.cv_results_)[["params", "mean_fit_time", "mean_train_score", "mean_test_score"]]


# In[ ]:


grid_search_SVC.best_params_  # Print best hyperparameters for the model


# <a id="525"></a>
# ### 5.2.5. K-NN

# In[ ]:


# Specifying hyperparameters' range for the model:
parameters_KNN = {"n_neighbors" : np.arange(2, 30)}

# Create a Grid Search Cross Validation instance ("n_jobs=-1" means use all cores of the processor):
grid_search_KNN = GridSearchCV(knn_cls, parameters_KNN, cv=6, n_jobs=-1, return_train_score=True)

# Fit Grid Search model with data:
grid_search_KNN.fit(x_train, y_train)

# Show results for all combinations:
pd.DataFrame(grid_search_KNN.cv_results_)[["params", "mean_fit_time", "mean_train_score", "mean_test_score"]]


# In[ ]:


grid_search_KNN.best_params_  # Print best hyperparameters for the model


# <a id="53"></a>
# ## 5.3. Rebuild Models with Tuned Hyperparameters
# 
# In this section, models with the best hyperparameter will be rebuilded and fitted with the data. Then the test data and test scores will be calculated. Let's do all this with one code:

# In[ ]:


# Logistic Regression Classifier:
logistic_regression_cls_tuned = LogisticRegression(C=0.1, penalty="l2", random_state=13)
logistic_regression_cls_tuned.fit(x_train, y_train)

# Gaussian Naive Bayes Classifier (Not Changed):
naive_bayes_cls_tuned = GaussianNB()
naive_bayes_cls_tuned.fit(x_train, y_train)

# Decision Tree Classifier:
decision_tree_cls_tuned = DecisionTreeClassifier(max_depth=14, random_state=13)
decision_tree_cls_tuned.fit(x_train, y_train)

# Random Forest Classifier:
random_forest_cls_tuned = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=13)
random_forest_cls_tuned.fit(x_train, y_train)

# Support Vector Classifier:
support_vector_cls_tuned = SVC(kernel="rbf", C=1, gamma=0.1, random_state=13)
support_vector_cls_tuned.fit(x_train, y_train)

# K-Nearest Neighbors Classifier:
knn_cls_tuned = KNeighborsClassifier(n_neighbors=3)
knn_cls_tuned.fit(x_train, y_train)

# Find test accuracy for all models:
print("Test Score for Logistic Regression: ", logistic_regression_cls_tuned.score(x_test, y_test))
print("Test Score for Gaussian Naive Bayes: ", naive_bayes_cls_tuned.score(x_test, y_test))
print("Test Score for Decision Tree: ", decision_tree_cls_tuned.score(x_test, y_test))
print("Test Score for Random Forest: ", random_forest_cls_tuned.score(x_test, y_test))
print("Test Score for SVC: ", support_vector_cls_tuned.score(x_test, y_test))
print("Test Score for K-NN: ", knn_cls_tuned.score(x_test, y_test))


# <a id="6"></a>
# # 6. EVALUATING MODELS
# 
# In this section we will do various measurements to test the performance of the models. Confusion Matrix for each model to be printed. Then there will be values such as precision and f1 score. To do this, we will take advantage of a function found in the Scikit-Learn library, which finds all of these values at the same time. Finally, the ROC Curve of the corresponding model will be drawn. Let's start by talking about some basic concepts. First, let's see what the Confusion Matrix is:
# 
# ![](https://cdncontribute.geeksforgeeks.org/wp-content/uploads/Confusion_Matrix1_1.png)
# 
# The "Actual" lines that you see in this image indicate the actual values of the classes, while the "Predicted" columns specify the estimates of the classes from the model. In this case, the values and meanings of the table are as follows:
# 
# * **True Positive(TP) :** Specifies that the data count belonging to the "Class 1" class is estimated to be "Class 1".
# * **True Negative(TN) :** Specifies that the data count belonging to the "Class 2" class is estimated to be "Class 2".
# * **False Positive(FP) :** Specifies that the data count belonging to the "Class 2" class is estimated to be "Class 1". (Type I Error)
# * **False Negative(FN) :** Specifies that the data count belonging to the "Class 1" class is estimated to be "Class 2". (Type II Error)
# 
# Some important information and formulas can be extracted from this matrix. Some of these formulas and their meanings are as follows:
# 
# * **Accuracy :** Specifies the percentage of total correct estimates. Formula: "_Accuracy = (TP + TN) / (TP + TN + FP + FN)_"
# * **Classification Error :** Specifies the percentage of total wrong estimates. Formula: "_Classification Error = (FP + FN) / (TP + TN + FP + FN)_"
# * **Sensitivity (True Positive Rate, Recall) :** Assuming that all values tested are "Class 1", it indicates the percentage of correct estimates. Formula: "_Recall = (TP) / (TP + FN)_"
# * **Specificity :** Assuming that all values tested are "Class 2", it indicates the percentage of correct estimates. Formula: "_Specificity = (TN) / (TN + FP)_"
# * **False Positive Rate :** Assuming that all values tested are "Class 2", it indicates the percentage of wrong estimates. Formula: "_True Positive Rate = (FP) / (TN + FP)_"
# * **Precision :** Specifies the proportion of correct estimates only in a case where the class "Class 1" is estimated. Formula: "_Precision = (TP) / (TP + FP)_"
# * **F1 Score :** Specifies the harmonic mean of Recall and Precision values. Formula: "_F1 Score = (2 * Precision * Recall) / (Precision + Recall)_"
# 
# Besides these values, there is a value called "Threshold" or "Cut-off". This value is usually valid in Binary Classification. Specifies the probability that the prediction in Binary Classification is true and the default value is 0.5. We will not take any action on optimization of the Threshold value here. Let's start by importing the library and functions we will use now:

# In[ ]:


from sklearn.metrics import confusion_matrix  # For find confusion matrix
from sklearn.metrics import classification_report  # For print evaluation report
from sklearn.metrics import roc_curve  # For drawing ROC curve
from sklearn.metrics import auc  # For find AUC


# <a id="61"></a>
# ## 6.1. Evaluating Logistic Regression Model
# 
# After giving the model test data and prediction, let's take out the Confusion Matrix for our model:

# In[ ]:


# Prediction of test dataset:
y_pred_LR = logistic_regression_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_LR = confusion_matrix(y_test, y_pred_LR)

# Plot confusion matrix with Heatmap:
cm_dataframe_LR = pd.DataFrame(confusion_matrix_LR, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_LR, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("Logistic Regression")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()


# Now let's print the various statistics that determine the performance of our model as a table:

# In[ ]:


report_LR = pd.DataFrame(classification_report(y_test, y_pred_LR, 
                                               output_dict=True, 
                                               target_names=["Edible", "Poisonous"]))
report_LR


# Finally, draw the ROC Curve and find the AUC (Area Under Curve) value:

# In[ ]:


# Find parameters for drawing ROC curve:
false_positive_rate_LR, true_positive_rate_LR, thresholds_LR = roc_curve(y_test, y_pred_LR)

# Find AUC value:
auc_LR = auc(false_positive_rate_LR, true_positive_rate_LR)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_LR, true_positive_rate_LR, label="AUC = %0.2f"%auc_LR)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for Logistic Regression")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")


# <a id="62"></a>
# ## 6.2. Evaluating Gaussian Naive Bayes Model
# 
# After giving the model test data and prediction, let's take out the Confusion Matrix for our model:

# In[ ]:


# Prediction of test dataset:
y_pred_GNB = naive_bayes_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_GNB = confusion_matrix(y_test, y_pred_GNB)

# Plot confusion matrix with Heatmap:
cm_dataframe_GNB = pd.DataFrame(confusion_matrix_GNB, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_GNB, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("Gaussian Naive Bayes")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()


# Now let's print the various statistics that determine the performance of our model as a table:

# In[ ]:


report_GNB = pd.DataFrame(classification_report(y_test, y_pred_GNB, 
                                                output_dict=True, 
                                                target_names=["Edible", "Poisonous"]))
report_GNB


# Finally, draw the ROC Curve and find the AUC (Area Under Curve) value:

# In[ ]:


# Find parameters for drawing ROC curve:
false_positive_rate_GNB, true_positive_rate_GNB, thresholds_GNB = roc_curve(y_test, y_pred_GNB)

# Find AUC value:
auc_GNB = auc(false_positive_rate_GNB, true_positive_rate_GNB)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_GNB, true_positive_rate_GNB, label="AUC = %0.2f"%auc_GNB)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for Gaussian Naive Bayes")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")


# <a id="63"></a>
# ## 6.3. Evaluating Decision Tree Classifier Model
# 
# After giving the model test data and prediction, let's take out the Confusion Matrix for our model:

# In[ ]:


# Prediction of test dataset:
y_pred_DT = decision_tree_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_DT = confusion_matrix(y_test, y_pred_DT)

# Plot confusion matrix with Heatmap:
cm_dataframe_DT = pd.DataFrame(confusion_matrix_DT, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_DT, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("Decision Tree Classifier")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()


# Now let's print the various statistics that determine the performance of our model as a table:

# In[ ]:


report_DT = pd.DataFrame(classification_report(y_test, y_pred_DT, 
                                               output_dict=True, 
                                               target_names=["Edible", "Poisonous"]))
report_DT


# Finally, draw the ROC Curve and find the AUC (Area Under Curve) value:

# In[ ]:


# Find parameters for drawing ROC curve:
false_positive_rate_DT, true_positive_rate_DT, thresholds_DT = roc_curve(y_test, y_pred_DT)

# Find AUC value:
auc_DT = auc(false_positive_rate_DT, true_positive_rate_DT)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_DT, true_positive_rate_DT, label="AUC = %0.2f"%auc_DT)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for Decision Tree Classifier")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")


# <a id="64"></a>
# ## 6.4. Evaluating Random Forest Classifier Model
# 
# After giving the model test data and prediction, let's take out the Confusion Matrix for our model:

# In[ ]:


# Prediction of test dataset:
y_pred_RF = random_forest_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_RF = confusion_matrix(y_test, y_pred_RF)

# Plot confusion matrix with Heatmap:
cm_dataframe_RF = pd.DataFrame(confusion_matrix_RF, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_RF, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("Random Forest Classifier")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()


# Now let's print the various statistics that determine the performance of our model as a table:

# In[ ]:


report_RF = pd.DataFrame(classification_report(y_test, y_pred_RF, 
                                               output_dict=True, 
                                               target_names=["Edible", "Poisonous"]))
report_RF


# Finally, draw the ROC Curve and find the AUC (Area Under Curve) value:

# In[ ]:


# Find parameters for drawing ROC curve:
false_positive_rate_RF, true_positive_rate_RF, thresholds_RF = roc_curve(y_test, y_pred_RF)

# Find AUC value:
auc_RF = auc(false_positive_rate_RF, true_positive_rate_RF)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_RF, true_positive_rate_RF, label="AUC = %0.2f"%auc_RF)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for Random Forest Classifier")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")


# <a id="65"></a>
# ## 6.5. Evaluating Support Vector Classifier Model
# 
# After giving the model test data and prediction, let's take out the Confusion Matrix for our model:

# In[ ]:


# Prediction of test dataset:
y_pred_SVC = support_vector_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_SVC = confusion_matrix(y_test, y_pred_SVC)

# Plot confusion matrix with Heatmap:
cm_dataframe_SVC = pd.DataFrame(confusion_matrix_SVC, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_SVC, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("SVC")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()


# Now let's print the various statistics that determine the performance of our model as a table:

# In[ ]:


report_SVC = pd.DataFrame(classification_report(y_test, y_pred_SVC, 
                                                output_dict=True, 
                                                target_names=["Edible", "Poisonous"]))
report_SVC


# Finally, draw the ROC Curve and find the AUC (Area Under Curve) value:

# In[ ]:


# Find parameters for drawing ROC curve:
false_positive_rate_SVC, true_positive_rate_SVC, thresholds_SVC = roc_curve(y_test, y_pred_SVC)

# Find AUC value:
auc_SVC = auc(false_positive_rate_SVC, true_positive_rate_SVC)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_SVC, true_positive_rate_SVC, label="AUC = %0.2f"%auc_SVC)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for SVC")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")


# <a id="66"></a>
# ## 6.6. Evaluating K-Nearest Neighbors Classifier Model
# 
# After giving the model test data and prediction, let's take out the Confusion Matrix for our model:

# In[ ]:


# Prediction of test dataset:
y_pred_KNN = knn_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_KNN = confusion_matrix(y_test, y_pred_KNN)

# Plot confusion matrix with Heatmap:
cm_dataframe_KNN = pd.DataFrame(confusion_matrix_KNN, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_KNN, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("K-NN Classifier")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()


# Now let's print the various statistics that determine the performance of our model as a table:

# In[ ]:


report_KNN = pd.DataFrame(classification_report(y_test, y_pred_KNN, 
                                                output_dict=True, 
                                                target_names=["Edible", "Poisonous"]))
report_KNN


# Finally, draw the ROC Curve and find the AUC (Area Under Curve) value:

# In[ ]:


# Find parameters for drawing ROC curve:
false_positive_rate_KNN, true_positive_rate_KNN, thresholds_KNN = roc_curve(y_test, y_pred_KNN)

# Find AUC value:
auc_KNN = auc(false_positive_rate_KNN, true_positive_rate_KNN)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_KNN, true_positive_rate_KNN, label="AUC = %0.2f"%auc_KNN)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for K-NN Classifier")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")


# <a id="7"></a>
# # 7. EXPLORATION OF RESULTS
# 
# In this section we will examine the results from the models. First, let's show the results of all models in a single table:

# In[ ]:


# Create list that keep model names:
model_names = ["Logistic Regression", 
               "Gaussian Naive Bayes",
               "Decision Tree Classification",
               "Random Forest Classification",
               "Support Vector Classification",
               "K-NN Classifiaction"]

# Shortening models' variables names for quick usage:
LR_model = logistic_regression_cls_tuned
GNB_model = naive_bayes_cls_tuned
DT_model = decision_tree_cls_tuned
RF_model = random_forest_cls_tuned
SVC_model = support_vector_cls_tuned
KNN_model = knn_cls_tuned

# Get model hyperparameters into variables:
LR_model_C = str(LR_model.get_params()["C"])
LR_model_penalty = str(LR_model.get_params()["penalty"])
DT_mode_max_depth = str(DT_model.get_params()["max_depth"])
RF_model_n_estimators = str(RF_model.get_params()["n_estimators"])
RF_model_max_depth = str(RF_model.get_params()["max_depth"])
SVC_model_kernel = str(SVC_model.get_params()["kernel"])
SVC_model_C = str(SVC_model.get_params()["C"])
SVC_model_gamma = str(SVC_model.get_params()["gamma"])
KNN_model_n_neighbors = str(KNN_model.get_params()["n_neighbors"])

# Create list that keep model hyperparameters:
model_params = ["C=" + LR_model_C + ", penalty=" + LR_model_penalty,
                "None",
                "max_depth=" + DT_mode_max_depth,
                "n_estimators=" + RF_model_n_estimators + ", max_depth=" + RF_model_max_depth,
                "kernel=" + SVC_model_kernel + ", C=" + SVC_model_C + ", gamma=" + SVC_model_gamma,
                "n_neighbors=" + KNN_model_n_neighbors]

# Create list that keep models' training accuracies:
model_training_accuracies = [LR_model.score(x_train, y_train),
                             GNB_model.score(x_train, y_train),
                             DT_model.score(x_train, y_train),
                             RF_model.score(x_train, y_train),
                             SVC_model.score(x_train, y_train),
                             KNN_model.score(x_train, y_train)]

# Create list that keep models' testing accuracies:
model_testing_accuracies = [LR_model.score(x_test, y_test),
                             GNB_model.score(x_test, y_test),
                             DT_model.score(x_test, y_test),
                             RF_model.score(x_test, y_test),
                             SVC_model.score(x_test, y_test),
                             KNN_model.score(x_test, y_test)]

# Create list that keep models' F1 scores:
model_f1_scores = [report_LR.iloc[0]["weighted avg"],
                   report_GNB.iloc[0]["weighted avg"],
                   report_DT.iloc[0]["weighted avg"],
                   report_RF.iloc[0]["weighted avg"],
                   report_SVC.iloc[0]["weighted avg"],
                   report_KNN.iloc[0]["weighted avg"]]

# Create list that keep models' precisions:
model_precisions = [report_LR.iloc[1]["weighted avg"],
                   report_GNB.iloc[1]["weighted avg"],
                   report_DT.iloc[1]["weighted avg"],
                   report_RF.iloc[1]["weighted avg"],
                   report_SVC.iloc[1]["weighted avg"],
                   report_KNN.iloc[1]["weighted avg"]]

# Create list that keep models' recalls:
model_recalls = [report_LR.iloc[2]["weighted avg"],
                report_GNB.iloc[2]["weighted avg"],
                report_DT.iloc[2]["weighted avg"],
                report_RF.iloc[2]["weighted avg"],
                report_SVC.iloc[2]["weighted avg"],
                report_KNN.iloc[2]["weighted avg"]]

# Create list that keep models' AUC values:
model_AUC_values = [auc_LR, auc_GNB, auc_DT, auc_RF, auc_SVC, auc_KNN]

# Generate table data with column names:
table_data = {"Parameters" : model_params,
              "Training Accuracy" : model_training_accuracies,
              "Testing Accuracy" : model_testing_accuracies,
              "F1 Score" : model_f1_scores,
              "Precision" : model_precisions,
              "Recall" : model_recalls,
              "AUC" : model_AUC_values}

# Create and print result table:
table_dataframe = pd.DataFrame(data=table_data, index=model_names)
table_dataframe


# As you can see, the results are quite nice. The reason for this is that the dataset has a sufficient number of data, or it can be tuned hyperparameters or PCA (Principal Component Analysis). Finally, let's graph all these results and finish the section:

# In[ ]:


table_dataframe.iloc[:, 1:].plot(kind="bar", ylim=[0.8, 1.0], figsize=(14, 9)) # Y Limit: 0.8 - 1.0
plt.legend(loc='lower right')
plt.show()


# <a id="8"></a>
# # 8. CONCLUSION
# 
# This kernel contains 6 different classification algorithms and works on "Mushroom Classification" dataset. In addition, EDA (Exploratory Data Analysis) was performed for dataset. I tried to give you the following information on this kernel:
# 
# * How to make a dataset's EDA (Exploratory Data Analysis) process simply
# * How libraries such as Pandas, Matplotlib, Seaborn and Scikit-Learn are used in classification models
# * How to convert category data to numerical data with Label Encoding
# * How we can graphically show the relationship between the various columns
# * How to draw graphics such as Pie Chart, Bar Chart, Stacked Bar Chart and Heatmap with Seaborn and Matplotlib
# * How to perform Scaling on data
# * What is Principal Component Analysis (PCA) and how to do it simply
# * Installation, use, optimization and evaluation of models such as Logistic Regression, Gaussian Naive Bayes, Decision Tree Classification, Random Forest Classification, SVC and K-NN Classification
# * Performing Hyperparameter Tuning to the models with the Grid Search method and finding the most suitable parameters
# * Extraction of Confusion Matrix for classification models and detailed description of Confusion Matrix
# * Calculation of classification models such as F1 Score, Recall, Precision and Accuracy
# * Drawing of ROC Curves of classification models and calculation of AUC (Area Under Curve) values
# * Simple comparison of classification models and their results
# 
# You can use the codes in this kernel as desired. Please read to its license for dataset usage. About 5 hours of writing this kernel was deleted and I had to write them all over again. So if you see some of my mistakes in between, that's why. Good work...

# <a id="9"></a>
# # REFERENCES
# 
# * https://datascienceplus.com/mushrooms-classification-part-1/
# * https://www.geeksforgeeks.org/confusion-matrix-machine-learning/
# 
# _Best regards..._ <br>
# **Mustafa YEMURAL**
