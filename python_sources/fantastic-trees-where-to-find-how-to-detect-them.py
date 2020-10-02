#!/usr/bin/env python
# coding: utf-8

# ![image](https://images.ctfassets.net/usf1vwtuqyxm/1SCzmQ07UgSmWegc2KWkmu/8b8bdf0779bc79769f202415be80fc45/FB-TRL3-87979.jpg?w=914)
# 
# 
# [Image Credits](https://www.wizardingworld.com/features/which-fantastic-beast-is-right-for-you) 

# This notebook aims to classify fantastic trees and give some clues about where to find them in the 4 wilderness areas of the Roosevelt National Forest of Northern Colorado! 
# 
# Our fantastic tree types are 7 in total and labeled  as `cover_type` in the dataset, using the other columns in the data set such as elevation, aspect, slope and some distance measures I will develop a model to differantiate fantastic trees. 
# 
# The notebook will follow the workflow suggested by Will Koehrsen in this [article](https://towardsdatascience.com/a-complete-machine-learning-walk-through-in-python-part-one-c62152f39420).
#     
# 1) Undserstand, Clean and Format Data
# 
# 2) Exploratory Data Analysis
# 
# 3) Feature Engineering & Selection
# 
# 4) Compare Several Machine Learning Models
# 
# 5) Perform Hyperparameter Tuning on the Best Model
# 
# 6) Evaluate the Best Model with Test Data
# 
# 7) Interpret Model Results
# 
# 8) Summary & Conclusions
# 
# If you are dying from curiosity, you can jump directly to *8. Summary & Conclusions*, but I cannot gurantee that you are not going to miss some beautiful visualizations and interesting insights about data science and machine learning. Enjoy Reading!

# In[ ]:


# for data manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 60)

# for visualization
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# to include graphs inline within the frontends next to code
import seaborn as sns
sns.set_context(font_scale=2)

# to bypass warnings in various dataframe assignments
pd.options.mode.chained_assignment = None

# machine learning models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# preprocessing functions and evaluation models
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 1. Understand, Clean and Format Data

# To understand how data is structured, I am going to look at:
# * First and last rows
# * Information
# * Descriptive statistics of the dataset.
# 
# and apply cleaning and formatting afterwards, if necessary.

# ## 1.1. First and last rows

# In[ ]:


trees = pd.read_csv("/kaggle/input/learn-together/train.csv")
print("Number of rows and columns in the trees dataset are:", trees.shape)


# In[ ]:


trees.head()


# In[ ]:


trees.tail()


# ## 1.2. Information (how many rows and columns, data types and non-null values) and descriptive statistics of the dataset

# In[ ]:


display(trees.info())


# In[ ]:


display(trees.describe())


# ## 1.3. Check for Anomalies & Outliers

# To help future ML model to grasp patterns in the data better, I am going to search for outliers. During this search, I will use the logic of [extreme outliers](https://people.richland.edu/james/lecture/m170/ch03-pos.html) to keep as much rows I can keep. So following data points will be dropped if they satisfy the following conditions:    
# - x < Q1 - 3 * IQR       
# - x > Q3 + 3 * IQR

# In[ ]:


def outlier_function(df, col_name):
    ''' this function detects first and third quartile and interquartile range for a given column of a dataframe
    then calculates upper and lower limits to determine outliers conservatively
    returns the number of lower and uper limit and number of outliers respectively
    '''
    first_quartile = np.percentile(np.array(df[col_name].tolist()), 25)
    third_quartile = np.percentile(np.array(df[col_name].tolist()), 75)
    IQR = third_quartile - first_quartile
                      
    upper_limit = third_quartile+(3*IQR)
    lower_limit = first_quartile-(3*IQR)
    outlier_count = 0
                      
    for value in df[col_name].tolist():
        if (value < lower_limit) | (value > upper_limit):
            outlier_count +=1
    return lower_limit, upper_limit, outlier_count


# In[ ]:


# loop through all columns to see if there are any outliers
for column in trees.columns:
    if outlier_function(trees, column)[2] > 0:
        print("There are {} outliers in {}".format(outlier_function(trees, column)[2], column))


# I am going to take a close look for the outlier elimination for the following columns:
# * Horizontal_Distance_To_Hydrology
# * Vertical_Distance_To_Hydrology
# * Horizontal_Distance_To_Roadways
# * Horizontal_Distance_To_Fire_Points
# 
# I am not going to consider other columns for potential outlier elimination because their data range is already fixed between 0 and 255 (e.g. Hillsahde columns)  or they seem like one-hot-encoded columns (e.g. Soil type and Wilderness areas).

# Recall the data ranges of those 4 columns:
# * Horizontal_Distance_To_Hydrology: 0, 1343
# * Vertical_Distance_To_Hydrology: -146, 554
# * Horizontal_Distance_To_Roadways: 0, 6890
# * Horizaontal_Distance_To_Firepoints: 0, 6993
# 
# Considering the Horizaontal_Distance_To_Firepoints having the highest number of outliers and widest data range, I am going to remove outliers only from that column.

# In[ ]:


trees = trees[(trees['Horizontal_Distance_To_Fire_Points'] > outlier_function(trees, 'Horizontal_Distance_To_Fire_Points')[0]) &
              (trees['Horizontal_Distance_To_Fire_Points'] < outlier_function(trees, 'Horizontal_Distance_To_Fire_Points')[1])]
trees.shape


# Number of the rows in the dataset is approximately 15000, after the removal.

# ## 1.4. Findings from Understand, Clean and Format Data
# Training dataset (trees dataframe) has 15120 entries and 56 columns with headers appropriately named. Dataset is clean and well-formatted, meaning it had no NA values and every column has a numeric (float or integer) data type. 
# 
# 4 columns had outliers, outliers of the `Horizontal_Distance_To_Fire_Points` is removed considering this column has a wider range and has the most number of outliers.
# 
# `Cover_Type` is our label/target column. `Wilderness_Area` and `Soil_Type` columns might have binary values (0,1) if so, they are the one-hot-encoded columns of 4 wilderness areas and 40 soil types respectively. I am going to start exploratory data analysis by seeking answer to that suspicion.

# # 2. Exploratory Data Analysis

# ## 2.1. Check if Wilderness_Area and Soil_Type columns have only binary values

# In[ ]:


# list of columns of wilderness areas and soil types
is_binary_columns = [column for column in trees.columns if ("Wilderness" in column) | ("Soil" in column)]
pd.unique(trees[is_binary_columns].values.ravel())


# Yes, they only have binary values.

# ## 2.2. Can one Fantastic Tree belong to multiple soil types and wilderness areas ?

# In[ ]:


# sum of all widerness area columns
trees["w_sum"] = trees["Wilderness_Area1"] + trees["Wilderness_Area2"] + trees["Wilderness_Area3"] + trees["Wilderness_Area4"]
print(trees.w_sum.value_counts())


# In[ ]:


# create a list of soil_type columns
soil_columns = [c for c in trees.columns if "Soil" in c]
trees["soil_sum"] = 0

# sum of all soil type columns
for c in soil_columns:
    trees["soil_sum"] += trees[c]

print(trees.soil_sum.value_counts())


# In[ ]:


trees.drop(columns=["w_sum", "soil_sum"], inplace=True)


# `Wilderness_Area` and `Soil_Type1-40` having only binary values and only one `soil_type` or `wilderness_area` being equal to 1, shows that they are one-hot-encoded columns.
# 
# One important thing about fantastic trees are, they can only belong to one soil type or one wilderness area.

# ## 2.3. Distribution of the Fantastic Trees

# In[ ]:


# set the plot size
figsize(14,10)

# set the histogram, mean and median
sns.distplot(trees["Cover_Type"], kde=False)
plt.axvline(x=trees.Cover_Type.mean(), linewidth=3, color='g', label="mean", alpha=0.5)
plt.axvline(x=trees.Cover_Type.median(), linewidth=3, color='y', label="median", alpha=0.5)

# set title, legends and labels
plt.xlabel("Cover_Type")
plt.ylabel("Count")
plt.title("Distribution of Trees/Labels/Cover_Types", size=14)
plt.legend(["mean", "median"])


# Distribution of fantastic trees shows perfect uniform distribution.
# 
# Here are the 7 types of the fantastic trees, numbered from 1 to 7 in the `Cover_Type` column:
# 
# 1) Spruce/Fir
# 
# 2) Lodgepole Pine
# 
# 3) Ponderosa Pine
# 
# 4) Cottonwood/Willow
# 
# 5) Aspen
# 
# 6) Douglas-fir
# 
# 7) Krummholz

# ## 2.4. Check if the Cover_Type shows non-uniform distribution among different Wilderness_Areas

# In[ ]:


# Create one column as Wilderness_Area_Type and represent it as categorical data
trees['Wilderness_Area_Type'] = (trees.iloc[:, 11:15] == 1).idxmax(1)

#list of wilderness areas
wilderness_areas = sorted(trees['Wilderness_Area_Type'].value_counts().index.tolist())

# distribution of the cover type in different wilderness areas
figsize(14,10)

# plot cover_type distribution for each wilderness area
for area in wilderness_areas:
    subset = trees[trees['Wilderness_Area_Type'] == area]
    sns.kdeplot(subset["Cover_Type"], label=area, linewidth=2)

# set title, legends and labels
plt.ylabel("Density")
plt.xlabel("Cover_Type")
plt.title("Density of Cover Types Among Different Wilderness Areas", size=14)


# Nother important finding about Fantastic Trees: Wilderness area is an important feature to determine the cover type:
# * Spruce/Fir, Lodgepole Pine and Krummholz (Cover_Type 1, 2, 7)  mostly found in Rawah, Neota and Comanche Peak Wilderness Area(1,2 and 3).
# * It is highly likely to find Ponderosa Pine (Cover_Type 3) in Cache la Poudre Wilderness Area (4) rather than other areas.
# * Cottonwood/Willow (Cover_Type 4) seems to be found only in Cache la Poudre Wilderness Area (4).
# * Aspen (Cover_Type 5) is equally likely to come from wilderness area Rawah and Comanche (1,3).
# * Douglas-fir (Cover_Type 6) can be found in any of the wilderness areas.
# 
# Note that, distribution of cover types extend more than the range because of the kernel density estimation.

# ## 2.5. Understanding the Soil_Type and Cover_Type relationship

# Since different soil types might appear in different wilderness areas, I am going to consider different wilderness areas while examining this relationship.

# In[ ]:


def split_numbers_chars(row):
    '''This function fetches the numerical characters at the end of a string
    and returns alphabetical character and numerical chaarcters respectively'''
    head = row.rstrip('0123456789')
    tail = row[len(head):]
    return head, tail

def reverse_one_hot_encode(dataframe, start_loc, end_loc, numeric_column_name):
    ''' this function takes the start and end location of the one-hot-encoded column set and numeric column name to be created as arguments
    1) transforms one-hot-encoded columns into one column consisting of column names with string data type
    2) splits string column into the alphabetical and numerical characters
    3) fetches numerical character and creates numeric column in the given dataframe
    '''
    dataframe['String_Column'] = (dataframe.iloc[:, start_loc:end_loc] == 1).idxmax(1)
    dataframe['Tuple_Column'] = dataframe['String_Column'].apply(split_numbers_chars)
    dataframe[numeric_column_name] = dataframe['Tuple_Column'].apply(lambda x: x[1]).astype('int64')
    dataframe.drop(columns=['String_Column','Tuple_Column'], inplace=True)


# In[ ]:


reverse_one_hot_encode(trees, 16, 56, "Soil_Type")


# In[ ]:


# plot relationship of soil type and cover type among different wilderness areas
g = sns.FacetGrid(trees, col="Wilderness_Area_Type", 
                  col_wrap=2, height=5, col_order=wilderness_areas)
g = g.map(plt.scatter,"Cover_Type", "Soil_Type", edgecolor="w", color="g")


# * Wilderness Area 3 is more diverse in soil type and cover type.
# * Only soil types 1 through 20 is represented in Wilderss Area 4, thus cover types in that area grew with them.
# * Cover type 7 seems to grow with soil types 25 through 40.
# * Cover Type 5 and 6 can grow with most of the soil types.
# * Cover Type 3 loves soil type 0 through 15.
# * Cover Type 1 and 2 can grow with any soil type.

# ## 2.6. Distribution and relationship of continuous variables (Elevation, Aspect, Slope, Distance and Hillsahde columns)

# In[ ]:


# store continious variables in a list
continuous_variables = trees.columns[1:11].tolist()

# Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)


# In[ ]:


# Create the pairgrid object
grid = sns.PairGrid(data = trees[continuous_variables])

# Upper is a correlation and kdeplot
grid.map_upper(corr_func);
grid.map_upper(sns.kdeplot, cmap = plt.cm.Greens)

# Diagonal is a histogram
grid.map_diag(plt.hist, color = 'green', edgecolor = 'white')

# Bottom is scatter plot
grid.map_lower(plt.scatter, color = 'green', alpha = 0.1)


# **Distributions:**
# * `Hillshade_9am` and `Hillshade_Noon` has bi-modal and left-skewed distributions.
# * `Horizontal_Distance_To_Firepoints`, `Horizontal_Distance_To_Roadways`, `Horizontal_Distance_To_Hydrology` has bi-modal and right-skewed distributions.
# * `Elevation` (height of a fantastic trees) resembles a uniform distribution.
# * `Slope`, `Vertical_Distance_To_Hydrology`, `Hillshade_3pm` shows a symmetric and bi-modal distribution.
# 
# **Some obvious relationships between the continuous features:**
# *  `Elevation` and shows positive trend with following variables:
#    * `Vertical_Distance_To_Hydrology`
#    * `Horizontal_Distance_To_Roadways`
#    * `Horizontal_Distance_To_Firepoints`
#    * `Horizontal_Distance_To_Hydrology`
# * As `Aspect` increases; `Hillshade_Noon` and `Hillshade_3pm` increases.
# * `Slope` has negative trend with:
#   * `Elevation`
#   * `Horizontal_Distance_To_Roadways`
#   * `Hillshade_9am`, `Hillshade_Noon` and `Hillshade_3pm`
#   * `Horizontal_Distance_To_Firepoints`
# * `Horizontal_Distance_To_Hydrology`  has positive trend with:
#   * `Horizontal_Distance_To_Firepoints`
#   * `Horizontal_Distance_To_Roadways`
#   * `Vertical_Distance_To_Hydrology`
# * `Vertical_Distance_To_Hydrology` - `Slope` and `Vertical_Distance_To_Hydrology` - `Horizontal_Distance_To_Hydrology` has obvious collinear relationship.
# * As `Horizontal_Distance_To_Roadways` increases, `Horizontal_Distance_To_Firepoints` increases and `Slope` decreases.
# * `Hillshade_9am` shows negative trend with `Hillshade_3pm` and `Aspect`, as `Hillshade_9am` increases `Elevation` increases.
# * `Hillshade_Noon` has positive trend with:
#   * `Elevation`
#   * `Aspect`
#   * `Horizontal_Distance_To_Roadways`
#   * `Hillshade_3pm` 
#   * `Horizontal_Distance_To_Firepoints`
# * `Hillshade_3pm` shows perfect negative relationship with `Hillshade_9am` and perfect positive relationship with `Hillshade_Noon`.
# 
# **Some Collinear features:**
# * hillshade noon - hillshade 3 pm
# * hillsahde 3 pm - hillshade 9 am
# * vertical distance to hydrology - horizontal distance to hydrology
# * elevation - slope

# ## 2.7. Visualize some collinear features with Cover_Type

# In[ ]:


figsize(24,10)

# plot the first subplot
plt.subplot(1,2,1)
sns.scatterplot(x="Vertical_Distance_To_Hydrology", y="Horizontal_Distance_To_Hydrology", 
                hue="Cover_Type", data=trees, 
                legend="full", hue_norm=(0,8), palette="Set1")
plt.title("Vertical_Distance_To_Hydrology VS Horizontal_Distance_To_Hydrology", size=14)

# plot the second subplot
plt.subplot(1,2,2)
sns.scatterplot(x="Elevation", y="Slope", 
                hue="Cover_Type", data=trees, 
                legend="full", hue_norm=(0,8), palette="Set1")
plt.title("Elevation VS Slope", size=14)


# In[ ]:


figsize(24,10)

# plot the first subplot
plt.subplot(1,2,1)
sns.scatterplot(x="Hillshade_Noon", y="Hillshade_3pm", 
                hue="Cover_Type", data=trees, 
                legend="full", hue_norm=(0,8), palette="Set1")
plt.title("Hillshade_Noon VS Hillshade_3pm", size=14)

# plot the second subplot
plt.subplot(1,2,2)
sns.scatterplot(x="Hillshade_9am", y="Hillshade_3pm", 
                hue="Cover_Type", data=trees, 
                legend="full", hue_norm=(0,8), palette="Set1")
plt.title("Hillshade_9am VS Hillshade_3pm", size=14)


# One of the features from the Hillshade_9am or Hillshade_3pm or Hillshade_Noon will be dropped when determining the training set. Which one to be eliminated will be determined after looking at the Pearson Coeffiecients with the label.

# ## 2.8. Pearson Coefficients of all features

# In[ ]:


plt.figure(figsize=(14,12))

# plot heatmap set the title
colormap = plt.cm.RdBu
sns.heatmap(trees.corr(),linewidths=0.1,vmax=1.0, 
            square=False, cmap=colormap, linecolor='white', annot=False)
plt.title('Pearson Correlation of All Features', y=1.05, size=14)


# None of the features are significantly different effect on determining the label cover type.
# 
# One interesting finding though, Soil Type 7 and 15 columns are blank in the heatmap, thus zero effect on determining the label Cover_Type. 
# 
# Approximately 5 (1 percent of all soil types) soil_type columns affects the cover type.
# 
# Can we get a better picture if we use soil_type as one numeric column rather than seperate one-hot-encoded columns?

# ## 2.9. Pearson coefficients with numeric Soil_Type representation

# In[ ]:


# make a list of numeric features and create a dataframe with them
all_features_w_label = continuous_variables + wilderness_areas + ["Soil_Type"] + ["Cover_Type"]
trees_w_numeric_soil = trees[all_features_w_label]

# pearson coefficients with numeric soil type column
correlations = pd.DataFrame(trees_w_numeric_soil.corr())

figsize=(16,14)

# plot the heatmap
colormap = plt.cm.RdBu
sns.heatmap(correlations,linewidths=0.1, 
            square=False, cmap=colormap, linecolor='white', annot=True)
plt.title('Pearson Correlation of Features with Numeric Soil_Type', size=14)


# ## 2.10. Findings From Exploratory Data Analysis
# <p> Data set have balanced labels, resulting in equal number of cover types. This will be an advantage when it comes to apply classification ML models because, the model will have good chance to learn patterns of all labels, eliminating the probability of underfitting. <p/>
# <p> Different wilderness areas consist of some specific trees. Interestingly, there is one fantastic tree, Cottonwood/Willow, specifically likes to grow in wilderness area 4. While cover types 1, 2, 5 and 6 can grow in any soil type, other cover types grows more with specific soil types. <p/>
# <p> Soil types are reverse-one-hot-encoded, meaning they are going to be included as numeric data in the training set and one-hot-encoded soil type columns will be excluded. With that way, there is a stronger correlation between soil type and Cover_Type. Numeric soil type column and other variables have pearson coefficients in the range of [-0.2, 0.1]. <p/>
# <p> Hillshade columns are collinear within each other and Hillshade_9am has the least importance in determining Cover_Type. Thus this column will be dropped in Part 3 for better interpretability of the future model. <p/>

# # 3. Feature Engineering & Selection

# ## 3.1. Add & Transform Features

# In the data set, the strongest positive pearson coefficient is 0.22 and -0.11 on the other end. After doing some Google Search about the features, maybe adding some features might help achieving stronger correlations. 
# 
# I decided to add linear combinations of the horizontal distance columns and  Euclidian distance of Horizontal_Distance_To_Hydrology and Vertical_Distance_To_Hydrology as suggested in this [presentation](https://www.slideshare.net/danielgribel/forest-cover-type-prediction-56288946?next_slideshow=2).
# 
# * Elevation and Vertical Distance to Hydrology
# * Horizontal Distance to Hydrology and Horizontal Distance to Firepoints
# * Horizontal Distance to Hydrology and Horizontal Distance to Roadways 
# * Horizontal Distance to Firepoints and Horizontal Distance to Roadways
# * Euclidian Distance of Horizontal Distance to Hydrology and Vertical Distance to Hydrology
# 
# After the addition, I will perform square root transformation to the features with positive data range. Square root transformation might help especially for the highly skewed distributions.
# 
# After the addition and transformation, I will check pearson coefficients again.

# In[ ]:


# add columns
trees_w_numeric_soil['Euclidian_Distance_To_Hydrology'] = (trees_w_numeric_soil['Horizontal_Distance_To_Hydrology']**2 + 
                                                           trees_w_numeric_soil['Vertical_Distance_To_Hydrology']**2)**0.5
trees_w_numeric_soil['Mean_Elevation_Vertical_Distance_Hydrology'] = (trees_w_numeric_soil['Elevation'] + 
                                                                      trees_w_numeric_soil['Vertical_Distance_To_Hydrology'])/2
trees_w_numeric_soil['Mean_Distance_Hydrology_Firepoints'] = (trees_w_numeric_soil['Horizontal_Distance_To_Hydrology'] + 
                                                              trees_w_numeric_soil['Horizontal_Distance_To_Fire_Points'])/2
trees_w_numeric_soil['Mean_Distance_Hydrology_Roadways'] = (trees_w_numeric_soil['Horizontal_Distance_To_Hydrology'] + 
                                                            trees_w_numeric_soil['Horizontal_Distance_To_Roadways'])/2
trees_w_numeric_soil['Mean_Distance_Firepoints_Roadways'] = (trees_w_numeric_soil['Horizontal_Distance_To_Fire_Points'] + 
                                                             trees_w_numeric_soil['Horizontal_Distance_To_Roadways'])/2


# In[ ]:


# add sqrt transformed columns to the trees_w_numeric_soil dataframe
for col in trees_w_numeric_soil.columns:
    if trees_w_numeric_soil[col].min() >= 0:
        if col == 'Cover_Type':
            next
        else:
            trees_w_numeric_soil['sqrt' + col] = np.sqrt(trees_w_numeric_soil[col])


# In[ ]:


correlations_transformed = pd.DataFrame(trees_w_numeric_soil.corr())
correlations_transformed = pd.DataFrame(correlations_transformed["Cover_Type"]).reset_index()

# format, and display sorted correlations_transformed
correlations_transformed.columns = ["Feature", "Correlation with Cover_Type"]
correlations_transformed = (correlations_transformed[correlations_transformed["Feature"] != "Cover_Type"]
                .sort_values(by="Correlation with Cover_Type", ascending=True))
display(correlations_transformed)


# So, in addition to the existing features, final features will be:
# * Instead of Horizontal_Distance_To_Hydrology, sqrtHorizontal_Distance_To_Hydrology
# * sqrtMean_Distance_Hydrology_Roadways
# * sqrtEuclidian_Distance_To_Hydrology
# * Mean_Elevation_Vertical_Distance_Hydrology
# * Mean_Distance_Firepoints_Roadways
# * Mean_Distance_Hydrology_Firepoints
# 
# Additionally, I will drop Hillshade_9am column since it is strongly correlated with Hillshadde_3pm.

# In[ ]:


# final list of features
transformed_features = ['sqrtHorizontal_Distance_To_Hydrology', 'sqrtMean_Distance_Hydrology_Roadways', 'sqrtEuclidian_Distance_To_Hydrology', 
                        'Mean_Elevation_Vertical_Distance_Hydrology', 'Mean_Distance_Firepoints_Roadways', 'Mean_Distance_Hydrology_Firepoints',  ]

all_features =  (['Elevation', 'Aspect', 'Slope', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
                  'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points' ] + wilderness_areas +
                 ['Soil_Type'] + transformed_features)


# ## 3.2. Seperate labels from features in the training set

# In[ ]:


trees_training = trees_w_numeric_soil[all_features]
labels_training = trees_w_numeric_soil["Cover_Type"].as_matrix()


# ## 3.3. Split training set as training and validation set

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(trees_training, labels_training, test_size=0.2, random_state=1)


# In[ ]:


print('Training Data Shape:', X_train.shape)
print('Validation Data Shape:', X_valid.shape)


# In[ ]:


print('Training Label Shape:', y_train.shape)
print('Validation Label Shape:', y_valid.shape)


# Training, validation and test set have the same number of columns.

# ## 3.4. Create a baseline metric

# Before diving deep into the ML classification algorithms, I am going to calculate a common sense baseline. A common sense baseline is defined in this [article](https://towardsdatascience.com/first-create-a-common-sense-baseline-e66dbf8a8a47) in simple terms, how a person has a knowledge in that field would solve the problem without using any data science tricks. Alternatively, as explained in this [post](https://machinelearningmastery.com/implement-baseline-machine-learning-algorithms-scratch-python/), it can be a dummy or simple algorithm, consisting of few lines of code, to use as a baseline metric.
# 
# Baseline metrics can be [different](https://machinelearningmastery.com/how-to-get-baseline-results-and-why-they-matter/) in regression and classification problems. Since fantastic trees will be classified into 7 groups and no expert wizards available around, I am going to use [dummy algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) from scikit-learn library. With that dummy algorithm, I will establish a baseline metric of accuracy which is percentage of correctly predicted trees among the test dataset .
# 
# Baseline metrics are important in a way that, if a ML model cannot beat the simple and intuitive prediction of a person's or an algorithm's guess, the original problem needs reconsideration or training data needs reframing.

# In[ ]:


# Create dummy classifer
dummy = DummyClassifier(strategy='stratified', random_state=1)

# train the model
dummy.fit(X_train, y_train)

# Get accuracy score
baseline_accuracy = dummy.score(X_valid, y_valid)
print("Our dummy algorithm classified {:0.2f} of the of the trees correctly".format(baseline_accuracy))


# Now, I expect that following ML models beat the accuracy score of 0.14!

# # 4. Compare Several Machine Learning Models

# I am going to use (with default parameters for now) and without discussing specifics of the models:
# 
# 1) K-Nearest Neighbors Classifier
# 
# 2) Light Gradient Boosting Machine (LightGBM) Classifier
# 
# 3) Random Forest Classifier
# 
# 4) Extra Trees (Random Forests) Classifier
# 
# 5) Extra Gradient Boosting (XGBoost) Classifier
# 
# and compare the results on accuracy score. Then I will select the best model with the highest accuracy score for use.
# 
# Since K-nearest neighbors classifier is using [euclidian distance](https://en.wikipedia.org/wiki/Euclidean_distance) to cluster labels, I am going to use normalized training set for those.

# ## 4.1. Z-Score normalization for K-Nearest Neighbors and LightGBM

# Here is the definition from the Scikit-Learn [documentation](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html):
# > Feature scaling through standardization (or Z-score normalization) can be an important preprocessing step for many machine learning algorithms. Standardization involves rescaling the features such that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one. Many algorithms (such as SVM, K-nearest neighbors, and logistic regression) require features to be normalized.

# In[ ]:


# create scaler
scaler = StandardScaler()

# apply normalization to training set and transform training set
X_train_scaled = scaler.fit_transform(X_train, y_train)

# transform validation set
X_valid_scaled = scaler.transform(X_valid)


# ## 4.2. Build models

# In[ ]:


# function to train a given model, generate predictions, and return accuracy score
def fit_evaluate_model(model, X_train, y_train, X_valid, Y_valid):
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_valid)
    return accuracy_score(y_valid, y_predicted)


# ### 4.2.1. K-Nearest Neighbor Classifier

# In[ ]:


# create model apply fit_evaluate_model
knn_classifier = KNeighborsClassifier()
knn_accuracy = fit_evaluate_model(knn_classifier, X_train_scaled, y_train, X_valid_scaled, y_valid)
print("Number of correct predictions made out of all predictions are:", knn_accuracy)


# ### 4.2.2. Light Gradient Boosting Machine (LightGBM) Classifier

# In[ ]:


# create model apply fit_evaluate_model
lgbm_classifier = LGBMClassifier()
lgbm_accuracy = fit_evaluate_model(lgbm_classifier, X_train_scaled, y_train, X_valid_scaled, y_valid)
print("Number of correct predictions made out of all predictions are:", lgbm_accuracy)


# ### 4.2.3. Random Forests Classifier

# In[ ]:


# create model apply fit_evaluate_model
rf_classifier = RandomForestClassifier()
rf_accuracy = fit_evaluate_model(rf_classifier, X_train, y_train, X_valid, y_valid)
print("Number of correct predictions made out of all predictions are:", rf_accuracy)


# ### 4.2.4. Extra Trees (Random Forests) Classifier

# In[ ]:


# create model apply fit_evaluate_model
xrf_classifier = ExtraTreesClassifier()
xrf_accuracy = fit_evaluate_model(xrf_classifier, X_train, y_train, X_valid, y_valid)
print("Number of correct predictions made out of all predictions are:", xrf_accuracy)


# ### 4.2.5. Extra Gradient Boosting (XGBoost) Classifier

# In[ ]:


# create model apply fit_evaluate_model
xgb_classifier = XGBClassifier()
xgb_accuracy = fit_evaluate_model(xgb_classifier, X_train, y_train, X_valid, y_valid)
print("Number of correct predictions made out of all predictions are:", xgb_accuracy)


# ## 4.3. Comparison of model performances

# In[ ]:


# create dataframe of accuracy and model and sort values
performance_comparison = pd.DataFrame({"Model": ["K-Nearest Neighbor", "LightGBM", "Random Forests", "Extra Trees", "XGBoost"],
                                       "Accuracy": [knn_accuracy, lgbm_accuracy, rf_accuracy, xrf_accuracy, xgb_accuracy]})

performance_comparison = performance_comparison.sort_values(by="Accuracy", ascending=True)

# set the plot
plt.figure(figsize=(10,10))
ax = sns.barplot(x="Accuracy", y="Model", data=performance_comparison, palette="Greens_d")

# set title arrange labels
plt.yticks(size = 14)
plt.xticks(size = 14)
plt.title("Accuracy Score of Different Models", size=14)


# Although it is known that gradient boosting algorithm outperforms others by loss, as plotted below heatmap, extreme (extra) random forests outperformed other algorithms with accuracy performance metric in this case. The reason might be, I did not focus on tuning the parameters of the each algorithm and used defaults values instead.
# 
# ![image](https://crossinvalidation.files.wordpress.com/2017/08/olson.jpg?w=900)
# 
# [Image Credit](https://crossinvalidation.com/2017/08/22/quantitative-comparison-of-scikit-learns-predictive-models-on-a-large-number-of-machine-learning-datasets-a-good-start/)
# 
# Do you remember our baseline metric produced by the dummy algorithm (0.14) ? Well, all 4 models beat that intuitive score and showed that machine learning is applicable to the fantastic tree classification problem!

# # 5. Perform Hyperparameter Tuning on the Best Model

# ## 5.1. Extra Trees Classifier

# Now, I am going to perform hyperparameter tuning on the best model (extra random forests classifier) and try to improve accuracy of the model. Searching and setting the best and optimal set of parameters for a machine learning model can be defined as hyperparameter tuning.
# 
# More than 80% accuracy can be interpreted as a reasonable score and managed not to fall the areas of underfitting or overfitting. One can call the model as underfit if s/he gets an accuracy score slightly more than the baseline metric, meaning the model fails to catch and learn from the patterns in the training set.
# 
# On the other hand, an accuracy score of more than 95% might show that the model already in the overfitted area. Meaning the model performed very well on the training data and captured the patterns but it might not show the same performance on the test data set. So one cannot conclude that the higher performance metric is always better.
# 
# Let's see if I can improve the accuracy of the model by playing with the parameters of [extra trees classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) without falling into overfitting area.

# ### 5.1.1. Hyperparameter Tuning with Random Search and Cross-Validation

# I am going to search for the best set of parameters with random search and cross validation.
# 
# In random serach, set of ML model's parameters are defined in a range and inputted to `RandomizedSearchCV`. This algorithm randomly selects some combination of the parameters and compares the defined `score` (accuracy, for this problem) with iterations. Random search runtime and iterations can be controlled with the parameter `n_iter`. This is in contrast to grid search iterations of every single combination of the given parameters. With intuition, one can say that, grid search requires more run-time than random search if a small number of n_iterations is defined. Generally, random search is better when there is a limited knowledge and of the best model hyperparameters and less time. 
# 
# K-fold Cross validation is the method used to assess the performance of the hyperparameters on the whole dataset. Rather than splitting the dataset set into 2 static subsets of training and test, dataset is divided equally for the given K, and with iterations different K subsets are trained and tested. In other words, divide the dataset into K folds, and follow the iterative process where first traininig is done on K-1 of the folds and then evaluate performance on the Kth fold. Process is repeated K times, so eventually dataset is tested on every example keeping in mind that each iteration is testing on a subset that did not train on before. At the end of K-fold cross validation, average of the performance metric on each of the K iterations substitutes the final performance measure.
# 
# A visualization of cross-validation:
# 
# ![image](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)
# 

# To perform hyperparameter tuning, I am going to define set of parameters and `RandomizedSearchCV` will look for the best combination with cross validation. So, randomly one single element is chosen from the below lists in each iteration. When the iteration is complete on each k-folds best set of parameters can be detected.

# In[ ]:


# The number of trees in the forest algorithm, default value is 10.
n_estimators = [50, 100, 300, 500, 1000]

# The minimum number of samples required to split an internal node, default value is 2.
min_samples_split = [2, 3, 5, 7, 9]

# The minimum number of samples required to be at a leaf node, default value is 1.
min_samples_leaf = [1, 2, 4, 6, 8]

# The number of features to consider when looking for the best split, default value is auto.
max_features = ['auto', 'sqrt', 'log2', None] 

# Define the grid of hyperparameters to search
hyperparameter_grid = {'n_estimators': n_estimators,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features}


# To find the best combination of the randomly set parameters and apply cross validation, I am going to use the `RandomizedSearchCV` with following arguments:
# * `estimator`: the model
# * `param_distributions`: the distribution of parameters we defined
# * `cv`: K in the K-fold cross validation, number of subsets to create
# * `n_iter`: the number of different combinations to try
# * `scoring`: which metric to use when evaluating candidates
# * `n_jobs`: number of cores to run in parallel (-1 will use all available)
# * `verbose`: how much information to display (1 displays a limited amount)
# * `return_train_score`: return the training score for each cross-validation fold
# * `random_state`: fixes the random number generator used so we get the same results every run

# In[ ]:


# create model
best_model = ExtraTreesClassifier(random_state=42)

# create Randomized search object
random_cv = RandomizedSearchCV(estimator=best_model,
                               param_distributions=hyperparameter_grid,
                               cv=5, n_iter=20, 
                               scoring = 'accuracy',
                               n_jobs = -1, verbose = 1, 
                               return_train_score = True, 
                               random_state=42)


# In[ ]:


# Fit on the all training data using random search object
random_cv.fit(trees_training, labels_training)


# In[ ]:


random_cv.best_estimator_


# Here is the best combination of parameters:
# * `n_estimators` = 300
# * `max_features` = None
# * `min_samples_leaf`= 1
# * `min_samples_split`= 2
# 
# Let's apply those parameters to the extra random forests classifier model and see observe the improvement on the accuracy score.

# In[ ]:


xrf_classifier_w_random_search = ExtraTreesClassifier(n_estimators=300, 
                                                     max_features=None, 
                                                     min_samples_leaf=1, 
                                                     min_samples_split=2,
                                                     random_state=42)

xrf_accuracy_opt_w_rand_search = fit_evaluate_model(xrf_classifier_w_random_search, X_train, y_train, X_valid, y_valid)


# In[ ]:


print("Accuracy score in the previous extra random forests model:", xrf_accuracy)
print("Accuracy score after hyperparameter tuning:", xrf_accuracy_opt_w_rand_search)


# After the hypermeter parameter tuning I increased the accuracy of the model by 2 to 3 points.

# ### 5.1.2. Possible further improvements with the GridSearch

# With the random search, I am able to define a best set of parameters (might change for a different case and set of parameters though) as mentioned above. 
# 
# To recap, first I used default parameter settings to find which algorithm yields best performance. Then, I improved performance of the best selected algorithm (random forest classifier) by narrowing down to the set of parameters with random search.
# 
# Now, I am going to look if there is any room left for further improvement in accuracy score in the algorithm. I am going to look for that improvement in the `n_estimator` parameter, (number of decision trees used in the extra random forests). Having the possibility of long run-times in mind, I will use GridSearch with parameter n_estimators and pass a 6-element list as input, to keep the run-time at reasonable minutes. 
# 
# Like random search, grid search also performs its search on whole data set with k-fold cross validation. I am going to use 5-fold cross validation as I did for random search.

# In[ ]:


# Create a range of trees to evaluate
trees_grid = {'n_estimators': [300, 500, 700, 900, 1200, 1500]}

# define all parameters except n_estimators
xrf_classifier_w_grid_search = ExtraTreesClassifier(max_features=None, 
                                                    min_samples_leaf=1, 
                                                    min_samples_split=2,
                                                    random_state=42)

# Grid Search Object using the trees range, the model and 5-fold cross validation
grid_search = GridSearchCV(estimator = xrf_classifier_w_grid_search, param_grid=trees_grid, 
                           cv = 5, scoring = 'accuracy', verbose = 1,
                           n_jobs = -1, return_train_score = True)


# In[ ]:


# fit the dataset to grid search object
grid_search.fit(trees_training, labels_training)


# In[ ]:


# Get the results into a dataframe
xrf_results = pd.DataFrame(grid_search.cv_results_)

# Plot the training and testing error vs number of trees
plt.figure(figsize=(10,10))
plt.plot(xrf_results['param_n_estimators'], xrf_results['mean_test_score'], label = 'Testing Accuracy')
plt.plot(xrf_results['param_n_estimators'], xrf_results['mean_train_score'], label = 'Training Accuracy')

# set title, labels and legend
plt.xlabel('Number of Estimators(Trees)'); plt.ylabel('Accuracy'); plt.legend();
plt.title('Performance vs Number of Trees', size=14);


# Training accuracy is very 100% percent, showing that the model studied and learned from the training set very well. 
# 
# When it comes to testing accuracy, accuracy drops 20 points, resulting in accuracy level around 80%. This shows that the model is performing worse in a newly-introduced dataset. This picture also gives a clue about the submission score, I expect it to be around 80%.
# 
# Another important message is, There are slight changes in the accuracy for the test set which means number of estimators trees can be improved further. Let's see what is the best n_estimator value.

# In[ ]:


xrf_results[["param_n_estimators", "params", "mean_test_score"]].sort_values(by="mean_test_score", ascending=False)


# Model performed best when n_estimators are 500. So I am going to update model with that parameter.

# In[ ]:


xrf_optimal_model = ExtraTreesClassifier(n_estimators=500, 
                                           max_features=None, 
                                           min_samples_leaf=1, 
                                           min_samples_split=2,
                                           random_state=42)

xrf_optimal_model_accuracy = fit_evaluate_model(xrf_optimal_model, X_train, y_train, X_valid, y_valid)


# In[ ]:


print("Accuracy score with random forests model when n_estimators=300:", xrf_accuracy_opt_w_rand_search)
print("Accuracy score with random forests model when n_estimators=500:", xrf_optimal_model_accuracy)


# ## 5.3. Visualization of the best model predictions

# I am going to use the [function](https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823) mentioned in the sci-kit learn documentation to print confusion_matrix. Confusion matrix will show the number of predictions made in each category with actual and predicted values, by comparing the actual labels and the prediected labels. 
# 
# Fantastic tree confusion matrix will be a 7x7 matrix. I will use normalized confusion matrix, so percentage of actual tree type correctly guessed out of all guesses in that particular category will appear in the diagonal of the matrix and non-diagonal elements will show misslabeled elements by the model. The higher the diagonal percentages of the confusion matrix the better, indicating many correct predictions.

# In[ ]:


# create set of y_predictions
y_predicted = xrf_optimal_model.predict(X_valid)


# In[ ]:


# make a list of cover_types
cover_types = sorted(trees['Cover_Type'].value_counts().index.tolist())


# In[ ]:


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
        # print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[ ]:


plot_confusion_matrix(y_valid, y_predicted, classes=cover_types, normalize=True,
                      title='Normalized confusion matrix')
plt.show()


# Model did pretty good detecting fantastic trees of type 3,4, 5, 6 and 7, and it seems a bit confused to detect types 1 and 2.

# # 6. Evaluate the Best Model with Test Data

# ## 6.1. Align test set with the training set

# In[ ]:


trees_test = pd.read_csv("/kaggle/input/learn-together/test.csv")


# In[ ]:


# add numeric soil type column
reverse_one_hot_encode(trees_test, 16, 56, "Soil_Type")


# In[ ]:


# add linear combinations of columns
trees_test['Euclidian_Distance_To_Hydrology'] = (trees_test['Horizontal_Distance_To_Hydrology']**2 + 
                                                 trees_test['Vertical_Distance_To_Hydrology']**2)**0.5
trees_test['Mean_Elevation_Vertical_Distance_Hydrology'] = (trees_test['Elevation'] + 
                                                            trees_test['Vertical_Distance_To_Hydrology'])/2
trees_test['Mean_Distance_Hydrology_Firepoints'] = (trees_test['Horizontal_Distance_To_Hydrology'] + 
                                                    trees_test['Horizontal_Distance_To_Fire_Points'])/2
trees_test['Mean_Distance_Hydrology_Roadways'] = (trees_test['Horizontal_Distance_To_Hydrology'] + 
                                                  trees_test['Horizontal_Distance_To_Roadways'])/2
trees_test['Mean_Distance_Firepoints_Roadways'] = (trees_test['Horizontal_Distance_To_Fire_Points'] + 
                                                   trees_test['Horizontal_Distance_To_Roadways'])/2


# In[ ]:


# transfrom columns 
trees_test['sqrt' + 'Horizontal_Distance_To_Hydrology'] = np.sqrt(trees_test['Horizontal_Distance_To_Hydrology'])
trees_test['sqrt' + 'Mean_Distance_Hydrology_Roadways'] = np.sqrt(trees_test['Mean_Distance_Hydrology_Roadways'])
trees_test['sqrt' + 'Euclidian_Distance_To_Hydrology'] = np.sqrt(trees_test['Euclidian_Distance_To_Hydrology'])


# In[ ]:


X_test = trees_test[all_features]
print(X_test.columns)


# ## 6.2. Make sure of the test data shape and there aren't any missing values

# In[ ]:


print('Test Data Shape:', X_test.shape)


# In[ ]:


print(X_test.isnull().sum())


# There are no NA values in the test data set, which is ready to be inputted in a ML model.
# In the training set, there are more than 12000 rows, and the test has much more rows (almost 500.000) than the tarining set. Let's see how the model will deal with a much bigger dataset!

# In[ ]:


# generate predictions for test data
test_predictions = xrf_optimal_model.predict(X_test)


# In[ ]:


# write results to the dataframe and create file for submission
output = pd.DataFrame({'Id': trees_test["Id"],
                       'Cover_Type': test_predictions})
output.to_csv('submission.csv', index=False)


# # 7. Interpret Model Results

# ## 7.1. Feature importances

# Remember, in the exploratory data analysis I looked at the pearson coeffients of the features. Since final results are generated, I will revisit this picture and observe features with the highest contribution to the model predictions. `feature_importances` attribute will be used for this.

# In[ ]:


#create list of features
features = list(trees_training.columns)

# Extract the feature importances into a dataframe
feature_results = pd.DataFrame({'feature': features, 
                                'importance': xrf_optimal_model.feature_importances_})

# Show the top 10 most important
feature_results = feature_results.sort_values('importance', ascending = False).reset_index(drop=True)
feature_results.head(10)


# Top 10 Pearson Correlations before building the model:

# In[ ]:


correlations_transformed.head(5)


# In[ ]:


correlations_transformed.tail(5)


# # 8. Summary & Conclusisons

# Now it is time to answer the headline!

# ## 8.1. Where to Find Fantastic Trees?

# Spruce/Fir, Lodgepole Pine and Krummholz loves to hangout in Rawah, Neota and Comanche Peak Wilderness Area.
# 
# Cache la Poudre Wilderness Area is perfect place for Ponderosa Pine and Cottonwood/Willow.
# 
# If you see an Aspen suspect that you might be at the Rawah or Comanche.
# 
# Douglas-fir is an easy going species, that goes along with any wilderness area.

# ## 8.2. How to Detect Fantastic Trees?

# To recognize fantastic trees, I analyzed them first, determined and transformed some of their features (like characteristics). 
# 
# To classify them, I implemented a extra random forest classifier model (how funny that fantastic trees are classifed with extra random forests model), fine tuned the model and generated predictions. 
# 
# **Most importantly: **
# 
# With the current workflow, and selection of features, extra random forests model and parameters I succesfully detected more than 75 percent of the fantastic trees correctly!
# <br>(Previously submitted accuracy score was 72% with random forests model)</br>
# 
# Extra random forests classification showed that:
# 
# * elevation
# * soil type
# * mean distance of elevation and vertical distance hydrology
# * wilderness_Area4
# * mean distance of horizontal distance to firepoints and roadways are the most important characteristics of a fantastic tree.
# 
# Top 5 features list stressed the importance of feature engineering and selection. 3 of the Top 5 features are created in the scope of this notebook.
# 
# ![end credits](https://i.ytimg.com/vi/WfvD-JZGlHs/maxresdefault.jpg)
# 
# [Image Credits](https://www.youtube.com/watch?v=WfvD-JZGlHs)

# ### And many thanks for reading until the end, if I am able to share my knowledge with you or at least created some inspiration in you, I will appreciate your upvote.
