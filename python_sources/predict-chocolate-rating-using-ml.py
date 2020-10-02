#!/usr/bin/env python
# coding: utf-8

# The dataset had total 9 attributes including the target attribute which is the rating for the chocolate. The problem statement is to classify the rating for the chocolated based on the value of the attributes. This problem was addressed as a classification problem since there were discrete values for the rating in the dataset. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# First of all lets import the dataset into our notebook.

# In[ ]:


import pandas as pd
import numpy as np
# Importing the csv data files 
data = pd.read_csv('../input/chocolate-bar-ratings/flavors_of_cacao.csv',error_bad_lines=False, warn_bad_lines=True)
data.head()


# The name of the attributes(column names) are renamed to make it easier to work with. The main reason is that in the original dataset the column names were taking two rows which would make our references to those columns clumsy as we need to include '\n' for the column names 

# In[ ]:


# Renaming the column name
new_col_names = ['company', 'bean_origin', 'REF', 'review_date', 'cocoa_percent',
                'company_location', 'rating', 'bean_typ', 'country_origin']
data_clean = data.rename(columns=dict(zip(data.columns, new_col_names)))
data_clean.head()


# Some functions are being created so that we avoid repetation of the code and to make our code more succinct.

# In[ ]:


# Function to count the number of null values in a column
def count_null_values(dataset, column_list):
    for i in range (len(column_list)):
        print ("The total number of null values in :",column_list[i])
        print (dataset[column_list[i]].isnull().sum())
    return

# Function to dispplay the unique counts in a column
def print_uniques(dataset, column_list):
    for i in range (len(column_list)):
        print ("Unique values for the column:",column_list[i])
        print (dataset[column_list[i]].unique())
        print ('\n')
    return

# Printing the null and unique values for each attribute in the dataset
print_uniques(data_clean, data_clean.columns)
count_null_values(data_clean, data_clean.columns)
#print len(data['Rating'].unique().tolist())


# ### Conducting Feature Engineering

# There were two attributes- Company(Maker-if known) and bean_typ which stored multiple infomation as a same instance value and thus required feature engineering so that we do not lose any valuable information during the data cleaning process. 

# In[ ]:


# Creating new column named maker 
data_clean['company_coffee'], data_clean['maker'] = data_clean['company'].str.split('(', 1).str
# Replacing the missing values with "Unknown"
print (data_clean.head(4))
data_clean['maker'].fillna(value='Unknown', inplace = True) 
#data_clean["maker"].replace(np.nan, "Unknown")
print (data_clean.head())
print (data_clean.info())
# Removing unwanted character
data_clean['maker'] = data_clean['maker'].apply(lambda x: x.split(')')[0])
# Dropping the original column
data_clean = data_clean.drop('company', 1)

print(data_clean.head(4))


# In[ ]:


# Converting the string values to lower case
data_clean['bean_typ'] = data_clean['bean_typ'].str.lower()

# Creating new column named sub_bean_type 
data_clean['bean_type'], data_clean['sub_bean_type'] = data_clean['bean_typ'].str.split(',', 1).str
# Replacing the missing values with "Unknown"
data_clean["sub_bean_type"].fillna("unknown", inplace = True) 
# Removing unwanted character
data_clean['sub_bean_type'] = data_clean['sub_bean_type'].apply(lambda x: x.split(')')[0])
# Dropping the original column
data_clean = data_clean.drop('bean_typ', 1)

print (data_clean.head(10))
print (data_clean['bean_type'].unique())


# ### Conducting Data Cleaning

# In the data cleaning stage the anomalies in the data were resolved

# In[ ]:


# Some data cleaning regarding bean type name
data_clean['bean_type'] = data_clean['bean_type'].replace('forastero (arriba) asss', 'forastero (arriba)')
data_clean['bean_type'] = data_clean['bean_type'].replace('forastero (arriba) ass', 'forastero (arriba)')
print (data_clean['bean_type'].unique())


# In[ ]:


# Data cleaning regarding the Broad Bean Origin column
print ("Before Cleaning country_origin column: ")
print (data_clean['country_origin'].unique())
data_clean['country_origin'] = data_clean['country_origin'].replace('Domincan Republic', 'Dominican Republic')
data_clean['country_origin'] = data_clean['country_origin'].replace('Carribean(DR/Jam/Tri)', 'Carribean')
data_clean['country_origin'] = data_clean['country_origin'].replace('Trinidad-Tobago', 'Trinidad, Tobago')
data_clean['country_origin'] = data_clean['country_origin'].replace("Peru, Mad., Dom. Rep.", "Peru, Madagascar, Dominican Republic")
data_clean['country_origin'] = data_clean['country_origin'].replace("Central and S. America", "Central and South America")
data_clean['country_origin'] = data_clean['country_origin'].replace("PNG, Vanuatu, Mad", "Papua New Guinea, Vanuatu, Madagascar")
data_clean['country_origin'] = data_clean['country_origin'].replace("Ven., Trinidad, Mad.", "Venezuela, Trinidad, Madagascar")
data_clean['country_origin'] = data_clean['country_origin'].replace("Ven.,Ecu.,Peru,Nic.", "Venezuela, Ecuador, Peru, Nicaragua")
data_clean['country_origin'] = data_clean['country_origin'].replace("Ven, Trinidad, Ecuador","Venezuela, Trinidad, Ecuador")
data_clean['country_origin'] = data_clean['country_origin'].replace("Ghana, Domin. Rep", "Ghana, Dominican Republic")
data_clean['country_origin'] = data_clean['country_origin'].replace("Ecuador, Mad., PNG","Ecuador, Madagascar, Papua New Guinea")
data_clean['country_origin'] = data_clean['country_origin'].replace("Mad., Java, PNG","Madagascar, Java, Papua New Guinea")
data_clean['country_origin'] = data_clean['country_origin'].replace("Gre., PNG, Haw., Haiti, Mad", "Grenada, Papua New Guinea, Hawaii, Haiti, Madagascar")

print ("After Cleaning country_origin column: ")
print (data_clean['country_origin'].unique())


# In[ ]:


# Data cleaning the bean origin column

data_clean['bean_origin'] = data_clean['bean_origin'].str.lower()

data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split(',')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('/')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('*')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('.')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('+')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split(';')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('-')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('(')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('#')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('1')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('2')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('3')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('4')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('5')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('6')[0])
data_clean['bean_origin'] = data_clean['bean_origin'].apply(lambda x: x.split('7')[0])
print (data_clean['bean_origin'].unique())
#print data_clean['bean_origin'].head()


# Previously we have seen that cocoa_percent attribute had the data type of object which was then tranfomed into integer 

# In[ ]:


# Converting cocoa_percent to numeric

print (data_clean.info())
data_clean['cocoa_percent'] = data_clean['cocoa_percent'].apply(lambda x: x.split('%')[0])
data_clean['cocoa_percent'] = pd.to_numeric(data_clean['cocoa_percent'], errors='coerce')
print (data_clean.info())


# In the dataset there were some special character epresent due to encoding such as '\xa0'. This characters were replaced with null

# In[ ]:


# Replacing the empty cells with null
data_clean = data_clean.replace('\xa0', np.nan)
count_null_values(data_clean, data_clean.columns)


# In[ ]:


# Changing the type for review_date from int to object
data_clean['review_date'] = data_clean['review_date'].astype(str)
data_clean['rating'] = data_clean['rating'].astype(str)
print(data_clean.info())


# ### Data Preprocessing

# In this dataset there were two numerical attributes which required scaling so that during training the machine learning model does not learn more from the attribute with the higher values when compared to the attribute with the low values.

# In[ ]:


# Normalizing the column with integer type

# Data Normalizing
from sklearn.preprocessing import StandardScaler 
data_norm = data
scaler_z = StandardScaler()
# Only the columns with integer and float type values are normalized
num_d = data_clean.select_dtypes(exclude=['object'])
data_clean[num_d.columns] = scaler_z.fit(num_d).transform(num_d)

# Getting information of the dataset after normalization
print (data_clean.head(10))
print (data_clean[num_d.columns].mean(axis= 0))
print (data_clean.info())


# ### Dealing with the null values for country_origin

# The attibutes with the null values were both categorical. In order to replace the missing values a KNN model was developed to predict the missing values. 

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Creating a temp dataset
temp_data_co = data_clean
# dropping the bean_type column since it has missing values
temp_data_co= temp_data_co.drop('bean_type', 1)

# Splitting the dataset into null and not null dataframe
test_data_co = temp_data_co[temp_data_co["country_origin"].isnull()]
train_data_co = temp_data_co[temp_data_co["country_origin"].notnull()]

# Label encoding only the categorical columns 
test_data_co_l = test_data_co.apply(LabelEncoder().fit_transform)
test_data_co_l['REF'] = data_clean['REF']
test_data_co_l['cocoa_percent'] = data_clean['cocoa_percent']

train_data_co_l = train_data_co.apply(LabelEncoder().fit_transform)
train_data_co_l ['REF'] = data_clean['REF']
train_data_co_l['cocoa_percent'] = data_clean['cocoa_percent']

# Defining the X and y 
X = train_data_co_l.drop('country_origin', axis=1).values
y = train_data_co_l['country_origin'].values

print (test_data_co.shape, train_data_co.shape)

X_train_co, X_test_co, y_train_co, y_test_co = train_test_split(X, y, test_size= 0.1, train_size=0.9, random_state=42)

print (X_train_co.shape, X_test_co.shape)

# Training a KNN machine leanring model to replace the missing values
from sklearn.neighbors import KNeighborsClassifier
# The model gave the best result at n_neighbors = 3
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train_co, y_train_co)
y_pred = knn.predict(X_test_co)
print (y_pred)
print (y_pred.size)

# Getting the accuracy metric
acc = accuracy_score(y_pred, y_test_co)
pre = precision_score(y_pred, y_test_co, average='micro')
rec = recall_score(y_pred, y_test_co, average='micro')
f1 = f1_score(y_pred, y_test_co, average='micro')

print ('Model performace for replacing the missing values: ')
print ('Accuracy: ', acc)
print ('Precision: ', pre)
print ('Recall: ',rec)
print ('F1 Score: ', f1)


# In[ ]:


# Predicting the null values for country_origin
pred_data_co_l = test_data_co_l.drop('country_origin', axis=1).values

# y_pred containing the predicted value
y_pred = knn.predict(pred_data_co_l)

# Storing the y_pred values in a column of a dataframe
temp1 = test_data_co_l
temp1["country_origin"] = y_pred

# Incrporating the prediction for the missing values into the main dataset
dataset_clean_co = pd.concat([train_data_co_l, temp1], join = 'inner')
print (dataset_clean_co.head())
print (dataset_clean_co.shape)


# ### Dealing with the null values for bean_type

# In[ ]:


# Preparing the dataset by propping the country_origin column since it had missing values
temp_data_bt = data_clean
temp_data_bt = temp_data_bt.drop('country_origin', 1)

# Splitting the dataset into null and not null dataframe
test_data_bt = temp_data_bt[temp_data_bt["bean_type"].isnull()]
train_data_bt = temp_data_bt[temp_data_bt["bean_type"].notnull()]

# Label encoding only the categorical columns 
test_data_bt = test_data_bt.apply(LabelEncoder().fit_transform)
test_data_bt['REF'] = data_clean['REF']
test_data_bt['cocoa_percent'] = data_clean['cocoa_percent']

train_data_bt = train_data_bt.apply(LabelEncoder().fit_transform)
train_data_bt['cocoa_percent'] = data_clean['cocoa_percent']
train_data_bt['REF'] = data_clean['REF']

# Defining the X and y 
X = train_data_bt.drop('bean_type', axis=1).values
y = train_data_bt['bean_type'].values

X_train_bt, X_test_bt, y_train_bt, y_test_bt = train_test_split(X, y, test_size= 0.2, train_size=0.8, random_state=42)


# Training a KNN machine leanring model to replace the missing values
from sklearn.neighbors import KNeighborsClassifier
# The model gave the best result at n_neighbors = 80
knn = KNeighborsClassifier(n_neighbors = 80)
knn.fit(X_train_bt, y_train_bt)
y_pred = knn.predict(X_test_bt)

# Getting the accuracy metric
acc = accuracy_score(y_pred, y_test_bt)
pre = precision_score(y_pred, y_test_bt, average='micro')
rec = recall_score(y_pred, y_test_bt, average='micro')
f1 = f1_score(y_pred, y_test_bt, average='micro')

print ('Model performace for replacing the missing values: ')
print ('Accuracy: ', acc)
print ('Precision: ', pre)
print ('Recall: ',rec)
print ('F1 Score: ', f1)


# In[ ]:


# Predicting the null values for bean_type with the KNN model

pred_data_bean_l = test_data_bt.drop('bean_type', axis=1).values

# y_pred storing the predicted values
y_pred = knn.predict(pred_data_bean_l)

temp = test_data_bt
temp["bean_type"] = y_pred

# Incrporating the prediction for the missing values into the dataset
dataset_clean_bean = pd.concat([train_data_bt, temp], join = 'inner')
print (dataset_clean_bean.head())
print (dataset_clean_bean.shape)


# ### Preparing the final dataset

# In[ ]:


data_clean_label_encoding = dataset_clean_bean
data_clean_label_encoding['country_origin'] = dataset_clean_co['country_origin']
# Checking the dataset for null values after data processing
count_null_values(data_clean_label_encoding, data_clean_label_encoding.columns)


# ### Exploratory Data Analysis

# Let's explore the correlation between the categorical values. This will give us a better understanding of how the attributes are related to one another.

# In[ ]:


# Finding out the correlation between the features in the dataset

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import LabelEncoder

sns.set(style="white")

# Compute the correlation matrix
corr = data_clean_label_encoding [['bean_origin','REF','review_date','cocoa_percent','company_location'                   ,'rating','maker','bean_type','maker', 'sub_bean_type',                   'country_origin']].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 8))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.set_context("paper", rc={"font.size":12,"axes.titlesize":12, "axes.labelsize":12}) 
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
ax.set_title('Correlation Between The Pairs of Columns ')
plt.show()


# Analysing the correlation matrix shown above we can notice that there are some high correlation between the attributes which means some of the attributes will not provide significant input to our machine leanring model. Moreover, there is a risk of having an overfitted model due to the presence of high correlation. This issue has been addressed by feature selection; discussed in the later part of the implementation.

# In[ ]:


# Finding the distribution of rating in the dataset
from scipy import stats
sns.distplot(data_clean["rating"], kde=False, fit=stats.gamma).set(title = 'Distribution for Chocolate Rating', xlabel = 'Rating', ylabel = 'Proportion Distribution' )
plt.show()


# The graph above shows the distribution of rating in the dataset. We can see that the pattern follows a normal distribution with having the peak rating around the value of 3

# ### Performing Feature Selection

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
get_ipython().run_line_magic('matplotlib', 'inline')
#Splitting the variables into features and target
X = data_clean_label_encoding.drop('rating', axis=1)
y = data_clean_label_encoding['rating'].values

print(X.shape)#printing dimensions of features
print(y.shape)#printing dimensions of label

#Printing the variability of all the features
print(X.var())
#Since the Variability of any column is not very low so selecting all the features based on variability


# Using ExtraTreesClassifier for feature selection
model = ExtraTreesClassifier()
model.fit(X,y)
print ("Dataset Size Before Feature Selection ")
print( X.shape)
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
clf.feature_importances_  
model = SelectFromModel(clf, prefit=True)
X_feat_select = model.transform(X)
print ("Dataset Size After Feature Selection ")
print (X_feat_select.shape )           
print ("Relative Feature importance for each of the Features- ")
print(clf.feature_importances_)

feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.title('Important Features')
plt.xlabel('Relative Importance')
plt.ylabel('Features')
plt.show()


# After applying feature selection we ended up with 5 features down from 10.

# ### Performing Data Partitioning

# In[ ]:


# Training- 80% Testing- 20%
X_train, X_test, y_train, y_test = train_test_split(X_feat_select, y, test_size= 0.2, random_state=42)
print ('Training and testing size')
print (X_train.shape)
print (X_test.shape)


# ### Random Forest Model Development

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
# Parameter for performing hyper tuning
parameters = {'n_estimators': [4, 5, 6], 'max_depth': [2, 3, 4],               'min_samples_split': [25, 30, 35], 'max_leaf_nodes': [4, 5, 6]}

random_forest_classifier = GridSearchCV(random_forest, parameters,  cv = 5)
random_forest_classifier.fit(X_train, y_train)
print(random_forest_classifier.best_params_)
print(random_forest_classifier.best_score_)


# The best test accuracy for the Random forest model is 25% which is very less as it can be interpreted as- The model will be able to predict the rating for the chocolate correctly once in every four attempts. However, this problem is a multiclass classification problem with 13 distinct values for the rating  which are- 3.75, 2.75, 3, 3.5, 4, 3.25, 2.5, 5, 1.75, 1.5, 2.25, 2, 1

# ### Rounding the values of rating

# We can increase the prediction of the model if we round the values for the ratings to its nearest whole number

# In[ ]:



temp = data_clean
temp['rating'] = temp['rating'].astype(float)

# Function to convert the values for the  rating to its nearest whole number
def round_rating(rating):
    if (rating < 1 ):
        return 0
    elif (rating > 0 ) and (rating < 2 ):
        return 1
    elif (rating >= 2 ) and (rating < 3 ):
        return 2
    elif (rating >= 3 ) and (rating < 4 ):
        return 3
    elif (rating >= 4 ) and (rating < 5 ):
        return 4
    else:
        return 5


temp['rating'] = temp['rating'].apply(round_rating)
print(temp['rating'].unique())


# In[ ]:


y_new = temp['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X_feat_select, y_new, test_size= 0.2, random_state=42)
print ('Suite-1 training and testing size')
print (X_train.shape)
print (X_test.shape)
print(y_new)


# In[ ]:


# Training the model
random_forest_classifier.fit(X_train, y_train)
print(random_forest_classifier.best_params_)
print(random_forest_classifier.best_score_)


# In[ ]:


# Testing the model
model = RandomForestClassifier(random_state=42, max_depth= 4, max_leaf_nodes= 4,                                       min_samples_split= 35, n_estimators= 6)
model.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)

# Getting the accuracy metric
acc = accuracy_score(y_pred, y_test)
pre = precision_score(y_pred, y_test, average='micro')
rec = recall_score(y_pred, y_test,average='micro')
f1 = f1_score(y_pred, y_test, average='micro')
print ('Model Performance Statistic: ')
print ('Accuracy: ', acc)
print ('Precision: ', pre)
print ('Recall: ',rec)
print ('F1 Score: ', f1)


# The accuracy of the model is now 70.75%
