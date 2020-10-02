#!/usr/bin/env python
# coding: utf-8

# # Wine Data Group Notebook
# We will be exploring a wine dataset to create visuals tha will allow us to visualize the relationship between different aspects of the data as well as create models to help predict beneficial wine making practices

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # visualization
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import eli5
from eli5.sklearn import PermutationImportance
from eli5 import show_weights
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
from wordcloud import WordCloud, STOPWORDS
import re

from nltk.tokenize import RegexpTokenizer
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Data Explanation/Exploration
# First we look at the raw data that we imported to wee what we are working with.

# In[ ]:


wine_file_path = '../input/wine-reviews/winemag-data-130k-v2.csv'
wine_data = pd.read_csv(wine_file_path)
wine_data


# Lets take a look at information regarding missing entries

# In[ ]:


#MATTHEW CADENA
print("Columns with Number of Missing Entries:")
print(wine_data.isnull().sum())


# In[ ]:


#LJS

#find outliers
def findOutliers(data_frame):
    #mean, standard deviation and three standard deviations
    data_mean = np.mean(data_frame)
    data_std = np.std(data_frame)
    three_stds = data_std *3

    #looking for outlier data that is 3 std above or below the mean
    lower_three_stds = data_mean - three_stds
    upper_three_stds = data_mean + three_stds
    
    lower_outliers = [x for x in data_frame if x < lower_three_stds]
    upper_outliers = [x for x in data_frame if x > upper_three_stds]
    
    without_outliers = [x for x in data_frame if x > lower_three_stds and x < upper_three_stds]
    
    num_lower = len(lower_outliers)
    num_higher = len(upper_outliers)
    num_total_outliers = num_lower + num_higher
    num_non_outliers = len(without_outliers)
    total = num_total_outliers + num_non_outliers
    
    print('Lower outliers: %d' % num_lower)
    print('Upper outliers: %d' % num_higher)
    print('Without outliers: %d' % num_non_outliers)
    print("*********************")
    print('num total outliers: %d'% num_total_outliers )
    print('total: %d' % total)
    print('**********************')
    print("Total PERCENT that are outliers: ", round(num_total_outliers/total, 9)* 100)
    


# In[ ]:


#LJS

#find the outlier data for points for all data
findOutliers(wine_data["price"])


# In[ ]:


#MATTHEW CADENA
wine_data.price.describe()
#USED TO SEE THE INFO BEFORE MANIP


# # Dealing with Outliers and Missing Data
# We nod trim the data to remove some extreme outliers for the price columns as well as fill empty entries with the mean value

# In[ ]:


#MATTHEW CADENA
q = wine_data["price"].quantile(0.999)
print(q)
wine_data_Outliers = wine_data[wine_data["price"]<q]
wine_data.price.describe()
#USED TO SEE THE INFO AFTER MANIP


# In[ ]:


#MATTHEW CADENA
#THIS FILLS ANY MISSING ENTRIES WITH THE AVERAGE
price_avg = wine_data["price"].mean()

wine_data['price'].fillna(price_avg, inplace = True)


# In[ ]:


#Cell removed #3
wine_data = wine_data.dropna(subset=['country'])

#replacing NaN with unknown for whole database
wine_data = wine_data.replace('Unknown', np.NaN)

wine_data.taster_name.fillna("Unknown")
wine_data.region_1.fillna("Unknown")

wine_data = wine_data[['country', 'province', 'region_1', 'winery', 'price', 'points', 'variety', 'title', 'taster_name', 'description']]
wine_data.rename(columns={'region_1':'region'}, inplace = True)
wine_data.head()


# In[ ]:


#TIFFANY TRAN
wine_data = wine_data.dropna(axis=0)


# In[ ]:


wine_data["country"] = wine_data["country"].astype('category')
wine_data["description"] = wine_data["description"].astype('category')
wine_data["province"] = wine_data["province"].astype('category')
wine_data["region"] = wine_data["region"].astype('category')
wine_data["taster_name"] = wine_data["taster_name"].astype('category')
wine_data["title"] = wine_data["title"].astype('category')
wine_data["variety"] = wine_data["variety"].astype('category')
wine_data["winery"] = wine_data["winery"].astype('category')


# In[ ]:


wine_data["country codes"] = wine_data["country"].cat.codes
wine_data["description codes"] = wine_data["description"].cat.codes
wine_data["province codes"] = wine_data["province"].cat.codes
wine_data["region codes"] = wine_data["region"].cat.codes
wine_data["taster codes"] = wine_data["taster_name"].cat.codes
wine_data["title codes"] = wine_data["title"].cat.codes
wine_data["variety codes"] = wine_data["variety"].cat.codes
wine_data["winery codes"] = wine_data["winery"].cat.codes

wine_data.head()


# In[ ]:


wine_data.describe()


# In[ ]:


#LJS

#information about price per country- how many, min, max for all data

from learntools.pandas.grouping_and_sorting import *
wine_data.groupby(['country']).price.agg([len,min,max])


# In[ ]:


cData = wine_data.copy()


# # Description Clean Up
# We will now begin our visuals by looking at the 15 most frequent relevant words and looking at how wine reviews containing those words affect their point spread

# In[ ]:


cData['description']= cData['description'].str.lower()
cData['description']= cData['description'].apply(lambda elem: re.sub('[^a-zA-Z]',' ', elem))  
cData['description']


# In[ ]:


tokenizer = RegexpTokenizer(r'\w+')
words_descriptions = cData['description'].apply(tokenizer.tokenize)
words_descriptions.head()


# In[ ]:


from collections import Counter
all_words = [word for tokens in words_descriptions for word in tokens]

count_all_words = Counter(all_words)
count_all_words.most_common(100)


# In[ ]:


stopword_list = stopwords.words('english')
ps = PorterStemmer()
words_descriptions = words_descriptions.apply(lambda elem: [word for word in elem if not word in stopword_list])
words_descriptions = words_descriptions.apply(lambda elem: [ps.stem(word) for word in elem])
cData['description_cleaned'] = words_descriptions.apply(lambda elem: ' '.join(elem))


# In[ ]:


all_words = [word for tokens in words_descriptions for word in tokens]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
count_all_words = Counter(all_words)
list_of_words = count_all_words.most_common(15)
list_of_words


# In[ ]:


first_ele =  [x[0] for x in list_of_words]
first_ele


# In[ ]:


index = 0
data1=pd.DataFrame()
data2=pd.DataFrame()
data3=pd.DataFrame()
data4=pd.DataFrame()
data5=pd.DataFrame()
data6=pd.DataFrame()
data7=pd.DataFrame()
data8=pd.DataFrame()
data9=pd.DataFrame()
data10=pd.DataFrame()
data11=pd.DataFrame()
data12=pd.DataFrame()
data13=pd.DataFrame()
data14=pd.DataFrame()
data15=pd.DataFrame()
dfs = [data1,data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15]
#x = [pd.DataFrame() for x in dfs]
while index < len(dfs):
    dfs[index] = cData[cData['description'].str.contains(first_ele[index])]
    index = index + 1


# In[ ]:


sns.set_style("dark")
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(30,30))

index = 0
row = 0
col = 0
while( index<len(dfs)):
    y = dfs[index]['points'].value_counts()
    x = y.index
    axes[row,col].bar(x,y)
    axes[row,col].set_title('Points Influenced by the keyword ' + first_ele[index])
    index = index + 1
    
    if col == 2:
        col = 0
        row = row + 1
        
    elif col != 2:
        col = col + 1


# # Data Visualization
# We will now look at various relationships between different aspects of the wine data set

# In[ ]:


#LJS

#distplot price distribution up to $500 by frequency for all data
sns.set_style("dark")
plt.figure(figsize=(30,10))

#graph = sns.distplot(wine_data['price'])
graph = sns.distplot(wine_data[wine_data['price'] < 500]['price'])
graph.set_title("Price Distribution up to $500", fontsize=18)
graph.set_xlabel("Price", fontsize=14)
graph.set_ylabel("Frequency Distribution", fontsize=14)


# In[ ]:


#LJS

#distplot price distribution up to $500 by frequency for non outlier data

plt.figure(figsize=(12,5))

#graph = sns.distplot(wine_data['price'])
graph = sns.distplot(wine_data_Outliers[wine_data_Outliers['price'] < 500]['price'])
graph.set_title("Price Distribution up to $500 without Outliers", fontsize=18)
graph.set_xlabel("Price", fontsize=14)
graph.set_ylabel("Frequency Distribution", fontsize=14)


# In[ ]:


#LJS

#regplot of Points vs distribution of Price for all data

plt.figure(figsize=(20,10))

graph = sns.regplot(x='points',y='price', data=wine_data, fit_reg = True)
graph.set_title("Points vs Distribution of Price", fontsize=18)
graph.set_xlabel("Points", fontsize=14)
graph.set_ylabel("Price Distribution", fontsize=14)


# In[ ]:


#MATTHEW CADENA
sns.set_style("dark")
plt.title('Wine Price vs Wine Points', fontsize=40)
plt.xlabel('Price Point', fontsize=20)
plt.ylabel('Points', fontsize=20)
#plt.show()
plt.xlim([0,200])   
plt.ylim([75,105])
g = sns.regplot(x = wine_data['price'], y = wine_data['points'],scatter=False, ci = 99.99)
g.figure.set_size_inches(20,10)


# In[ ]:


#TIFFANY TRAN
graph = wine_data[wine_data.price < 100].dropna().sample(5000)
sns.kdeplot(graph.price, graph.points, shade=True, cmap="bone_r")
plt.xlabel("Price", fontsize=15)
plt.ylabel("Points", fontsize=15)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.title("Price and Points Correlation", fontsize= 15)
plt.suptitle('Sample Size 5000', fontsize= 10)


# In[ ]:


#TIFFANY TRAN
graph = wine_data_Outliers[wine_data_Outliers.price < 100].dropna().sample(5000)
sns.kdeplot(graph.price, graph.points, shade=True, cmap="bone_r")
plt.xlabel("Price", fontsize=15)
plt.ylabel("Points", fontsize=15)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.title("Price and Points Correlation without Outliers", fontsize= 15)
plt.suptitle('Sample Size 5000', fontsize= 10)
plt.figure(figsize=(10,10))


# In[ ]:


#MATTHEW CADENA
#TOP 15 COUNTRIES BY WINE REVIEWS
sns.set(font_scale = 1.5)
g=sns.catplot(x="country",kind="count", palette="Spectral",data = wine_data,order=pd.value_counts(wine_data['country']).iloc[:15].index)
g.set_xticklabels(rotation=60)

g.fig.set_size_inches(30,10)
g.fig.suptitle('Top 15 Countries by Wine Reviews', fontsize=40)
plt.ylabel("Wine Reviews", fontsize = 25)
plt.xlabel("Country", fontsize = 25)


# In[ ]:


#TIFFANY TRAN
points_counts = wine_data['points'].value_counts()
plt.figure(figsize=(25,10))
sns.barplot(x = points_counts.index, y = points_counts)
plt.title("Number of Wines in each Point Value", fontsize = 40)
plt.ylabel("Number of Wines", fontsize = 25)
plt.xlabel("Point Value", fontsize = 25)


# In[ ]:


#LILLIAN
#MATTHEW CADENA: ADDED THE ORDER PARAMETER TO CLEAN UP THE PLOT
#TOP 15 VARIETY OF WINES
sns.set(font_scale = 1.5)
g=sns.catplot(x="variety",kind="count", palette="inferno_r",data = wine_data,order=pd.value_counts(wine_data['variety']).iloc[:15].index)
g.set_xticklabels(rotation=60)
g.fig.set_size_inches(30,10)
plt.title("Most Reviewed Wine Varieties", fontsize = 40)
plt.ylabel("Reviews", fontsize = 25)
plt.xlabel("Wine Variety", fontsize = 25)


# In[ ]:


#BOXPLOT FOR THE TOP 20 VARIETIES OF WINE BASED ON POINTS
#LILLIAN
#MATTHEW CADENA: ADDED THE ORDER PARAMETER TO CLEAN UP GRAPH
fig = plt.figure(figsize=(35,10))
g = sns.boxplot(x="variety", y="points", data=wine_data,order=pd.value_counts(wine_data['variety']).iloc[:150].index)
g.set_xticklabels(wine_data.variety, rotation = 270)
plt.ylim([75,100])


# In[ ]:


#MATTHEW CADENA
fig = plt.figure(figsize=(35,10))
g = sns.boxplot(x="winery", y="points", data=wine_data,order=pd.value_counts(wine_data['winery']).iloc[:100].index)
g.set_xticklabels(wine_data.winery, rotation = 270)
plt.ylim([75,100])
g.set_title("Top 100 Wineries by Points", fontsize = 40)
g.set_xlabel("Winery", fontsize=20)
g.set_ylabel("Points", fontsize =20)


# In[ ]:


#MATTHEW CADENA
fig = plt.figure(figsize=(35,10))
g = sns.boxplot(x="province", y="points", data=wine_data,order=pd.value_counts(wine_data['province']).iloc[:64].index)
g.set_xticklabels(wine_data.winery, rotation = 270)
plt.ylim([75,100])
g.set_title("Top Wine Producing Provinces by Points", fontsize = 40)
g.set_xlabel("Province", fontsize=20)
g.set_ylabel("Points", fontsize =20)


# In[ ]:


sns.set_palette(sns.color_palette("colorblind"))

plt.figure(figsize=(10,10))

country = wine_data.country.value_counts()[:5]

graph = sns.boxplot(x='country', y = "points", data=wine_data[(wine_data.country.isin(country.index.values))],order=pd.value_counts(wine_data['country']).iloc[:5].index)
graph.set_title("Top 5 Counries Based on Points", fontsize=20)
graph.set_xlabel("Country", fontsize=15)
graph.set_ylabel("Points", fontsize=15)
graph.set_xticklabels(graph.get_xticklabels(), rotation = 45)
plt.ylim([79,101])


# In[ ]:


#THIS NEEDS FIXING#
#KINDA FIXED....*

#Wine prices by country 
countries =['Argentina','Italy', 'France', 'Spain', 'US', 'Chile', 'Portugal', 'New Zealand', 'Germany', 'South Africa']
sub_data = wine_data[wine_data['country'].isin(countries)]
plt.figure(figsize=(15,10))
sns.set_context("paper", font_scale=2.5)
#sns.violinplot(x="country", y="price", data=sub_data,inner=None)
sns.violinplot(x="country", y="price", data=wine_data,order=pd.value_counts(wine_data['country']).iloc[:15].index ,inner=None)
plt.ylabel("Price", fontsize=25)
plt.xlabel("Country", fontsize=25)
plt.title("Top Ten Wine Producing Countrie and Their Wine Price Spread", fontsize=40)
#MATTHEW CADENA: I ADDED THE ORDER PARAMETER TO ONLY DISPLAY THE COUNTRIES IN THE 'COUNTRIES' LIST INSTEAD ALL THE COUNTRIES
#Cris L
plt.ylim(0,450)
plt.xticks(rotation =90)


# # US Data
# We will take a quick look at wine data pertaining to the US, seeing as it contains the most reviews

# In[ ]:


#CRISTIAN LYNCH

#Wines in the US
wine_data_US = wine_data[wine_data['country'] == 'US']
wine_data_US.head()


# In[ ]:


#CRISTIAN LYNCH

#Top 5 wine variety in the US
value_counts = wine_data_US["variety"].value_counts()
value_counts.head()


# In[ ]:


#Plot of reviews by US provinces
sns.set_style("dark")
plt.figure(figsize=(20, 10))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
sns.countplot(x="province", data=wine_data_US,order=pd.value_counts(wine_data_US['province']).iloc[:5].index)
plt.ylabel("Number of Reviews", fontsize=25)
plt.xlabel("Province", fontsize=25)
plt.title("Count of Reviews by province in US", fontsize=40)
plt.xticks(rotation=0)
plt.show()
#MATTHEW CADENA: I ADDED THE ORDER PARAMETER TO ONLY DISPLAY THE TOP 5 RATHER THAN EVERY PROVINCE
#Cris L


# # Model Visualization
# We have just looked at all the graphs and data visualization for this dataset. We will now work with the data to build models and predict certain things. Frst we will create a basic model look at the Mean Absolute Error.

# In[ ]:


y = wine_data.points
features = ['price', 'country codes', 'province codes', 'variety codes', 'winery codes', 'region codes']
x = wine_data[features]
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state = 0)
basic_model = DecisionTreeRegressor()
basic_model.fit(train_x, train_y)
val_predictions = basic_model.predict(val_x)
print("Printing MAE for Basic Decision Tree Regressor:", mean_absolute_error(val_y, val_predictions))
#Chris and Tiffany worked 


# Now we will be using the basic Decision Tree and adding Leaf Nodes to try and focus the data

# In[ ]:


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    leaf_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    leaf_model.fit(train_x, train_y)
    preds_val = leaf_model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %f" %(max_leaf_nodes, my_mae))
#Chris worked on this


# Now we will look at the Random Forest Regressor model

# In[ ]:


forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_x, train_y)
forest_preds = forest_model.predict(val_x)
print("Printing MAE for RandomForest Model:",mean_absolute_error(val_y, forest_preds)) #TIFFANY ADDED THIS LINE
#Chris worked on this


# In[ ]:


perm = PermutationImportance(basic_model, random_state=1).fit(val_x, val_y)
eli5.show_weights(perm, feature_names = val_x.columns.tolist())
#Chris worked on this


# In[ ]:


#TIFFANY
#choosing the prediction target
y = wine_data.points

#choosing features
wine_features = ['price', 'country codes', 'province codes', 'variety codes', 'winery codes', 'region codes']
X = wine_data[wine_features]

#testing
#X.describe()
X.head()


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
wine_model = DecisionTreeRegressor(random_state=1)

wine_model.fit(X,y)


# In[ ]:


print("Making point predictions for the following 5 Wines:")
print(X.head())
print("The points predictions are")
print(wine_model.predict(X.head()))


# Prediciting The Error, How far I am off from the actual. [error = actual - predicted]

# In[ ]:


from sklearn.metrics import mean_absolute_error

predicted_wine_points = wine_model.predict(X)
print("Printing the mean absolute error", mean_absolute_error(y, predicted_wine_points))


# In[ ]:


Currently we will be splitting the data to have better results


# In[ ]:


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X,y, random_state = 0)

wine_model = DecisionTreeRegressor()

wine_model.fit(train_X,train_y)

#getting predicted points
val_predictions = wine_model.predict(val_X)
print("Using the DecisionTreeRegressor.. Now\nPrinting the mean absolute value ",mean_absolute_error(val_y, val_predictions))


# # Using a Pipeline

# In[ ]:


#CHRISTOPHER MUCHKENFUSS


pipe_data = pd.read_csv(wine_file_path)

pipe_data.dropna(axis=0, inplace=True)
y = pipe_data.points
#pipe_data.drop(['price'], axis=1, inplace=True)

 


X_train_full, X_valid_full, y_train, y_valid = train_test_split(pipe_data, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)
# Select categorical columns
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10000 and 
                    X_train_full[cname].dtype == "object"]

 

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

 

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
#X_test = X_test_full[my_cols].copy()

 

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

 

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

 

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

 

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

 

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

 

# Preprocessing of training data, fit model 
clf.fit(X_train, y_train)

 

# Preprocessing of validation data, get predictions
preds = clf.predict(X_valid)

print('MAE Using Pipeline:', mean_absolute_error(y_valid, preds))


# In[ ]:


#TIFFANY
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model',
                               RandomForestRegressor(n_estimators=50,random_state=0))])


# In[ ]:


#TIFFANY
from sklearn.model_selection import cross_val_score

points_CV = -1 * cross_val_score(my_pipeline, X, y, cv=5, 
                              scoring = 'neg_mean_absolute_error')
print("Using Cross Validation..\nNow Printing Mean Absolute Error points:\n",
      points_CV)


# In[ ]:


#TIFFANY TRAN
print("Using Cross Validation..\nNow Printing Average Mean Absolute Error points across all experiments: \n", points_CV.mean())


# In[ ]:


#TIFFANY TRAN
from xgboost import XGBRegressor

xgbr_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
xgbr_model.fit(train_X, train_y, 
             early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)], 
             verbose=False)


# In[ ]:


#TIFFANY TRAN
from sklearn.metrics import mean_absolute_error

predictionsXBGR = xgbr_model.predict(val_X)
print("Mean Absolute Error: " + str(mean_absolute_error(predictionsXBGR, val_y)))

