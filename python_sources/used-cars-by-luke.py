#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pydotplus')


# In[ ]:


import pandas as pd


# ## Data
# We need some data, true...
# Data for analysis will be the dataset you can find at Kaggle, containing prices of some used cars (Janusze motoryzacji of USA). You can notice relationships there, f.e. draw conclusions what factors make a car expensive 
# 
# Example used here you may find at Kaggle https://www.kaggle.com/austinreese/craigslist-carstrucks-data

# # E for Extract

# `pandas` offers lots of `read_...` functions, here we're using one of them, to read a CSV file `read_csv`

# In[ ]:


path = "../input/craigslist-carstrucks-data/vehicles.csv"
df = pd.read_csv(path)


# ## EDA
# EDA stands for Exploratory Data Analysis. This is an important part of data analysis process. You want to learn what are the characteristics of your data, what is its nature. First a sample of data with `head`

# In[ ]:


df.head()


# `shape` is telling you dimensions of data, what is the number of observations (rows) and features (columns)

# In[ ]:


df.shape


# You may be interested what are features we have available together with types

# In[ ]:


df.dtypes


# Let us take a look on how numbers are distributed

# In[ ]:


df.describe()


# ... but sometimes more intense look into data is needed, and then one may decide to use an external library like `pandas_profiling`

# In[ ]:


from pandas_profiling import ProfileReport
profile = ProfileReport(df, minimal=True, title='Used Cars Profiling Report', html={'style':{'full_width':True}})
profile


# # T is for Transform
# With `pandas` it is so easy to transform your data. Cleaning and munging is so important, because only then we may draw valuable conclusions

# In[ ]:


columns_to_skip = ['id','url','region','region_url','title_status','vin','image_url','description', 'county', 'state', 'lat', 'long']
df = df.drop(columns=columns_to_skip)
df.dtypes


# We got rid of some columns so now it seems we are free to deal with missing values

# ## Missing values
# `pandas` is an excellent tool to clean and process your data. We will get rid of some features that seem to be not needed for this calculation.
# 
# One of the things you have to decide about your data is how to deal with missing data. Sometimes we decide to **remove** missing values. On other occasions we could say we **aproximate** them. It is up to the task we have. Here I do not care so much about loosing some information so I will get rid off null values.

# In[ ]:


df.shape


# In[ ]:


df.isna().sum().plot.bar()


# It is clear there is one column containing a lot of null values (`size`) so I will just remove it.

# In[ ]:


columns_to_skip_because_of_null_quantity = ['size']


# In[ ]:


df = df.drop(columns=columns_to_skip_because_of_null_quantity)


# In[ ]:


df.shape


# We ended up with a data set containing 12 features.
# 
# Now it is time to take our strategy against null values, we will remove all rows which have some null values, with a simple yet powerful `dropna` function. What is cool, this function has plenty of parametrization options that help you to adjust its strategy according to your needs.

# In[ ]:


df = df.dropna()


# After this *destruction* shape has changed, we lost some of the observations.

# In[ ]:


df.shape


# A quick look into the top part of the dataset

# In[ ]:


df.head()


# And let us count the unique number of elements

# In[ ]:


df.nunique()


# ## Feature engineering

# Our goal is to make classification. We are not wishing to play with regression of real values, we need categories.
# 
# There are some columns we may adapt, we can take a range of it and split it into chunks.
# 
# Other columns can make it impossible, so we need to remove them too. Such a column is `model`. Sorry, we `drop` you man.

# In[ ]:


df = df.drop(columns='model')


# Now it is time to generalize some values. `odometer` is an example of that. First, what kind of values do we have there? We can easily print a *histogram* using `hist` function of a series.

# In[ ]:


df['odometer'].hist()


# We can see big disproportions, skewness od this data. What are extreme values? 
# 
# We can learn when we sort values, with the *descending* order.

# In[ ]:


df.sort_values('odometer', ascending=False).head(500)


# Basing on myy **domain knowledge** I can judge, mileage of 10M miles is too much. Let us look again at histogram now, checking mileage of 300k and less.
# 
# We may easily filter pandas dataframe using square brackets and putting a condition inside. It works like SQL `where` clause.

# In[ ]:


df[df.odometer < 300000]['odometer'].hist()


# Now we make some arbitrary decision, we divide mileage into 3 groups

# In[ ]:


categories = [
    ('0_light', 60000),
    ('1_medium', 120000),
    ('2_heavy', 9999999999)
]

def odocategories(distance):
    for name,value in categories:
        if distance < value:
            return name
    return categories[-1][0]
df['distance'] = df['odometer'].apply(odocategories)
df.head()


# With machine learning classification tasks, we need to have features and labels, input and output. The algorithm will try to find patterns in input affecting the output changes. 
# 
# In our case we want to make things simpler, so we will use a simple output type - binary one. We will just say whether the car is cheap or expensive. We will introduce then a new column `expensiveness` telling you only `True` if the car is **expensive**, and `False` otherwise.
# 
# Once again we need to make our continuous values to be spread into some categories. So we have to find the expensiveness trigger.

# In[ ]:


df['price'].hist()


# Distribution is skewed, too. So to make our visualization clear, we need to take a look into dataset exluding extremely high values

# In[ ]:


df.sort_values('price', ascending=False).head(100)


# It is effective to look only at cars, which cost less then 100k

# In[ ]:


df[df.price < 100000]['price'].hist(bins=100)


# Hmm.. it seems we do have some zero-valued cars.

# In[ ]:


df.sort_values('price').head()


# If a car was **sold** there was some **price** so it is an error, thus I will remove such cars

# In[ ]:


df = df[df.price > 0]


# With pandas you may easily create a new columns as a result of lambda expression

# In[ ]:


expensiveness_trigger = 20000

df['expensive'] = df.price.map(lambda price: 0 if (price < expensiveness_trigger) else 1)


# In[ ]:


df.head()


# Here checking sample of only expensive cars

# In[ ]:


df[df.price > expensiveness_trigger].head()


# Total distribution of classes in a column can be easily retrieved by `value_counts`

# In[ ]:


df.expensive.value_counts()


# One can also create a column as a result of function execution

# In[ ]:


def cylinder_text_to_number(txt):
    first_letter = txt[0]
    if first_letter.isdigit():
        return int(first_letter)
    else:
        return None
df['cylinder_number'] = df.cylinders.apply(cylinder_text_to_number)
# df.isna()['cylinder_number'].sum()
df = df.dropna()
df.head()


# One more column we will treat indepe

# In[ ]:


df.condition.value_counts()


# From the scikit-learn library we will use `LabelEncoders` to change our categories into number. Only one column I iwll treat in some special manner.  `condition` will be transformed using the dictionary/map I will provide

# In[ ]:


conditions = {
    'salvage' : 0,
    'fair': 1,
    'good': 2,
    'excellent': 3,
    'like new': 4,
    'new': 5
}
df['cat_condition'] = df.condition.apply(lambda v:conditions[v])


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le_manufacturer = LabelEncoder()
df['cat_manufacturer'] = le_manufacturer.fit_transform(df['manufacturer'])
le_fuel = LabelEncoder()
df['cat_fuel'] = le_fuel.fit_transform(df['fuel'])
le_transmission = LabelEncoder()
df['cat_transmission'] = le_transmission.fit_transform(df['transmission'])
le_type = LabelEncoder()
df['cat_type'] = le_type.fit_transform(df['type'])
le_paint_color = LabelEncoder()
df['cat_paint_color'] = le_paint_color.fit_transform(df['paint_color'])
le_distance = LabelEncoder()
df['cat_distance'] = le_distance.fit_transform(df['distance'])
df.head()


# # Train ML model
# Our matrix is ready, we have numerical values for each feature. Time to do some Machine Learning.

# In[ ]:


df.dtypes


# With pandas you can easily select some columns, also using their type.

# In[ ]:


from sklearn.model_selection import train_test_split

y = df['expensive']
X = df.select_dtypes('number').drop(columns=['price', 'expensive', 'odometer'])

X


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

feature_names = list(X.columns)
feature_names


# In[ ]:


X_train.shape


# In[ ]:


y_test.shape


# Matrices ready

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=5, random_state=0)


# In[ ]:


dtc.fit(X_train, y_train)


# ## Test model quality

# In[ ]:


dtc.score(X_test, y_test)


# In[ ]:


from sklearn.metrics import classification_report
y_true = y_test
y_pred = dtc.predict(X_test)
print(classification_report(y_true, y_pred))


# ## Visualize tree

# In[ ]:


dtc.classes_


# In[ ]:


labelencoder_classes = lambda le: list(zip(le.classes_, range(len(le.classes_))))
print("type")
print(labelencoder_classes(le_type))
print("fuel")
print(labelencoder_classes(le_fuel))


# In[ ]:


import sklearn.tree as tree
import pydotplus
from sklearn.externals.six import StringIO 
from IPython.display import Image
dot_data = StringIO()
tree.export_graphviz(dtc, 
 out_file=dot_data, 
 class_names=['cheap','expensive'],
 feature_names=feature_names,
 filled=True, # Whether to fill in the boxes with colours.
 rounded=True, # Whether to round the corners of the boxes.
 special_characters=True,
                     proportion=True
                    )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())


# monochromatic only
# https://stackoverflow.com/questions/42891148/changing-colors-for-decision-tree-plot-created-using-export-graphviz

# In[ ]:


graph.write_png("used_cars.png")


# Variance of each variable explained

# In[ ]:


from sklearn.feature_selection import chi2


# In[ ]:


y.value_counts()


# In[ ]:


chi2_results = chi2(X, y)
df_chi2 = pd.DataFrame(chi2_results)
df_chi2.columns = X.columns
df_chi2.index = ['cheap', 'expensive']
df_chi2 = df_chi2.T
df_chi2.cheap.plot.bar()


# # Pretty charting

# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


df = df[df.year > 1980]


# In[ ]:


df.groupby('year').count()['price'].iplot()


# In[ ]:


df_count_yearly = pd.DataFrame(df.groupby(['year', 'type']).count()['price'])
df_count_yearly.columns = ['count']
df_count_yearly.head()


# In[ ]:


df_pivot = pd.pivot_table(df_count_yearly, columns='type', index='year')['count']
df_pivot.head()


# In[ ]:


df_pivot.iplot()

