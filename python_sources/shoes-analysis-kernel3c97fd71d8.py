#!/usr/bin/env python
# coding: utf-8

# # The Question:
# 
# Which factors are most influential to Shoe Price?
# 
# # Intended Audience:
# 
# Consumers who want more information on how to save money when looking for shoes
# 
# # The factors (or feature variables) we'll be looking at:
# 
# Brand, category, color, rating, dateAdded, dateUpdated, merchant, size
# 
# # Hypothesis:
# 
# Brand and rating are the most influential factors for shoe price.
# 
# 
# 

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv('../input/7210_1.csv', parse_dates=['dateAdded', 'dateUpdated'])
df.head()


# Let's drop all of the columns we won't be using:

# In[2]:


df = df[['brand', 'categories', 'colors', 'dateAdded', 'dateUpdated', 'prices.amountMin', 'reviews', 'prices.merchant','prices.size']] #I'm not taking the Max Prices, because most of them are the same
df.head()


# I'm going to rename some of these colum names so that they make more sense:

# In[3]:


df = df.rename(columns={'prices.merchant': 'Merchant', 'brand': 'Brand', 'categories': 'Categories','colors': 'Colors', 'prices.amountMin' : 'Price', 'reviews': 'Rating', 'prices.size': 'Size'}) 
df.head()


# Now, let's have a look at the empty spaces, and figure out how to handle them:

# In[4]:


df.info()


# In[5]:


print("Null values per column:")
df.isnull().sum()


# The "Colors", "Rating", and "Size" columns have an  incredible amound of nulls.  However, even if we drop thos rows, we'll still have a few thousand left over to work with.  I think I'll create a few other DataFrames that contain these dropped rows so that I can use them in Machine Learning later on:

# In[6]:


Brand_df = df[['Brand', 'Price']].dropna()
Colors_df = df[["Colors", "Price"]].dropna()
Rating_df = df[["Rating", "Price"]].dropna()
Merchant_df = df[['Merchant', 'Price']].dropna()
Size_df = df[["Size", "Price"]].dropna()
NoNull_df = df.dropna()


# We'll need to clean up the "Categories" and "Ratings" columns quite a bit:

# In[7]:


# Honestly I'm not yet skilled enough to clean it up the way I want to.  Basically I was wanting to have only the "rating" info show up in the Rating
# column and have the "categorical" info (such as "Boots" and "Athelitic") show up in the Category column.  If someone can show me how to do this,
# I'd greatly appreciate it!  

df.drop(['Categories', 'Rating'], axis=1, inplace=True)


# In[8]:


df.head()


# Now let's visualize some of the dates here:

# In[9]:


print("How many entries we have over a period of time:")

df['dateAdded'].value_counts().resample('Y').sum().plot.line()


# In[10]:


dfp = df[['dateAdded', 'Price']].dropna()
dfp = dfp.drop_duplicates(subset=['dateAdded', 'Price'], keep=False)
dfp = dfp.sort_values(by='dateAdded')

import numpy as np
import scipy.stats as stats
import scipy.special as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.plot(dfp['dateAdded'],dfp['Price'])
plt.xticks(rotation=45)
plt.xlabel('dateAdded')
plt.ylabel('Price')
plt.title('Price over the Dates Added:')


# The prices seem to increase over time, but peaking at the end of 2015 and beginning of 2016

# In[11]:


dfp = df[['dateUpdated', 'Price']].dropna()
dfp = dfp.drop_duplicates(subset=['dateUpdated', 'Price'], keep=False)
dfp = dfp.sort_values(by='dateUpdated')

import numpy as np
import scipy.stats as stats
import scipy.special as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.plot(dfp['dateUpdated'],dfp['Price'])
plt.xticks(rotation=45)
plt.xlabel('dateUpdated')
plt.ylabel('Price')
plt.title('Price over the Dates Updated:')


# In[12]:


dfp = df[['dateAdded', 'dateUpdated', 'Price']].dropna()
dfp = dfp.drop_duplicates(subset=['dateUpdated', 'dateAdded', 'Price'], keep=False)
dfp = dfp.sort_values(by='dateAdded')

import numpy as np
import scipy.stats as stats
import scipy.special as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.plot(dfp['dateUpdated'],dfp['Price'], label='dateUpdated')
plt.plot(dfp['dateAdded'],dfp['Price'], label='dateAdded')
plt.xticks(rotation=45)
plt.xlabel('Dates Added/Updated')
plt.ylabel('Price')
plt.title('Price over the Dates Added/Updated:')
plt.legend(loc='upper right')


# The intent here was to visualize if there was any increase or decrease in the price if there was more or less time between the dateAdded and dateUpdated, but it is very difficult to get any insights from the visualization here as is.
# 
# 
# 
# Let's try something new:

# In[13]:


dfp.head(20)


# In[14]:


dfa = dfp[['dateAdded', 'Price']].dropna()
dfu = dfp[['dateUpdated', 'Price']].dropna()
dfd = dfu['dateUpdated'] - dfa['dateAdded']
dfdwp = [dfd, dfa[['Price']]]
dfdwp = pd.concat(dfdwp, axis=1)
dfdwp.head(20)


# In[15]:


dfdwp = dfdwp.rename(columns={0: 'Time_Until_Update'})
dfdwp.head(20)


# In[16]:


dfdwp = dfdwp.reset_index().groupby("Time_Until_Update").mean()
dfdwp.head()


# In[17]:


dfdwp.loc[dfdwp['index'] == 11157]  #This is a test to make sure we did it right


# Now, I have created a DataFrame where one column contains the amount of time between the dateAdded and the dateUpdated, while the Price column contains an average of the prices where the times were the same, since we need to do that for our visualization:

# In[18]:


dfdwp.plot(y='Price', use_index=True)
plt.xticks(rotation=45)  #I'll need someone to show me how to get the x tick mark labels to show up here
plt.xlabel('Time Until Update')
plt.ylabel('Price')
plt.title('Price over the Time Until Update:')
plt.legend(loc='upper right')


# As we can pretty well see, there is no obviously visible change in the amount of time that passes until the product is updated.

# In[19]:


dfdwp.reset_index(level=0, inplace=True)
dfdwp.head()


# In[20]:


plt.plot(dfdwp['Time_Until_Update'],dfdwp['Price'])
plt.xticks(rotation=45)  # Never mind, I've figured it out :)
plt.xlabel('Time_Until_Update')
plt.ylabel('Price')
plt.title('Price over the Time Until Update:')


# I've figured out how to fix that 'index problem' I was having in my code earlier, but now my 'typical coded matplotlib' plot looks way different than the 'index based x axis coded' plot I had made before.
# 
# I'll reproduce the 'index based x axis coded' plot below so that you can see what I mean:

# In[21]:


dfdwp.plot(y='Price', use_index=True)


# Maybe it has to do with the range on the x-axis.  I noticed that the plot before had '1e17' at the bottom right corner with a range of 0.0 to 0.8, while this one has a range of 0 to 4000.
# 
# Let's try changing the range on the x-axis:

# In[22]:


plt.plot(dfdwp['Time_Until_Update'],dfdwp['Price'])
plt.xticks(rotation=45) 
plt.xlabel('Time_Until_Update')
plt.ylabel('Price')
plt.title('Price over the Time Until Update:')
plt.xlim([0, 4000])


# In[23]:


dfdwp.plot(y='Price', use_index=True)
plt.xlim([0, 10000])


# In[24]:


dfdwp.info()


# Nope, looks like that didn't work.  I'll need some extra help on this one, so if anyone has any suggestions, please feel free to let me know. 
# 
# 
# 
# I think that I will use this graph for reference in my analysis because the x ticks appear to be based on the time values instead of their positional indexes:

# In[25]:


plt.plot(dfdwp['Time_Until_Update'],dfdwp['Price'])
plt.xticks(rotation=45) 
plt.xlabel('Time_Until_Update')
plt.ylabel('Price')
plt.title('Price over the Time Until Update:')


# Now, it looks like the the more time has passed, the less expensive the shoes are.  Let's remove some of the outliers (anything over $500) and see how it looks:

# In[26]:


from scipy import stats

dfdwpno = dfdwp[dfdwp["Price"] < 501]
dfdwpno.head()


# In[27]:


print("Number of outliers:")
print("")
dfdwpoo = dfdwp[dfdwp["Price"] > 500]
dfdwpoo.info()


# In[28]:


plt.plot(dfdwpno['Time_Until_Update'],dfdwpno['Price'])
plt.xticks(rotation=45) 
plt.xlabel('Time_Until_Update')
plt.ylabel('Price')
plt.title('Price over the Time Until Update (no outliers):')


# It looks like this same trend holds through even when we remove the outliers

# Now that we are finished exploring the date values, let's have a look at our categorical values: Brand, Color, Merchant, Size

# Let's start with Brand:

# In[29]:


Brand_dfm = Brand_df.reset_index().groupby("Brand").mean()
Brand_dfm.reset_index(level=0, inplace=True)
Brand_dfm.head(10)


# It looks like some of the brands are the same, but are treated differently by the program since there is a difference in capitalization (for example "143 GIRL" and "143 Girl").  I'm going to fix that here:

# In[30]:


Brando = Brand_dfm['Brand'].apply(lambda x: x.upper())
Brando.head()


# In[31]:


Brand_dfm = [Brando, Brand_dfm[['Price', 'index']]]
Brand_dfm = pd.concat(Brand_dfm, axis=1)
Brand_dfm.head(10)


# In[32]:


Brand_dfm = Brand_dfm.reset_index().groupby("Brand").mean()
Brand_dfm.reset_index(level=0, inplace=True)
Brand_dfm.head(10)


# In[33]:


plt.bar(Brand_dfm['Brand'],Brand_dfm['Price'])
plt.xticks(rotation=45) 
plt.xlabel('Brand')
plt.ylabel('Price')
plt.title('Average Price for each Brand:')


# Looks like ther's way too many brands to list clearly in a bar graph.  I'm going to reorder the DataFrame so that it's sorted by the price:

# In[34]:


Brand_dfm = Brand_dfm.sort_values(by='Price')
Brand_dfm.head(10)


# In[35]:


Brand_dfm.describe()


# So it looks like the average shoe price is $95.90, but it can range from $0.99 to $3322.19!  Quite a range, let's see how it looks visually:

# In[39]:


# some dummy lists with unordered values 
x_axis = Brand_dfm['Brand']
y_axis = Brand_dfm['Price']

def barplot(x_axis, y_axis): 
    # zip the two lists and co-sort by biggest bin value         
    ax_sort = sorted(zip(y_axis,x_axis), reverse=True)
    y_axis = [i[0] for i in ax_sort]
    x_axis = [i[1] for i in ax_sort]

    # the above is ugly and would be better served using a numpy recarray

    # get the positions of the x coordinates of the bars
    x_label_pos = range(len(x_axis))

    # plot the bars and align on center of x coordinate
    plt.bar(x_label_pos, y_axis,align="center")

    # update the ticks to the desired labels
    plt.xticks(x_label_pos,x_axis)


barplot(x_axis, y_axis)
plt.show()


# It looks like most are $100 or less.  Brand still seems to hold quite a sway on price, especially once it gets into the higher range.  Let's have a look at Color next:

# In[42]:


Colors_dfm = Colors_df.reset_index().groupby("Colors").mean()
Colors_dfm.reset_index(level=0, inplace=True)
Colors_dfm.head(10)


# In[43]:


Colors_dfm.info()


# That first "," color doesn't make any sense, let's remove it:

# In[44]:


Colors_dfm = Colors_dfm[Colors_dfm.Colors != ","]
Colors_dfm.head()


# Now, let's pull up some statistics:

# In[46]:


Colors_dfm.describe()


# Average price based on color is $74, with a standard deviation of $95.66, a minimum of $0.01 and a maximum of $1483.12.  Let's see how this looks visually:

# In[47]:


# some dummy lists with unordered values 
x_axis = Colors_dfm['Colors']
y_axis = Colors_dfm['Price']

def barplot(x_axis, y_axis): 
    # zip the two lists and co-sort by biggest bin value         
    ax_sort = sorted(zip(y_axis,x_axis), reverse=True)
    y_axis = [i[0] for i in ax_sort]
    x_axis = [i[1] for i in ax_sort]

    # the above is ugly and would be better served using a numpy recarray

    # get the positions of the x coordinates of the bars
    x_label_pos = range(len(x_axis))

    # plot the bars and align on center of x coordinate
    plt.bar(x_label_pos, y_axis,align="center")

    # update the ticks to the desired labels
    plt.xticks(x_label_pos,x_axis)


barplot(x_axis, y_axis)
plt.show()


# This looks very similar to our previous graph on Brands.  Let's see what the top 10 colors are:

# In[52]:


Colors_dfm = Colors_dfm.sort_values(by='Price')
print(Colors_dfm.nlargest(10, 'Price'))


# And now the bottom 10 colors:

# In[53]:


Colors_dfm.head(10)


# Next, let's look at Merchant:

# In[54]:


Merchant_dfm = Merchant_df.reset_index().groupby("Merchant").mean()
Merchant_dfm.reset_index(level=0, inplace=True)
Merchant_dfm.head(10)


# In[55]:


Merchant_dfm.describe()


# Average Merchant DataFrame gives us an average of 84.26 per merchant, with a Standard Deviation of 231.13, a minimum of 3.56, and a maximum of 365.49.  Let's see how this looks visually:

# In[56]:


# some dummy lists with unordered values 
x_axis = Merchant_dfm['Merchant']
y_axis = Merchant_dfm['Price']

def barplot(x_axis, y_axis): 
    # zip the two lists and co-sort by biggest bin value         
    ax_sort = sorted(zip(y_axis,x_axis), reverse=True)
    y_axis = [i[0] for i in ax_sort]
    x_axis = [i[1] for i in ax_sort]

    # the above is ugly and would be better served using a numpy recarray

    # get the positions of the x coordinates of the bars
    x_label_pos = range(len(x_axis))

    # plot the bars and align on center of x coordinate
    plt.bar(x_label_pos, y_axis,align="center")

    # update the ticks to the desired labels
    plt.xticks(x_label_pos,x_axis)


barplot(x_axis, y_axis)
plt.show()


# It looks very similar to our previous two graphs with Brand and Color, but there is definitely a higher peak towards the left hand side.  Let's see what the top 10 merchants are:

# In[57]:


Merchant_dfm = Merchant_dfm.sort_values(by='Price', ascending=False)
Merchant_dfm.head(10)


# It looks like JewelsObsession and Shoesbyclair take the top 2 by far, and the rest gradually decrease.  Let's see how the bottome 10 merchants look like:

# In[58]:


Merchant_dfm = Merchant_dfm.sort_values(by='Price', ascending=True)
Merchant_dfm.head(10)


# Let's explore the Size category now.  Personally, I don't think that this will be a very reliable indicator of size, since I wouldn't think that a Small shoe size will cost any more or less than a Large shoe size, as long as it's the exact same shoe otherwise, but let's see what our EDA shows:

# In[59]:


Size_dfm = Size_df.reset_index().groupby("Size").mean()
Size_dfm.reset_index(level=0, inplace=True)
Size_dfm.head(10)


# In[60]:


Size_dfm.info()


# In[61]:


Size_dfm.describe()


# I may yet stand corrected: Average Size Prices DataFrame shows a minimun of 5.99 with a maximum of 995, and a Standard Deviation of 318.76.  Let's see how this looks visually:

# In[63]:


# some dummy lists with unordered values 
x_axis = Size_dfm['Size']
y_axis = Size_dfm['Price']

def barplot(x_axis, y_axis): 
    # zip the two lists and co-sort by biggest bin value         
    ax_sort = sorted(zip(y_axis,x_axis), reverse=True)
    y_axis = [i[0] for i in ax_sort]
    x_axis = [i[1] for i in ax_sort]

    # the above is ugly and would be better served using a numpy recarray

    # get the positions of the x coordinates of the bars
    x_label_pos = range(len(x_axis))

    # plot the bars and align on center of x coordinate
    plt.bar(x_label_pos, y_axis,align="center")

    # update the ticks to the desired labels
    plt.xticks(x_label_pos,x_axis)


barplot(x_axis, y_axis)
plt.show()


# This graph certainly looks a little different than the other 3 categories.  There appears to be longer plateaus towards the top.  This may be because shoe size is listed 'specially' or 'different' for different brands.  Let's see what the top and bottom 10 shoe sizes are:

# In[66]:


Size_dfm = Size_dfm.sort_values(by='Price', ascending=False)
print("Top 10 Shoe Sizes:")
Size_dfm.head(10)


# Hmm, it looks like nearly all of the Top 10s have a "B" at the end.

# In[67]:


Size_dfm = Size_dfm.sort_values(by='Price', ascending=True)
print("Bottom 10 Shoe Sizes:")
Size_dfm.head(10)


# It appears that many of the smaller shoe sizes, and ones with a range appear to be in the Bottom 10s

# Now that we have explored out data, let's go into Machine Learning:

# # Machine Learning:
# 
# This is a regression problem, and we're going to start off using scikit-learn's Linear Regression algorithm.  This is a simple algorithm that can be regularized to avoid overfitting, however it doesn't perform very well with non-linear relationships, and it's not flexible enough to capture more complex patterns.

# First, we are going to create a model that uses all of our feature variables (Brand, color, dateAdded, dateUpdated, merchant, size) against our target variable (Price):

# In[73]:


NoNull_df.head()


# In[78]:


from sklearn.model_selection import cross_val_score

LABELS = ['Brand', 'Colors', 'Merchant', 'Size']

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
NoNull_df[LABELS] = NoNull_df[LABELS].apply(categorize_label, axis=0)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(NoNull_df[LABELS])

NoNull_df[['dateAdded', 'dateUpdated']] = NoNull_df[['dateAdded', 'dateUpdated']].astype(int)
merged = pd.concat([NoNull_df[['dateAdded', 'dateUpdated']], label_dummies], axis=1)

X = merged
y = NoNull_df[['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, 
                                                    random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Create the regressor: reg_all
reg = LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("All Feature Variable Model:")
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)


# RMSE of 43.61, and an R^2 of 0.059.  Not very good if you ask me.

# Now, we are going to create 6 Simple Linear Regression Models where we use only one feature variable (Brand, color, dateAdded, dateUpdated, merchant, or size) for each model against our target variable, Price:

# In[81]:


from sklearn.model_selection import cross_val_score

LABELS = ['Brand']

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
Brand_df[LABELS] = Brand_df[LABELS].apply(categorize_label, axis=0)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(Brand_df[LABELS])

X = label_dummies
y = Brand_df[['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, 
                                                    random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Create the regressor: reg_all
reg = LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("Brand Feature Variable Model:")
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)


# That didn't do very well, let's have a look at Colors:

# In[82]:


from sklearn.model_selection import cross_val_score

LABELS = ['Colors']

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
Colors_df[LABELS] = Colors_df[LABELS].apply(categorize_label, axis=0)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(Colors_df[LABELS])

X = label_dummies
y = Colors_df[['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, 
                                                    random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Create the regressor: reg_all
reg = LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("Color Feature Variable Model:")
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)


# That didn't do to well either.  Let's have a look at Merchant:

# In[86]:


from sklearn.model_selection import cross_val_score

LABELS = ['Merchant']

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
Merchant_df[LABELS] = Merchant_df[LABELS].apply(categorize_label, axis=0)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(Merchant_df[LABELS])

X = label_dummies
y = Merchant_df[['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, 
                                                    random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Create the regressor: reg_all
reg = LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("Merchant Feature Variable Model:")
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)


# Not having too much luck, let's try Size:

# In[85]:


from sklearn.model_selection import cross_val_score

LABELS = ['Size']

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
Size_df[LABELS] = Size_df[LABELS].apply(categorize_label, axis=0)

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(Size_df[LABELS])

X = label_dummies
y = Size_df[['Price']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, 
                                                    random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Create the regressor: reg_all
reg = LinearRegression()

# Fit the regressor to the training data
reg.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg.predict(X_test)

# Compute and print R^2 and RMSE
print("Size Feature Variable Model:")
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)


# Size has worked the best so far, surprisingly enough.  Let's try our numericals dateAdded and dateUpdated:

# In[89]:


df.info()


# In[93]:


dfml = df

dfml[['dateAdded', 'dateUpdated']] = df[['dateAdded', 'dateUpdated']].astype(int)

y = dfml.Price
X = dfml.dateAdded

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))


# In[95]:


# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

# Fit the model to the data
reg.fit(X, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print('R^2 Value:')
print(reg.score(X, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.xlabel("Economy")
plt.ylabel("Happiness Score")
plt.ylim([0,10])
plt.show()


# In[97]:


# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))


# In[98]:


# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg_all, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


# R^2 value of 0.379, and an RMSE of 147.63 for using dateAdded as a feature variable.  Let's see how dateUpdated does:

# In[100]:


X = dfml.dateUpdated
X = X.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("dateUpdated Model:")
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg_all, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


# In[101]:


dfdwpno.head()


# In[103]:


dfdwpno[['Time_Until_Update']] = dfdwpno[['Time_Until_Update']].astype(int)

X = dfdwpno.Time_Until_Update
X = X.reshape(-1,1)

y = dfdwpno.Price
y = y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("Time Until Update Model:")
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg_all, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))


# So far, the best models we had so far is where we used Multiple Linear Regression with all of the feature variables, and the Simple Linear Regression model with Time Until Update as the feature variable, since they both had by far the lowest RMSE compared to the other models.

# # Conclusion:

# The machine learning models I ran didn't turn out so well, so it is difficult for me to gain much insight on those.  Based on the EDA, it is pretty clear that Brand and Merchant have a disting impact on the price, based on the Averages DataFrame for each.  Other factors such as Color and Size had such a wide variety of saying basically the same thing (i.e. fancier/different ways of saying "black", or different variations of saying "size 8") that I think that these differences are likely correlated with the way the Brand and/or Merchant named them.   
# 
# As for the time between dateAdded and dateUpdated, it is clear that generally the more time that passes, the lower the price will be.

# In[ ]:




