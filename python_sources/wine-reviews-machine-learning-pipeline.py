#!/usr/bin/env python
# coding: utf-8

# ### Dataset: 150,000 wine reviews scrapped from Wine Enthusiast Magazine (winemag.com)
# 
# ### I took interest in this dataset as I have a WSET Level 3 blind tasting exam coming up next month, I'm curious to know if machine can perform deductive wine tasting as a trained human can do
# 
# ###  Goal: 
# ## (1) Use wine review description to predict grape variety 
# ### Wine reviews usually discuss level of tannin, acidity, aromas, color, intensity, which are important indicators of grape variety in a blind tasting situation
# 
# ## (2) Explore relationship between wine price and and points/wine region/ grape variety, 
# ### Unfortunately, the dataset do not show vintage, which is an important pricing factor

# In[ ]:


# Load CSV 
import csv
import numpy
import pandas
get_ipython().run_line_magic('pylab', 'inline')
filename = '../input/winemag-data_first150k.csv'
winedata = pandas.read_csv(filename)
print(winedata.shape)


# In[ ]:


winedata.head()


# ### Clean up the data - Remove duplicaet and Null value, left with 89K data entries

# In[ ]:


# Find out the duplicaet value
print("Total number of examples: ", winedata.shape[0])
print("Number of examples with the same description: ", winedata[winedata.duplicated(['description'])].shape[0])


# In[ ]:


Cleanwine = winedata.drop_duplicates('description', keep='first', inplace=False)
Cleanwine.dropna(subset=['description','points','price'], inplace=True)
Cleanwine.shape


# In[ ]:


## Finding missing values
total = Cleanwine.isnull().sum().sort_values(ascending = False)
total


# ### Now we made sure that all entreis that missing variety, price, points, and descriptions (review) are deleted from the dataset
# ### Next step: Scanning the data for any bais or skewness

# In[ ]:


CountryList = Cleanwine['points'].groupby(winedata['country']).mean()
CountryList = CountryList.to_frame().reset_index()
CountryList.columns = ['country', 'AvgPoints']
CountryList.sort_values(by='AvgPoints', ascending=False).head(10)
# Calculate average points for wines from each country


# ### The list of countries that have highest rated wines looks odd, as England, India, Morocco, and Slovenia are not even considered major wine regions
# ### -> Look into number of wines represented in this dataset for each country
# 
# ### Scanning the data - (1) representitiveness of wine region

# In[ ]:


count = Cleanwine.country.value_counts()
count = count.to_frame('count').reset_index()
count.columns = ['country', 'count']
CountryList = CountryList.merge(count, on="country", how='left')
CountryList.sort_values(by='count',ascending=True).head(10)


# In[ ]:


CountryList.sort_values(by='count',ascending=False).head(10)


# In[ ]:


Cleanwine['country'].value_counts()[:10].plot(kind='bar',figsize=(12,8));
plt.xticks(rotation=45)
plt.xlabel('country')
plt.ylabel('Number of country count')
plt.show()


# ### Some countries only have 1-2 wines represented in this dataset, and top country, US, has over 40k wines represented....
# ### Therefore, executive decisions are made to remove all countries with less than 500 observations

# In[ ]:


Cleanwine2 = Cleanwine.merge(CountryList, on="country", how='left')
Major = Cleanwine2[Cleanwine2['count'] >= 500]
Major.shape


# ### 13 countries representing the major wine regions are left

# In[ ]:


MajorCountry = Major['points'].groupby(Major['country']).mean()
MajorCountry= MajorCountry.to_frame().reset_index()
MajorCountry.columns = ['country', 'AvgPoints']
MajorCountry.sort_values(by='AvgPoints', ascending=False)


# ### Scanning the data - (1) representitiveness of grape variety

# In[ ]:


count2 = Major.variety.value_counts()
count2 = count2.to_frame('count').reset_index()
count2.columns = ['variety', 'countgrape']
count2.shape


# ### There are over 10,000 kinds of grape varieties in the world 
# ### 567 kinds of grape varieties are represented in[](http://) this dataset

# In[ ]:


count2.sort_values(by='countgrape', ascending=False).head(10)


# In[ ]:


count2.sort_values(by='countgrape', ascending=True).head(10)


# ### Executive decision are made to remove all grape varieties with less than 500 observations

# In[ ]:


Cleanwine3 = Major.merge(count2, on="variety", how='left')
Major2 = Cleanwine3[Cleanwine3['countgrape'] >= 500]
Major2. head(10)


# ### we are left with 31 grape varieties representing the most commonly used grape varieties in wine production

# In[ ]:


Major2.variety.nunique()


# In[ ]:


Major2.variety.value_counts()


# ### Now we are happy with the dataset representing only wines from major wine regions and commonly seen grape varieties.
# ### As shown in below chart, most wines are rated between the range of 84-92, "wine points system" has been a very controversial topic in the wine industry, as you can see here there are not much differentiation here as you would expect.

# In[ ]:


import seaborn as sns
sns.countplot(x='points',data = Major2, palette='hls' )
plt.show()


# In[ ]:


CountryList2 = Major2['price'].groupby(Major2['country']).mean()
CountryList2 = CountryList2.to_frame().reset_index()
CountryList2.columns = ['country', 'price']
CountryList2.sort_values(by='price', ascending=False).head(10)


# ### Above list shows the countries making the most expensive wines on average.... It's not surprising to see France come to the top, but I would expect to see Italy and Spain rank lower than where they are here. 

# In[ ]:


plt.figure(figsize=(20,25))
plt.subplot(2,1,1)
g = sns.boxplot(x='country', y='price',data=Major2)
g.set_title("Which country has the most expensive wines", fontsize=25)
g.set_xlabel("Country", fontsize=20)
g.set_ylabel("Price ($)", fontsize=20)
g.set_xticklabels(g.get_xticklabels(),rotation=45)
                
plt.show()


# In[ ]:


plt.figure(figsize=(20,5))
plt.title("Distribution of price")
ax = sns.distplot(Major2["price"])


# In[ ]:


# if we want to see better price distribution we have to scale our price or drop the tail.  

plt.figure(figsize=(20,5))
plt.title("Distribution of price")
ax = sns.distplot(Major2[Major2["price"]<200]['price'])

percent=Major2[Major2['price']<200].shape[0]/Major2.shape[0]*100
print("There are :", percent, "% wines less than 200 USD")


# In[ ]:


percent=Major2[Major2['price']>200].shape[0]/Major2.shape[0]*100
print("There are :", percent, "% wines more expensive than 200 USD")


# In[ ]:


print("Number of wines costs more than 200USD:", Major2[Major2['price']>200].shape[1])


# In[ ]:


plt.figure(figsize=(20,16))
plt.subplot(2,1,2)
g1 = sns.boxplot(x='country', y='points',data=Major2)
g1.set_title("Which country has the highest rated wines", fontsize=25)
g1.set_xlabel("Country's ", fontsize=20)
g1.set_ylabel("Points", fontsize=20)
g1.set_xticklabels(g1.get_xticklabels(),rotation=45)
plt.subplots_adjust(hspace = 0.6,top = 0.9)
plt.show()


# ## Goal (1): Use wine review description to predict grape variety
# 
# ### often wine review mentions grape variety, therefore first step is to classify grape variety as stop words
# ### -> Eliminate grape variety from  wine review description, so that description can serve as blind tasting notes, which is the ground for making judgement on grape variety

# In[ ]:


variety = Major2.variety.unique().tolist()
variety[:5]
# Extract a list of grape varieties


# ### Even with the variety included with stopwords, I still get unconvincingly high scores for Logistic Regression and Decision Tree Regression...
# ### After some investigation I realized that varieties are mentioned in wine reviews often in lower case, so need to include lower case varietites as stop words, break down the hyphenated words...

# In[ ]:


lcvariety = [x.lower() for x in variety]
lcvariety[:5] #convert the grape varieties to lowercase


# In[ ]:


bkvariety = [i.replace('-', ' ').replace(',', ' ').split(' ') for i in lcvariety] #generated a list of list
bkvariety = [item for sublist in bkvariety for item in sublist] # convert list of list to a python list
bkvariety[:8] #break down the grape variety names into separate words


# ### But I don't think it's fair to consider 'red' or 'white' as stop word, as in "blind tasting" you do see the color of the wine, color descriptions should not be included in the stopwords
# ### Color descriptions like ruby, garnet, gold, lemon-green and color intensity descriptions are all important clues in deductive blind tasting

# In[ ]:


bkvariety.remove('red')
bkvariety.remove('white')


# In[ ]:


import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.update(variety,lcvariety,bkvariety) #update stop words


# In[ ]:


print(stop)


# ## Training a Logistic Regression Model
# ### Spliting the data: 80% for traning and 20% for testing

# In[ ]:


import sklearn
sklearn.__version__


# In[ ]:


from sklearn.model_selection import train_test_split
X = Major2.description
y = Major2.variety

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# ### 58K data entries for training, 15K data entries for testing

# In[ ]:


Major2.info()


# In[ ]:


#Heuristic data exploration
print(type(X))
print(type(y))
print(len(X))
print(len(y))


# In[ ]:


set(y) #grape varieties as the target


# In[ ]:


#take a look at some sample data
print(X[17])
print(y[17])


# ## Bag of Words - Vectorization

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer(stop_words = stop, ngram_range=(1, 3), max_features=3000) 
#vec = CountVectorizer(stop_words = stop, ngram_range=(1, 3), min_df=3, max_features=5000) 
X_train_vec = vec.fit_transform(X_train)
X_train_vec


# ## Without TF-IDF (Term Frequency-Inverse Document Frequency)

# In[ ]:


from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X_train_vec, y_train)


# ### Test on the test set

# In[ ]:


X_test_vec= vec.transform(X_test)
y_pred = logit.predict(X_test_vec)
list(zip(y_pred, y_test))[:10]


# In[ ]:


from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)


# In[ ]:


print(metrics.precision_score(y_test, y_pred, average='weighted'))
print(metrics.recall_score(y_test, y_pred, average='weighted'))
print(metrics.f1_score(y_test, y_pred, average='weighted'))


# In[ ]:


print(metrics.classification_report(y_test, y_pred))


# ### The score is not particularly high, but it's reasonable. 
# ### Even well trained sommelier and wine masters would not be able to achieve 100% accuracy, so I'm very happy with the result so far

# ## Also wanted to test on some of external wine reviews that are not from this dataset... 
# ## so I wrote this fluffy wine review on Cloudy Bay Sauvignon Blanc 2017:
# 
# ### "This wine is clear pale lemon-green. The nose is crisp clean, full of lovely youthful aromas of grapfruit peel, gooseberry, lime zest, and apricot. The signature herbaceous flavor is reminiscent of asparagus, green pepper, and a hint of crushed white stones. The wine offers high acidity, balanced with pronounced fruitiness and decent length. Can drink now, but has potential for aging to further develop complexity and tertiary flavors. It pairs well with seafood - shows off the umami in oysters or cuts off the fattiness in salmon. It would also be a great choice to pair with Asian hotpot and Sichuan cuisine with its refreshing acidity."

# In[ ]:


Myreview =['This wine is clear pale lemon-green. The nose is crisp clean, full of lovely youthful aromas of grapfruit peel, lemon zest, and apricot. The signature herbaceous flavor is reminiscent of asparagus, green pepper, and a hint of crushed wet stones. The wine offers high acidity, balanced with pronounced fruitiness and decent length. Can drink now, but has potential for aging to further develop complexity and tertiary flavors. It pairs well with seafood - show off the umami in oysters or cut off the fattiness in salmon. It would also be a great choice to pair with Asian hotpot and sichuan cuisine with its refreshing acidity.']
X_test_vec= vec.transform(Myreview)
y_pred = logit.predict(X_test_vec)


# In[ ]:


print(y_pred)


# ## Excited to see the model predicted correctly on my wine review!

# ## With TF-IDF (Term Frequency-Inverse Document Frequency)

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer()
X_tf = tf.fit_transform(X_train_vec)
X_tf.shape


# In[ ]:


from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X_tf, y_train)


# ### Test on the test set

# In[ ]:


X_test_vec= vec.transform(X_test)
X_test_tf = tf.transform(X_test_vec)
y_pred = logit.predict(X_test_tf)


# In[ ]:


from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)


# In[ ]:


print(metrics.precision_score(y_test, y_pred, average='weighted'))
print(metrics.recall_score(y_test, y_pred, average='weighted'))
print(metrics.f1_score(y_test, y_pred, average='weighted'))


# In[ ]:


print(metrics.classification_report(y_test, y_pred))


# 
# ## Decision Tree Classifier Model
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=13)
clf.fit(X=X_train_vec,y=y_train)


# In[ ]:


clf.score(X=X_test_vec, y=y_test)


# ### In this case, Logistic Regreesion Model with TF-IDF > Logistic Regreesion Model without TF-IDF > Decision Tree Classifier Model

# ## (2) Explore relationship between wine price and and points/wine region/ grape variety
# 
# ### Relationship between points and price:
# #### Pearson's correlation coefficient between two variables is defined as the covariance of the two variables divided by the product of their standard deviations.

# In[ ]:


df = Major2[['price','points']]
df.dropna(subset=['price','points'], inplace=True)


# In[ ]:


from scipy.stats import pearsonr
print("Pearson Correlation:", pearsonr(df.price, df.points))


# ### Happy to report that wine price do show positive relationship with wine rating (implying quality), but it's not a strong relationship! 
# ### [](http://)There are other factors to be identified...

# In[ ]:


import statsmodels.api as sm
print(sm.OLS(df.points, df.price).fit().summary())


# In[ ]:


sns.lmplot(y = 'price', x='points', data=df)


# ### As shown in above chart, most great wines are not going to cost you a fortune. Even the highest priced wine in this dataset did not fetch the highest rating...
# ## *******To be continued....

# In[ ]:




