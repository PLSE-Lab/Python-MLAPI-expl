#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor


# # About the data
# The data contains physical attributes of around 55000 diamonds and their price. We will train ML algorithm to predict the price. There is no description of features, but most of them are self-explanatory. Table and depth could be googled, but I will not use any deep domain knowledge here. 

# # Assumptions about the model
# In my opinion, this is classic linear regression problem. I see in Kaggle many people throw more sophisticated tools at every problem, but I think here it is not needed, as regression does a good job. At least from the reading, more sophisticated algorithms are needed when concrete issues in the model cannot be captured well by simple methods, but I didnt identify such issues here.
# 
# Let us discuss whether linear regression assumptions are met here. The most delicate assumption is about normal distributioin of additive error of the target variable. (linearity and homoscedacity will be seen from visualizations). Note that, unlike what many people do in their kernels, target variable or dependent variables do not have to be normally distributed!
# 
# Let us think how is the noice in the price obtained. The seller of diamonds has some price in her mind, and depending on client's negotiating skills she may add or reduce up to 10-20% of the price. This is our noise, which we can assume is normally distributed(supposedly, distribution of good and bad negotiators is normal). However, this error is not additive, but multiplicative! By adding 20% to the price, we actually multiply the price by 1.2. In other words, for more expensive diamonds the error will usually be bigger, so linear regression will be more sensitive to noise in the high price range. Therefore, in order to apply linear regression,  we must take log of the price (which will transform multiplicative error to additice). We will see other indicators why that will be a good choice. 

# # How to measure succesfull prediction?
# 
# To continue the discussion above, we should not count absolute values of errors as the mistake. If someone tells you that their algorithm determines the price up to $\pm 1000$ dollars, that is not of good use if you plan to buy a cheap diamond. It would be more meaningful, if one could determine the price up to 10% error. Hence, we will try to minimize the error in terms of % of price. Luckily, this is the same as minimizing $R^2$ for log_price prediction.
# 
# Let us start now.

# In[ ]:


data = pd.read_csv('../input/diamonds/diamonds.csv')
np.random.seed(42)
data.head()


# In[ ]:


data.info()


# No missing values. Great!

# # Split data for data analysis/model training and final test
# We split our data into training set and test set (90-10). We will not touch the test set till the final test. 

# In[ ]:


df_train,  final_data, df_test, final_result = train_test_split(data.drop('price', axis=1), data.price, test_size=0.1)


# 0 column is clearly not relevant.

# In[ ]:


df = df_train.join(df_test)
df = df.drop('Unnamed: 0', axis=1)


# # Exploration and visualization
# Let us check linear pair correlations of numerical values. 

# In[ ]:


df['log_price'] = np.log1p(df.price)
print(round(df.corr(),2))
sns.heatmap(df.corr())


# We see very high correlations among all values, except for table and depth that seem to predict very little. Notice that log_price is also better linearly correlated with most of the variables than price. Let us draw pair correlations

# In[ ]:


sns.pairplot(df.head(5000))


# # Observations from the graphs:
# 
# Outliers:  x<1, y<1, y>20, z<1, z>15, depth<50, depth>77, table< 50, table > 74. Carat>3.5  very few values, also remove.
# 
# x,y, z are clearly linearly corelated. Carat seems polynomial in x,y,z. Price seems exponentially dependent on carat and x,y,z - sizes (but log_price does seems linear, which is good for the use of linear regression in x,y,z). Also, note that carat is not linear with x,y,z - rather behaves like square root. We should consider adding square root of carat to our model as a feature.
# 
# Table and depth do not seem like good predictors, while dimensions and carat are. 
# 
# ## Outliers
# We will remove outliers, since linear regression is very sensitive to them and we have enough data without them.

# In[ ]:


df.drop(df[df.carat>3.5].index, inplace = True)
df.drop(df[df.x<1].index, inplace = True)
df.drop(df[df.y<1].index, inplace = True)
df.drop(df[df.y>20].index, inplace = True)
df.drop(df[df.z<1].index, inplace = True)
df.drop(df[df.z>15].index, inplace = True)
df.drop(df[df.depth<50].index, inplace = True)
df.drop(df[df.depth>77].index, inplace = True)
df.drop(df[df.table<50].index, inplace = True)
df.drop(df[df.table>74].index, inplace = True)
df.shape


# In total we removed 40 samples, so we should not be overly concerned. 

# # Check categorical data.

# In[ ]:


print(df[ 'cut'].describe())
print(df['cut'].unique())
print(df[ 'color'].describe())
print(df['color'].unique())
print(df[ 'clarity'].describe())
print(df['clarity'].unique())


# Start with cut type. It is categorical value, but it might be ordinal, since ideal sounds better than fair. Let us see.

# In[ ]:


fig, axs = plt.subplots(ncols=3, figsize=(18,4))
sns.boxplot(y='price', x = 'cut', data = df, ax = axs[0]).set_title('Price distribution by cut type')
sns.countplot( x = 'cut',   data = df,  ax = axs[1]).set_title('Frequencies of different cut types')
sns.boxplot( y = 'x', x = 'cut',   data = df,  ax = axs[2]).set_title('Size distribution of different cut types')


# From the above, it is not obvous that better cut type results in higher price, the average price and price distributions do not seem to differ. In fact, ideal seems to be on average cheaper than everything, that is  because ideal cuts are most frequent and on average are smaller (as we see in the third graph). In general there are many diamonds of different sizes of each cut type, and the size has bigger influence on the price. Since x,y,z are correlated, let us see how the graph price-x depends on the cut. 

# In[ ]:


plt.figure(figsize=(17,7))
sns.scatterplot('x','log_price', hue= 'cut', data = df[0:10000]).set_title('Price vs x_dimension, by cut')


# Now we can see more clearly, that diamonds of the same dimension with ideal cut usually cost more than others. There is clear distinction between ideal, 
# and fair, while good, very good and premium seem to be mixed. We could bin good, very good and premium together, since there is no clear separation, but we can also leave them in the order. The order that looks right to me is: Ideal > Premium > Very Good  > Good > Fair. I have to admit that the separation among some cuts in the picture is not very clear, so I went with the common sense. (I tried to exchange the order of Premium and Very Good, which resulted in much worse predictions)

# Also, this graph supports the assumption that log_price homoscedatic (we see that at each x-size range noise in price looks roughly the same).

# In[ ]:


fig, axs = plt.subplots(ncols=2, figsize=(15,4))
sns.scatterplot('table','x', hue= 'cut', data = df, ax = axs[0]).set_title('Price vs table by cut')
sns.scatterplot('depth','x', hue= 'cut', data = df, ax = axs[1]).set_title('Dimension vs table by cut')


# We see that cut quality is does not really depend on the dimensions, but depends on combination of table and depth(in the next graph we see clear separation of circles).
# So we see that while table and depth do not seem to influence the price directly, they do so through affecting the cut type. The graph below shows how cut is determined by table and depth

# In[ ]:


plt.figure(figsize=(10,10))
sns.scatterplot('table','depth', hue= 'cut', data = df).set_title('Depth vs table by cut')
plt.xlim(50,70)
plt.ylim(55,70)


# We see circles of different colors. This is clear separation  between cuts and their order (better = closer to table,depth = (55,62)). Moreover, the separation is not linear. We could probably remove table and depth (since we already have the cut quality). On the other hand, if we use nonlinear separation methods, we can extract from table/depth continuos values to cut quality, and not discrete, as we have now.

# Check the color now:

# In[ ]:


fig, axs = plt.subplots(ncols=4, figsize=(20,4))
sns.boxplot(y='price', x = 'color', data = df, ax = axs[0]).set_title('Price distribution by color')
sns.countplot( x = 'color',   data = df,  ax = axs[1]).set_title('Frequencies of different colors')
sns.boxplot( y = 'x', x = 'color',   data = df,  ax = axs[2]).set_title('Size distribution of different colors')
sns.countplot(x='color', hue = 'cut', data = df, ax = axs[3]).set_title('Distribution of cuts for different colors')


# As in the case with cut type, color does not seem to be ordinal ,this is because the size is the main factor in the price, and different colors appear in different sizes. So we cannot see which color is more valuable just from distributions of prices. We also see that cut types are similarly distributed among all colors and each color is distributed along very wide range of sizes.
# 
# Let us draw size-price graph.

# In[ ]:


plt.figure(figsize=(17,7))
sns.scatterplot('x','log_price', hue= 'color', data = df).set_title('Price vs x-size graph, by color')


# Now clearly we see that for a diamond of the same size, certain colors imply higher prices. Moreover, we can notice that the order that we find from our graph is alphabetical (D - being the most expensive and J - the cheapest), which explains why the colors are labeled in this way by letters. (we could also easily discover this online).
# 
# The same story goes with clarity attribute, we cannot learn much from clarity distributions, but from looking how price is influenced by clarity given the dimension, we can easily observe the order in which clarity labels demand higher prices. (again, we could learn this online, but hey, this is visualization part!)

# In[ ]:


plt.figure(figsize=(17,7))
sns.scatterplot('x','log_price', hue= 'clarity', data = df).set_title('Price vs x-size graph, by clarity')


# Also here, we certain alphabetical order, which validates that we identified the quality-clarity relationship correctly. 
# 
# Another visualization we could to is to fix x-dimension, and see how price depends on categorical values. Clarity is the value that is more obviously separated:

# In[ ]:


dfx = df[(df.x<7.3) & (df.x>7.2)]
fig, axs = plt.subplots(ncols=3, figsize=(17,5))
sns.boxplot(y='price', x='clarity', data =  dfx, ax = axs[0]).set_title('For 7.2<x<7.3')
sns.boxplot(y='price', x='color', data =  dfx, ax = axs[1]).set_title('For 7.2<x<7.3')
sns.boxplot(y='price', x='cut', data =  dfx, ax = axs[2]).set_title('For 7.2<x<7.3')


# Now we can encode the categorical variables in the order that we discovered from analyzing the graphs

# In[ ]:


cut_dict = {'Good':1, 'Premium':3, 'Very Good':2, 'Ideal':4, 'Fair':0}
df.cut = df.cut.map(cut_dict)
clarity_dict = {'SI1':2, 'VS1':4, 'VVS1':6, 'SI2':1, 'VS2':3, 'IF':7, 'VVS2':5, 'I1':0}
df.clarity = df.clarity.map(clarity_dict)
color_dict = {'E':5, 'H':2, 'D':6, 'F':4, 'I':1, 'G':3, 'J':0}
df.color = df.color.map(color_dict)


# # How good can our prediction be?
# Let us focus on most common categorical features: color = G, cut = Ideal, clarity = SI1. Let us look at some small x-size interval. Since x,y,z are strongly linearly correlated, and table and depth do not seem to predict much, all the samples that we single out with those features should have roughly the same price. Let us graph all such (around 100) samples from our whole data.

# In[ ]:


most_freq = data[(data['color']=='G') & (data['cut']=='Ideal') & (data['clarity']=='SI1') 
                 & (data['x']>5.67) & (data['x']<6)]
sns.scatterplot('x', 'price', data = most_freq)
plt.title('Prices of diamonds with most common features by size')


# We see, that the price for diamond with exactly the same features (ignoring table and depth) fluctuates between 2300 and 3100. With average price around 2700, this is an error of 15% of the price. So we are not expecting any model to have less than 15% error. Such graphs will be a good way to check whether our assumption about normality of the noise holds, here the sample is maybe a bit small, but it does seem that more customers are getting lower than average price than the customers getting higher than average price. 

# # First model

# For educational reasons, I will start with linear regression predicting price (and not log_price). In my opinion, this is **INCORRECT MODEL**, as I discussed in the beginning (because we cannot assume that the noise is normally distributed). What we will see is that the regression score is very high, nevertheless, residual error graph does not look good, implying that this prediction is bad. This is a lesson that high regression score does not imply that we are giving good answer to the problem.
# 
# We will use polynomial features, because, as we saw from exploratory analysis, there is curvature in many pairwise plots. This will add many dimensions, so we wish to avoid using unneccesary data, like table and depth columns. Also, we will apply PCA to reduce dimensions. (I tried to include table and depth, it slows everything down and does not improve the results, because most likely they get dropped by PCA).

# In[ ]:


X = df.drop(['price', 'table', 'depth', 'log_price'], axis = 1)
y = df.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# We use scaler, since we will use polynomials that are not scale invariant, so better be safe then sorry. We add PCA analysis, to reduce the number of features. I also tried Ridge regression to deal with co-linearity of x,y,z (that is another reason for presence of the scaling), but in fact it makes worse prediction.

# In[ ]:


pipe_lr = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree = 4), PCA(n_components=80) , LinearRegression())
pipe_lr.fit(X_train, y_train)

print('Training score:', pipe_lr.score(X_train,y_train))
print('Test score: ',pipe_lr.score(X_test, y_test))
plt.title('Residual graph')
plt.scatter(y_test.values, 100*(y_test.values - pipe_lr.predict(X_test))/y_test.values)
plt.ylim(-100,100)
x = [50 * i for i in range(380) ]
plt.plot(x,[0]*380, color = 'red')
plt.ylabel('Error in % of the price')
plt.xlabel('Real price')    


# Regression score here is quite high, but it is irrelevant for our scoring system. We do not care about absolute value errors. Looking at the residual graph, we see that relative errors are huge for lower price range, as big as 100%, so this model is useless for people buying a diamond for less than 2000 dollars. This is because we are not working with the right scale. 

# ## Correct linear regression
# Let us do the same predicting log_price. But first, we will write a function to draw the residual graph

# In[ ]:


def evaluate_graph(estimator, X_test, y_test):
    """sketches a graph to evaluate residual errors and prints regression score"""
    
    print('Test score: ',estimator.score(X_test, y_test))
    plt.title('Residual graph')
    plt.scatter(np.expm1(y_test.values), 
                100*(np.expm1(y_test.values) - np.expm1(estimator.predict(X_test)))/np.expm1(y_test.values))
    plt.ylim(-100,100)
    x = [50 * i for i in range(380) ]
    plt.plot(x,[0]*380, color = 'red')
    plt.ylabel('Error in % of the price')
    plt.xlabel('Real price')    


# # Linear regression for predicting log_price

# In[ ]:


X = df.drop(['price', 'table', 'depth',  'log_price'], axis = 1)
y = df.log_price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr=LinearRegression()
lr.fit(X_train, y_train)

print('Training score:', lr.score(X_train, y_train))
evaluate_graph(lr, X_test, y_test)


# The regression score is roughly the same, but the residual graph here looks much better. The error is more uniform along the price range.
# 
# There is some bias, the algorithm tends to more often overprice expensive diamonds, and underprice cheaper ones. Also, the discounts seem to be bigger than the overpricing. 
# 
# We should definitely stick with logarithmic model. 

# # Feature engineering
# Experiments show that adding table and depth to linear model does not help. The problem is that table and depth do not have linear relation with the price (as could be seen on the graph table-depth by cut type). However, as we saw in visualization, table-depth combination determines the cut.
# 
# ## Continuos cut feature
# Below, we would try to make use of table and depth data and produce a continuous cut variable (which might give more accurate information on pricing than the cut). We will train KNN regression algorithm.

# In[ ]:


feat_data = df[['table', 'depth', 'cut' ]]
feat_data.head()


# In[ ]:


Xf = feat_data[['table', 'depth']]
yf = feat_data['cut']
Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.33)

#svm = LinearSVR(epsilon=0.0, tol=0.0001, C=10, loss='epsilon_insensitive', 
#                intercept_scaling=1.0, dual=True, verbose=0, max_iter=1000)
clf = KNeighborsRegressor(n_neighbors=150, algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None)
clf.fit(Xf_train, yf_train)
clf.score(Xf_test, yf_test)


# Do not be pessimistic about the low score, it does not measure our success. To test if our algorithm works, we will plot decision regions and see if it resembles the original picture.

# In[ ]:


def plot_decision_regions(classifier, resolution = 1):
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:5])
    
    x1_min = 50
    x2_min =50
    x1_max = 70
    x2_max = 70
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.5, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.xlim(xx2.min(), xx2.max())
    plt.xlabel('table')
    plt.ylabel('depth')
    plt.title('Decision regions for cut type')

    #Credit to Sebastian Raschka, Vahid Mirjalili - Python Machine Learning. 2nd ed-Packt Publishing (2017)


# In[ ]:


plot_decision_regions(clf)


# We see that decision regions resemble a lot the cut type (except for he lower left corner, where there are no samples anyway). 

# So we engineered new feature: continuous cut-type. Let us add it to ur data and see how it correlates with discrete cut type

# In[ ]:


df['new_cut'] = clf.predict(df[['table', 'depth']])


# In[ ]:


df.head()


# Look how close cut and new_cut features are for first 5 samples. If we round new_cut, it will most likely coincide with cut most of the times.

# In[ ]:


df[['cut', 'new_cut', 'log_price']].corr()


# Note, discrete and continous cut type are highly correlated, but not fully. Also, new_cut seems to be more corellated with log_price than cut. This is good news for us, as this feature can improve our model. However, if we restrict our sample to diamonds with most frequent features, the difference is less visible.

# In[ ]:


dfpr = df[(df.x>5.67) & (df.x<6) & (df['color']==3) & (df['clarity']==2) ]
print('Correlations for diamonds with same most common features:')
dfpr[['cut', 'new_cut', 'log_price']].corr()


# In[ ]:


print('Mean continuos cut by cut type:')
df.new_cut.groupby(df.cut).mean()


# We notice, that as we struggled to distinguis Premium and Very Good cuts from the graphs, it is indeed a delicate task, as seen here from our model. This is one case where continuous cut can be useful. It is also safe to assume that such diamonds can be often missclassified due to human error.

# In[ ]:


sns.boxplot(x= 'cut', y = 'new_cut',  data = df)


# This new feature makes sense. It is correlated to the cut, but also continuous, and captures some features of table and depth. For example, this feature could potentially fix human error mistakes of incorectly classifying the cut. Let us run regression with our new feature.

# In[ ]:


X = df.drop(['price', 'table', 'depth', 'log_price'], axis = 1)
y = df.log_price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr= LinearRegression()
lr.fit(X_train, y_train)

print('Training score:', lr.score(X_train, y_train))
evaluate_graph(lr, X_test, y_test)


# * We see that the score is roughly the same. We can keep our new feature.

# ## Other fetaures
# Two more features we can add is area and volume. Intuitively, this features are important (area - how visually big the diamond looks like, volume - the amount of material) and are not be captured by x,y,z size features, since we are not including polynomial features in our model, We can just add these two manually. Additionally, we noticed that square root of carat might be better suited for linear regression then carat, so we will add this feature as well.

# In[ ]:


df['volume'] = df.x*df.y*df.z
df['area'] = df.x*df.y
df['root_carat'] = np.sqrt(df.carat)
X = df.drop(['price', 'table', 'depth', 'log_price'], axis = 1)
y = df.log_price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lr= LinearRegression()
lr.fit(X_train, y_train)

print('Training score:', lr.score(X_train, y_train))
evaluate_graph(lr, X_test, y_test)


# We see that the score increased even further, so new features are useful.

# # Model evaluation
# We already looked at residual graph, which looks satisfactory. Let us check the distribution of residuals

# In[ ]:


residual_error = 100*(np.expm1(y_test.values) - np.expm1(lr.predict(X_test)))/np.expm1(y_test.values)
plt.figure(figsize=(7,6))
plt.title('Residual error distribution')
plt.xlim(-100,100)
plt.xlabel('Residual error in %')
plt.ylabel('Density')
sns.distplot(residual_error, bins = 100, kde = True, fit=norm)
plt.show()


# The black curve is the normal distribution, seems pretty close. Ideally, we should check that errors are normally distributed at every price range. 

# In[ ]:


print(residual_error.mean(), residual_error.std())


# The mean of the error is -1%, which means that our algorithm overprices the diamonds by 1% on average. Standard deviation is 15%.

# Another test is to check whether residual error correlates with any data. Let us add residual error to test data, and return cut, table and depth as well.

# In[ ]:


X_test.loc[:,'residual_error'] = residual_error
X_test.join(df[[ 'table', 'depth', 'price', 'log_price']]).corr()['residual_error'].sort_values()


# We see that residual error has very low correlations with our data. As we see, the error does grow with the price, which is not a good sign, but the corellation coefficient is rather small.

# All the above point that linear regression was a good model for this problem.

# # Conclusions
# 

# In conclusion, the new_cut feature I engineered does not seem to add much to the model, but I am really happy with the visualization of it, so I will keep it. Also, our prediction has standard deviation of seems to miss the real price by at most $\pm 25\%$, and we saw from simple visualization that we certainly can not expect prediction with less than $15 \%$ error, so I feel like the result is not too bad. Moreover, the sample we took of diamonds with most common features, we saw that price distribution is slightly skewed and does not completely look like normal distribution, so it is not unreasonable that our residual error is not exactly normally distributed. Overall, residual error graph looks good to me. We also saw that training and test scores were quite similar, so there is not much evidence for over/underfitting.

# # Final test
# Prepare the test data

# In[ ]:


final_data.cut = final_data.cut.map(cut_dict)
final_data.clarity = final_data.clarity.map(clarity_dict)
final_data.color = final_data.color.map(color_dict)
final_data['new_cut']  = clf.predict(final_data[['table', 'depth']])
final_data['volume'] = final_data.x*final_data.y*final_data.z
final_data['area'] = final_data.x*final_data.y
final_data['root_carat'] = np.sqrt(final_data.carat)
final_data = final_data.drop(['Unnamed: 0', 'table', 'depth'], axis=1)


# In[ ]:


linreg = LinearRegression()
linreg.fit(X,y)


# In[ ]:


plt.title('Residual graph')
plt.scatter(final_result.values, 100*(final_result.values - np.expm1(lr.predict(final_data)))/final_result.values)
plt.ylim(-100,100)
plt.ylabel('Error in % of the price')
plt.xlabel('Real price')
x = [50 * i for i in range(380) ]
plt.plot(x,[0]*380, color = 'red')
print('Regression score:', lr.score(final_data, np.log1p(final_result.values)))


# ****I like that regression score is consistent on data that was previously unseen, so the model is likely to generalize well.

# ## P.S.
# I am new in ML and this is my first kernel. Feel free to comment.
