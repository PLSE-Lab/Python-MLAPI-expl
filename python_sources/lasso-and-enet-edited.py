#!/usr/bin/env python
# coding: utf-8

# # House Prices: Regression with Lasso and Elastic Net  
# ### Compiled by the EDSA Out! Liars, JHB
# 
# This challenge uses the Ames, Iowa housing data set to create a model that could predict the sale prices of houses. We decided on a simple methodology that uses Lasso and Elastic Net regression techniques to make those predictions. Along the way we tested a few different models and tried a few different attacks on feature engineering- but realised that simpler was more effective sometimes. We did, however, average our predictions from the 2 models for a more accurate prediction- which is really just the simplest form of ensembling.

# These are some of the kernels that we consulted, and learned from in the creation of this model. 
# 
# 1. [Notebook on stacked regressions](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)
# 1. [House Price Regression and Feature Engineering
# ](https://www.kaggle.com/thevachar/house-price-regression-and-feature-engineering)
# 1. [House Prices Prediction with Ensemble](https://www.kaggle.com/kyen89/house-prices-prediction-with-ensemble)

# In[1]:


#import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Lasso, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Why is the shape of the data important?
# We need the shape of the data so we can see that as we make changes, these changes are reflected in the dataframe, i.e, if I drop a column and the shape of my data is still as it was, I know that I have not correctly executed the command. The other reason is that later on we will be concatenating the datasets, and saving the shape of the train data allows us to seperate them correctly later.

# In[2]:


#Reading in training data and getting shape
train = pd.read_csv("../input/train.csv")

train_shape = train.shape[0]

train.shape


# In[3]:


#Reading in test data and getting shape
test = pd.read_csv("../input/test.csv")
test.shape


# ### Finding and eliminating outliers  
# We noticed that there were a few houses with a very large living area that sold at a much lower price- as demonstrated below, pinpointing the houses that had a living area greater than 4000 square feet and sold at less than $300000. The reason we drop outliers this early is so that they don't affect the rest of our analysis, and so that we have consistency in the length of the variables and target.

# In[4]:


#checking for outliers in the data
sns.scatterplot('GrLivArea', 'SalePrice', hue='OverallQual', data=train);
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[5]:


#filter dataset
train[(train['SalePrice'] < 300000) & (train['GrLivArea'] > 4000)]


# In[6]:


#dropping outliers and rechecking the shape of the data- saving the shape to use to split the data later
train.drop(train[(train['SalePrice'] < 300000) &                  (train['GrLivArea'] > 4000)].index,inplace=True)
train_shape = train.shape[0]
train.shape


# In[7]:


#Ensuring the outliers have been removed
sns.scatterplot('GrLivArea', 'SalePrice', hue='OverallQual', data=train)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ### Target- Sale Price
# Taking a look at our target, we see that the distribution is skewed. In order to get a more accurate model it is advisable to normalise the target, as we have done below. We used log normalisation for this problem.

# In[8]:


#checking the distribution of the of the sale prices to see if it needs to be normalised
plt.figure(figsize=(8, 5))
sns.distplot(train['SalePrice'])
plt.title('SalePrice Distribution')
plt.ylabel('Frequency')
plt.xticks(rotation=45)


# In[9]:


#Normalising the sale price because it was skewed, and assigning it to variable y
y = train.SalePrice
y = np.log1p(y)
len(y)


# In[10]:


#Plotting the results to view the transformation
plt.figure(figsize=(8, 5))
sns.distplot(y)
plt.title('Normalized SalePrice Distribution')
plt.ylabel('Frequency')
plt.xticks(rotation=45)


# ### Correlations
# We took a look at a heatmap of the top 10 most correlated values to sale price- we used these in our very first linear regression test. As you can see there are a few variables that are highly correlated to each other, but research suggests that this is not a problem when predicting the target value when using more complex models. It would be more problematic if using linear regression.

# In[11]:


#computing the correlations of all the features against the sale price
corr = pd.DataFrame(train.corr(method='pearson')['SalePrice'])
corr1 = train.corr().drop('Id')

# Top 10 Heatmap
k = 10 #number of variables for heatmap
cols = corr1.nlargest(k, 'SalePrice')['SalePrice'].index
to_plot = np.corrcoef(train[cols].values.T)
plt.figure(figsize=(10, 10))
sns.set(font_scale=1.25)
sns.heatmap(to_plot, cbar=True, annot=True, square=True, fmt='.2f',
            annot_kws={'size': 9}, yticklabels=cols.values, 
            xticklabels=cols.values, cmap='coolwarm')


# ### The smoosh
# The reason we combine our test and train sets is so that any changes we make to the features are made to both sets. We did attempt this with the data sets seperate but there was always something that was missed. We remedied this by concatenating test and train datasets so what is done to one is also done to the other
#  Doing this in this manner allows for easier manipulation of the data. 

# In[12]:


all_columns = test.columns
train = train[all_columns]

data = pd.concat([train, test])
data.shape


# ### Where is it?
# Missing data is always a problem because it creates inconsistencies in our model. A model will not run if NaNs or NULLS are present. Here we find the missing values and plot them out to see where they lie- and find the most efficient and effective way to fill them.  
# For most of the NaNs, the value is not actually missing, it is just that the property does not have that feature. In the data description document, each feature has a corresponsing code or abbreviation that essentially says "feature not present." This is predominantly NA or none, and in the numerical data we have replaced missing values with zeros unless otherwise stated.  
# For catagorical features we chose to fill with the mode value of that column, and for one feature we filled on a conditional average, which takes the average of that feature based on other features that it is highly correlated to.

# In[13]:


#count of missing values per variable

#sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))
missing = data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(ascending=False, inplace=True)
missing.plot.bar()
#missing


# In[14]:


#Filling null values with appropriate code as seen in data description doc
nones = [
    'PoolQC', 'MiscFeature', 'Alley','Fence', 
    'FireplaceQu', 'GarageType','GarageFinish',
    'GarageQual','GarageCond','BsmtQual','BsmtCond',
    'BsmtExposure','BsmtFinType1','BsmtFinType2'
]

for none in nones:
    data[none].fillna('NA',inplace=True)


# In[15]:


#This feature used 'None' and 'TA' instead of NA
data.MasVnrType.fillna('None', inplace=True)
data.KitchenQual.fillna('TA', inplace=True)


# In[16]:


#Filling nulls in numerical data with zero
naughts = [
    'GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1',
    'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath',
    'BsmtHalfBath','MasVnrArea' 
]

for naught in naughts:
    data[naught].fillna(0, inplace=True)


# In[17]:


#we've filled nulls in some of of the catagorical data using the mode value
modes = [
    'MSZoning','Exterior1st','Exterior2nd',
    'SaleType','Electrical','Functional'
]

for m in modes:
    data[m].fillna(data[m].mode()[0], inplace=True)


# ### Conditional Filling of NaNs  
# We saw that lot frontage seems to have been influenced by lot area and neighborhood so we used conditional averages to fill nulls here. The 1st plot shows the distribution of the feature before the nulls are filled. The 2nd shows that the distribution did not change, there were just more values to show. 

# In[18]:


sns.distplot(data['LotFrontage'],kde=True,bins=70,color='b')


# In[19]:


data['LotFrontage'] = data.groupby(['LotArea','Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))


# In[20]:


sns.distplot(data['LotFrontage'],kde=True,bins=70,color='b')


# In[21]:


#Check the data again to ensure there are no nulls left
data.isnull().sum().sum()


# ### Is variability important?
# It came to our attention that there was very little variability in the utilities column and as shown below, all but one row shows the same value and so it doesn't add anything to our model, therefore we dropped the column

# In[22]:


data['Utilities'].value_counts()


# In[23]:


data.drop('Utilities',axis=1, inplace=True)


# ### Just basic engineering
# We didnt do too much feature engineering, as we felt it overcomplicated an already large dataset. We did however make some changes to how certain features were classified and binned some of the ordinal data that had over 5 categories. We made new features with the total square footage, and the bathrooms as well because as individual features that were not adding as much value as they could have combined. We assumed this would factor into the improvement of our score but it did not perform as we expected and so we removed many of these items.

# In[24]:


#classifying all catagorica data 
boxs = [
    'MSSubClass', 'MoSold', 'GarageYrBlt', 'LotConfig', 'Neighborhood', 
    'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 
    'MasVnrArea', 'Foundation', 'SaleCondition', 'SaleType', 'Exterior2nd'
]
for box in boxs:
    data[box] = data[box].astype('category')


# In[25]:


#Simplifying some of the ordinal features
#    data['OverallQual_binned'] = data.OverallQual.replace({1:1, 2:1, 3:1, # bad
#                                                            4:2, 5:2, 6:2, # ok
#                                                            7:3, 8:3, 9:4, 10:4 # good, great
#                                                           })
#    data['OverallCond_binned'] = data.OverallCond.replace({1:1, 2:1, 3:1, 
#                                                            4:2, 5:2, 6:2, 
#                                                            7:3, 8:3, 9:4, 10:4
#                                                           })


# ### Creating new features
# Creatng new features from multiple features in the same group which could be summed together. Half baths couning as half did not add value to the model, yet they did have an impact on the price, so we made them count as a full bathroom by multipying by 0.5. This, however did not have a positive impact on our model and actually caused it to perform poorly, so they were removed.

# In[26]:


#data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
#data.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

#data['TotBathrooms'] = data.FullBath + (data.HalfBath*0.5) + data.BsmtFullBath + (data.BsmtHalfBath*0.5)

#basements = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF']
#data['TotalBsmt'] = data[basements].sum(axis=1)


# ### Splitting and Encoding
# At this point we need to filter our data into nominal and ordinal groups. Nominal data has categories but follows no particular order, whilst ordinal data is split into ordered categories, as we've seen above with the overall quality and overall condition columns. This means that they must be encoded in different ways. Ordinal data can be assigned integers of increasing value, but nominal data must be dummy encoded. Encoding is necessary because our model is only able to read numerical vaues.  
# At this point we will also normalise the skewed numerical features using log transformation, as we did previously with our target.

# In[27]:


#Nominal data
nominals = [
    'MSSubClass','MSZoning','Street','Alley','LandContour','LotConfig','Neighborhood','Condition1',
    'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl', 'Exterior1st','Exterior2nd',
    'MasVnrType','Foundation','Heating','CentralAir','GarageType','MiscFeature','SaleType',
    'SaleCondition','MoSold','YrSold'
]


# In[28]:


#Ordinal data
ordinals = [
    'LotShape','LandSlope','ExterQual','ExterCond','BsmtQual','BsmtCond',
    'BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','Electrical',
    'KitchenQual','Functional','FireplaceQu','GarageFinish','GarageQual',
    'GarageCond','PavedDrive','PoolQC','Fence'
]


# In[29]:


#encoding ordinal data with a regular label encoder
from sklearn.preprocessing import LabelEncoder
for ordinal in ordinals:
    lab = LabelEncoder()
    lab.fit(data[ordinal])
    data[ordinal] = lab.transform(data[ordinal])


# In[30]:


#Splitting catagorical and numerical features
categorical_features = data.select_dtypes(include = ["object", "category"]).columns
numerical_features = data.select_dtypes(exclude = ["object", "category"]).columns
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
data_num = data[numerical_features]
data_cat = data[categorical_features]


# In[31]:


data_num.isnull().values.sum(), data_cat.isnull().values.sum()


# In[32]:


#dummy encoding catagorical data
data_cat = pd.get_dummies(data_cat)
print(data_cat.shape)


# In[33]:


#Normalising skewed features for accuracy
from scipy.stats import skew

skewness = data_num.apply(skew)
skewness = skewness[abs(skewness) > 0.5]
print("There are {} skewed numerical features to log transform".format(skewness.shape[0]))
skewed_features = skewness.index
data_num[skewed_features] = data_num[skewed_features].applymap(np.log1p)


# In[34]:


#skewness = skewness[abs(skewness) > 0.75]

#from scipy.special import boxcox1p
#skewed_features = skewness.index
#lam = 0.15
#for feat in skewed_features:
    #all_data[feat] += 1
#    data[feat] = boxcox1p(data[feat], lam)


# In[35]:


#joining the numerical and catagorical data, abd scaling features. Splitting the test and train sets
data = pd.concat([data_num, data_cat], axis=1)

scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

X_train = data[:train_shape]
X_test = data[train_shape:]
print(X_train.shape, X_test.shape)


# ### The Models
# Its time to create our predictive models and feed in our prepared data. We will be using Lasso and Elastic Net- But before we get there we need to import the necessary libraries for this step. We then used KFold cross validation to test the expected performance- for a quick explanation of how that works please consult [this](https://machinelearningmastery.com/k-fold-cross-validation/) article.  
# 

# In[36]:


from sklearn.model_selection import GridSearchCV,learning_curve, cross_val_score, KFold


# In[37]:


#Declaring kfold variable
kfold = KFold(n_splits=20, random_state=42, shuffle=True)


# In[38]:


#Function to measure RMSE
#n_folds = 20

#def rmsle_cv(model):
#    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
#    rmse= np.sqrt(-cross_val_score(model, X_train.values, y, scoring="neg_mean_squared_error", cv = kf))
#    return(rmse)


# ### Lasso
# We chose the lasso model because we realised that with the time we had, it was the most efficient way to rid ourselves of the task of complex feature selection. It was a model that fascinated us and whe testing showed very good results.

# In[40]:


#Lasso model 
lasso = LassoCV(alphas=[
    0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.003, 0.007, 
    0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.2, 0.4, 0.6, 
    0.8, 1, 1.2
                ], random_state=1, n_jobs=-1, verbose=1)

lasso.fit(X_train, y)
alpha = lasso.alpha_
alphas = lasso.alphas_
mse = lasso.mse_path_
print("Optimized Alpha:", alpha)

plt.plot(mse)

lasso = LassoCV(alphas=alpha * np.linspace(0.5,1.5,20),
                cv = kfold, random_state = 1, n_jobs = -1)
lasso.fit(X_train, y)
alpha = lasso.alpha_
coeffs = lasso.coef_
intercpt = lasso.intercept_
print("Final Alpha:", alpha)
print("Intercept:", intercpt )


# In[41]:


#print("Lasso mean score:", rmsle_cv(lasso).mean())
#print("Lasso std:", rmsle_cv(lasso).std())


# In[42]:


lasso_4 = np.expm1(lasso.predict(X_test))
lasso_preds = pd.DataFrame(dict(SalePrice=lasso_4, Id=test.Id))


# ### Elastic-Net
# We found this model when looking for examples of a lasso and ridge ensemble- it was the best of both worlds. Although it did not perform as well as Lasso did, the averaging of the two models proved to be fruitful. Read the documentation for Elastic-Net and how it works [here.](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)

# In[43]:


#elastic net model
elnet = ElasticNetCV(alphas = [
    0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.003, 0.007,
    0.009, 0.01, 0.03, 0.05, 0.07, 0.09, 0.2, 
    0.4, 0.6, 0.8, 1, 1.2
                        ] 
                ,l1_ratio = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
                ,cv = kfold, random_state = 1, n_jobs = -1)

elnet.fit(X_train, y)
alpha = elnet.alpha_
alphas = elnet.alphas_
ratio = elnet.l1_ratio_
print("Optimized Alpha:", alpha)
print("Optimized l1_ratio:", ratio)

elnet = ElasticNetCV(alphas = alpha * np.linspace(0.5,1.5,20),
                     l1_ratio = ratio * np.linspace(0.9,1.3,6), 
                     cv = kfold, random_state = 1, n_jobs = -1)
elnet.fit(X_train, y)
coeffs2 = elnet.coef_
coeffs2 = elnet.intercept_
alpha = elnet.alpha_
ratio = elnet.l1_ratio_

print("Final Alpha:", alpha)
print("Final l1_ratio:", ratio)


# In[44]:


#print("ElasticNet mean score:", rmsle_cv(elnet).mean())
#print("ElasticNet std:", rmsle_cv(elnet).std())


# In[45]:


ElNet_2 = np.expm1(elnet.predict(X_test))
#ElNet_preds = pd.DataFrame(dict(SalePrice=ElNet_2, Id=test.Id))


# In[46]:


#simple stacked model- Averaging the models together for a more accurate prediction
stacked_sub = (lasso_4 + ElNet_2)/2


# In[47]:


stacked_preds = pd.DataFrame(dict(SalePrice=stacked_sub, Id=test.Id))
stacked_preds.to_csv('stacked_submission.csv', index=False)
#ElNet_preds.to_csv('ElNet_1.csv')
#lasso_preds.to_csv('Lasso_4.csv',index=False)


# In[48]:


model = ['Linear Regression','Decision Tree', 'Lasso', 'Ridge', 'Elastic-Net', 'Lasso + E-Net']
score = ['0.46485', '0.21068', '0.12131', '0.12301', '0.12216', '0.12042']

df = pd.DataFrame(list(zip(model, score)), 
               columns =['Model', 'Score']) 
df 


# ### Final thoughts
# From this model we obtained a score of 0.1204, and we believe that it was the best use of our resources with the time that we had. In future, we would consider delving further into ensembling and exploring the realm of boosting. We have also learnt the importance of testing multiple models at every point to gauge what has made a difference and what has not. I think that with experience will come a more intuitive selection process as we will have a better understanding of a wider range of models, but until that point, test, test, test!

# In[ ]:




