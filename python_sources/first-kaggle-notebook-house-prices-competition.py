#!/usr/bin/env python
# coding: utf-8

# I am someone who is not from data science background but have keen on it. Therefore, I started my self learning path by learning through any platform available across the internet since early 2020. This is the my first experience to run through a quite complete data science process from Analysing Data till Making Prediction. I believe that there are still a lot of improvement can be made for my model and is open to recieve any advises and comments. Do leave your advises or comments, really appreciate it.

# # Import Modules

# In[ ]:


#Basic Modules
from datetime import datetime
import pandas as pd
import numpy as np

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#PreProcessing
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import category_encoders as ce

#Model Building
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

#Optional Modules
from IPython.display import display
pd.options.display.max_columns = None
import warnings
warnings.filterwarnings("ignore")


# # Read input file

# In[ ]:


Sample = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv",index_col = 'Id')
Test = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv",index_col = 'Id')


# # 1. Exploratory Data Analysis (EDA)

# ### 1.1 Simple understanding about the data

# In[ ]:


Sample.shape


# In[ ]:


Sample.head()


# In[ ]:


Tgt_Col = 'SalePrice'
Num_Col = Sample.select_dtypes(exclude='object').drop(Tgt_Col,axis=1).columns
Cat_Col = Sample.select_dtypes(include='object').columns

print("Numerical Columns : " , len(Num_Col))
print("Categorical Columns : " , len(Cat_Col))


# Information obtain:
# 1. There are 1460 observations in this sample data
# 2. There are 80 attributes (79 features and 1 target).
# 3. Among the 79 features, there are 36 are numeric and 43 are categorical

# ### 1.2 Analyse Target Variable

# In[ ]:


sns.distplot(Sample[Tgt_Col])
plt.ticklabel_format(style='plain', axis='y')
plt.title("SalePrice's Distribution")
plt.show()

print('Skewness : ' , str(Sample[Tgt_Col].skew()))


# In[ ]:


sns.distplot(np.log(Sample[Tgt_Col]+1))
plt.ticklabel_format(style='plain', axis='y')
plt.title("SalePrice's Distribution")
plt.show()

print('Skewness : ' , str(np.log(Sample[Tgt_Col]+1).skew()))


# Information get:
# 1. Target column is skewed which might affect the accuracy of predictive models.
# 2. Log transformation on target column can effectively reduce the skewness.

# ### 1.3 Numerical Columns

# In[ ]:


Sample[Num_Col].describe().round(decimals=2)


# In[ ]:


fig = plt.figure(figsize=(12,18))
for idx,col in enumerate(Num_Col):
    fig.add_subplot(9,4,idx+1)
    sns.distplot(Sample[col].dropna(), kde_kws={'bw':0.1})
    plt.xlabel(col)
plt.tight_layout()
plt.show()


# In[ ]:


cor = Sample.corr()

import matplotlib.style as style
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30,20))

mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(cor, cmap=sns.diverging_palette(8, 150, n=10),mask = mask, annot = True,vmin=-1,vmax=1);
plt.title("Heatmap of all the Features", fontsize = 30);


# In[ ]:


fig = plt.figure(figsize=(12,18))
for idx,col in enumerate(Num_Col):
    fig.add_subplot(9,4,idx+1)
    if abs(cor.iloc[-1,idx])<0.1:
        sns.scatterplot(Sample[col],Sample[Tgt_Col],color='red')
    elif abs(cor.iloc[-1,idx])>=0.5:
        sns.scatterplot(Sample[col],Sample[Tgt_Col],color='green')
    else:
        sns.scatterplot(Sample[col],Sample[Tgt_Col],color='blue')
    plt.title("Corr to SalePrice : " + (np.round(cor.iloc[-1,idx],decimals=2)).astype(str))
plt.tight_layout()
plt.show()


# In[ ]:


Sample[Num_Col].isna().sum().sort_values(ascending=False).head()


# Information get:
# 1. There are some columns which are actually categorical variables.
# 2. From histogram, there are some features are skewed.
# 2. From heatmap, there are few features are highly correlated to each others.
# 3. From scatter plot, there are few certain outliers.
# 4. From scatter plot, those low correlated features.
# 5. There are some columns having missing values.

# ### 1.4 Categorical Features

# In[ ]:


Sample[Cat_Col].describe()


# In[ ]:


for col in Sample[Cat_Col]:
    if Sample[col].isnull().sum() > 0 :
        print (col , " : ", Sample[col].isnull().sum() , Sample[col].unique())


# Information get:
# 1. There are columns having missing values.
# 2. These missing value are actually means Not Available(NA) after looking at the data description.
# 4. Only the missing value for Electrical feature does not means NA.

# # 2. Data Cleaning and Preprocessing

# ### 2.1 Missing Values and Outliers
# Fill NA to those categorical columns with missing values. <br>
# For MasVnrArea, it should be 0 since the MasVnrType is NA. <br>
# There is 3 columns left with missing values, will handle it with imputer later. <br>
# Drop those columns which are considered as outliers.

# In[ ]:


Sample_copy = Sample.copy()

Sample_copy['MasVnrArea'] = Sample['MasVnrArea'].fillna(0)

Cat_Cols_Fill_NA = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType',
                      'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',
                      'GarageQual', 'GarageFinish', 'GarageType','GarageCond']

for cat in Cat_Cols_Fill_NA:
    Sample_copy[cat] = Sample_copy[cat].fillna("NA")


# In[ ]:


Sample_copy.isna().sum().sort_values(ascending = False).head()


# In[ ]:


Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['LotFrontage']>200].index)
Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['LotArea']>100000].index)
Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['MasVnrArea']>1200].index)
Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['BsmtFinSF1']>4000].index)
Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['TotalBsmtSF']>4000].index)
Sample_copy = Sample_copy.drop(Sample_copy[(Sample_copy['GrLivArea']>4000) & (Sample_copy[Tgt_Col]<300000)].index)
Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['BsmtFinSF2']>1300].index)
Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['1stFlrSF']>4000].index)
Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['EnclosedPorch']>500].index)
Sample_copy = Sample_copy.drop(Sample_copy[Sample_copy['MiscVal']>5000].index)
Sample_copy = Sample_copy.drop(Sample_copy[(Sample_copy['LowQualFinSF']>600) & (Sample_copy[Tgt_Col]>400000)].index)


# ### 2.2 Transform Target and Redefine Features
# Target variable is log transformed to reduce skewness. <br>
# The remaining missing values will be imputed later. <br>
# Redefine numerical and categorical features.

# In[ ]:


Sample_copy[Tgt_Col] = np.log(Sample_copy[Tgt_Col]+1)
Sample_copy = Sample_copy.rename(columns={'SalePrice': 'SalePriceLog'})
Tgt_features = 'SalePriceLog'


# In[ ]:


Sample_copy['MSSubClass'] = Sample_copy['MSSubClass'].astype(str)
Sample_copy['OverallQual'] = Sample_copy['OverallQual'].astype(str)
Sample_copy['OverallCond'] = Sample_copy['OverallCond'].astype(str)


# In[ ]:


Num_features = Sample_copy.select_dtypes(exclude='object').drop(Tgt_features,axis=1).columns
Cat_features = Sample_copy.select_dtypes(include='object').columns


# # 3. Feature Selection and Engineering
# Highly correlated features needed to find out and exclude one of it. <br>
# Numerical features with low correlations with target variable should be excluded. <br>
# Redefine highly skewed, numeric and categorical features.

# In[ ]:


cor = Sample_copy.corr()
cor_list = cor.abs().unstack()
cor_list[cor_list>0.75].sort_values(ascending=False)[34:].drop_duplicates()


# In[ ]:


Collinear = ['GarageArea','TotRmsAbvGrd','GarageYrBlt','1stFlrSF']


# In[ ]:


Low_Corr = []

for idx,col in enumerate(Num_features):
    if abs(cor.iloc[-1,idx])<=0.1:
        Low_Corr.append(col)


# In[ ]:


features_drop = ['SalePriceLog'] + Low_Corr + Collinear

X = Sample_copy.drop(features_drop, axis=1)
y = Sample_copy[Tgt_features]


# In[ ]:


numeric_features = X.select_dtypes(exclude='object').columns
categorical_features = X.select_dtypes(include='object').columns

skewed_feats = X[numeric_features].apply(lambda x: x.skew())
high_skew = skewed_feats[skewed_feats > 0.5]
skew_features = high_skew.index

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# # 4. Machine Learning Algorithm

# ### 4.1 Setup
# Define the scaler to standardize the data. <br>
# Define reverse log function to inverse transform the log(saleprice). <br>
# Define all the models to try for predict the saleprice. <br>
# Setup preprocessing process using pipeline.

# In[ ]:


RobustScaler = preprocessing.RobustScaler
PowerTransformer = preprocessing.PowerTransformer

model_list = {'RF_Model' : RandomForestRegressor(random_state=5),
              'XGB_Model' : XGBRegressor(objective ='reg:squarederror',n_estimators=1000, learning_rate=0.05),
              'Lasso_Model' : Lasso(alpha=0.0005), 
              'Ridge_Model' : Ridge(alpha=0.002), 
              'Elastic_Net_Model' : ElasticNet(alpha=0.02, random_state=5, l1_ratio=0.7), 
              'GBR_Model' : GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)}

def inv_y(transformed_y):
    return np.exp(transformed_y)


# In[ ]:


skew_transformer = Pipeline(steps=[('imputer', SimpleImputer()),
                                   ('scaler', PowerTransformer())])

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer())])

categorical_transformer = Pipeline(steps=[
    ('imputer1', SimpleImputer(strategy='constant', fill_value='NA')),
    ('encoder', ce.one_hot.OneHotEncoder()),
    ('imputer2', SimpleImputer())])
    
preprocessor = ColumnTransformer(
    transformers=[('skw', skew_transformer, skew_features),
                  ('num', numeric_transformer, numeric_features),
                  ('cat', categorical_transformer, categorical_features)])


# ### 4.2 Evaluate Each Model and Cross Validate Top Model
# Loop through all the models to get the each model's predictive performance. <br>
# Cross validate the top 2 models which give the best performance. <br>

# In[ ]:


model_score = pd.Series()
    
for key in model_list.keys():
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('scaler', RobustScaler()),
                           ('model', model_list[key])])
    
    model = pipe.fit(train_X, train_y)
    
    pred_y = model.predict(val_X)
    val_mae = mean_absolute_error(inv_y(pred_y), inv_y(val_y))
    model_score[key] = val_mae
    
top_2_model = model_score.nsmallest(n=2)
print(top_2_model)


# In[ ]:


from sklearn.model_selection import cross_val_score

imputed_X = preprocessor.fit_transform(X,y)
n_folds = 10

for model in top_2_model.index:
    scores = cross_val_score(model_list[model], imputed_X, y, scoring='neg_mean_squared_error', 
                             cv=n_folds)
    mae_scores = np.sqrt(-scores)

    print(model + ':')
    print('Mean RMSE = ' + str(mae_scores.mean().round(decimals=3)))
    print('Error std deviation = ' + str(mae_scores.std().round(decimals=3)) + '\n')


# ### 4.3 Choose Algorithm and Fine Tune
# Decide the best model by considering the MAE score and RMSE score. <br>
# Search for best combination of hyperparameter for the model.

# In[ ]:


param_grid = [{'alpha': [0.001, 0.0005, 0.0001]}]
top_reg = Lasso()

grid_search = GridSearchCV(top_reg, param_grid, cv=5, 
                           scoring='neg_mean_squared_error')

grid_search.fit(imputed_X, y)

grid_search.best_params_


# # 5. Making Prediction

# ### 5.1 Pre-processing Test Data
# Repeat the manual preprocessing defined before on test data.

# In[ ]:


test_X = Test.copy()

test_X['MasVnrArea'] = test_X['MasVnrArea'].fillna(0)

cat_cols_fill_na = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType',
                      'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtQual', 'BsmtCond',
                      'GarageQual', 'GarageFinish', 'GarageType','GarageCond']

for cat in cat_cols_fill_na:
    test_X[cat] = test_X[cat].fillna("NA")
    
test_X['MSSubClass'] = test_X['MSSubClass'].astype(str)
test_X['OverallQual'] = test_X['OverallQual'].astype(str)
test_X['OverallCond'] = test_X['OverallCond'].astype(str)

if 'SalePriceLog' in features_drop:
    features_drop.remove('SalePriceLog')

test_X = test_X.drop(features_drop, axis=1)


# ### 5.2 Create Final Model and Predict
# Define the model using the best hyperparameter and predict it.

# In[ ]:


final_model = Lasso(alpha=0.0005, random_state=5)

pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('scaler', RobustScaler()),
                       ('model', final_model)])

model = pipe.fit(X, y)

test_preds = model.predict(test_X)

output = pd.DataFrame({'Id': Test.index,
                       'SalePrice': inv_y(test_preds)})

output.to_csv(str(datetime.now().strftime('%Y%m%d_%H%M%S')) + '.csv', index=False)


# # 6. Reference
# 
# [House Prices: 1st Approach to data science process](https://www.kaggle.com/cheesu/house-prices-1st-approach-to-data-science-process/notebook) by [Chee Su Goh](https://www.kaggle.com/cheesu)
