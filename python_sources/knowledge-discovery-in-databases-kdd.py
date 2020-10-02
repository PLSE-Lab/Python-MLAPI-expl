#!/usr/bin/env python
# coding: utf-8

# # Wine Points Prediction
# Prepared By: </br>
#     Latoya Clarke ,
#     </br>
#     Daniella Mcalla ,
#     Mardon Bailey 

# ## Scenario:
# Roger Voss is  a famous master sommelier,  european editor at Wine Enthusiast and author of many books about wine.  **Wine Merchants** is a local wine retailer in France. They realized a trend in their wine sales. This trend shows that the higher the wine scores given by Mr. Roger Voss, the higher the sales of the wine. Wine Merchants wants to expand their wine offerings but are unsure of how these other wines would perform (in sales). Wine Merchant now wants to predict score Mr. Roger Voss would give these newer wine offerings inorder to select which wines will substantially increase their bottom line. 
# 
# 
# ### Food for thought
# Based on research, the taste, smell, texture and look of a bottle of wine is what affects the point/rating a sommelier (wine expert) gives the wine. If these characteristics of the wine drives the points given to it, what are the factors that drive these characteristics to be of a certain quality.  
# 

# ## Selection of Data

# #### Description of Dataset:
# The Wine dataset consists of data about wine tasting reviews scraped from the Wine Enthusiast Magazine https://www.winemag.com/?s=&drink_type=wine&page=12466on  . The dataset consists of only wines that have received a point between 80 and 100 inclusive. The dataset consists of 129, 971 observations (rows) and  14 attributes (columns) of which only a subset of these observations were used in the actual analysis. The sample of the dataset used in the analysis has a dimension of (25514, 14) and is exclusive to wines evaluated by Roger Voss.  From the 14 features in the sample dataset, only a few were selected to predict the points Roger Voss alloted to each wine.  Some of these features include the wine's designation, regions in which the grapes used to make the wine are grown and the type of grapes used to make the wine (variety) etc. Other features such as the vintage year were created via feature engineering in aid of developing models to predict the points ratings Roger Voss would give a wine. 
# 
# Since the aim is to create predictive models that will predict points Roger Voss is likely to give a wine, the sample dataset was not randomly selected. The selection was rather deliberate. 
# 
# * ###### Metadata:
# 
# More information on the dataset  such as the metadata may be accessed via this link https://www.kaggle.com/zynicide/wine-reviews
# 
# #### Definition of Terms: 
# 
# * ###### Sommelier :
# A wine expert/specialist. A knowledgeable wine professional 
# 
# * ###### Wine Tasting Review :
# An event where sommeliers perform a sensory examination and evaluation of a wine. (i.e. taste, smell, feel & look of wine) 
# 
# * ###### Vintage Year : 
# The year on a bottle of  wine which denotes that most if not all the grapes used to make that bottle of wine were harvested in that pecified year
# 
# 
# 
# 

# In[ ]:


# Libraries needed 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import warnings
import re
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
#random_state = 1
plt.rcParams['figure.figsize']=8,8
import os
print(os.listdir("../input"))


# In[ ]:


# load wine dataset into notebook
data_path = "../input/winemag-data-130k-v2.csv"
wine_data = pd.read_csv(data_path)


# In[ ]:


print("Wine Data Dimension:",wine_data.shape)
wine_data.head(3) #disply first 5 elements in dataset


# In[ ]:


wine_data.tail(3) #displays last 3 elements in dataset


# In[ ]:


# check to see if there are spaces in column names
wine_data.columns


# In[ ]:


wine_data.describe()


# In[ ]:


# Useful information about the data
wine_data.info()


# In[ ]:


# Disttribution of points for initial dataset
sns.distplot(wine_data.points) 
plt.xlabel("Points",size=15)
plt.title("Fig.1: Points Distribution", size=20)


# In[ ]:


# Wine Tasters and the Amount of Wines Evaluated
wine_data['taster_name'].value_counts().plot(kind='bar')
plt.xticks(fontsize=11)
plt.xlabel("Taster Names",size=15)
plt.ylabel('No. of wines tasted', size=15)
plt.title("Fig. 2: Wine Tasters and the Amount of Wines Evaluated", size=20)


# ### Sample Dataset (i.e. wines evaluated by Roger Voss)

# In[ ]:


# Selecting sample of the data that will be used (i.e. the wines evaluated by Roger Voss)
wine_data = wine_data[(wine_data['taster_name']=='Roger Voss') | (wine_data['taster_twitter_handle']=='@vossroger')]
print("Wine Data Sample Dimension:",wine_data.shape,"(i.e. Wines evaluated by Roger Voss)") #dataset dimension
wine_data.head()


# In[ ]:


# Measures of the numerical data (i.e. wines tasted by Roger Voss)
wine_data.describe()


# In[ ]:


wine_data.info()


# In[ ]:


# Distribution of points for wines Roger Voss evaluated
sns.distplot(wine_data.points) 
plt.xlabel("Points",size=15)
plt.title("Fig.3: Points Distribution for Wines Evaluated by Roger Voss ", size=20)


# In[ ]:


sns.countplot(x='country',data=wine_data, orient="h")
plt.ylabel('Country Count',size=12)
plt.xlabel("Country",size=12)
plt.xticks(rotation=45)
plt.suptitle("Fig.4: Countries per bottle of Wine Evaluted by Roger Voss ", size=20)


# ## Preprocessing of Data

# In[ ]:


# Rename column 'serial' to 'wine_id'
wine_data.rename(columns={'serial':'wine_Id'}, inplace=True)
wine_data.head(1)


# In[ ]:


#check for duplicates in dataset and remove if any
print(wine_data.duplicated(subset=None, keep='first').sum(),"duplicate record(s)")


# In[ ]:


# Perform feature extraction to impute the year of each wine
wine_data['year'] = wine_data['title'].str.extract('(\d\d\d\d)', expand=True)


# In[ ]:


# Check to see if there are any null years
wine_data['year'].isnull().value_counts()


# In[ ]:


# wines that does not have a year in the title
#Wines without a year are classified as Non-Vintage wines
wine_data.title[wine_data['year'].isnull()].head()


# In[ ]:


# convert year to int so as to make searches for preprocessing easier
wine_data.year = pd.to_numeric(wine_data.year, errors='coerce').fillna(0).astype(np.int64)


# In[ ]:


# check fo erroneous years (NB: its year 2018, any year above this is invalid)
print((wine_data['year']>2018).sum(),"invalid year(s)")


# In[ ]:


# Applying feature engineering to create type of wine (Vintage/n\Non-Vintage)
wine_data['type']= None
wine_data.type[wine_data['year']!=0] = 'Vintage'
wine_data.type[wine_data['year']==0] = 'Non-Vintage'


# In[ ]:


# Create loation by feature extraction from title
no_location = wine_data['title'].str.split('(', expand=True, n=1)
#wine_data['location'] = no_location.str.extract('(', expand=True)
#wine_data
no_location=no_location[1].str.split(')', expand=True, n=1)
wine_data['location']=no_location[0]

#wine_data[wine_data['location'].isnull()==True]


# In[ ]:


# impute location from region_2,region_1,province 
wine_data['location'].fillna(wine_data.region_2, inplace = True) 
wine_data['location'].fillna(wine_data.region_1, inplace = True)
wine_data['location'].fillna(wine_data.province, inplace = True)


# In[ ]:


# look for missing locations and country
print(wine_data['location'].isnull().sum(),"missing location(s) and",wine_data['country'].isnull().sum(),"missing countries") #check for null locations

#impute missing missing location and country from title research
wine_data.location.fillna('Bordeaux',inplace=True)
wine_data.country.fillna('France',inplace=True)

print("are attributed to 'Bordeaux' region in 'France' based on research of wine titles")


# In[ ]:


# look for missing prices
print(wine_data['price'].isnull().sum(),"missing price(s)") #check for null prices

#impute missing prices with the median price
wine_data.price.fillna(wine_data['price'].median(),inplace=True)
print("imputed from median price")


# In[ ]:


# Drop columns that are not needed
wine_data_2 = wine_data.drop(['designation','region_1','region_2','taster_twitter_handle','description','province','taster_name'],axis=1)
wine_data_2.head()


# In[ ]:


wine_data_2.info() # confirm that there are no missing values


# ## Transformation of Data

# In[ ]:


# Transformation
# Label encoder transforms nominal features into numerical labels which algorithms can make sense of
def create_label_encoder_dict(df):
    from sklearn.preprocessing import LabelEncoder
    
    label_encoder_dict = {}
    
    for column in df.columns:
        if not np.issubdtype(df[column].dtype, np.number) and column != 'year':
            label_encoder_dict[column]= LabelEncoder().fit(df[column])
    return label_encoder_dict


# In[ ]:



label_encoders = create_label_encoder_dict(wine_data_2)
#print("Encoded Values for each Label")
#print("="*32)
#for column in label_encoders:
 #   print("="*32)
 #   print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))
  #  print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)
    


# In[ ]:


### Apply each encoder to the data set to obtain transformed values
wd3 = wine_data_2.copy() # create copy of initial data set
for column in wd3.columns:
    if column in label_encoders:
        wd3[column] = label_encoders[column].transform(wd3[column])

print("Transformed data set")
print("="*32)
wd3.head()


# ## Mining of Data

# In[ ]:


# Function to do K-Fold Cross Validation
def cross_validate(x,y,kf_split):
    from sklearn.model_selection import KFold
    
    #K-Fold Cross Validation
    kf =KFold(n_splits=kf_split,shuffle=True,random_state=1)
    
    for train_index, test_index in kf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    return [X_train,y_train,X_test,y_test]


# 
# #### ====JUSTIFICATION OF USE====
# 
# A cross validataion was done to help reduce the liklihood of selection bias and overfitting
# and to give insights as to how the predictive models will generalise to an unknown dataset.
# 
# Apart from popularity, there is no specific reason for choosing the K-Fold Cross Validation.
# 

# In[ ]:


# Algorithms without Hyper Parameter Tuning
def pred_techniques(x,y,kf_split): 
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neural_network import MLPRegressor

    train_test = cross_validate(x,y,kf_split) #perform kfold cross validation

    # Decision Tree Regressor
    reg = DecisionTreeRegressor(random_state=1) 
    reg.fit(train_test[0], train_test[1])
    
    
    # Multi-Layer Perceptron Regressor
    clf = MLPRegressor(solver='adam', alpha=1e-5, activation='relu',learning_rate_init =0.01,shuffle=True,
                    hidden_layer_sizes=(7, 4),random_state=1)
    clf.fit(train_test[0],train_test[1])
    

    return [reg,clf,train_test[2],train_test[3],train_test[0],train_test[1]]


# In[ ]:


# separate data into dependent (Y) and independent(X) variables
feature_cols =  ['variety','winery','location', 'year']
x_data = wd3[feature_cols]
y_data = wd3['points']

l = pred_techniques(x_data,y_data,100)
print("Fig.5: Feature Significance") 
pd.DataFrame([ "%.2f%%" % perc for perc in (l[0].feature_importances_ * 100) ], index = x_data.columns, columns = ['Feature Significance in Decision Tree'])    


# ## Interpretation/Evaluation

# In[ ]:


# Accuracy Test Scores of both techniques 
r2_tree = l[0].score(l[2],l[3])
r2_nn = l[1].score(l[2],l[3])

print("Decision Tree Regressor")
print("="*32)
print("R Square:",r2_tree )


print("\nMulti-Layer Perceptron Regressor")
print("="*32)
print("R Square:",r2_nn)


# In[ ]:


# Actual points and Predicted points by both models plus 4 predictor variables on left
results= l[2].copy()
results['Actual Points']=l[3]
pred_tree=l[0].predict(l[2])
pred_mlp=l[1].predict(l[2])
results['Dec_Tree_Reg']=pred_tree
results['MLP_Regressor']=pred_mlp
print("Fig.6: Actual Points and Predicted Points yielded from both Models")
results.head()


# In[ ]:


# Calculate Variance in both models
mse_treg = mean_squared_error(l[3],pred_tree)
mse_nn = mean_squared_error(l[3],pred_mlp)

# Calculate Standard Deviation in both models
rmse_treg = math.sqrt(mean_squared_error(l[3],pred_tree))
rmse_nn = math.sqrt(mean_squared_error(l[3],pred_mlp))

# Calcualte Mean Absolute Error in both models
mae_treg = mean_absolute_error(l[3],pred_tree)
mae_nn = mean_absolute_error(l[3],pred_mlp)

# Print evaluation metrics of both models
print("Decision Tree Regressor")
print("="*32)
print("MSE:",mse_treg)
print("RMSE:",rmse_treg)
print("MAE:",mae_treg)

print("\nMulti-Layer Perceptron Regressor")
print("="*32)
print("MSE:",mse_nn)
print("RMSE:",rmse_nn)
print("MAE:",mae_nn)


# In[ ]:


print("Decision Tree Number of Perfect Predictions:")
results[results['Dec_Tree_Reg']==results['Actual Points']].Dec_Tree_Reg.count()


# In[ ]:


print("Neural Network Number of Perfect Predictions:")
results[results['MLP_Regressor']==results['Actual Points']].MLP_Regressor.count()


# In[ ]:


sns.distplot( results["Actual Points"] , color="skyblue", label="Actual Points")
sns.distplot( results["Dec_Tree_Reg"] , color="orange", label="Decision Tree Predicted Points")
plt.legend()
plt.xlabel("Points",size=15)
plt.title("Fig.7: Actual Points vs Decision Tree Predicted Points", size=20)


# In[ ]:


sns.distplot( results["Actual Points"] , color="skyblue", label="Actual Points")
sns.distplot( results["MLP_Regressor"] , color="red", label="NN Predicted Points")
plt.legend()
plt.xlabel("Points",size=15)
plt.title("Fig.8: Actual Points vs Neural Network Predicted Points", size=20)


# In[ ]:


print("Fig.9: Summary of Evaluation Metrics")
pd.DataFrame(dict(R_Square= [r2_tree,r2_nn],
                  MSE=[mse_treg,mse_nn], RMSE=[rmse_treg,rmse_nn],MAE=[mae_treg,mae_nn]),
                index=['Dec Tree Reg','MLP Reg'])


# 
# The two techniques used to predict the points Roger Voss would give a wine are Decision Tree Regressor (DTR) and Multi-Layer Perceptron Regressor (MLPR).  Both techniques were used because the response variable (points) is of numeric datatype and both are able to do multiple regression.
# 
# Decision Tree
# 
# The Decision Tree Algorithm builds a tree like structure as a model which uses a top-down, greedy search through the space of possible branches with no backtracking.  The model breaks down the dataset into smaller and smaller subsets while the associated decision tree is created incrementally. The Decision Tree algorithm is simple to understand and interpret, has value even with little hard data, helps determine worst, best and expected values for different scenarios and can be combined with other decision techniques.
# 
# Multi-Layer Perceptron (MLP) 
# 
# The Multi-Layer Perceptron is the sum of several perceptions together. The input layer reads in the data and the output layer creates the resulting output. The Multi-Layer Perceptron model trains using backpropagation with no activation function in the output layer, which can also be seen as using the identity function as activation function. The Multi-Layer Perceptron uses a parameter alpha for regularization (L2 regularization) term which helps in avoiding overfitting by penalizing weights with large magnitudes. The Multi-layer Perceptron algorithm is capable of learning non-linear models in real-time and requires tuning a number of hyperparameters such as the number of hidden neurons, layers, and iterations. Multi-Layer Perception algorithm is also sensitive to feature scaling.
# 
# 
#  The Decision Tree Regressor outperformed the MLP regressor  in this experiment. The R2 score  or "coefficient of determination" which depicts how well the data fits the model had a value of 0.278 for the Decision Tree Regressor and 0.020657 for the MLP Regressor.  These values are relatively low which suggest that the models did not fit the data very well. However, the standard deviations or "RMSE" of both models were relatively low.  This is an indication that the observations are not spread out but rather closer to the actual points which further suggests that the model did not perform as bad.  The Decision Tree Regressor had a standard deviation of 2.551697 and the MLP Regressor had a standard deviation of 2.972425. 

# In[ ]:


results.describe()


# ## Conclusion
# 
# The wine retailer is now able to know what wines to purchase and where to purchase them.

# ## References
# 
# Attending a Wine Tasting Event - dummies. (2018). Retrieved from https://www.dummies.com/food-drink/drinks/wine/attending-a-wine-tasting-event/
# 
# Metadata 
# https://www.kaggle.com/zynicide/wine-reviews
# 
# The Different Types of Wine (Infographic) | Wine Folly. (2018). Retrieved from https://winefolly.com/review/different-types-of-wine/

# ## Appendices

# In[ ]:


sns.regplot(x="price", y="points", data=wine_data, fit_reg = False)
plt.xlabel("Price",size=12)
plt.ylabel("Points",size=12)
plt.title("Fig.10: Correlation between Price and Points",size=20)


# In[ ]:


# Coverage of Vintage vs. Non-Vintage
wine_data['type'].value_counts().plot(kind="pie",autopct='%1.0f%%')
labels = 'Vintage', 'Non-Vintage'
plt.legend(labels)
plt.suptitle("Fig.11: Vintage vs. Non-Vintage Wine", size=20)
plt.ylabel('')


# In[ ]:


# Non-Vintage Wine Points Distribution
#wine_data[wine_data['year']== 0]
no_year= wine_data[wine_data['year']==0]
sns.boxplot(x=no_year.points)
plt.xlabel("Points",size=15)
plt.title("Fig.12: Non-Vintage Wine Points Distribution  ", size=20)


# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(wd3.corr(),cmap=plt.cm.Reds,)
plt.xticks(size=12,rotation=45)
plt.yticks(size=12)
plt.title('Fig.13: Correlation between Transformed Data Columns ',size=20)


# 

# 

# In[ ]:




