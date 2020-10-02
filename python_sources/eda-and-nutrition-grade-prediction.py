#!/usr/bin/env python
# coding: utf-8

# _Wilson Goma  July 2017_

# ## contents
#    
# **Introduction**
# 
# Issues of the study          
# Import          
# Loading data
#    
# **Preprocessing** 
# 
# Cleaning   
# Training/Testing Set
#    
# **Analysis**
# 
# Univariate analysis   
# Multivariate analysis  
# Features Engineering
#    
# **Modeling and Prediction**
# 
# Prediction
#    
# **Conclusion**
# 

# ## Introduction
# 
# For my first notebook on kaggle I chose the open food facts data set. 
# In this kernel I will mainly do an EDA and use a simple KNN predictive model.
# All feedback is welcome! :)

# ### I. Issues of the study
# 
# The objective of this project is to carry out an exploratory analysis and supplement the database by predicting the nutritional grade of unannotated foods.

# ### II. Import

# In[ ]:


import numpy as np , matplotlib as plt 
get_ipython().run_line_magic('pylab', 'inline')
from scipy import stats
from scipy.stats import chi2_contingency

import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import preprocessing , decomposition , neighbors
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
import random

import warnings
warnings.filterwarnings('ignore')


# ### III. Loading Data

# In[ ]:


# Data load
RawData_Nutri = pd.read_csv('../input/en.openfoodfacts.org.products.tsv',  sep='\t')
RawData_Nutri.shape


# In[ ]:


RawData_Nutri.describe()


# ## Preprocessing

# ### I. Cleaning

# First, there are many missing values in the dataset. We can view it.

# In[ ]:


plt.figure(figsize=(13, 40))
plt.rcParams['axes.facecolor'] = 'black'
plt.rc('grid', color='#202020')
(RawData_Nutri.isnull().mean(axis=0)*100).plot.barh(color ="#FF6600")
plt.xlim(xmax=100)
plt.title("Missing values rate",fontsize=18)
plt.xlabel("percentage",fontsize=14)


# I chose to discard features with more than 60% missing values, these variables are **inconsistent**.

# In[ ]:


# Keep only consistent features (less of 60% NaN values)
de = RawData_Nutri.isnull().mean(axis=0)
l = []
for i in range(0,len(de)):
    if de[i] < 0.6:
        templist = list(de[de==de[i]].index) 
        for i in range (0,len(templist)):
            l.append(templist[i])

variable_consistante = list(set(l)) 
Dnutri_NewFeat = RawData_Nutri.loc[:, lambda df: variable_consistante] 
Dnutri_NewFeat.shape


# Some variables are redundant or not useful for the issue.

# In[ ]:


# This list contains unwanted features
NoList = ["code","url","states_en", "countries_tags","additives","brands",
          "last_modified_datetime","creator","additives_tags",
          "states","states_tags","ingredients_text","created_datetime",
          "serving_size","created_t","nutrition-score-uk_100g","countries",
          "last_modified_t","brands_tags","additives_en",
          "ingredients_that_may_be_from_palm_oil_n"]

for i in range (0,len(NoList)):
    variable_consistante.remove(NoList[i])

Dnutri_NewFeat = RawData_Nutri.loc[:, lambda df: variable_consistante]
Dnutri_NewFeat.shape


# In[ ]:


#variable_consistante


# In[ ]:


l = ["product_name","countries_en","nutrition_grade_fr","nutrition-score-fr_100g"]
featlist = list(Dnutri_NewFeat)
for i in range(0,len(l)):
    featlist.remove(l[i])

# Replace NaN value by 0 for nuremic features.
for i in range(len(featlist)):
    Dnutri_NewFeat[featlist[i]].fillna(0, inplace=True)


# In[ ]:


# Replace NaN value by Unknow for categorial features.
Dnutri_NewFeat["countries_en"].fillna("Unknow", inplace=True)
Dnutri_NewFeat["product_name"].fillna("Unknow", inplace=True)


# ### II. Training / Testing Set

# Make **Training Set**

# In[ ]:


Dnutri_Nano = Dnutri_NewFeat.dropna(axis=0, how='any') 
Dnutri_Nano = Dnutri_Nano.sort_values(by=["nutrition_grade_fr"] , ascending=[True])
Dnutri_Nano.shape


# In[ ]:


plt.figure(figsize=(13,4))
(Dnutri_Nano.notnull().mean(axis=0)*100).plot.barh(color ="#33CC66")
plt.xlim(xmax=100)
plt.title("Not null value rate (Dnutri_Nano) ")


# Make **Testing Set**

# In[ ]:


Dnutri_score_less = Dnutri_NewFeat[pd.isnull(Dnutri_NewFeat['nutrition_grade_fr'])]
Dnutri_score_less.shape


# In[ ]:


plt.figure(figsize=(13,3))
(Dnutri_score_less.isnull().mean(axis=0)*100).plot.barh(color ="#33CCFF")
plt.xlim(xmax=100)
plt.title("Missing values rate (Dnutri_score_less)")


# ## Analysis

# ### I. Univariate analysis

# Let's see the distributions and check the **outliers**.

# In[ ]:


def boxplot_univ (feature,plotColor="#CC9900"):
    """
    Generates a boxplot from a given features and color
    """
    plt.figure(figsize=(8,3)) 
    plt.rc('grid', color='#202020') 
    plt.rc('axes', facecolor='black')
    plt.rc('text', color='black')
    sns.boxplot(data=Dnutri_Nano, y=feature, color=plotColor) 


# In[ ]:


boxplot_univ("energy_100g")
plt.ylim(0, 5000)

boxplot_univ("fat_100g","#FFCC33")
plt.ylim (0, 200)

boxplot_univ("sugars_100g","#33CCFF")
plt.ylim (-50, 150)

boxplot_univ("salt_100g","#F5F5DC")
plt.ylim (0, 10)

boxplot_univ("fiber_100g","#33CC33")
plt.ylim (0, 100)

boxplot_univ("additives_n","purple")
plt.ylim (0, 20)

boxplot_univ("proteins_100g","red")
plt.ylim (0, 100)

boxplot_univ("calcium_100g","#CCCCCC")
plt.ylim (0, 0.5)


# There are some outlier in the distributions. 
# 

# In[ ]:


#Outliers Treatment
Dnutri_Nano.loc[Dnutri_Nano.energy_100g > 4000, 'energy_100g'] = 4000
Dnutri_Nano.loc[Dnutri_Nano.fat_100g > 100, 'fat_100g'] = 100
Dnutri_Nano.loc[Dnutri_Nano.carbohydrates_100g > 100, 'carbohydrates_100g'] = 100
Dnutri_Nano.loc[Dnutri_Nano.sugars_100g > 100, 'sugars_100g'] = 100
Dnutri_Nano.loc[Dnutri_Nano.sugars_100g < 0, 'sugars_100g'] = 0
Dnutri_Nano.loc[Dnutri_Nano.salt_100g > 100, 'salt_100g'] = 100
Dnutri_Nano.loc[Dnutri_Nano.sodium_100g > 100, 'sodium_100g'] = 100
Dnutri_Nano.loc[Dnutri_Nano.fiber_100g >100, 'fiber_100g'] = 100
Dnutri_Nano.loc[Dnutri_Nano.proteins_100g >100, 'proteins_100g'] = 100
Dnutri_Nano.loc[Dnutri_Nano.proteins_100g < 0, 'proteins_100g'] = 0


# "Trans-fat_100g" and "ingredient_from_palm_oil_n" have almost 100% zero value. They will be removed, they do not provide information for the rest of the analysis.

# In[ ]:


# Delete features
Dnutri_Nano = Dnutri_Nano.drop('trans-fat_100g',1)
Dnutri_Nano = Dnutri_Nano.drop('ingredients_from_palm_oil_n',1)


# In[ ]:


nutriGrd = Dnutri_Nano['nutrition_grade_fr'].value_counts(normalize=True)
plt.figure(figsize=(6, 6))
pie(nutriGrd.values, labels=nutriGrd.index,
                autopct='%1.1f%%', shadow=True, startangle=90)
title('Nutrigrade Rate')
show()


# The vast majority of food comes from the United States or France. It is noted that certain foods are attributed to several nationalities. Note that the top 5 nationalities account for about 95% of food.

# ### II. Bivariate & Multivariate analysis

# In[ ]:


def boxplot_multiv (feature,plotColor="#CC9900"):
    """
    Generate boxplot from nutrition_grade_fr and a given feature
    """
    plt.figure(figsize=(15, 4)) 
    plt.rc('grid', color='#202020') 
    plt.rc('axes', facecolor='black')
    plt.rc('text', color='black')
    sns.boxplot(data=Dnutri_Nano, x="nutrition_grade_fr",y=feature, color=plotColor)


# In[ ]:


# bivariate boxplot
boxplot_multiv("energy_100g")
boxplot_multiv("sugars_100g","#33CCFF")
boxplot_multiv("fat_100g","yellow")
boxplot_multiv("additives_n","purple")
plt.ylim (0,10)
boxplot_multiv("fiber_100g","green")
plt.ylim (0,15)
boxplot_multiv("salt_100g","white")
plt.ylim (0,5)
boxplot_multiv("proteins_100g","red")
plt.ylim (0,30)


# The distributions are obtained according to the nutritional grade.
# 1. Energy, it is no surprise that average an annotated food "a" (good quality) has little calorie and conversely an annotated food "has" much more. Overall, this hierarchy is generally respected.
# 2. Sugar, Fat and Salt, follow the same logic. This is what you would expect.
# 3. Additives, the case of the additive is less obvious there does not seem to have a clear difference between the categories not set for the grade "a" where the food mostly do not have additives.
# 4. Fiber, the case of fibers is interesting, this time it seems that the more a food is advised "d", "e" minus it has fiber. Food "a" has on average more fiber than the rest.
# 5. Proteins, Finally the proteins do not seem to dicriminate a particular nutritional grade. It's not totally odd, a holy food like bad little contain a lot or a lot of protein.

# In[ ]:


corr = Dnutri_Nano.corr()
corr = corr.round(1)
plt.figure(figsize=(10, 9))
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

sns.set(font_scale=1.2)
with sns.axes_style("white"):
    sns.heatmap(corr,  annot = True ,vmax=1, cmap="RdBu_r",square=True, mask=mask)


# A heatmap corr, to interpret the features correlations .
# Strong expected correlations such as salt and sodium, the both features are very similar, one will keep only one to facilitate the learning phase.
# Sugars and carbohydrates as expected. Calories, sugars, fats are strongly correlated.
# 
# Some features have no correlation and seem to be isolated from the phenomenon, especially metals (iron and calcium) and vitamins. We decide to reduce the dimensions by removing them.

# In[ ]:


l = ["vitamin-c_100g","vitamin-a_100g","iron_100g","calcium_100g","cholesterol_100g",
    "salt_100g"]

for i in range(0,len(l)):
    Dnutri_Nano = Dnutri_Nano.drop(l[i],1)


# **Nationality Impact**

# I want to study the impact of nationality on nutritional grade. To do this, I need to make a contingency table and perform a chi-square test.

# In[ ]:


usa = Dnutri_Nano[Dnutri_Nano["countries_en"]== "United States"]['nutrition_grade_fr'].value_counts()
fr = Dnutri_Nano[Dnutri_Nano["countries_en"]== "France"]['nutrition_grade_fr'].value_counts()
swi = Dnutri_Nano[Dnutri_Nano["countries_en"]== "Switzerland"]['nutrition_grade_fr'].value_counts()
germ = Dnutri_Nano[Dnutri_Nano["countries_en"]== "Germany"]['nutrition_grade_fr'].value_counts()


# In[ ]:


# Create contingence dataframe
contingence = {'USA' : pd.Series(usa, index=usa.index),
               'France' : pd.Series(fr, index=fr.index),
               'Suisse' : pd.Series(swi, index=swi.index),
               'Germany' : pd.Series(germ, index=germ.index)
              }
# Contingence table
khi_cont = pd.DataFrame(contingence)
khi_cont


# In[ ]:


khi_array = khi_cont.as_matrix(columns=None)

chi2, pvalue, degrees, expected = chi2_contingency(khi_array,correction=True)
expected


# In[ ]:


chi2, degrees, pvalue


# P-value = 4e-34, Nationality would have a significant effect on nutrition grade.

# Now to process the row with severals nationalities,I reused a solution proposed in the following kernel :
#     
# https://www.kaggle.com/nadiinchi/visualizing-data

# In[ ]:


def splitDataFrameList(df, target_column, separator):
    '''Split rows with several countries
    '''
    def splitListToRows(row, row_accumulator, target_column, separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df

Dnutri_Nano = splitDataFrameList(Dnutri_Nano, "countries_en", ",")


# ### III. Features Engineering

# First we would like to encode our categorical feature.

# In[ ]:


index_list = Dnutri_Nano["countries_en"].value_counts().index
index_list = index_list.drop(['United States', 'France', 'Switzerland', 
                              'Germany'])
for i in index_list:
    Dnutri_Nano["countries_en"].replace({i : "Other_Pays" }, inplace=True)


# In[ ]:


#One hot Encoding
Dnutri_Nano = pd.get_dummies(Dnutri_Nano, columns=["countries_en"], prefix=["From"])


# **Can we create new features that impact the issue ?**

# In[ ]:


badFood =  ((Dnutri_Nano["sugars_100g"]) + (Dnutri_Nano["sodium_100g"]*5) + (Dnutri_Nano["saturated-fat_100g"])) / (Dnutri_Nano["fiber_100g"]+0.1)
goodFood =  (Dnutri_Nano["fiber_100g"])/(Dnutri_Nano["saturated-fat_100g"]+0.1)

Dnutri_Nano["badFood"]= badFood
Dnutri_Nano["goodFood"]= goodFood

for i in ["badFood","goodFood"]:
    if i == "badFood":
        colordist = "red"
    else:
        colordist = "green"
    plt.rcParams['axes.facecolor'] = 'black'
    plt.rc('grid', color='#202020')
    sns.distplot(Dnutri_Nano[i], kde=True, color=colordist)
    plt.show()


# Let us check if our two new variables are correlated with the target variable "nutritional score".

# In[ ]:


sns.set(font_scale=4)
flatui = ["green", "#99FF33", "#FFFF33", "#FF6600", "#FF0000"]
plt.rcParams['axes.facecolor'] = 'black'
plt.rc('grid', color='#202020')
sns.pairplot(Dnutri_Nano[["goodFood","badFood","nutrition-score-fr_100g","nutrition_grade_fr"]],
             hue="nutrition_grade_fr", diag_kind="kde",size =12, palette=flatui)


# In[ ]:


corr = Dnutri_Nano[["goodFood","badFood","nutrition-score-fr_100g"]].corr()
corr = corr.round(1)
plt.figure(figsize=(7, 6))

sns.set(font_scale=1.3)
with sns.axes_style("white"):
    sns.heatmap(corr,  annot = True ,vmax=1, cmap="BrBG",square=True)


# **Dimensionality reduction**

# We want to visualize the impact of our features, for which we choose to perform a PCA.

# In[ ]:


# little preprocessing
headers = list(Dnutri_Nano)
index = Dnutri_Nano["product_name"]
nutrigrade = Dnutri_Nano["nutrition_grade_fr"]
D = Dnutri_Nano.drop("nutrition_grade_fr",1)
D = D.drop("product_name",1)
D = D.drop("nutrition-score-fr_100g",1)


# The Data must be standardized.

# In[ ]:


# Data Standardizing
std_scale = preprocessing.StandardScaler().fit(D)
nutri_scaled = std_scale.transform(D)


# In[ ]:


# Run PCA
pca = decomposition.PCA()
pca.fit(nutri_scaled)
print (pca.explained_variance_ratio_)


# In[ ]:


print (pca.explained_variance_ratio_.cumsum())


# The first component catch 17% of the variance. 80% of the variance can be explained with the 8 first components.

# In[ ]:


plt.figure(figsize=(12, 7))
sns.set(font_scale=2)
plt.rcParams['axes.facecolor'] = 'black'
plt.rc('grid', color='#202020')

plt.step(range(16), pca.explained_variance_ratio_.cumsum(), where='mid',color="#66FFFF")
sns.barplot(np.arange(1,17),pca.explained_variance_ratio_,palette="PuBuGn_d")


# For the choice of the components, we use the Kaiser method, that is to say that we retain the axes whose inertia is greater than the mean inertia (here 1/16), this is the case for the first 8 axes. The Kaiser method therefore recommends 8 axes here.

# In[ ]:


pca2 = decomposition.PCA(n_components=8)
pca2.fit(nutri_scaled)


# **We can visualize our data on 2 principals components**

# In[ ]:


pcs = pca2.components_
def PCA_plot (components,comp1,comp2):
    plt.figure(figsize=(12, 12))
    for i, (x, y) in enumerate(zip(components[comp1, :], components[comp2, :])):
        # Display origine segment (x, y)
        plt.plot([0, x], [0, y], color='#00FFFF')
        
        plt.text(x, y, D.columns[i], fontsize='12', color='#FFFF99')

    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

# To assign each individual a color corresponding to their nutrition grade 
# This will be useful when vizualing
conv = Dnutri_Nano["nutrition_grade_fr"].replace('a', "green")
conv = conv.replace('b', "#99FF33")
conv = conv.replace('c', "#FFFF33")
conv = conv.replace('d', "#FF6600")
conv = conv.replace('e', "#FF0000")
Dnutri_Nano['nutrigrade_num'] = pd.Series(conv, index=Dnutri_Nano.index)

def scatterP_c (x,y):
    plt.figure(figsize=(12, 12))
    X_projected = pca2.transform(nutri_scaled)

    plt.scatter(X_projected[:, x], X_projected[:, y],
    c=Dnutri_Nano.get('nutrigrade_num'))
    plt.xlim([-15, 40])
    plt.ylim([-25, 40])
    plt.rcParams['axes.facecolor'] = 'k'
    plt.rc('grid', color='#202020')


# Some vizualisation

# In[ ]:


scatterP_c(0,1)
PCA_plot(pcs,0,1)


# We can see that some features have small impact on principal compenent.
# We will delete them.

# In[ ]:


D = D.drop("From_Germany",1)
D = D.drop("From_Switzerland",1)
D = D.drop("From_Other_Pays",1)


# ## Modeling and Prediction

# ### I. KNN model

# In[ ]:


#Sampling
NutriSpl = (random.sample(list(D.index),150000))

data_entry = D.loc[NutriSpl]
#data_entry = data_entry.drop("nutrigrade_num",1)
data_target = nutrigrade.loc[NutriSpl]


# In[ ]:


print(data_entry.shape)
print(data_target.shape)


# We need to determine the k optimal for KNN

# In[ ]:


#Testing set / Trainning set
Xtrain, Xtest, ytrain, ytest = train_test_split(data_entry, data_target, train_size=0.8,random_state=1)
# Training with KNN
# Optimisation du score
error_list = []
for k in range(2,15):
    knn = neighbors.KNeighborsClassifier(k) # training !
    error_list.append(100*(1-knn.fit(Xtrain, ytrain).score(Xtest,ytest))) # compute error on testing set
    
# Display KNN performence in term of K
plt.figure(figsize=(15, 15))
plt.plot(range(2,15), error_list,'go-', markersize =8)
plt.ylabel('error (%)')
plt.xlabel('k')
plt.show()


# **k = 3 is the best parameter**

# In[ ]:


NutriSpl = (random.sample(list(D.index),230000))

data_entry = D.loc[NutriSpl]
data_target = nutrigrade.loc[NutriSpl]

Xtrain, Xtest, ytrain, ytest = train_test_split(data_entry, data_target, train_size=0.8,random_state=1)

# Training  KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(Xtrain, ytrain) 
error = (1 - knn.score(Xtest, ytest))*100  # compute error on testing set
error


# **Error rate : 22 %**

# **Prediction**

# In[ ]:


# Apply the same treatment on our testing set.
Dnutri_score_less = splitDataFrameList(Dnutri_score_less, "countries_en", ",")

index_list = Dnutri_score_less["countries_en"].value_counts().index
index_list = index_list.drop(['United States','France', 'Switzerland','Germany'])
for i in index_list:
    Dnutri_score_less["countries_en"].replace({i : "Other_Pays" }, inplace=True)
Dnutri_score_less = pd.get_dummies(Dnutri_score_less, columns=["countries_en"], prefix=["From"])

Dnutri_score_less = Dnutri_score_less.drop("vitamin-c_100g",1)
Dnutri_score_less = Dnutri_score_less.drop("vitamin-a_100g",1)
Dnutri_score_less = Dnutri_score_less.drop("iron_100g",1)
Dnutri_score_less = Dnutri_score_less.drop("calcium_100g",1)
Dnutri_score_less = Dnutri_score_less.drop("cholesterol_100g",1)
Dnutri_score_less = Dnutri_score_less.drop("salt_100g",1)
Dnutri_score_less = Dnutri_score_less.drop("ingredients_from_palm_oil_n",1)
Dnutri_score_less = Dnutri_score_less.drop("product_name",1)
Dnutri_score_less = Dnutri_score_less.drop("nutrition-score-fr_100g",1)
Dnutri_score_less = Dnutri_score_less.drop("nutrition_grade_fr",1)
Dnutri_score_less = Dnutri_score_less.drop("trans-fat_100g",1)
Dnutri_score_less = Dnutri_score_less.drop("From_Germany",1)
Dnutri_score_less = Dnutri_score_less.drop("From_Switzerland",1)
Dnutri_score_less = Dnutri_score_less.drop("From_Other_Pays",1)

badFood =  ((Dnutri_score_less["sugars_100g"]) + (Dnutri_score_less["sodium_100g"]*5) + (Dnutri_score_less["saturated-fat_100g"])) / (Dnutri_score_less["fiber_100g"]+0.1) 
goodFood =  (Dnutri_score_less["fiber_100g"])/(Dnutri_score_less["saturated-fat_100g"]+0.1)

Dnutri_score_less["badFood"]= badFood
Dnutri_score_less["goodFood"]= goodFood

dd = Dnutri_score_less[D.columns.tolist()]


# In[ ]:


# Prediction
knn.predict(dd)


# In[ ]:


Dnutri_score_less["nutrition_grade_fr"] = knn.predict(dd)
Dnutri_score_less.head(5)


# ## Conslusion

# Thank you for reading. 
# All feedback is very welcome! :)
