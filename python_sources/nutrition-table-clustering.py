#!/usr/bin/env python
# coding: utf-8

# # Introduction

# In this Kernel we want to look closer at the data collected by the App "OpenFoodFacts" (https://www.kaggle.com/openfoodfacts/world-food-facts). This is an free app where you can scan your groceries to see their ingredients and nutritions. This can be really helpful if you can't decide whether you should buy some kind of product or not while being in the supermarket.
# Before the app can show you information about a product it has to be scanned and registered by some other user. Just like that you can also register products in the databank if you one that is not included yet.
# 
# But there is a problem with this app: You have to type in the ingredient list and the nutrition table yourself. This is really time consuming and annoying for the user. Thereby you can easily get wrong entries because the user doesn't want to type in the values anymore and starts to type in wrong entries like 0g instead of the real value, but you can also get typing errors like turned numbers.
# You also have to type in the categories of a product by yourself. There are also so many categrories that kind of mean the same that this process gets even more complicated and time-consuming.
# 
# Therefore it would be great if the app could detect strange looking entries and could then ask the user himself or the next user if these values are really correct; and it would be also useful if the app could predict the category of a product based on its values.
# 
# We now want to work out a model that can predict these outliers and can cluster the data in categories. We will use the nutrition-table for this task because it is easier to enter that than to implement the ingredients-list with it complex additives-names. 

# # Getting started

# First we have to import all important libraries and of course the OpenFoodFacts-data.

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

from sklearn.mixture import GaussianMixture
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

import os
print(os.listdir("../input"))


# Now that we have don that let's have a look at our data:

# In[ ]:


original = pd.read_csv("../input/en.openfoodfacts.org.products.tsv",delimiter='\t', encoding='utf-8', nrows = 50000)
original.head()


# As we can see there is a lot of missing data, some columns even seem to have no entries at all. There are also some columns like the "url" one we don't really need for our further procedure.
# Because of that we have to clean our data before starting our analysis:

# # Cleaning the data

# ### Drop features with high percentage of missing values:

# First we want to clean the columns with a really high amount of missing values because these are really hard to fill up and our model can't work with NaN-data.
# Therefore we now want to look at the distribution of missing features:

# In[ ]:


nan_values = original.isnull().sum()
nan_values = nan_values / original.shape[0] *100

plt.figure(figsize=(20,8))
sns.distplot(nan_values, kde=False, bins=np.int(original.shape[0]/100), color = "Red")
plt.xlabel("Percentage of nan_values", size = 15)
plt.ylabel("Frequency", size = 15)
plt.title("Missing values by feature", size = 20)


# As we can see there is a high amount of columns with almost no entires at all. There is also a little peak at about 20-30% of nan_values; we don't want to keep these features because they're missing too many values.
# The data with low missing values can be found at about 10% of nan_values and lower. So let's take 10% as a limit for deleting the affected features:

# In[ ]:


def split_data_by_nan(cutoff):
    cols_of_interest = nan_values[nan_values <= cutoff].index
    data = original[cols_of_interest]
    return data.copy()

low_nan_data = split_data_by_nan(10)

print("Original number of features: " + str(original.shape[1]))
print("Number of features with less than 10 % nans: " + str(low_nan_data.shape[1]))


# There are only 29 features with less than 10% of missing values left. This sounds way better than the 163 before!
# Let's see which features survived our first round of cleaning:

# In[ ]:


low_nan_data.columns


# We can see that there is still a good amount of important features! But there are also some features we probably don't really need for our further analyis.
# 
# So now let's only take the following features:
# 
# *  fat_100g
# * carbohydrates_100g
# * sugars_100g
# * proteins_100g
# * salt_100g
# * energy_100g
# 
# These seem to be the most important features for our purpose.

# In[ ]:


nutrition_table_cols = ["fat_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g", "salt_100g", "energy_100g"]
nutrition_table = low_nan_data[nutrition_table_cols].copy()


# ## Dropping products with incomplete nutrition table:

# Now we can almost start our analysis!
# But before we can do that we have to take into account that we still have some missing values in our data.
# Let's see how many products don't have the information for all features:

# In[ ]:


nutrition_table["isempty"] = np.where(nutrition_table.isnull().sum(axis=1) >= 1, 1, 0)
percentage = nutrition_table.isempty.value_counts()[1] / nutrition_table.shape[0] * 100
print("Percentage of incomplete tables: " + str(percentage))


# So there are about 11% of not completed product-information.
# Let's delete these products now so our data is complete and ready for our analysis.

# In[ ]:


nutrition_table = nutrition_table[nutrition_table.isempty==0].copy()
nutrition_table.isnull().sum()


# In[ ]:


nutrition_table.drop("isempty", inplace=True,axis=1)


# # Reconstructing the energy

# Now we have our cleaned data we really can work with!
# But before we start we want to look closer at the individual features. We know that the energy a product contains can mainly be calculated through his amount of carbs, fat and proteins.
# We also know the following:
# 1g of fat contains about 39 kJ of energy;  1g of carbohydrates and proteins both contain about 17 kJ of energy .
# 
# Because of the complicated input-process in the OpenFoodFacts app we already said that some users start to type in wrong values. The energy is also often given in kJ and kcal, so the users can get even more confused. By calculating the energy based on the features fat, carbohydrates and proteins and comparing it to the given value of energy we can detect if some entries might be wrong and could probably even correct them in some cases:

# In[ ]:


nutrition_table["reconstructed_energy"] = nutrition_table["fat_100g"] * 39 + nutrition_table["carbohydrates_100g"] * 17 + nutrition_table["proteins_100g"] * 17

nutrition_table.head()


# It seems like the reconstructed energy is almost matching the given amout of energy. But let's look at a plot of both amounts of energy for a better conclusion:

# In[ ]:


plt.figure(figsize = (10,10))
plt.scatter(nutrition_table["energy_100g"], nutrition_table["reconstructed_energy"], s = 0.6, c= "goldenrod")
plt.xlabel("given energy")
plt.ylabel("reconstructed energy")


# We can see that both amounts of energy are mostly close to each other or identical, but there are still some information that don't fit that good, because they differ from the linear interrelation we can see. Everything that differs really greatly from the straight line we can see seems to be an outlier too.
# 
# There are some products with a given amount of energy of 0 kJ while the reconstructed energy is way higher. This means that there are values of fat, carbohydrates and proteins higher that 0g given, so that the given amount of energy is probably wrong. 
# On the other hand there are also amounts of reconstructed energy which equals to 0 kJ, while the given amount energy is higher. In these cases the values of fat, carbohydrates and proteins in our data have to be 0g, but they are probably higher in reality.
# 
# Because of these findings it would be good if we also consider the reconstructed energy for our clustering:

# # Another obvious mistake
# 
# Apart from the mistakes we detected through our reconstructed energy we can also easily detect another mistake: Our features fat, carbohydrates and proteins are given based on 100g. So if the sum of these features is higher than 100 we would also know that there is something wrong with our given data.
# 
# So let's see if we can find any of these mistakes:

# In[ ]:


nutrition_table["g_sum"] = nutrition_table.fat_100g + nutrition_table.carbohydrates_100g + nutrition_table.proteins_100g

nutrition_table["exceeded"] = np.where(nutrition_table.g_sum.values > 100, 1, 0)


# In[ ]:


nutrition_table[nutrition_table["exceeded"] == 1].head()


# So as we can see there are really some products registered with this mistake. As we can see below there are 94 products with this mistake in our data alone. This really seems to be a huge source of errors.

# In[ ]:


nutrition_table.exceeded.value_counts() 


# In[ ]:


nutrition_table["product"] = original.loc[nutrition_table.index.values]["product_name"] 


# In[ ]:


nutrition_table.to_csv("nutrition_table.csv", header=True, index=True) 


# # Looking at our given features
# 
# In the plot below you can see the distribution of our features. You can choose which feature you want to look at by changing the name of the string in the feature-list below.
# You can choose between the following features:
# * fat_100g
# * carbohydrates_100g
# * sugars_100g
# * proteins_100g
# * salt_100g
# * energy_100g
# * reconstructed_energy
# * g_sum
# 

# In[ ]:


feature = "g_sum"


# In[ ]:


colors_dict = {"fat_100g": "lightskyblue", "carbohydrates_100g": "limegreen", "sugars_100g": "hotpink", "proteins_100g": "mediumorchid", "energy_100g": "gold", "salt_100g": "gray", "reconstructed_energy": "orange", "g_sum": "m"}


# In[ ]:


plt.figure(figsize=(20,5))
sns.distplot(nutrition_table[feature], kde = False, color = colors_dict[feature])
plt.xlabel(feature)
plt.ylabel("frequency")


# 
# 
# As we can see all distributions have its highest peak at about 0g (or a bit higher) per 100g which seems to be a bit odd at first sight. This is probably coming from the fact that not all products include all of the nutritions above. For example: products with a high amount of carbohydrates (e.g. pasta or rice) often don't contain much fat, products with a high amount of fat(e.g. bacon) often don't contain much carbohydrates neither.
# 
# Now let's have a closer look at the individual distributions:
# 
# **Carbohydrates** seem to have the widest distribution: Every amount of carbohydrates between 0g and 100g can be found in products, while most products contain up to 15-20g carbohydrates per 100g. There is also a small increase between 50g and 70g. This wide distribution makes sense becauseso many product contain at least a little bit of sugar (a type of carbohydrates), but there are also some products like pasta that almost only consits of carbohydrates.
# 
# The only other distribution that reaces 100g per 100g is the distribution of **fat**, but here we can see that most products contain almost no fat and products with an higher amount of fat are way more uncommon. This is really surprising because it always seems that most of our groceries contain way too much fat than it should.
# 
# The **protein**-distribution shows that the occurrence of protein in product is also rather low. The highest amout is about 50g per 100g and most products contain an amount of protein lower than 30g per 100g. This makes sence because there are not that many products with an high amount of protein on the market. Peer groups that consume large quantities of protein buy special products which are not that common in the "normal" grocery-stores so that they don't appear in the dataset of Open Food Facts that often.
# 
# The distribution of **salt** has only one high peak at about 0g per 100g. This occurs because almost every product only contains up to about 1g of salt per 100g.
# 
# A bit different is the distribution of **energy**. Though we measure the amount of energy per 100g, energy is not measured in grams but in kJ. Because of that we can't really compare it to the other distributions. What we can see is that the distribution includes two pikes - one at 0 calories and one at about 1500 kJ. Between these two pikes the distribution describes something similar to a parabel. The maximum amount seems to be at about 4500 kJ., but most products only contain up to 2500 kJ.
# The distribution of our **reconstructed energy** is really similar, but by comparing both distributions we can see some small differences (like the height of some individual bars) just like we discovered in the scatterplot above.
# 

# # Gaussian Mixture Model

# For our analysis we only want to use the features we already selected above.
# Now we have to decide how many clusters we are going to expect:
# There will probably be clusters for fruits, vegetables, grain products, sweets, cookies & cakes, beverages, meat, sausages, fish, milk products, oils, nuts&seeds and probably some more.
# So we want our model to make 15 clusters because that seems to be a good amount for our data:

# In[ ]:


features = ["fat_100g", "carbohydrates_100g", "sugars_100g", "proteins_100g", "salt_100g", "energy_100g", "reconstructed_energy", "g_sum"]


# In[ ]:


X_train = nutrition_table[features].values

model = GaussianMixture(n_components=14, covariance_type="full", n_init = 5, max_iter = 200)
model.fit(X_train)
log_prob = model.score_samples(X_train)
results = nutrition_table[features].copy()
results["cluster"] = model.predict(X_train)
results["product_name"] = original.loc[nutrition_table.index.values, "product_name"]

probas = np.round(model.predict_proba(X_train), 2)
cluster_values = results.cluster.values
certainty = np.zeros(cluster_values.shape[0])
for n in range(len(certainty)):
    certainty[n] = probas[n,cluster_values[n]]
    
results["certainty"] = certainty


# As we can see our model clustered our products. In the column "certainty" we can also see how sure our model is by predicting the affilation of a product to one cluster.
# 
# If you want to you can also have a closer look at what kind of products were clustered together by changing the first number in the code line below from 0 to 14 and changing the second number to view more or less products:

# In[ ]:


results[results.cluster == 2].head(15)


# # Anomaly Detection

# ## Choosing the outlier treshold:

# Before we can ask our model to find outliers in our data we have to tell him hom much the affected data has to differ from the normal data. We want to do this by choosing a suitable epsilon.
# Therefore we want to visualize the log probabilities of our data so we can decide how we want to choose this epsilon.
# I chose the 12%-percentile because it divides the data where the log-probability falls sharply.
# 
# You can look at different percentiles by changing the amount of your_choice below. It's the light purple line in the plot below.

# In[ ]:


def get_outliers(log_prob, treshold):
    epsilon = np.quantile(log_prob, treshold)
    outliners = np.where(log_prob <= epsilon, 1, 0)
    return outliners 


# In[ ]:


your_choice = 0.12


# In[ ]:


plt.figure(figsize=(20,5))

sns.distplot(log_prob, kde=False, bins=50, color="Red")
g1 = plt.axvline(np.quantile(log_prob, 0.25), color="Green", label="Q_25")
g2 = plt.axvline(np.quantile(log_prob, 0.5), color="Blue", label="Q_50 - Median")
g3 = plt.axvline(np.quantile(log_prob, 0.75), color="Green", label="Q_75")
g4 = plt.axvline(np.quantile(log_prob, your_choice), color="Purple", label="Q_ %i" % (int(your_choice*100)))
handles = [g1, g2, g3, g4]
plt.xlabel("log-probabilities of the data spots")
plt.xlim((-50,0))
plt.ylabel("frequency")
plt.legend(handles) 


# ## Detecting the outliers:

# Now we can finally detect the outliers in our data. In the column "anomaly" you can see if a product is an outlier if the value of this feature is 1. If the value is 0 we know that this product is not an outlier:

# In[ ]:


results["anomaly"] = get_outliers(log_prob, your_choice)
results.head()


# ## Visualize abnormal data:

# In the plot below you can see the scatterplot of two of our featues. You can again change the features by changing the strings in the features-list.
# The plot also shows which products were labeled as outliers (red) and which were labeled as normal (blue).

# In[ ]:


features = ["energy_100g", "reconstructed_energy"]


# In[ ]:


plt.figure(figsize = (10,10))
plt.scatter(results[features[0]], results[features[1]], c=results.anomaly.values, cmap = "coolwarm", s = 5.5)
plt.xlabel(features[0])
plt.ylabel(features[1])


# In[ ]:


results.to_csv("clustering_and_anomalies.csv", header=True, index=True) 


# ## Looking at the abnormal data

# Now we want to look at the products that were listed as outliers to see if there are any patterns we can detect:

# In[ ]:


results[results.anomaly == 1].head(50)


# In the table above we can see that there are mostly natural products like nuts, fruits, oats, grains, seasonings and even vanilla. That could come from the fact that such natural products often don't have completed nutrition tables or no nutrition-tables at all because these natural products vary so much in their nutritions.
# We can also see that there are some products containing chocolate or milk. In these cases both energys are mostly close to each other while the g_sum is higher than 100g in some cases.
