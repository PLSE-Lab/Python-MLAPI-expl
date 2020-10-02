#!/usr/bin/env python
# coding: utf-8

# Check out my website here: https://expail.weebly.com

# # King County House Sales Analysis

# The goal of this house sales analysis is to gain some business insights from the house sales in King County.
# First, let's clean the data so we can frame appropiate questions.

# ## Loading, Viewing, & Cleaning Data

# ### Loading Data

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')


# ### Viewing First Version of Data

# In[ ]:


data.head()


# We need to remove the ID:

# In[ ]:


data.drop('id',axis=1,inplace=True)


# And make sure the date is properly formatted into year:

# In[ ]:


def get_year(date):
    return int(str(date)[:4])
data['year'] = data['date'].apply(get_year)
data.drop('date',axis=1,inplace=True)


# Final newly formatted head:

# In[ ]:


data.head()


# Now we can formulate some business questions!

# ## Which features contribute the most to the final price?

# ### Creating a Model

# #### Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X = data.drop('price',axis=1)
y = data['price']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# #### Training Models

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

lr = LinearRegression()
lr.fit(X_train, y_train)

rr = Ridge(alpha=0.01)
rr.fit(X_train, y_train)

rr100 = Ridge(alpha=100) #  comparison with alpha value
rr100.fit(X_train, y_train)

lasso = Lasso()
lasso.fit(X_train,y_train)

dtr = DecisionTreeRegressor()
dtr.fit(X_train,y_train)

mlpr = MLPRegressor()
mlpr.fit(X_train,y_train)


# #### Get Accuracies

# In[ ]:


from sklearn.metrics import mean_absolute_error as mae
#lr, rr, rr100, lasso
print("LR")
print(mae(y_test,lr.predict(X_test)))
print("RR")
print(mae(y_test,rr.predict(X_test)))
print("RR100")
print(mae(y_test,rr100.predict(X_test)))
print("LASSO")
print(mae(y_test,lasso.predict(X_test)))
print("Decision Tree Regression")
print(mae(y_test,dtr.predict(X_test)))
print("Neural Network Regression")
print(mae(y_test,mlpr.predict(X_test)))


# Decision Tree Regression is the best regression model.

# Now that we have a good model, let's use two techniques to find the importance of features - Permutation Importance and SHAP.

# ### SHAP

# In[ ]:


import shap

explainer = shap.TreeExplainer(dtr)
shap_values = explainer.shap_values(X_test)


# In[ ]:


shap.summary_plot(shap_values, X_test, plot_type="bar")


# ### Permutation Importance

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(dtr, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())


# From these two tests, we can confidently say that the most important features are:
# - lat & long (location)
# - grade
# - sqft_living
# 
# 
# Let's graph these out using a PDP (Partial Dependence Plot) to see how their values affect house price.

# ### PDPs for grade, sqft_living, lat & long

# In[ ]:


from pdpbox import pdp, info_plots
import matplotlib.pyplot as plt

base_features = data.columns.values.tolist()
base_features.remove('price')

feat_name = 'sqft_living'
pdp_dist = pdp.pdp_isolate(model=dtr, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()

feat_name = 'grade'
pdp_dist = pdp.pdp_isolate(model=dtr, dataset=X_test, model_features=base_features, feature=feat_name)

pdp.pdp_plot(pdp_dist, feat_name)
plt.show()


# Analysis:
# 
# sqft_living - the largest jump is somewhere between 1200 and 1500 square feet. However, this isn't very evident though - it's pretty linear.
# 
# grade - the largest jump is from grade 8 to grade 9. This can increase the house price by almost 175,000 dollars.

# In[ ]:


inter1  =  pdp.pdp_interact(model=dtr, dataset=X_test, model_features=base_features, 
                            features=['sqft_living', 'grade'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['sqft_living','grade'], plot_type='contour')
plt.show()


# In[ ]:


inter1  =  pdp.pdp_interact(model=dtr, dataset=X_test, model_features=base_features, 
                            features=['lat', 'long'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['lat','long'], plot_type='contour')
plt.show()


# Wow, there seems to be a pretty clean-cut difference in price value by location! Let's graph it out on a map. Where are the most expensive or cheap houses?

# In[ ]:


import folium
from folium.plugins import HeatMap

# Create basic Folium crime map
crime_map = folium.Map(location=[47.5112,-122.257], 
                       tiles = "Stamen Terrain",
                      zoom_start = 9)

# Add data for heatmp 
data_heatmap = data[['lat','long','price']]
data_heatmap = data.dropna(axis=0, subset=['lat','long','price'])
data_heatmap = [[row['lat'],row['long']] for index, row in data_heatmap.iterrows()]
HeatMap(data_heatmap, radius=10, 
        gradient = {.35: 'blue',.55: 'purple',.68:'lime',.78:'red'}).add_to(crime_map)
# Plot!
crime_map


# The price goes up in this order: blue, purple, lime, red.
# We've got some interesting insights into which places have the most expensive houses. The Seattle area has the strongest red (and therefore the highest prices). This makes sense; houses in Seattle are more expensive than the surrounding area.

# ### PDPs for All Factors

# Although some factors are not very important, in terms of return over investment it performs higher. Let's do a PDP on all the columns.

# In[ ]:


for column in data.columns.drop('price'):
    feat_name = column
    pdp_dist = pdp.pdp_isolate(model=dtr, dataset=X_test, model_features=base_features, 
                               feature=feat_name)

    pdp.pdp_plot(pdp_dist, feat_name)
    plt.show()


# **Bedrooms:** The amount of bedrooms doesn't matter very much, although variability is very high.
# 
# **Bathrooms:** The amount of bathrooms increases a little bit per additional bathroom, but only after 3 bathrooms.
# 
# **Square Feet of Living Space:** The amount of square feet of living space increases a little bit for the first 2100 square feet, then rises rather significantly from 2100 square feet to 3000 square feet, then rises only a little bit per addition square feet of living.
# 
# **Square Feet Lot:** The price increases drastically for the first 20,000 square feet, and then doesn't increase very much after this.
# 
# **Floors:** The amount of floors doesn't matter very much (although variability is high).
# 
# **Waterfront:** Having a waterfront drastically increases the house price by 500,000.
# 
# **View:** As the view score increases, the house price increases, in a linear fashion.
# 
# **Condition:** The condition only starts to increase the house price semi-significantly after the condition reaches 3.0.
# 
# **Grade:** The grade only starts to increase the house price significantly after a grade of 7.
# 
# **Square Feet Above:** The square feet is pretty linear, not increasing the house price by very much.
# 
# **Square Feet Basement:** The square feet of the basement does not impact the house price at all, all though variability is relatively high.
# 
# **Year Built:** It seems that houses built before 1930 have about the same house price as ones built after (about) 2015, and that there is a dip in between that lowers the house price.
# 
# **Zip Code:** Although Zip Code cannot be treated as a linear variable and the number itself as a cause of price difference, in general the higher the zip code number, the less the price is.
# 

# In[ ]:


data.head()


# In[ ]:


inter1  =  pdp.pdp_interact(model=dtr, dataset=X_test, model_features=base_features, 
                            features=['sqft_living', 'sqft_lot'])

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['sqft_living','sqft_lot'], 
                      plot_type='contour')
plt.show()


# In[ ]:


def heart_disease_risk_factors(model, patient):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient)
    shap.initjs()
    return shap.force_plot(explainer.expected_value, shap_values, patient)

data_for_prediction = X_test.iloc[8,:].astype(float)
heart_disease_risk_factors(dtr, data_for_prediction)


# In[ ]:


shap_values = explainer.shap_values(X_train.iloc[:50])
shap.force_plot(explainer.expected_value, shap_values, X_test.iloc[:50])


# In[ ]:




