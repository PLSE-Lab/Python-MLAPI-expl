#!/usr/bin/env python
# coding: utf-8

# # **Which populations assessed should stay home and which should see an HCP?**

# Since this question only has a few submissions, I think this notebook will be able to add a new perspective. The question we are trying to answer is "Which populations assessed should stay home and which should see an HCP?" When viewing this research question this sparked the question which what would be a metric to measure which communities should see a HCP? I decided to use death rate to determine which populations should stay home versus see an HCP. <br/>
# Disclaimer: This analysis is only using data from the United States. However, I think some of the conclusions we can make can be generalized to other countries. 

# In[ ]:


#Imports for the notebook
import numpy as np
import pandas as pd
import seaborn as sns
# import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn import preprocessing
import plotly.graph_objects as go


# # **Importing the correct datasets**<br/>
# This analysis uses three main datasets:
# * The USA Facts Confirmed Cases dataset 
# * The USA Facts Deaths dataset
# * The us-county-health-rankings-2020.csv dataset provided by Kaggle
# 
# These USA Facts datasets are similar to the confirmed-covid-19-cases-in-us-by-state-and-county.csv and the confirmed-covid-19-deaths-in-us-by-state-and-county.csv dataset provided by Kaggle. However, the datasets used are from the usafacts.org website. These datasets were used because they contained the most up to date information. <br/>
# URL to USAFacts website with datasets: https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/

# In[ ]:


us_cases_county = pd.read_csv('/kaggle/input/usafacts-updated/covid_confirmed_usafacts_June1.csv')
us_deaths_county = pd.read_csv('/kaggle/input/usafacts-updated/covid_deaths_usafacts_June1.csv')

#Importing data on each county
health_by_county = pd.read_csv('/kaggle/input/county-health-rankings/us-county-health-rankings-2020.csv')
health_by_county.rename(columns={'fips': "countyFIPS"}, inplace=True)
health_by_county.head()


# # Data Preprocessing

# Once we read in the data, we want to get what is the latest date. This is because the data is the cumulative number of confirmed cases or deaths per county in the United States. That way we will be able to conduct our analysis with the most up to date data. We will filter out all the data except the data most recently available death and confirm cases number along with the county and state information. 

# In[ ]:


cum_cases_county = us_cases_county.filter(['countyFIPS', 'County Name', 'State', 'stateFIPS', '6/1/20'], axis=1)
cum_deaths_county = us_deaths_county.filter(['countyFIPS', 'County Name', 'State', 'stateFIPS', '6/1/20'], axis=1)


# We will need to filter the data by an arbitrary number of deaths, since death rates can be inaccurate when there have only been only a few cases or deaths in a given county. These counties will not be able to provide useful information and can mess up any models we are using. 

# In[ ]:


cum_deaths_county[cum_deaths_county['6/1/20'] >= 20].sort_values(by='6/1/20', ascending=False).shape


# Here we will create a new dataframe and create the new column for the death rate which to reiterate will be the metric used to determine in which populations people should visit an HCP if they have symptoms. 

# In[ ]:


#Changed names for some columns
cum_cases_county.rename(columns={'6/1/20': "Confirmed Cases"}, inplace=True)
cum_deaths_county.rename(columns={'6/1/20': "Deaths"}, inplace=True)
#Building new Dataframe 
cum_cases_deaths_county = pd.DataFrame(cum_cases_county)
cum_cases_deaths_county['Deaths']= cum_deaths_county['Deaths']
cum_cases_deaths_county['Death Rate'] = cum_cases_deaths_county['Deaths'] / cum_cases_deaths_county['Confirmed Cases']


# Now we can clean up the data and check to see where in the country, there are teh highest death rates. Below the cell will output the top ten highest death rates for counties with 20 or more deaths. 

# In[ ]:


cum_cases_deaths_county = cum_cases_deaths_county[(cum_cases_deaths_county['countyFIPS'] != 0) 
                                                  & (cum_cases_deaths_county['Death Rate'] <= 1)]
cum_cases_deaths_county[cum_cases_deaths_county['Deaths'] >= 20].sort_values(by='Death Rate', ascending=False).head(10)


# In[ ]:





# Let's look at the death rate by state.

# In[ ]:


death_rate_state = cum_cases_deaths_county.groupby(['State']).agg(
    {'Death Rate': 'mean'}).reset_index().sort_values(by='Death Rate', ascending=False)

death_rate_state.head()


# Below is a map of the death rate by state.

# In[ ]:


#Geographical map representation death rates
fig = go.Figure(data=go.Choropleth(locations=death_rate_state['State'],
                                   z=death_rate_state['Death Rate'].astype(float),
                                  locationmode='USA-states', 
                                  colorscale='Reds',
                                  colorbar_title='Death Rate of Covid-19'))
fig.update_layout(title_text='Average Death Rate of Covid-19 by State', 
                  geo_scope='usa')
fig.show()


# The data from the us-county-health-rankings-2020 dataset is incredibly extensive so some of the features would be needed to be filtered out. It originally had over 500 features, and after the feature selection process there were only a little over 100 features. Below is the process used to chose the features that will be used in the model. An example of features that were removed were something like teen birth rates. Some features didn't seem to have any obvious connection to the death rate. Features such as obesity, health, and smoking were of course kept for the model.

# In[ ]:


#Getting filtered county data by index
index_lst = []
index_lst.extend(range(7))
index_lst.extend(range(23,31))
index_lst.extend(range(55,71))
index_lst.extend(range(103,112))
index_lst.extend(range(120,122))
index_lst.extend(range(134,141))
index_lst.extend(range(163,167)) #Income
index_lst.extend(range(203,215)) #Housing
index_lst.extend(range(326,330)) #Food
index_lst.extend(range(371,382))
index_lst.extend(range(394,397))
index_lst.append(412) #reduced lunch
index_lst.extend(range(485,507))

#Total 106 features (columns)
simp_health_county = health_by_county.iloc[:, index_lst]
#Outputs our newly filtered dataframe
simp_health_county.head()


# Now the us-county-health-rankings-2020 dataset will be joined with the cumulative death rates dataframe where the county FIPS is the same.

# In[ ]:


#Now do a join on simp_health_county and death rate
county_data = pd.merge(simp_health_county, cum_cases_deaths_county, on='countyFIPS')
county_data.drop(['County Name', 'State'], axis=1, inplace=True)
county_data.head()


# In[ ]:


#Filter the data for more than 15 deaths (Gives us 448 samples (not much :( )
#10 or more gives us 561 samples 
min_deaths = 20
filtered_county_data = county_data[county_data['Deaths'] >= min_deaths]

#Add random column (used in feature importance)
filtered_county_data['random'] = np.random.random(size=len(filtered_county_data))

#Ouput Data
labels = pd.DataFrame(filtered_county_data['Death Rate'])

#Get input data
x_data = filtered_county_data.drop(['state',
                                    'county', 'Death Rate',
                                    'primary_care_physicians_ratio',
                                    'other_primary_care_provider_ratio',
                                    'Deaths','Confirmed Cases'], axis=1)
#Dealing with the NANS
#average of that column by state 
x_data.fillna(x_data.groupby(['stateFIPS']).transform('mean'), inplace=True)

#Only effective if the state has no value (Put in average for the entire column)
x_data.fillna(x_data.mean(), inplace=True)


#Train Test split 
x_train, x_test, y_train, y_test = train_test_split(x_data, labels, 
                                                    test_size=0.2, random_state=101)

#View our training input data
x_train.head()


# # Data Normalization

# In[ ]:


#Normalizing the x training data
scaler = preprocessing.MinMaxScaler()
x_train_norm = scaler.fit_transform(x_train)
x_train_norm_df = pd.DataFrame(x_train_norm, columns=x_train.columns)

#From Towards Data Science Article
#Normalizing x test data
scaler = preprocessing.MinMaxScaler()
x_test_norm = scaler.fit_transform(x_test)
x_test_norm_df = pd.DataFrame(x_test_norm, columns=x_test.columns)


# # Random Forest Regression Model

# To model the data, the input features will be the 104 columns from us-county-health-rankings-2020 dataset and the labels will be the death rates from each county calculated from the USAFacts datasets. The regression model used is the random forest regression model from SKLearn. 
# <br/>
# 

# In[ ]:


#Will use Random Forests
rf = RandomForestRegressor(n_estimators=400, max_features='sqrt', n_jobs=1, oob_score=True,
                           bootstrap=True, random_state=101)
model = rf.fit(x_train_norm_df, y_train.values.ravel())
print('R^2 Training Score: {:.2f}'.format(rf.score(x_train_norm_df, y_train)))
print('OOB Score: {:.2f}'.format(rf.oob_score_))
print('Validation Score: {:.2f}'.format(rf.score(x_test_norm_df, y_test)))


# # Feature Importance Exploration

# In this section of the notebook, I will use three seperate methods to determine feature importance. 
# 1. Impurity-based feature importance (built into SKLearn)
# 2. Permutation feature importance (built into SKLearn)
# 3. Linear regression coeficients for each feature and the labels
# 

# # Impurity-Based Feature Importance

# In[ ]:


#Using SKLearn feature importance
#Need feature names to be with feature_importances 
feature_importances_init = rf.feature_importances_
feature_importances = []
feat_cols = []
for i in range(feature_importances_init.shape[0]):
    if feature_importances_init[i] >= 0.01:
        feature_importances.append(feature_importances_init[i])
        feat_cols.append(x_train.columns[i])

#Convert lists to numpy arrays 
feature_importances = np.asarray(feature_importances)
feat_cols = np.asarray(feat_cols)

num_features = len(feature_importances)

#We want to sort the importances and in order to plot them
sorted_importances_indices = feature_importances.argsort()


# Plotting the feature importances in a horizontal bar graph

# In[ ]:


fig, ax = plt.subplots()
fig.set_figheight(15)
ax.barh(range(num_features), feature_importances[sorted_importances_indices], color='b', align='center')
ax.set_yticks(range(num_features))
ax.set_yticklabels(feat_cols)
ax.invert_yaxis()
ax.set_title("Covid-19 Death Rate Feature Importances")

plt.show()


# There is a lot we can get from this graph. However, it is important to note that the impurity-based feature importance may be deceptive when there are there are large amounts of unique categorical data. The other thing to keep in mind about the horizontal bar graph above is that these are the most important features in the random forest regression model. The model right now has an R squared, validation value of 0.41 which still needs to be substantially improved before the features in the graph can be trusted more. <br/>
# On the other hand, if we look at our most important features based on the graph, these features do not seem to far off of reality. Many of the counties with the highest death rates are from counties in more rural states such as Ohio, Louisiana, and Indiana. The next four highest rated features all have to do with the whether or not the county has proficient English speakers. This once again is not too far off of reality where immigrants often do not have the same accessability to relief options. Interestingly, it appears that the percentage of females in a county has a important role in the prediction of our model. After the English proficiency, it appears that race/ ethnicity, age distributions, insurance, and overcrowding all have an impact.

# # Permutation Feature Importance

# Permutation feature importance is the decrease in the score of the model when a specific feature is shuffled randomly. If there is a change in the model, this means that the feature that was randomly shuffled had an impact on the outcome of the model. There is the n_repeats parameter in the permutation_imporance function which determines how many times each feature is randomly shuffled.

# In[ ]:


r = permutation_importance(model, x_test, y_test, n_repeats=30, random_state=42)


# In[ ]:


permutation_df = pd.DataFrame(columns=['Feature', 'Importance Mean', 'Importance'])

for i in r.importances_mean.argsort()[::-1]:
    #Checking if it is within two standard deviations of the mean
    if (r.importances_mean[i] - 2 * r.importances_std[i]) > 0:
        importance_val = str(r.importances_mean[i]) + " +/- " + str(r.importances_std[i])
        permutation_df = permutation_df.append({'Feature': x_train.columns[i], 'Importance Mean': r.importances_mean[i],
                                                'Importance': importance_val}, ignore_index=True)

#Sorts the features in permutation_df from largest to smallest importance
permutation_df.sort_values(by='Importance Mean', ascending=False)


# This table above shows some of the similar information to the impurity-based feature importance analysis. Once again the number of non proficient English speakers is at the top. 95percent_ci_low_39 is the lower confidence interval for the number of nonproficient English speakers. This makes sense because people who are not proficient in the language are less likely to want to seek medical attention because of the language barrier. This goes along with many what has been in the news where immigrants have been hit hard by the pandemic. We also see that once again the percent of native hawaiian or other pacific islanders seems to have a large impact on the death rate. The percentage of native Americans also appears to be of some importance. Once again, this makes sense since native American reservations have been ravaged by Covid-19. However, this approach does not show the importance of a county being rural compared to the impurity-based approach. 
# <br/>
# Quick note: <br/>
# The features that appeared in this approach changed after normalizing the data with the MinMaxScaler approach. Before the normalization, one of the other important features was whether or not someone was insured. I wanted to include this because this feature made a lot of intuitive sense. These individuals who do not have health insurance may only seek the guidance of a health care professionals after their symptoms have worsened. Also, the other features pertaining to the percentage of speakers not proficient in English were also percieved to be important before I applied data normalization.

# # Linear Regression Feature Importance

# In[ ]:


def feature_lin_correlation(x_data, y_data):
    correlation_df = pd.DataFrame(columns=['Feature', 'Correlation'])
    for cols in x_data.columns:
        reg = LinearRegression()
        #Input and output data for linear regression
        x = x_data[cols].to_numpy()
        y = y_data.to_numpy()
        
        #Reshaping data
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        fitReg = reg.fit(x, y) 

        #Adds feature and its correlation value to the correlation dataframe
        correlation_df = correlation_df.append({'Feature': str(cols), 'Correlation': float(fitReg.coef_)}, 
                                               ignore_index=True)
    return correlation_df


# In[ ]:


correlation_df = feature_lin_correlation(x_data, labels)

#Orders the correlation dataframe from highest postive correlation to highest negative correlation
correlation_df.sort_values(by='Correlation', ascending=False)


# This output seems to be somewhat less helpful than the previous two feature importance methods as a result of the random feature. It still has a stronger correlation than many other values, but it is formed of all values randomly generated from zero to one. This means any value with a lower absolute value of the correlation value should not be included in important features. <br/>
# The average number of physically healthy days, 95percent_ci_high_3 (upper confidence interval for average number of physically healthy days), 95percent_ci_low_3 (lower confidence interval for the average number of physically healthy days) were all in the top five positive correlations. This shows that previous unhealthiness seems to be an important part of determining the death rate. Quartile 10 is the percent of people with access to exercise opportunities while quartile 9 is the percent of physically inactive people. The percentage of physically inactive people having a positive correlation makes sense because inactive people are more likely to have preexisting conditions.<br/>
# In terms of the negative correlation the only feature that seems to be important is the percentage of native hawaiian or other pacific islanders. It seems that this feature has a large impact on the death rate given its large negative coefficient.

# It is important to note that the impurity based and permutation feature based feature importance analysis is based on the random forest regressor model. These approaches only tell us the importance of each feature in the model based off of that model. Since the random forest regressor model does not necesarrily fit the data super well, it is important to know that the features that are the most important are not necessarily the most important features in determining the death rate.

# # Conclusion

# From the feature extraction analysis, we are able to see some features that may have a higher correlation with death rate than others. These are the features that will be most important for policy makers to keep in mind when trying to make decisions in the United States. If there are attributes of these communities that are leading to higher death rates, it is up to the others to realize that these are the populations that need to go to the hospital if they have any symptoms at all. Hopefully, by trying to focus on these features of counties in the United Sates, we will be able to help mitigate the impact on Covid-19 on communities that are yet to be ravaged by the virus. 

# # Next Steps

# With the information we gathered from the feature importance methods, we can use other data from the us county health rankings in order to try to make more accurate models. The model that I used had 106 features. It would be interesting to try using fewer features since our data is not very large. We can also experiment with using different minimum death filters. For example in my model, I made an arbitrary decision to use the information with counties with 20 or more deaths. However, it may be that a filter of 10 will lead to more enlightening information about the impact of death rates, and thus the populations who should seek a HCP. 
