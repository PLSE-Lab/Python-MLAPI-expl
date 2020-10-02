#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression, RandomizedLasso
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import cross_validation
from sklearn.preprocessing import OneHotEncoder, Imputer, LabelEncoder
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline


# In[ ]:


plot_params = {
 "font.size":14.0,
 "figure.figsize": (6, 4),
  "axes.labelsize": "large",
  "figure.autolayout": True,
  "patch.edgecolor":"white",
  "axes.facecolor": "#f0f0f0",
  "patch.edgecolor": "#f0f0f0",
    "figure.facecolor": "#f0f0f0",
 "grid.linestyle": "-",
 "grid.linewidth": 1.0,
 "grid.color": "#cbcbcb",
 "savefig.edgecolor": "#f0f0f0",
 "savefig.facecolor": "#f0f0f0"
}
sns.set(rc=plot_params)


# In[ ]:


# get airbnb & test csv files as a DataFrame
airbnb_df  = pd.read_csv('../input/train_users.csv')
test_df    = pd.read_csv('../input/test_users.csv')

# preview the data
airbnb_df.head()


# In[ ]:


# drop unnecessary columns, these columns won't be useful in analysis and prediction
airbnb_df  = airbnb_df.drop(['date_account_created','timestamp_first_active'], axis=1)
test_df    = test_df.drop(['date_account_created','timestamp_first_active'], axis=1)


# In[ ]:


airbnb_df['booked'] = (airbnb_df['country_destination'] != 'NDF').astype(int)
gp = airbnb_df[['id','country_destination']].groupby('country_destination').count()
ax = gp.sort('id').plot(kind='barh', color=['#0059b3'])
ax.set_xlabel('# of bookings')
ax.set_ylabel('country destination')
ax.legend_.remove()


# In[ ]:


# date_first_booking

def get_year(date):
    if date == date: 
        return int(str(date)[:4])
    return date

def get_month(date):
    if date == date: 
        return int(str(date)[5:7])
    return date

# Create Year and Month columns
airbnb_df['Year']  = airbnb_df['date_first_booking'].apply(get_year)
airbnb_df['Month'] = airbnb_df['date_first_booking'].apply(get_month)

test_df['Year']  = test_df['date_first_booking'].apply(get_year)
test_df['Month'] = test_df['date_first_booking'].apply(get_month)

# fill NaN
airbnb_df['Year'].fillna(airbnb_df['Year'].median(), inplace=True)
airbnb_df['Month'].fillna(airbnb_df['Month'].median(), inplace=True)

test_df['Year'].fillna(test_df['Year'].median(), inplace=True)
test_df['Month'].fillna(test_df['Month'].median(), inplace=True)

# convert type to integer
airbnb_df[['Year', 'Month']] = airbnb_df[['Year', 'Month']].astype(int)
test_df[['Year', 'Month']]   = test_df[['Year', 'Month']].astype(int)


# In[ ]:


# gender

i = 0
def get_gender(gender):
    global i
    if gender != 'FEMALE' and gender != 'MALE':
        return 'FEMALE' if(i % 2) else 'MALE'
    i = i + 1
    return gender

# replace all values other than 'FEMALE' and 'MALE'
airbnb_df['gender'] = airbnb_df['gender'].apply(get_gender)
test_df['gender']   = test_df['gender'].apply(get_gender)


# In[ ]:


# age

# assign all age values > 100 to NaN, these NaN values will be replaced with real ages below
airbnb_df["age"][airbnb_df["age"] > 100] = np.NaN
test_df["age"][test_df["age"] > 100]     = np.NaN

# get average, std, and number of NaN values in airbnb_df
average_age_airbnb   = airbnb_df["age"].mean()
std_age_airbnb       = airbnb_df["age"].std()
count_nan_age_airbnb = airbnb_df["age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test_df["age"].mean()
std_age_test       = test_df["age"].std()
count_nan_age_test = test_df["age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_airbnb - std_age_airbnb, average_age_airbnb + std_age_airbnb, size = count_nan_age_airbnb)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# fill NaN values in Age column with random values generated
airbnb_df["age"][np.isnan(airbnb_df["age"])] = rand_1
test_df["age"][np.isnan(test_df["age"])]     = rand_2

# convert type to integer
airbnb_df['age'] = airbnb_df['age'].astype(int)
test_df['age']   = test_df['age'].astype(int)


# In[ ]:


# signup_method
airbnb_df["signup_method"] = (airbnb_df["signup_method"] == "basic").astype(int)
test_df["signup_method"]   = (test_df["signup_method"] == "basic").astype(int)

# signup_flow
airbnb_df["signup_flow"] = (airbnb_df["signup_flow"] == 3).astype(int)
test_df["signup_flow"]   = (test_df["signup_flow"] == 3).astype(int)

# language
airbnb_df["language"] = (airbnb_df["language"] == 'en').astype(int)
test_df["language"]   = (test_df["language"] == 'en').astype(int)

# affiliate_channel
airbnb_df["affiliate_channel"] = (airbnb_df["affiliate_channel"] == 'direct').astype(int)
test_df["affiliate_channel"]   = (test_df["affiliate_channel"] == 'direct').astype(int)

# affiliate_provider
airbnb_df["affiliate_provider"] = (airbnb_df["affiliate_provider"] == 'direct').astype(int)
test_df["affiliate_provider"]   = (test_df["affiliate_provider"] == 'direct').astype(int)


# In[ ]:


for f in airbnb_df.columns:
    if f == "country_destination" or f == "id": continue
    if airbnb_df[f].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(np.unique(list(airbnb_df[f].values) + list(test_df[f].values)))
        airbnb_df[f] = lbl.transform(list(airbnb_df[f].values))
        test_df[f]   = lbl.transform(list(test_df[f].values))


# In[ ]:


X = airbnb_df.drop(["country_destination", "id", 'booked'],axis=1)
y = airbnb_df["country_destination"]
test  = test_df.drop("id",axis=1).copy()


# In[ ]:


# modify country_destination to numerical values
country_num_dic = {'NDF': 0, 'US': 1, 'other': 2, 'FR': 3, 'IT': 4, 'GB': 5, 'ES': 6, 'CA': 7, 'DE': 8, 'NL': 9, 'AU': 10, 'PT': 11}
num_country_dic = {y:x for x,y in country_num_dic.items()}
y = y.map(country_num_dic)


# In[ ]:


model = Pipeline([("imputer", Imputer(missing_values='NaN',
                                          strategy="mean",
                                          axis=1)),
                     ('feature', RandomizedLasso()),
                      ("model", LogisticRegression())])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)
model.fit(X_train, y_train)
ypred = model.predict(X_test)
print(metrics.classification_report(y_test, ypred))


# In[ ]:


model = Pipeline([("imputer", Imputer(missing_values='NaN',
                                          strategy="mean",
                                          axis=1)),
                     ('feature', RandomizedLasso()),
                      ("model", GradientBoostingClassifier())])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=0)
model.fit(X_train, y_train)
ypred = model.predict(X_test)
print(metrics.classification_report(y_test, ypred))


# In[ ]:


# convert type to integer
ypred = ypred.astype(int)
# change values back to original country symbols
ypred = Series(ypred).map(num_country_dic)


# In[ ]:



# Create submission

country_df = pd.DataFrame({
        "id": test_df["id"],
        "country": ypred
    })

submission = DataFrame(columns=["id", "country"])

# sort countries according to most probable destination country 
for key in country_df['country'].value_counts().index:
    submission = pd.concat([submission, country_df[country_df["country"] == key]], ignore_index=True)

submission.to_csv('airbnb.csv', index=False)

