#!/usr/bin/env python
# coding: utf-8

# This hotel occupancy rate data is based on real Portuguese market research data. Because as the relevant data that can be used to study the hotel market research, in fact, in the teaching industry or school, we have little contact. Therefore, this data actually has very high research value. If we can analyze this data in depth, we can have a more comprehensive grasp of the correlation of various parameters such as hotel market conditions, occupancy conditions, seasonal changes, and popular population.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# This is a strait forward data analysis and visualization on datas from hotels booking information.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


import pandas as pd
df = pd.read_csv("../input/hotel-booking-demand/hotel_bookings.csv")


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.hotel.value_counts()


# In[ ]:


df.isnull().any()


# In[ ]:


df.isnull().sum()


# In[ ]:


nan_replacements = {"children:": 0.0,"country": "Unknown", "agent": 0, "company": 0}
df_cln = df.fillna(nan_replacements)


df_cln["meal"].replace("Undefined", "SC", inplace=True)


zero_guests = list(df_cln.loc[df_cln["adults"]
                   + df_cln["children"]
                   + df_cln["babies"]==0].index)
df_cln.drop(df_cln.index[zero_guests], inplace=True)


# In[ ]:


df_cln.shape


# In[ ]:


df = df_cln


# # 1.EDA

# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


sns.countplot(df.arrival_date_year)


# In[ ]:


arrival = df.groupby(['arrival_date_year','arrival_date_month'])[['hotel']].count()
arrival


# In[ ]:


average_children = round(df["children"].mean())


# We can see the lack of some months of 2015 and 2016, so the visualized cannot tell us the growth of each year. We should move on.

# In[ ]:


df.groupby(['arrival_date_year', 'arrival_date_month']).size().plot.bar(figsize=(15,5))


# It is explained here that the hotel's scheduled heat has seasonal characteristics. It can be seen from the histogram that the number of reservations has increased significantly during the summer or summer vacation. When this period of time passed, the number of reservations fell again.

# In[ ]:


sns.countplot(df.hotel)


# The City Hotel is more popular than Resort Hotel.

# In[ ]:


rate = df['is_canceled'].value_counts().tolist()
bars =  ['Not Cancel','Cancel']
y_pos = np.arange(len(bars))
plt.bar(y_pos,height=rate , width=0.3 ,color= ['red','blue'])
plt.xticks(y_pos, bars)
plt.xticks(rotation=90)
plt.title("Cancel Rate", fontdict=None, size = 'large')
plt.show()


# We can see the number of canceled booking is more than the half the 'Not Canceled' booking.

# In[ ]:


plt.title('Cancellation')
plt.ylabel('Cancel_Sum')

df.groupby(['hotel','arrival_date_year'])['is_canceled'].sum().plot.bar(figsize=(10,5))


# hrough the analysis of the cancellation rate of two hotels, no matter what type of hotel or location, we can find that the cancellation rate in 2016 is very high.

# In[ ]:


sns.countplot(df.stays_in_weekend_nights)


# In[ ]:


sns.countplot(df.stays_in_week_nights)


# We can realize the trend that majority of people prefer to stay in hotel during weekdays. 

# In[ ]:


top_countries = df["country"].value_counts().nlargest(10).astype(int)


# In[ ]:


df.groupby(['arrival_date_month','arrival_date_year'])['children', 'babies'].sum().plot.bar(figsize=(15,5))


# There are far more families with children than with babies.

# In[ ]:


plt.figure(figsize=(25,10))
sns.barplot(x=top_countries.index, y=top_countries, data=df)


# This shows that the hotel mainly accepts Portuguese tourists, followed by European tourists.

# After the general analysis and visualization, we can see that we are trying to figure out whether the booking will be finally canceled or not. It is a simple classification problem. I decided to predict it with Logistic Regression Model, and put the 'is_canceled' column as the dependent variable. Since this model requires all input to be numerical, I will turn them into categories through One Hot Encode. It will increase the dimention of regression. I will choose PCA the have a process of decreasing dimensions.

# # 2. Predict Cancelation

# In[ ]:


cancel_corr = df.corr()["is_canceled"]
cancel_corr.abs().sort_values(ascending=False)[1:]


# In[ ]:


df.groupby("is_canceled")["reservation_status"].value_counts()


# 

# In[ ]:


#When implementing data engineering, we need to distinguish and classify the nature of features.
num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month",
                "stays_in_weekend_nights","stays_in_week_nights","adults","children",
                "babies","is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled","agent","company",
                "required_car_parking_spaces", "total_of_special_requests", "adr"]

cat_features = ["hotel","arrival_date_month","meal","market_segment",
                "distribution_channel","reserved_room_type","deposit_type","customer_type"]


# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

features = num_features + cat_features
X = df.drop(["is_canceled"], axis=1)[features]
y = df["is_canceled"]


num_transformer = SimpleImputer(strategy="constant")

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                               ("cat", cat_transformer, cat_features)])


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

base_models = [("DT_model", DecisionTreeClassifier(random_state=42)),
               ("RF_model", RandomForestClassifier(random_state=42,n_jobs=-1)),
               ("LR_model", LogisticRegression(random_state=42,n_jobs=-1)),
               ("XGB_model", XGBClassifier(random_state=42, n_jobs=-1))]


# In[ ]:


from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score

kfolds = 4
split = KFold(n_splits=kfolds, shuffle=True, random_state=42)

for name, model in base_models:
    
    model_steps = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
    
    
    cv_results = cross_val_score(model_steps, 
                                 X, y, 
                                 cv=split,
                                 scoring="accuracy",
                                 n_jobs=-1)
   
    min_score = round(min(cv_results), 4)
    max_score = round(max(cv_results), 4)
    mean_score = round(np.mean(cv_results), 4)
    std_dev = round(np.std(cv_results), 4)
    print(f"{name} cross validation accuarcy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")


# So we can see the RandomForest model is the best.

# # 3. Feature Importance

# In[ ]:



model_pipe.fit(X,y)


onehot_columns = list(model_pipe.named_steps['preprocessor'].
                      named_transformers_['cat'].
                      named_steps['onehot'].
                      get_feature_names(input_features=cat_features))


feat_imp_list = num_features + onehot_columns


feat_imp_df = eli5.formatters.as_dataframe.explain_weights_df(
    model_pipe.named_steps['model'],
    feature_names=feat_imp_list)
feat_imp_df.head(10)

