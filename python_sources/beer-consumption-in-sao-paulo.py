#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import os
print(os.listdir("../input"))
raw_data = pd.read_csv("../input/beer-consumption-sao-paulo/Consumo_cerveja.csv")
raw_data.head()


# Date,
# Average Temperature (C),
# Minimum Temperature (C),
# Maximum Temperature (C),
# Rainfall (mm),
# Weekend,
# Beer Consumption (liters),

# #### Let's first translate column names to english

# In[ ]:


raw_data.columns = ["Date", "Avg Temp", "Min Temp", "Max Temp", "Rainfall", "Weekend", "Beer Consumption"]
raw_data.head()


# # Questions-
# 1. Is beer consumption more in begining of month?
# 2. how is beer consumption spread throughout the week?
# 3. how is beer consumption spread throughout the months?
# 4. how is consumption related to avg, min and max temperature?
# 5. how is consumption related to rainfall?
# 6. What is the impact of weekend on consumption?

# In[ ]:


raw_data.shape


# In[ ]:


raw_data.info()


# 

# ### DATA TIDYING

# ##### So we have 5 object type features. Other than "Date" rest features have ", " instead of "." decimal and the difference in shape and infor data entries suggest presence of a lot of NULL data. Let's get into some tidying of our data.

# In[ ]:


raw_data.isnull().sum()


# In[ ]:


pd.set_option("display.max_rows", 15)


# In[ ]:


raw_data.iloc[364:]


# ##### As data for only the first year is present need to drop rest entries

# In[ ]:


after_drop_data = raw_data.drop(range(365, 941))
after_drop_data.shape


# In[ ]:


after_drop_data.tail()


# In[ ]:


temp_data = after_drop_data[["Min Temp", "Max Temp", "Avg Temp", "Rainfall"]]
temp_data.head()


# In[ ]:


import re
for index, row in temp_data.iterrows():
    for j in ["Min Temp", "Max Temp", "Avg Temp", "Rainfall"]:
        row[j] = re.sub(r',', '.', row[j])
temp_data.head()


# In[ ]:


temp_data = temp_data.astype("float64")


# In[ ]:


tidy_data = after_drop_data
for j in ["Min Temp", "Max Temp", "Avg Temp", "Rainfall"]:
    tidy_data[j] = temp_data[j]
tidy_data.head()


# In[ ]:


tidy_data.info()


# ##### Now let's find out if there is any information to be extracted from the dates in relation with and beer consumption
# ##### To do that first extract day and month from the date feature

# In[ ]:


date_data = pd.DataFrame(tidy_data["Date"])
date_data.head()


# In[ ]:


months = np.array([])
days = np.array([])
for index, row in tidy_data.iterrows():
    months = np.append(months, re.search(r'-(.+?)-', row["Date"]).group(1))
    days = np.append(days, re.search(r'-..-(.+?)$', row["Date"]).group(1))
date_data["Month"] = months
date_data["Day"] = days
date_data[["Month", "Day"]] = date_data[["Month", "Day"]].astype("float64")
date_data.head()


# In[ ]:


final_data = pd.merge(tidy_data, date_data, on="Date", how="inner")
final_data = final_data.drop(["Date"], axis=1)
#Re-order to have a y on right side 
final_data = final_data[["Avg Temp", "Min Temp", "Max Temp", "Rainfall", "Weekend", "Month", "Day", "Beer Consumption"]]
final_data.head()


# # Exploratory Data Analysis

# #### first take out test set and train set on fixed seed 25

# In[ ]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(final_data, test_size = 0.2, random_state = 25) 
print(train.shape)
print(test.shape)


# In[ ]:


df = train
df.describe()


# In[ ]:


df.hist(figsize = (15, 18))


# #### The temp features are almost normal can conduct a normality test to verify. Will put it on hold for future versions of kernel. Rest don't provide much inference

# In[ ]:


df.boxplot(figsize=(8, 8), column = ["Min Temp", "Avg Temp", "Max Temp", "Beer Consumption"])


# #### As expected from the temperature data has incresing quantile distribution. One outlier can bee seen from the whiskers of Max Temp feature, don't think it is an issue.

# In[ ]:


corr = df.corr()
sns.heatmap(corr)


# #### Heatmap shows good corelation in temperature data and consumption. But rest has to be further investigated. Moreover, Temperature data is inter-corelated. Multicolinearity could hence be a problem here. Adding Pricipal Component Analysis for Dimensionality reduction to my to do list

# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].scatter(df["Beer Consumption"], df["Min Temp"])
ax[1].scatter(df["Beer Consumption"], df["Avg Temp"])
ax[2].scatter(df["Beer Consumption"], df["Max Temp"])


# #### Scatterplot reassures positive corelation between beer consumption and temperature data. We can infer that people prefer a beer more on hotter days rather than cold ones

# ### PCA for temperature data - (Giving bogus results hence)
# #### This execution is on hold for future versions of this kernel 
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 2)
# dec_temp = pca.fit_transform(df[["Avg Temp", "Min Temp", "Max Temp"]])
# print(pca.explained_variance_ratio_)
# dec_temp[:5]
# #### after end of preparation pieline
# dec_temp = pd.DataFrame(dec_temp, columns=["PC1", "PC2"])
# df = df.drop(["Min Temp", "Max Temp", "Avg Temp"], axis=1)
# df["Temp1"] = dec_temp["PC1"]
# df["Temp2"] = dec_temp["PC2"]
# df = df[["Temp1", "Temp2", "Rainfall", "Weekend", "Month", "Day", "Beer Consumption"]]
# df.head()

# #### Let's try figure out how beer consumtion is spread across different months

# In[ ]:


by_month = df[["Month", "Beer Consumption"]]
by_month = by_month.groupby("Month").mean().reset_index()
by_month["Name"] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
by_month.sort_values(by=["Beer Consumption"], inplace=True, kind="heapsort")
by_month.reset_index(inplace=True)
by_month.drop(["index"], axis=1, inplace=True)
plt.figure(figsize=(12, 4))
plt.bar(by_month["Name"], by_month["Beer Consumption"])


# #### We can infer that people drink more during holiday season that is new year's eve, christmas and i believe october has halloween
# #### So i wll engineer a feature utilising this inference

# In[ ]:


order_Month = by_month["Month"].tolist()
order_Month = {v : k+1 for k, v in enumerate(order_Month)}
order_Month


# In[ ]:


def editMonth(df):
    for index, row in df.iterrows():
        #print(row["Month"], order_Month[row["Month"]])
        row["Month"] = order_Month[row["Month"]]
    return df
df = editMonth(df)
df.head()


# In[ ]:


plt.scatter(df["Month"], df["Beer Consumption"])


# In[ ]:


print(final_data[["Month", "Beer Consumption"]].corr())
print(df[["Month", "Beer Consumption"]].corr())


# #### So we were able to catch significant co-relation. We will do same with days

# In[ ]:


by_day = df[["Day", "Beer Consumption"]]
by_day = by_day.groupby("Day").mean().sort_values(by="Beer Consumption").reset_index()
plt.figure(figsize=(16, 4))
plt.bar(range(1, 32), by_day["Beer Consumption"], tick_label=by_day["Day"])


# #### Quite interestingly month consumption is more for 31. Could it be because of 31st Dec? As 30 is on  the lowes side. Can't explain highest for 21 though. 
# #### I have two options either to go with this or not. For this version i'll choose the prior.

# In[ ]:


order_Day = by_day["Day"].tolist()
order_Day = {v : k+1 for k, v in enumerate(order_Day)}
print(order_Day)
def editDay(df):
    for index, row in df.iterrows():
        #print(row["Day"], order_Day[row["Day"]])
        row["Day"] = order_Day[row["Day"]]
    return df
df = editDay(df)
df.head()


# In[ ]:


plt.scatter(df["Day"], df["Beer Consumption"])


# In[ ]:


print(final_data[["Day", "Beer Consumption"]].corr())
print(df[["Day", "Beer Consumption"]].corr())


# #### Well it's better than nothing. that new year's eve could create a bias!

# In[ ]:


co = df.corr()
sns.heatmap(co)


# In[ ]:


rest_data = df[["Rainfall", "Weekend", "Beer Consumption"]]
fig, ax = plt.subplots(1, 2)
ax[0].scatter(rest_data["Weekend"], rest_data["Beer Consumption"], c=rest_data["Weekend"])
ax[1].scatter(rest_data["Rainfall"], rest_data["Beer Consumption"])


# In[ ]:


rest_data.corr()


# In[ ]:


plt.hist(x = [rest_data["Beer Consumption"].where(rest_data["Weekend"]==1).dropna(), rest_data["Beer Consumption"].where(rest_data["Weekend"]==0).dropna()], color=["blue", "red"], histtype="step")


# #### As expected distribution for weekend is quite separated but for significance will do a chi-square test!

# In[ ]:


plt.hist(rest_data["Rainfall"], log=True)


# In[ ]:


def logRainfall(df):
    df["Rainfall"] = df["Rainfall"].apply(np.log)
    #df["Rainfall"].replace(to_replace = (-np.inf), value=0)
    df.loc[df["Rainfall"] == -np.inf, "Rainfall"] = 0
    return df
df = logRainfall(df)
df["Rainfall"]


# #### Log transformation on the right skewed rainfall distribution generated lots of "-inf" as there are lots of 0s in data

# In[ ]:


sns.heatmap(df.corr())
print(df.corr()["Beer Consumption"])


# #### Let's do all of the above to test set

# In[ ]:


test = editMonth(test)
test = editDay(test)
test = logRainfall(test)


# In[ ]:


test


# In[ ]:


x_train, y_train = df.drop(["Beer Consumption"], axis=1), df["Beer Consumption"]
x_test, y_test = test.drop(["Beer Consumption"], axis=1), test["Beer Consumption"]
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# #### Let's fit our linear regression line to it

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr.coef_


# (x_test.isnull()==True).Rainfall.sum()

# In[ ]:


predict = lr.predict(x_test)
print(predict[:5])
print(y_test[:5])


# In[ ]:


lr.score(x_test, y_test)#r2 score


# In[ ]:


from sklearn.metrics import mean_squared_error
ans = np.sqrt(mean_squared_error(y_test, predict))


# In[ ]:


ans


# #### we were able to get a descent l2score on this.
# # Questions / Answers-
# 1. Is beer consumption more in begining of month? Ans: No particular inference for this question
# 2. how is beer consumption spread throughout the week? Ans: On hold
# 3. how is beer consumption spread throughout the months? Ans: corelation can be found in terms of months with holidays & exams
# 4. how is consumption related to avg, min and max temperature? Ans: all are corelated to beer consumption
# 5. how is consumption related to rainfall? Ans: Highly right skewed due to more days with almost no rainfall
# 6. What is the impact of weekend on consumption? Ans: More consumption on weekend but the difference is not steep

# #### FUTURE VERSIONs will contain-
# 1. Dealing with multicolinearity of temperature
# 2. pipeling for feature enggineering and model evlauation
# 3. Cross validation and model selection using algorithms like SVM and Trees
# 4. Different analysis for days and distribuution on week days

# In[ ]:




