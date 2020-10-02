#!/usr/bin/env python
# coding: utf-8

# # Regression Challange: Day 1 
# Day 1 mission - "How to pick the right regression technique for your data" 
# 

# First of all, let's pick up a dataset!
# 
# We are going to use the szeged-weather dataset ("Historical weather around Szeged, Hungary - from 2006 to 2016"), which contains timestamped records of weather features, some numerical, some categorical. The **Regression Challenge: Day 1**  proposed us to take 1 variable as target $Y$ and 1 variable as the feature $X$ used to predict it. 

# In[ ]:


# importing dataset
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
weather_df = pd.read_csv('../input/szeged-weather/weatherHistory.csv', encoding='utf-8')
weather_df.head(3)


# Before looking for a model/technique, let's briefly investigate what is the basic idea of regression. In statistics' community, the regression task is the procedure to estimating the relationship between variables. In the machine learning community, you'll se the comparisons with classification task where regression will be often associated as the task of predict numerical targets, and classification predict categorical variables (check out [this quora's answer](https://www.quora.com/What-is-the-main-difference-between-classification-problems-and-regression-problems-in-machine-learning)). For this challange, you should consider the statistical regression culture, where the relationship between input and output is the problem itself (check out [this other one](https://www.quora.com/What-is-the-main-difference-between-classification-and-regression-problems)).
# 
# Before take any pick up of regression technique, let's take a minute to breathe and reflect about hypothesis. The first step to master the relationship between two variables is to *choose what relationship could be interesting to be investigated*. Then, we need to setup the variables that could have any surprising information to be analyzed or hypothesis to be confirmed.
# 
# 
# 

# To gain in terms of time information, I will decompose the **Formatted Date** to month, week of year and hour. Additionaly, a T variable will be added to represent a linear sampling time.

# In[ ]:


datetime = pd.to_datetime(weather_df["Formatted Date"])
datetime = datetime.apply(lambda x: x+pd.Timedelta(hours=2)) #Correcting +2 GMT
weather_df["Month"] = datetime.apply(lambda x: x.month)
weather_df["WoY"] = datetime.apply(lambda x: x.week)
weather_df["Hour"] = datetime.apply(lambda x: x.hour)
weather_df["T"] = weather_df.index
weather_df[["Formatted Date","Month","WoY","Hour","T"]].head()


# We'll analyze the variables with plots in order to try to find a pattern that we want to investigate. To do a correlation plot, for instance, including all the variables, a problem will appear. The categorical ones **("Summary", "Precip Type" and "Daily Summary")** would be included only if we encoding them to a numerical space, however it will be meaningless, as they aren't ordinal variables. To avoid this, let's do it separately: 
# 
# * a pairplot including some of the numerical variables (including the ones about date)
# * a correlation heatmap to support the pairplot
# * two separated violin (similar to boxplot) analysis
# 

# In[ ]:


import seaborn as sns
sns.pairplot(weather_df[["Precip Type","Temperature (C)","Apparent Temperature (C)","Humidity","Hour","T"]],
             hue="Precip Type",
             palette="YlGnBu");


# In[ ]:


corr = weather_df.drop('Loud Cover', axis=1).corr() # dropping Loud Cover because it never change
sns.heatmap(corr,  cmap="YlGnBu", square=True);


# In[ ]:


sns.violinplot(x="Precip Type", y="Temperature (C)", data=weather_df, palette="YlGnBu");


# In[ ]:


sns.violinplot(x="Precip Type", y="Humidity", data=weather_df, palette="YlGnBu");


# With these plots we can choose two variables to study the relationship betweem them. In this study let's focus to explore *Temperature* as a function of *Humidity*, i.e., "how humidity influences in temperature?". The correlation plot gives us the information that they're strongly opposite related. Our hypothesis could be: "
# 
# After we aim **Now it's time to pick up a regression technique!**. We have a prior knowledge on both variables, but what's the best way to create a function (a regression) to relate them, and even when we don't have a certain humidity, we could estimate temperature, based on our regression.
# 
# We have three simple options and our choice will depend exclusively on what kind of output we want to predict:
# * **Linear**: When predicting a continuous value. 
# * **Logistic**: When predicting a category.
# * **Poisson**: When predicting counts.
# 
# In our case, *Temperature* is a continuous value, so we choose the **Linear Regression** model to tackle. If the predicted variable were *Precip Type*, we should use Logistic Regression, but there isn't countable variables to apply *Poisson* on such configuration of data.
# 
# Looking for those violinplots before, I think if we apply a linear model to just one category of *Precip Type*, the model may be more accurate, considering the noise from the others patterns of humidity vs temperature.

# In[ ]:


sns.jointplot("Humidity", "Temperature (C)", data=weather_df.where(weather_df['Precip Type']=='null'), kind="hex");


# The focus on **linear regression**, when you are predicting with a single feature, is to fit the best line, minimizing the square error between all the samples. This line $\hat{Y}$ is defined as an estimative of the ground truth $Y$:
# 
# $$ Y = \alpha X + \beta + \epsilon $$
# 
# $$ \hat{Y} = \hat{\alpha} X + \hat{\beta} $$ 
# 
# Where $X$ is the set of samples, $\alpha$ is the inclination of the curve and $\beta$ its intercepts. The last parameter, $\epsilon$ is the one that our model can't explain. As we'll see, our model $\hat{Y}$ won't be 100% accurate and this $\epsilon$ is the model's residual, that you should keep in mind for next diagnostics analysis on Day 2. Remeber that ^ stands for an estimative. 

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

ls = linear_model.LinearRegression()

# Our model will only predict temperature for non-raining/snowing configuration.
# I would recommend you to change 'null' for 'rain' or 'snow' and verify
# quality metrics, and see how efficient the models would be, only by filtering.
data = weather_df.where(weather_df['Precip Type']=='null')
data.dropna(inplace=True)

X = data["Humidity"].values.reshape(-1,1)
y = data["Temperature (C)"].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.33, 
                                                    shuffle=True, random_state=0)
print("Linear Regression")
ls.fit(X_train, y_train)
print("alpha = ",ls.coef_[0])
print("beta = ",ls.intercept_)
print("\n\nCalculating some regression quality metrics, which we'll discuss further on next notebooks")
y_pred = ls.predict(X_test)
print("MSE = ",mean_squared_error(y_test, y_pred))
print("R2 = ",r2_score(y_test, y_pred))


# Finally, we finished our regression. It means that now we have a relationship between Humidity and Temperature, where we can predict how hot/cold would be, considering we have measured the humity change. I thing it's that was a good way to wake up the regression instincts in your data heart! Next analysis will show you how to interpret if those fits were good enough to trust in critical scenarios. Let's test it!

# In[ ]:


hypothetical_humidity = 0.7
temperature_output = ls.predict(hypothetical_humidity)[0][0]
print("For such {} humidity, Linear Regression predict a temperature of {}C".format(hypothetical_humidity, round(temperature_output,1)))

