#!/usr/bin/env python
# coding: utf-8

# # Drive for Show, Putt for Dough
# By: Tom Allen

# This analysis serves to explore historical PGA data from 2010-2018 that was collected from the PGA Tour website. The reason I was interested in analyzing this data set was for two reasons. One reason being, I am new to using Python for data analysis and wanted to practice on data I am genuinely curious in. The second reason being, I wanted to see whether putting had a bigger impact on score/winnings than driving due to a light argument I had with a friend.

# ### Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load Dataset
# Assign PGA_Data_Historical.csv file to DataFrame: pga_df.

# In[ ]:


pga_df = pd.read_csv('../input/pga-tour-20102018-data/PGA_Data_Historical.csv')
pga_df


# ### Check DataFrame
# Over 2 million records with 2,083 different variables (subsets in the Variable column).

# In[ ]:


pga_df.info()


# ### Reshape DataFrame
# Unstacks the Variable column for easier use of handling key-value pairs.

# In[ ]:


pga_df = pga_df.set_index(['Player Name','Variable','Season'])['Value'].unstack('Variable').reset_index()
pga_df.head()


# ### Grab Useful Variables
# I picked the 11 variables that I thought had the most impact on score/winnings:
# 
# - Player Name: Player Name 
# 
# - Season: Year
# 
# - Top 10 Finishes - (TOP 10): Number of top 10 finshes the player had in that year
# 
# - Top 10 Finishes - (EVENTS): Number of total events in that year for that player
# 
# - Driving Distance - (AVG.): Average driving distance per drive for that year
# 
# - Driving Accuracy Percentage - (%): % of fairways hit per drive (fairways hit / total drives) for that year
# 
# - Scoring Average (Actual) - (AVG): Player's average score per round in that year
# 
# - 3-Putt Avoidance > 25' (2 PUTT OR BETTER): Player's probability of making 2 putts or better per hole if on the green and 25 feet away from pin
# 
# - Total Money (Official and Unofficial) - (MONEY): Player's total winnings in that year

# In[ ]:


col = ["Player Name","Season","Top 10 Finishes - (TOP 10)","Top 10 Finishes - (EVENTS)","Driving Distance - (AVG.)",
          "Driving Accuracy Percentage - (%)","Scoring Average (Actual) - (AVG)", "3-Putt Avoidance > 25' - (2 PUTT OR BETTER %)",
          "Putting Average - (AVG)", "Total Money (Official and Unofficial) - (MONEY)"]


# ### Assign the 11 Useful Variables to DataFrame

# In[ ]:


pga_df=pga_df[col]


# ### Rename Variables to Simpler Labels

# In[ ]:


pga_df.rename(columns = {'Player Name':'PlayerName'}, inplace = True)
pga_df.rename(columns = {'Top 10 Finishes - (TOP 10)':'Top_10_Finishes'}, inplace = True)
pga_df.rename(columns = {'Top 10 Finishes - (EVENTS)':'Events'}, inplace = True)
pga_df.rename(columns = {'Driving Distance - (AVG.)':'Avg_Drive_Dist'}, inplace = True)
pga_df.rename(columns = {'Driving Accuracy Percentage - (%)':'Avg_Drive_Acc'}, inplace=True)
pga_df.rename(columns = {'Scoring Average (Actual) - (AVG)':'Avg_Score'}, inplace=True)
pga_df.rename(columns = {"3-Putt Avoidance > 25' - (2 PUTT OR BETTER %)":'Three_Putt_Avoid'}, inplace=True)
pga_df.rename(columns = {'Putting Average - (AVG)':'Avg_Putt'}, inplace=True)
pga_df.rename(columns = {'Total Money (Official and Unofficial) - (MONEY)':'Money'}, inplace=True)


# ## Cleaning the Data

# ### Replace String Values
# Important in order to convert Money column into numbers.

# In[ ]:


pga_df = pga_df.replace({'\$':'',',':''},regex = True)


# ### Convert Columns to Numbers

# In[ ]:


for col in pga_df.columns[2:]:
    pga_df[col] = pga_df[col].astype(float)


# ### Create New Avg_Money and Avg_Finish Columns
# Create average variables per event since some players golf at more events than others. The Avg_Finish variable calculates the percentage a player will finish in the top 10 in that event. 

# In[ ]:


pga_df['Avg_Money'] = round(pga_df['Money']/pga_df['Events'])
pga_df['Avg_Finish'] = pga_df['Top_10_Finishes']/pga_df['Events']


# ### Convert Avg_Finish to Boolean Variable
# The percentage of a random player finishing in the top 10 per event is roughly 5%. I convert the Avg_Finish variable into a Boolean variable, so we can easily check if a player finishes above (True) or below (False) the mean of all the players in the dataset. 

# In[ ]:


pga_df['Avg_Finish'] = pga_df['Avg_Finish'] > pga_df['Avg_Finish'].mean()
print('Per event, the average percentage a player finishes in the top 10 is: {0:.0%}'.format(pga_df['Avg_Finish'].mean()))


# ### Fill NaN Values in Events Column
# The dataset had many records in Events column that were NaN (missing values). This for loop checks if there is a NaN value, and would predict the # of events based on their total winnings that season / their average winnings per event.

# In[ ]:


original_record_count = pga_df.shape[0]
for num in pga_df['Events']:
    pga_df['Events'] = round(pga_df['Events'].fillna(value=pga_df['Money']/pga_df['Avg_Money']))


# ### Drop Top_10 Finishes Column

# In[ ]:


pga_df.drop(['Top_10_Finishes'],axis = 1, inplace = True)


# ### Drop Remaining NaN Values in DataFrame

# In[ ]:


pga_df=pga_df.dropna()


# ### Check How Many Records Were Dropped
# Some players only had data on a couple variables per year, so dropping the records with NaN values gave me only the players with values for each variable I wanted.

# In[ ]:


new_record_count = pga_df.shape[0]
print(f'Original record count: {original_record_count}')
print(f'New record count: {new_record_count}')
print('Number of records dropped: '+str(original_record_count - new_record_count))


# In[ ]:


pga_df.head(5)


# ### Display Correlations Between Features
# Created a scaled copy of the original pga_df for future use of analyzing correlations with the variables.

# In[ ]:


pga_scale = pga_df.copy(deep=True)
corr = pga_scale.corr()
corr


# # Data Visualizations

# ### Plotting Driving vs Money and Putting vs Money
# At first glance, looks like driving distance average and putting average have fairly strong correlations with money earned.

# In[ ]:


sns.set_style('whitegrid')
f, ax = plt.subplots(1,2, figsize = (12,5))
ax[0].scatter('Avg_Drive_Dist','Money',data=pga_df,color='green')
ax[0].set_title("Driving for Money")
ax[0].set_xlabel('Driving Avg. (Yards)')
ax[0].set_ylabel('Money ($)')
ax[1].scatter('Avg_Putt','Money',data=pga_df, color = 'green')
ax[1].set_title("Putting for Money")
ax[1].set_xlabel('Putting Avg. (Putts per Hole)')
ax[1].set_ylabel('Money ($)')


# ### Check Actual Correlations
# Putting is more strongly correlated to winnings, although negatively sloped, which makes sense since the fewer putts per hole would yield a better score.

# In[ ]:


corr_drive = round(corr['Avg_Money']['Avg_Drive_Dist'],4)
corr_putt = round(corr['Avg_Money']['Avg_Putt'],4)
print(f'The correlation between Average Driving Distance and Money earned is: {corr_drive}')
print(f'The correlation between Average Putts and Money earned is: {corr_putt}')


# ### Display Player's Driving Ability (Distance vs Accuracy)
# This scatterplot is inversely correlated showing that players are usually either good at driving the ball far or driving the ball accuratley, but not at both.

# In[ ]:


all_avg_acc = pga_df['Avg_Drive_Acc']
all_avg_dist = pga_df['Avg_Drive_Dist']
sns.set_style('dark')
sns.jointplot(x=all_avg_acc,y=all_avg_dist,data=pga_df)


# I wanted to quickly check the player who seemed to be an outlier in this graph since they were able to drive far AND accuratley. Turns out to be Gary Woodland, 2019 U.S. Open champ!

# In[ ]:


pga_df[(pga_df['Avg_Drive_Dist']>310) & (pga_df['Avg_Drive_Acc']>62)]


# Here we see there is a relativley strong negative correlation between driving distance and accuracy. Seems like there is almost a trade-off for players either wanting to hit the ball far or accurately. 

# In[ ]:


print('The correlation between driving distance and driving accuracy is: ' + str(round(corr['Avg_Drive_Dist']['Avg_Drive_Acc'],4)))


# ### Standardize Features by Scaling to Unit Variance
# Standardizing the features is important because some of the values can be widely different (think driving distance and putting average).

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Feature_columns = ['Avg_Drive_Dist','Avg_Drive_Acc', 'Avg_Score','Three_Putt_Avoid','Avg_Putt',
                   'Money','Avg_Money','Avg_Finish']
pga_scale[Feature_columns]=scaler.fit_transform(pga_scale[Feature_columns])


# ### Display Heatmap of Correlations
# The darker or lighter the shade of green, the stronger the correlation

# In[ ]:


sns.heatmap(pga_scale.corr(),linewidth=1,linecolor='white',cmap='Greens')


# # Decision Tree

# ### Set Up Decision Tree
# Use Feature_Columns without Avg_Finish as input variables and Avg_Finish as target variable. Then split split the data into training and testing sets. 
# 
# The goal is to predict whether the model can accuratley predict if a player finishes above the mean or not. 

# In[ ]:


from sklearn.model_selection import train_test_split
X = pga_df[Feature_columns].drop('Avg_Finish', axis=1)
y = pga_df['Avg_Finish']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)


# ### Create Decision Tree Model
# Fits the training set to the Decision Tree model

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# ### Predicitions and Evaluation
# Displays confusion matrix and classification report.
# 
# Model accurately predicits if player does or does not finish Top 10 above the mean 85% of the time (True Positive: 163 + True Negative: 412)/Total: 693

# In[ ]:


predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


# # Multiple Linear Regression

# ### Assign Training Features (X) and Target Variable ('Avg_Score')

# In[ ]:


X = pga_df[['Avg_Drive_Dist','Avg_Drive_Acc','Avg_Putt','Three_Putt_Avoid',]]
y = pga_df['Avg_Score']


# ### Train Test Split
# 80% of the data will be allocated to the train set, therefore 20% to the test set

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# ### Create and Train the Linear Regression Model

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression() # creates an object from the LinearRegression class
lm.fit(X_train,y_train) # fits the training set to the model


# ### Check Coefficients
# High Avg_Putt makes sense because an average increase of one putt (stroke) per hole can be dentrimental to the final score. 

# In[ ]:


coeff_df = round(pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient']),3)
coeff_df


# ### Predictions
# Data points seem relatively fit to the regression line, only a few major outliers that could cause some mispredictions. 

# In[ ]:


predictions = lm.predict(X_test)
plt.scatter(y_test, predictions, color = 'green')


# ### Loss Functions
# The Mean Squared Error (MSE) is 0.16 which is relatively low indicating the values are dispersed closely to the mean

# In[ ]:


from sklearn import metrics
MSE = round(metrics.mean_squared_error(y_test, predictions),4)
print(f'Mean Squared Error = {MSE}')


# # PGA Tour Trends
# A few observations:
# - Average driving distance has increased throughout the years (better club technology, stronger players?)
# - Money has also increased throughout the years. Looks like the shaded region is a bit higher indicating there may be outliers that don't accuratley predict the regression line
# - The average score has decreased meaning players are getting better at all facets of the game (time for more challenging courses?)

# In[ ]:


s = sns.PairGrid(pga_df, y_vars = ['Season'],x_vars = ['Avg_Drive_Dist','Money','Avg_Score'], height = 4)
s.map(sns.regplot)


# # Conclusion
# Overall, this was a fun dataset to play around with, especially because I follow golf and get to see how accurate some of my previous inclinations were. 
# 
# For instance, I can now say that average putts per hole has a bigger impact on money winnings than average driving distance per hole. I also had a feeling that driving distance had been increased over the years due to players becoming stronger with more instense workout regiments (Brooks Koepka, Rory McIlroy, Dustin Johnson) and the advancement in technology with the clubs themselves.
# 
# I did learn a great deal throughout this analysis. One thing I didn't realize was how negativley correlated average driving distance and average driving accuracy were. I figured players were either good at driving altogether or not, meaning they could drive far and accurate instead of having that trade-off. Another interesting chart to look at was the increase in money winnings throughout the years. Many people say golf is a dying sport that is being washed away with the older generation. Since there is an increase in money winnings, that may indicate that more people are watching the sport, therefore more companies/advertisers sponsoring events. 
# 
# There is definitley a lot more to analyze with this vast data set, and I look forward to using it more. If you notice any errors or have any recommendations in areas to improve in, please feel free to reach out. 

# In[ ]:




