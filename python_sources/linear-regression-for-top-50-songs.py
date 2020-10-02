#!/usr/bin/env python
# coding: utf-8

# # Quick Linear Regression for Spotify's Top 50 Songs
# 
# As 2019 comes to a close, the top 50 songs for 2019 from spotify has been released.
# 
# Interestigly, spotfify has a unique [tool](http://organizeyourmusic.playlistmachinery.com/) that can organize the dataset by outputting some variables that may of interest.
# 
# * Genre - the genre of the track
# * Year - the release year of the recording. Note that due to vagaries of releases, re-releases, re-issues and general madness, sometimes the release years are not what you'd expect.
# * Added - the earliest date you added the track to your collection.
# * Beats Per Minute (BPM) - The tempo of the song.
# * Energy - The energy of a song - the higher the value, the more energtic. song
# * Danceability - The higher the value, the easier it is to dance to this song.
# * Loudness (dB) - The higher the value, the louder the song.
# * Liveness - The higher the value, the more likely the song is a live recording.
# * Valence - The higher the value, the more positive mood for the song.
# * Length - The duration of the song.
# * Acousticness - The higher the value the more acoustic the song is.
# * Speechiness - The higher the value the more spoken word the song contains.
# * Popularity - The higher the value the more popular the song is.
# 
# However, for the dataset provided the 'year' category wasn't included.
# It can be assumed all these songs were released in 2019.
# 
# With all this information. I wondered if there was a clear distinction for the variables provided and the popularity scored provided by spotify?
# 
# Let's find out!

# In[ ]:


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
from scipy import stats


# In[ ]:


songs_df = pd.read_csv('../input/top50spotify2019/top50.csv',encoding='ISO-8859-1')


# # Inspection/Cleaning
# We will begin by inspecting and cleaning the dataset. 
# 1. Inspection: Detect any incorrect and incosistent data.
# 2. Cleaning: Fix or remove any anomalies found.

# In[ ]:


songs_df.describe()


# In[ ]:


songs_df.isnull().sum()


# We will drop the 'Unnamed: 0' column as the column isn't connected to what we're interested in. 

# In[ ]:


#Dropping Unnamed column
songs_df = songs_df.drop('Unnamed: 0', axis = 1)


# Now we want to check the data types to make sure the data type matches with the values Spotify is saying the tool returns.

# In[ ]:


#checking the data types 
songs_df.dtypes


# We can visually plot the number of unique values per variable or we can obtain a quick count.

# In[ ]:


songs_df.nunique()


# # Visualization
# Now that the data set checks off our initial Inspection/Cleaning. We will plot some graphs to observe the distribution for our variables.
# 
# Once again, we're interested in seeing if there's a correlatino for the variables provided and the popularity obtained for a particular song.
# 
# For this we won't include "Track.Name" or "Artist.Name" in our model.

# In[ ]:


songs_df =  songs_df.drop(['Track.Name', 'Artist.Name'], axis = 1)


# In[ ]:


categorical = ['Genre']
numerical= ['Beats.Per.Minute', 'Energy', 'Danceability', 'Loudness..dB..', 'Liveness', 'Valence.', 'Length.', 'Acousticness..', 
            'Speechiness.']
target = 'Popularity'


# In[ ]:


#Obtaining the counts for each category from every variable
for i in songs_df.columns:
    print(songs_df[i].value_counts())


# In[ ]:


fig = plt.figure(figsize = (18, 12)) 
count  = 1
for i in numerical:
    ax = fig.add_subplot(5, 2, count)
    ax.hist(songs_df[i])
    ax.set_title(i)
    count += 1

fig.tight_layout()
plt.show()


# One of the assumptions when building a Linear Regression model is:
# * No or little multicollinearity
# 
# Therefore, we want to check if any is present amongst our variables. If any is seen, we will choose one of the variables that shares correlation with another and drop the other.

# In[ ]:


#correlation for numerical values
songs_corr = songs_df.corr()


# In[ ]:


# heatmap of the correlation 
plt.figure(figsize=(10,10))
plt.title('Correlation heatmap')
sns.heatmap(songs_corr,annot=True)


# It appears no multicollinearity is present.
# 
# Let's visualize it by creating some scatter plots.

# In[ ]:


g = sns.PairGrid(songs_df)
g = g.map(plt.scatter)


# # Dummy Coding
# 
# We have a nominal variable "Genre". Before we include the variable into our model we will obtain dummy variables for the Genres and see if "Genre" plays a role in popularity. Followed by dropping the "Genre" that has the most instances in our data set, to use as referance in our model.

# In[ ]:


#from out value counts we know dance_pop is our most popular, we will drop and use as referance
dummies_dropped_one = pd.get_dummies(songs_df['Genre'])
dummies_dropped_one = dummies_dropped_one.drop(columns = ['dance pop'])
#dropping genre column as we used it for our dummies
songs_df = songs_df.drop('Genre', axis = 1)
#combining our dummies with our other variables
songs_df = pd.concat([songs_df, dummies_dropped_one], axis = 1)


# # The Model
# 
# Perfect, Now we can build our model and see if there's a correlation amongst the variables chosen and "Popularity".

# In[ ]:


X = songs_df.loc[:, songs_df.columns != target]
y = songs_df[target].loc[:,]
X_1 = sm.add_constant(X, prepend = True, has_constant = 'add')
#%%
#Using SkLearn to create out training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.3, random_state=0)


# In[ ]:


#Statsmodels Linear Regression
method = sm.regression.linear_model.OLS(y_train.values.ravel(), X_train, has_constant = True)
result = method.fit()
print(result.summary())


# # Tuning
# 
# Our first model we can see our R-Square returns a high correlation meaning there is some connection between our variables and 'Popularity'. However, our adjusted R-square is relatively low, as well as many of our variables don't appear significant in our initial model. Let's perform some feature selection and see if we can generalize our model. 
# 
# Another thing we will do is standardize our data, Our variables have different scales and another assumption of Linear regression is:
# * Our data has a Normal Distribution

# In[ ]:


#Feature Selection
cols = list(X.columns)
pmax = 1 #placeholder for new p-value max
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1,prepend = True, has_constant = 'add')
    model = sm.OLS(y,X_1, hasconst = True).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols) #not idexing the constant column     
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)


# In[ ]:


#selecting our significant variables
X = songs_df.loc[:, selected_features_BE]
y = songs_df.loc[:,target]

#scaling our variables
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#adding a constant
X_1 = sm.add_constant(X, prepend = True, has_constant = 'add')
#Using SkLearn to create out training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.3, random_state=0)


# In[ ]:


#Statsmodels Logistic Regression
method = sm.regression.linear_model.OLS(y_train.values.ravel(), X_train, hasconst = True)
result = method.fit()
print(result.summary())


# # Final Takeaways
# 
# It appears there is a correlation between the variables chosen and "Popularity". More interestingly, the genres:
# * dfw rap 
# * electropop 
# * latin 
# * panamanian pop 
# * reggaeton
# * reggaeton flow
# 
# All seemed to have a higher popularity than our referance "dance pop".
# 
# Furthermore, it appears the aspects that describe the songs such as "Beats Per Minute" and others didn't show correlation with "Popularity". 
# 
# Obviously, this quick analysis is only on 50 songs. It would be interesting for re-do this for a larger data set of songs.
# 
# Cheers!
