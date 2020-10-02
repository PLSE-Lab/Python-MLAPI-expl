#!/usr/bin/env python
# coding: utf-8

# # Downloading Data

# There are three ways of getting hold of the data. If you're running this notebook locally, you can download the data manually or through the Kaggle API. If you're running this notebook as a kernel on Kaggle, the data is already there and you don't need to do anything. In all three cases you need to have [signed in](https://www.kaggle.com/) to Kaggle and [accepted the rules](https://www.kaggle.com/c/tmdb-box-office-prediction/rules) of the competition in order to be able to do anything. 
# 
# ### Downloading manually
# 
# Download the [data](https://www.kaggle.com/c/10300/download-all) next to this notebook. Then unzip it to a directory called `input`:

# ### Downloading with the Kaggle API

# If you already set up the [Kaggle API](https://github.com/Kaggle/kaggle-api), you can download the data for this competition with the following commands. If you get a *403 Permission Denied* error, you didn't [accept the rules](https://www.kaggle.com/c/tmdb-box-office-prediction/rules) of the competition yet.

# ### Running this notebook as a kernel on Kaggle
# 
# You don't need to do anything. The data is already there.

# # Loading Data

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
import os


# In[ ]:


if os.getcwd() == '/kaggle/working':
    print("Running on Kaggle kernel")
    data_folder = '../input/'
else:
    data_folder = ''

train = pd.read_csv(data_folder + 'train.csv')
test = pd.read_csv(data_folder + 'test.csv')

print('train dataset size:', train.shape)
print('test dataset size:', test.shape)

train.sample(4)


# ## Feature engineering
# 
# 

# In[ ]:


train.info()


# ### Process JSON-style features
# 
# There are 8 JSON-style features, 4 numerical, 4 text, and 1 date feature. At first, convert JSON-styled features into string/category/list ones.
# 
# * **`belongs_to_collection`**: convert `name` into string
# * **`genres`, `production_companies`**: convert `name` values into comma-separated string list
# * **`production_countries`**: convert `iso_3166_1` values into comma-separated string list
# * **`spoken_languages`**: convert `iso_639_1` values into comma-separated string list
# * **`Keywords`**: convert `name` values into comma-separated string list
# * **`cast`, `crew`**: get their lengths, as its detailed information is very unlikely relevant to the revenue 

# In[ ]:


def proc_json(string, key):
    try:
        data = eval(string)
        return ",".join([d[key] for d in data])
    except:
        return ''

def proc_json_len(string):
    try:
        data = eval(string)
        return len(data)
    except:
        return 0

train.belongs_to_collection = train.belongs_to_collection.apply(lambda x: proc_json(x, 'name'))
test.belongs_to_collection = test.belongs_to_collection.apply(lambda x: proc_json(x, 'name'))

train.genres = train.genres.apply(lambda x: proc_json(x, 'name'))
test.genres = test.genres.apply(lambda x: proc_json(x, 'name'))

train.production_companies = train.production_companies.apply(lambda x: proc_json(x, 'name'))
test.production_companies = test.production_companies.apply(lambda x: proc_json(x, 'name'))

train.production_countries = train.production_countries.apply(lambda x: proc_json(x, 'iso_3166_1'))
test.production_countries = test.production_countries.apply(lambda x: proc_json(x, 'iso_3166_1'))

train.spoken_languages = train.spoken_languages.apply(lambda x: proc_json(x, 'iso_639_1'))
test.spoken_languages = test.spoken_languages.apply(lambda x: proc_json(x, 'iso_639_1'))

train.Keywords = train.Keywords.apply(lambda x: proc_json(x, 'name'))
test.Keywords = test.Keywords.apply(lambda x: proc_json(x, 'name'))

train.cast = train.cast.apply(proc_json_len)
test.cast = test.cast.apply(proc_json_len)

train.crew = train.crew.apply(proc_json_len)
test.crew = test.crew.apply(proc_json_len)


# ### Genres
# As a movie can have multiple genres, it is not a reasonable way to convert `genres` column into category type. It might make the same genres different, e.g. 'Drama,Romance' and 'Romance,Drama' would be categorized differently. Therefore we make dummy columns for all of the genres.

# In[ ]:


# get total genres list
genres = []
for idx, val in train.genres.iteritems():
    gen_list = val.split(',')
    for gen in gen_list:
        if gen == '':
            continue

        if gen not in genres:
            genres.append(gen)
            

genre_column_names = []
for gen in genres:
    col_name = 'genre_' + gen.replace(' ', '_')
    train[col_name] = train.genres.str.contains(gen).astype('uint8')
    test[col_name] = test.genres.str.contains(gen).astype('uint8')
    genre_column_names.append(col_name)


# ### Normalize Revenue and Budget
# 
# Budget and revenue are highly skewed and they need to be normalized by logarithm.

# In[ ]:


train['revenue'] = np.log1p(train['revenue'])
train['budget'] = np.log1p(train['budget'])
test['budget'] = np.log1p(test['budget'])

train.sample(5)


# ## Select features
# 
# For this benchmark solution, we are manually selecting the following features:

# In[ ]:


features = ['budget', 'popularity', 'runtime', 'genre_Adventure', 'genre_Action', 'genre_Fantasy', 
            'genre_Drama', 'genre_Family', 'genre_Animation', 'genre_Science_Fiction']


# ### Fix missing values
# 
# The `runtime` column has 2 and 4 missing values for the train and test dataset respectively:

# In[ ]:


print('-'*30)
print(train[features].isnull().sum())
print('-'*30)
print(test[features].isnull().sum())


# So let's fill the missing values with the mean of the other runtimes.

# In[ ]:


train.runtime = train.runtime.fillna(np.mean(train.runtime))
test.runtime = test.runtime.fillna(np.mean(train.runtime))


# In[ ]:


train[features].sample(5)


# ## Train model
# 
# Let's use cross validation on a linear regression model using the selected features to check the explained variance:

# In[ ]:


X, y = train[features], train['revenue']
model = LinearRegression()
result = cross_validate(model, X, y, cv=10, scoring="explained_variance", verbose=False, n_jobs=-1)
print(f"The variance explained is {np.mean(result['test_score']):.1%}")


# ## Make predictions
# 
# Now let's train the model and make predictions for the test set:

# In[ ]:


model.fit(X, y)
predict = model.predict(test[features])


# ## Create submission file
# 
# Create a submission file from the predictions:

# In[ ]:


submit = pd.DataFrame({'id': test.id, 'revenue':np.expm1(predict)})
submit.to_csv('submission.csv', index=False)


# ## Submit your results to Kaggle
# 
# ### With the Kaggle API
# 
# Run the following command:

# Then go to [your submissions](https://www.kaggle.com/c/tmdb-box-office-prediction/submissions) to see your latest results.
#  
# ### By uploading your submission file manually
# 
# Go to the [submission page](https://www.kaggle.com/c/tmdb-box-office-prediction/submit) on Kaggle and click the *'Upload Files'* button below to upload your .csv and have it scored, so you can see your new ranking!
# 
# ### When you ran the notebook as a kernel on Kaggle
# 
# Click the *'Commit'* button at the top to commit (and save) your work and run the notebook. Note that it should run successfully from start to finish, so make sure you remove any cells that were meant to be run from a local machine for example.
# 
# Once it ran, click the blue *'Open Version'* button at the right, then click on *'Output'* at the left, which will point you to the *'Output'* section of this page. There you will see a *'Submit to competition'* button. Clicking that will score your submission file and show you your new ranking!

# In[ ]:




