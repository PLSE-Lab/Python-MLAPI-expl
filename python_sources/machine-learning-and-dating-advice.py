# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
"""
hi, as the fun title implies, this script will walk through the process of making data on the fly
and then doing predictions on that data. now this data isnt from any original source and as such,
doesnt have any significance but this example serves as a reference for doing simple pre-processing
and model-fitting to a data consisting of the 3 basic types people might encounter while doing machine
learning in the wild - number, text, category
"""
"""
function :  create_random_data_from_vals
it takes in a list of values and returns a list of the same values
spread randomly accross the length of that list so as to construct 
a dataset. this is the first step to building the dataset by script
inputs : 
list_inp(the list of values that we will spread accross the data)
size(the size of the resulting list that`ll hold the data to be fed to the dataframe)

outputs:
list of values of specified size and values as specified in the input list

"""
def create_random_data_from_vals(list_inp,size):
    return [list_inp[random.randint(0,len(list_inp)-1)]
            for a in range(size)]
"""
function :  convert_data
this function takes in our dataset and converts the columns
to text and categoricals as per the column names we provide in the 
parameters 
inputs : 
df(dataframe), 
categoricals(list of column names to be converted as categorical),
text(list of column names to be converted as strings)
outputs:
the converted dataframe

"""

def convert_data(df,categoricals,text):
    #convert the categorical columns into 
    #appropriate types
    for a in categoricals:
        df.loc[:,a]=df.loc[:,a].astype('category')
    #convert the string columns into 
    #appropriate types
    for a in text:
        df[a]=df[a].astype(str)
    
    #df.info()
    df = pd.get_dummies(df,
                          columns=categoricals,
                          drop_first=True)
    return df
"""
following are the features that will go into our dataset.

'age'	
'restaurant_type'
'rating'	
'date'	
'mood'	
'her_comments'	
'calls_u_back'	- target

the dataset will be constructed by assembling the lists that have randomly 
scattered values in their respective columns, so e.g. age can be between 20 - 28, 
hence we can create 200 different rows having  age between 20 and 28.
Same is for the rest of the columns.
"""
age = [a for a in range(20,28)]
restaurant_type = [
        'chinese',
        'oriental',
        'western',
        'steak',
        'vegan',
        'fast food'
        ]
rating=[a for a in range(2,5)]
date=[
      'evening',
      'afternoon'
      ]
mood=[
      'casual',
      'serious'
      ]
her_comments=[
        'good',
        'ho hum',
        'boring',
        'romantic',
        'fun']
calls_u_back=[
        'yes',
        'no']
"""
here`s where we create the data from the values provided above.
Again , this dataset is synthesized, it doesnt have any significance
or anything. It just showcases a prototyping approach to doing 
ML or practicing pre-processing on generated data.
"""
age_data	=	create_random_data_from_vals(age,200)
restaurant_type_data	=	create_random_data_from_vals(restaurant_type,200)
rating_data	=	create_random_data_from_vals(rating,200)
date_data	=	create_random_data_from_vals(date,200)
mood_data	=	create_random_data_from_vals(mood,200)
her_comments_data		=	create_random_data_from_vals(her_comments,200)
calls_u_back_data		=	create_random_data_from_vals(calls_u_back,200)


"""
specify the categoricals and text and then pass it in the 
dataframe conversion program
"""
categoricals=['restaurant_type',
                  'date',
                  'mood',
                  'calls_u_back']
text=['her_comments']

data={
'age'	:	age_data	,
'restaurant_type'	:	restaurant_type_data	,
'rating'	:	rating_data	,
'date'	:	date_data	,
'mood'	:	mood_data	,
'her_comments'	:	her_comments_data	,
'calls_u_back'	:	calls_u_back_data	
}
df=pd.DataFrame(data)
df=convert_data(df,categoricals,text)

"""
one important thing in this dataset are the comments
which are treated like text. hence we have to go about
using a CountVectorizer for having that data in our 
dataset as well.
"""
cnt_vect=CountVectorizer()

"""
her_comments is the only text column in our data. 
so we fit and transform that accordingly. After all,
her comments matter :-)
"""

d=cnt_vect.fit_transform(df.her_comments)

tfidf_transformer = TfidfTransformer().fit_transform(d)
"""
time to attach her comments to the dataframe on which 
we`ll do the predictions 
"""
df_text=pd.DataFrame(data=tfidf_transformer.A,
                     columns=cnt_vect.get_feature_names())
df=pd.concat([df,df_text],axis=1)
"""
no need to keep the her_comments column now that we have
them in a neat sparse matrix.
"""
df=df.drop(text,axis=1)
"""
when we performed pre-processing on the data,
it appended a column and now the target column is available as
'calls_u_back_yes' .we have to set that as our target in the
target variable y
"""

y=df.loc[:,'calls_u_back_yes']

"""
once we have the target column safe and sound in our target variable, 
we can safely discard that from our training data.
"""

df=df.drop('calls_u_back_yes',axis=1)
"""
do normalization on an ALL NUMBERS data now
"""

X=normalize(df)
"""
split the train test data accordingly.
"""
X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.3,
        random_state=42
        )
"""
now to do the model fitting.
"""

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print(knn.score(X_test,y_test))