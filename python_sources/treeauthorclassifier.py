# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd
from sklearn.cross_validation import train_test_split
from nltk.corpus import stopwords # for stopwords (commonly used words to ignore)
import nltk # for stopwords
import re
import numpy as np

postsDF = pd.read_csv('../input/posts.csv')

# split into test and train 


train, test = train_test_split(postsDF, test_size = 0.2) #creates train and test dataframes 

train = train.reset_index()
test = test.reset_index()




def post_to_words( raw_review ):
    # Function to convert a text to a processed string of words
    # The input is a single string (a raw blog post), and 
    # the output is a single string (a preprocessed blog post)
    #

    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))  


num_text= train["text"].size # number of rows
# a list for clean blog text to go in 
clean_train_text = []

# print train['text']

for i in range( 0, num_text ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    text=train["text"][i]
    t=post_to_words( text )
    clean_train_text.append(t)
    # print to confirm working every 50 posts
    # if( (i+1)%50 == 0 ): 
    # 	print "text %d of %d\n" % ( i+1, num_text )     


from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(analyzer = "word",  
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 10000) 


train_data_features = vectorizer.fit_transform(clean_train_text)

train_data_features = train_data_features.toarray()

# print train_data_features.shape

vocab = vectorizer.get_feature_names()
# print vocab


# dist = np.sum(train_data_features, axis=0)
# for tag, count in zip(vocab, dist):
#     print count, tag

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit( train_data_features, train["author"] )

# print test.shape

num_posts = len(test["text"])

clean_test_text = [] 

for i in range(0,num_posts):
    # if( (i+1) % 1000 == 0 ):
        # print "Review %d of %d\n" % (i+1, num_posts)
    clean_post = post_to_words( test["text"][i] )
    clean_test_text.append( clean_post )


test_data_features = vectorizer.transform(clean_test_text)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)

# print test.columns.values


output = pd.DataFrame( data={"id":test["index"], "author":result} )

output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

print(sum(output['author']=='Alex Tabarrok'))

print(sum(test['author']=='Alex Tabarrok'))

# print test


# Any results you write to the current directory are saved as output.