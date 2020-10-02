#!/usr/bin/env python
# coding: utf-8

# ### An attempt at Modeling Stock market prices with NLP

# ![enter image description here][1]
# 
# 
#   [1]: http://cdn.images.express.co.uk/img/dynamic/22/590x/secondary/dow-jones-stocks-810706.jpg

# In this Kernel we will make an attempt at predicting the DJIA trend with various methods ranging from Natural langage Processing to sequence based models.
# We will also try to inspect the models in order to undestand the methodology involved in each algorithm.

# In[ ]:


import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
import plotly.plotly as py
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')


# In[ ]:


Table = pd.read_csv("../input/Combined_News_DJIA.csv")


# In[ ]:


Table.info()


# The metadata for the file show no missing data ,there is one numeric column Label containing the target we will try to predict. 

# In[ ]:


Table.Date = pd.to_datetime(Table.Date) # First convert The Date col to Date format
fig = plt.figure(figsize=(20,10))
plt.plot(Table.Date,Table.Label);


# This plot is not very informative but we can observe that **Bullish/bearish days seem following each other**, this is a well known phenomena in financial econometrics [(see Arch/Garch models )][1], we will see later if we can do somthing to capture this...
# This is much clearer using a calendar plot, from the library calmap for example, I added a snapshot below for a few years.
# 
# 
#   [1]: https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity

# ![enter image description here][1]
# 
# 
#   [1]: https://cloud.githubusercontent.com/assets/22575341/26532195/9ba3fdaa-43fa-11e7-825c-e369f75cd4af.PNG

# In[ ]:


Visualizing the actual Series can also be useful 


# In[ ]:


index_price = pd.read_csv("../input/DJIA_table.csv")
index_price.Date = pd.to_datetime(index_price.Date)
plt.figure(figsize=(10,8))
plt.plot(index_price.Date, index_price.Close,label = "DJIA closing price");
plt.plot(index_price.Date, index_price.Volume/100000,label = "Volumes");# scale volumes for readability
plt.legend();
plt.title("DJIA stocks");


# In[ ]:


print("Porportion of bullish days: {0:.2f}%".format(Table.Label.mean()))


# ## Let's start with a very basic bag-of-words model ##
# The intuition behind this simple model is that the occurence of some particular words or sequence of words in the reddit News is linked with an event which itself had an impact on the index.
# 
# 
#  - We first need to clean the data, to avoid modeling too much noise we will use the python natural langage toolkit nltk
# 
# 

# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from string import punctuation
from nltk.corpus import stopwords
snowball = SnowballStemmer("english")


# In[ ]:


stem = lambda word: snowball.stem(word)
punctuation = ''.join([p for p in punctuation if p not in ['.','!',"?","-"]])


# In[ ]:


def clean_text(text,stem=False,remove_punct='punct',remove_stopwords=False,return_list=False):
    text=str(text)
    if text.startswith("b'"): # remove the byte types strings which have been converted to text
        text=text[2:]
    words = word_tokenize(text)
        
    # Optionally use stemmer
    if stem:
        words = [stem(w) for w in words]
    if remove_punct:
        if remove_punct=='all':
            words = [re.sub("[^a-zA-Z\.\?\!]"," ", w) for w in words]
        elif remove_punct=='punct':
            punct = set(punctuation)
            words = [''.join(ch for ch in w if ch not in punct) for w in words]
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    
    words = [w for w in words if w not in ['','b']]
    if return_list:
        return(words)
    else:
        return ' '.join(words)


# In[ ]:


print("                 Raw                               ---               Cleaned")
for i ,(word1,word2) in enumerate(zip(Table.Top1,Table.Top1.map(lambda x: clean_text(x)))):
    if i>15: break
    print(word1[:50],'---',word2[:50])


# **Apply the cleaning over all news**

# In[ ]:


get_ipython().run_cell_magic('time', '', "cols = [t for t in list(Table.columns) if t not in ['Label','Date']]\nCleanedTable = Table[cols].apply(lambda i: i.map(lambda x: clean_text(x)))")


# **We now concatenate all columns in order to get a single document per day, made of all top 25 news We might drop the information related to the rank of the News but we thus have a dictionnary containing all words for a given day**

# In[ ]:


corpus = []
for i,line in enumerate(CleanedTable.index):
    line_doc=''
    for col in CleanedTable:
        line_doc += " "+str(CleanedTable.ix[i,col])
    corpus.append(line_doc)


# We now vectorize those documents into a bag of words model counting the occurence of each word of the vocabulary.
# The second methods use tfidf Vectorizer, this methods weights the words by assuming that:
# 
#  - The frequency of word within a document represents the importance of this word in the document.
#  - Terms occuring often across documents have a low discriminative power and then should be given a lower weight  
# 
#  You can find more information about this functions in the scikitlearn [Documentation][1] and [User Guide][2]
# 
# 
#   [1]: http://We%20now%20vectorize%20those%20documents%20into%20a%20bag%20of%20words%20model%20counting%20the%20occurence%20of%20each%20word%20of%20the%20vocabulary.%20The%20second%20methods%20use%20tfidf%20Vectorizer,%20this%20methods%20weights%20the%20words%20by%20assuming%20that:%20%201.%20The%20frequency%20of%20word%20within%20a%20document%20represents%20the%20importance%20of%20this%20word%20in%20the%20document.%202.%20Terms%20occuring%20often%20across%20documents%20have%20a%20low%20discriminative%20power%20and%20then%20should%20be%20given%20a%20lower%20weight%20%20%20%20%20You%20can%20find%20more%20information%20about%20this%20functions%20in%20the%20scikitlearn%20Documentation%20and%20User%20Guide%20%20%20%20%20http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
#   [2]: http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer\ndata = pd.Series(corpus)\ncv = CountVectorizer(\n    analyzer ='word',\n    ngram_range=(1,3), # we include (1,3) ngrams since they might have a higher predictive power than single words \n    stop_words='english',\n    max_df = 0.7,\n    min_df=5 # from my experience the minimum robust occurence frequency for a word is in the range [4,15]\n)\ntf=TfidfVectorizer(\n    analyzer ='word',\n    ngram_range=(1,3),\n    stop_words='english',\n    max_df = 0.7,\n    min_df=5 \n)\n\ncount_matrix = cv.fit_transform(data)\ntfidf_matrix = tf.fit_transform(data)")


# In[ ]:


# train test split according to description
X_train = tfidf_matrix[:1611]
X_test = tfidf_matrix[1610:]
y_train = Table.Label[:1611]
y_test = Table.Label[1610:]


# We will now fit a Logistic Regression with the tfidf matrix as input using gridsearch to find the optimal parameters.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.linear_model import LogisticRegression\nimport scipy\nfrom sklearn.model_selection import GridSearchCV\nlm = LogisticRegression()\n\nparam_grid = {\n    \'C\': np.logspace(0.01,20,10),\n    "penalty" :[\'l1\',\'l2\']\n}\n\ngs = GridSearchCV(lm, param_grid,\n                        cv=3,\n                        n_jobs=-1,\n                        scoring="roc_auc")\ngs.fit(X_train, y_train)')


# In[ ]:


lm = gs.best_estimator_
print('best params:',gs.best_params_)
print('best CV score:', gs.best_score_)

lm.fit(X_train,y_train)
print("Nb of significative features",sum(np.abs(lm.coef_)[0]>0))
print("{0:.2f}% of features excluded by regularization".format((1-sum(np.abs(lm.coef_)[0]>0)/len(lm.coef_[0,:]))*100))


# In[ ]:


pred = lm.predict(X_test)
from sklearn.metrics import accuracy_score,auc,roc_auc_score

print("acc:",accuracy_score(y_test,pred))
print("auc:",roc_auc_score(y_test,pred))


# The accuracy on the test set is very close from a random guess (0.5) and from the number of Positive Labels in the sample, the predictive power of this model is highly questionable.
# Cross validation suggests that the best set of hyperparameters induce almost no regularization, meaning that all variables are meaningful...
# 
# Using the CountVectorizer instead similarly yields  poor results
# 
# Nevertheless we can inspect the model weights and try to see which words are used for prediction.

# In[ ]:


var_imp = pd.DataFrame({"features":cv.get_feature_names(),
              "coefs":pd.Series(lm.coef_[0,])})
var_imp.index = var_imp.features
var_imp['color'] = var_imp.coefs.map(lambda l: l>0)

plot_table = pd.concat([var_imp.sort_values(by='coefs').head(20),var_imp.sort_values(by='coefs').tail(20)])
dic = {True:'g',False:'r'}
plot_table["coefs"].plot(kind='barh',figsize = (10,13),
                         color = plot_table.color.map(dic),
                         title = "Bag of word Feature Importance",
                         label="color");


# We can now check if some expressions are included in the model and what is their "impact" on the DJIA price

# In[ ]:


def print_coef(terms,print_errors=True,threshold=0.5):
    for term in terms:
        term=term.lower()
        try:
            if np.abs(var_imp.ix[term,"coefs"])>threshold:
                print("Coef value for term",'"{}"'.format(term),"is : {0:.2}".format(var_imp.ix[term,"coefs"]))
            else:
                if print_errors:
                    print('The term "{}"is not highly significative'.format(term))
        except:
            if print_errors:
                print('"{}" not in variables'.format(term))

terms = ['global crisis','putin says','china trying','job losses',"germany says"
         ,'killed civilians','globalization','afghan military','avian flu','tsunami hit',"civil war"]
print_coef(terms)


# We can confirm the poor results of the model since some words seem to have an opposite impact as what we would expect for example "tsunami hit" has a positive impact, the coefficients suggest that when germany is speaking this yields has a bad impact on the shares price of american industrial companies (this could be debated...).
# Nevertherless some coefficients make sense such as "afghan military" or "avian flu".
# ## Let's check the influence of the mention of Country names in the model ##

# In[ ]:


from nltk.corpus import gazetteers
# we import a list of all country names from nltk 
# and pass it through our function
countries = gazetteers.words(fileids="countries.txt")
print_coef(countries,print_errors=False,threshold=5)


# Here again we can see some curious results, but we won't take the risk to comment them...

# In[ ]:




