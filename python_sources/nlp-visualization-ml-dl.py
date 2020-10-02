#!/usr/bin/env python
# coding: utf-8

# # IMPORTANT
# **This kernel has heavy html pages generated from ScatterText libirary, might cause the browser to take time to load please wait for the page to finish loading **

# ## Introduction
# in this kernel i'm going to explore the movies reviews data and try some new visualization for text , i'm also going to try different machine learning and Deep learning approches for prediction

# In[ ]:


## install empath libirary needed for scattertext
get_ipython().system('pip install empath')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np 
import seaborn as sns


# In[ ]:


train = pd.read_csv('../input/train.tsv', sep="\t")
test = pd.read_csv('../input/test.tsv', sep="\t")
sub = pd.read_csv('../input/sampleSubmission.csv', sep=",")


# In[ ]:


train.head(5)


# In[ ]:


test.head(5)


# lets see the imbalance between classes

# In[ ]:


train.groupby('Sentiment').Phrase.count().plot(kind='bar')


# i want to see how many sentence we have 

# In[ ]:


len(train.SentenceId.unique())


# ## Term Importance metrics
# **tf.idf difference** (not recommended)
# $$ \mbox{Term Frquency}(\mbox{term}, \mbox{category}) = \#(\mbox{term}\in\mbox{category}) $$$$ \mbox{Inverse Document Frquency}(\mbox{term}) = \log \frac{\mbox{# of categories}}{\mbox{# of categories containing term}} $$$$ \mbox{tfidf}(\mbox{term}, \mbox{category}) = \mbox{Term Frquency}(\mbox{term}, \mbox{category}) \times \mbox{Inverse Document Frquency}(\mbox{term}) $$$$ \mbox{tfidf-difference}(\mbox{term}, \mbox{category}) = \mbox{tf.idf}(\mbox{term}, \mbox{category}_a) - \mbox{tf.idf}(\mbox{term}, \mbox{category}_b) $$
# 
# Tf.idf ignores terms used in each category. Since we only consider two categories (positive, negative), a large number of terms have zero (log 1) scores. The problem is Tf.idf doesn't weight how often a term is used in another category. This causes eccentric, brittle, low-frequency terms to be favored.
# 
# This formulation does take into account data from a background corpus.
# $$ \#(\mbox{term}, \mbox{category}) \times \log \frac{\mbox{# of categories}}{\mbox{# of categories containing term}} $$

# 
# **Scaled F-Score**
# 
# Associatied terms have a relatively high category-specific precision and category-specific term frequency (i.e., % of terms in category are term)
# Take the harmonic mean of precision and frequency (both have to be high)
# We will make two adjustments to this method in order to come up with the final formulation of Scaled F-Score
# 
# Given a word $w_i \in W$ and a category $c_j \in C$, define the precision of the word $w_i$ wrt to a category as: $$ \mbox{prec}(i,j) = \frac{\#(w_i, c_j)}{\sum_{c \in C} \#(w_i, c)}. $$
# 
# The function $\#(w_i, c_j)$ represents either the number of times $w_i$ occurs in a document labeled with the category $c_j$ or the number of documents labeled $c_j$ which contain $w_i$.
# 
# Similarly, define the frequency a word occurs in the category as:
# $$ \mbox{freq}(i, j) = \frac{\#(w_i, c_j)}{\sum_{w \in W} \#(w, c_j)}. $$
# 
# The harmonic mean of these two values of these two values is defined as:
# $$ \mathcal{H}_\beta(i,j) = (1 + \beta^2) \frac{\mbox{prec}(i,j) \cdot \mbox{freq}(i,j)}{\beta^2 \cdot \mbox{prec}(i,j) + \mbox{freq}(i,j)}. $$
# 
# $\beta \in \mathcal{R}^+$ is a scaling factor where frequency is favored if $\beta <1$, precision if $\beta >1$, and both are equally weighted if $\beta = 1$. F-Score is equivalent to the harmonic mean where $\beta = 1$.
# 

# 

# There are some problem with harmonic means, they are dominated by the precision.
# a solution for that is to take the normal CDF of precision and frequency percentage scores, which will fall between 0 and 1, which scales and standardizes both scores.
# but i'm not going to try this approach here.

# Lets look at the most positive words and most negative with respect to the scaled F score 

# In[ ]:


import scattertext as st
import spacy
from pprint import pprint
from IPython.display import IFrame
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:98% !important; }</style>"))

data=train[(train['Sentiment']==0)|(train['Sentiment']==4)]
data['cat']=data['Sentiment'].astype("category").cat.rename_categories({0:'neg',4:'pos'})


# In[ ]:


corpus = st.CorpusFromPandas(data, 
                             category_col='cat',                               
                             text_col='Phrase',
                             nlp=st.whitespace_nlp_with_sentences).build()


# In[ ]:


term_freq_df = corpus.get_term_freq_df()
term_freq_df['positive Score'] = corpus.get_scaled_f_scores('pos')
pprint(list(term_freq_df.sort_values(by='positive Score', 
                                      ascending=False).index[:10]))


# In[ ]:


term_freq_df['negative Score'] = corpus.get_scaled_f_scores('neg')
pprint(list(term_freq_df.sort_values(by='negative Score', 
                                      ascending=False).index[:10]))


# ### Visualizing term associations
# here i'm making a scatter plot for the terms that associated with the (positive,negative) classes .
# Each dot corresponds to a word or phrase in the two classes. The closer a dot is to the top of the plot, the more frequently it was used in positive reviews. The further right a dot, the more that word or phrase was used in negative reviews. Words frequently used by both parties, like "cast" and "film" and even "movie" tend to occur in the upper-right-hand corner. Although very low frequency words have been hidden to preserve computing resources, a word that neither party used, like "implausible" would be in the bottom-left-hand corner.

# In[ ]:


html = st.produce_scattertext_explorer(corpus,
         category='pos',category_name='positive',         
        not_category_name='neg',width_in_pixels=1000,
          metadata=data['cat'])
open("Convention-Visualization.html", 'wb').write(html.encode('utf-8'))
IFrame(src='Convention-Visualization.html', width = 1300, height=700)


# ### Visualizing Empath topics and categories
# 
# Often the terms of most interest are ones that are characteristic to the corpus as a whole. These are terms which occur frequently in all sets of documents being studied, but relatively infrequent compared to general term frequencies.

# In[ ]:


feat_builder = st.FeatsFromOnlyEmpath()
empath_corpus = st.CorpusFromParsedDocuments(data,
                                              category_col='cat',
                                              feats_from_spacy_doc=feat_builder,
                                              parsed_col='Phrase').build()
html = st.produce_scattertext_explorer(empath_corpus,
                                        category='pos',
                                        category_name='Positive',
                                        not_category_name='Negative',
                                        width_in_pixels=1000,
                                        metadata=data['cat'],
                                        use_non_text_features=True,
                                        use_full_doc=True,
                                        topic_model_term_lists=feat_builder.get_top_model_term_lists())
open("Convention-Visualization-Empath.html", 'wb').write(html.encode('utf-8'))
IFrame(src='Convention-Visualization-Empath.html', width = 1300, height=700)


# ### lexicalized semiotic squares 
# The idea behind the semiotic square is to express the relationship between two opposing concepts and concepts things within a larger domain of a discourse. Examples of opposed concepts life or death, male or female, or, in our example, positive or negative sentiment. Semiotics squares are comprised of four "corners": the upper two corners are the opposing concepts, while the bottom corners are the negation of the concepts.
# 
# Circumscribing the negation of a concept involves finding everything in the domain of discourse that isn't associated with the concept. For example, in the life-death opposition, one can consider the universe of discourse to be all animate beings, real and hypothetical. The not-alive category will cover dead things, but also hypothetical entities like fictional characters or sentient AIs.

# In[ ]:


data=train[(train['Sentiment']!=1)&(train['Sentiment']!=3)]
data['cat']=data['Sentiment'].astype("category").cat.rename_categories({0:'neg',2:'neu',4:'pos'})


# In[ ]:


corpus = st.CorpusFromPandas(
    data,
    category_col='cat',
    text_col='Phrase',
    nlp=st.whitespace_nlp_with_sentences
).build().get_unigram_corpus()

semiotic_square = st.SemioticSquare(
    corpus,
    category_a='pos',
    category_b='neg',
    neutral_categories=['neu'],
    scorer=st.RankDifference(),
    labels={'not_a_and_not_b': 'Neutral', 'a_and_b': 'Reviews'})

html = st.produce_semiotic_square_explorer(semiotic_square,
                                           category_name='Positive',
                                           not_category_name='Negative',
                                           x_label='pos-neg',
                                           y_label='neu-Review',
                                           neutral_category_name='neutral',
                                           metadata=data['cat'])


# In[ ]:


open("lexicalized_semiotic_squares.html", 'wb').write(html.encode('utf-8'))
IFrame(src='lexicalized_semiotic_squares.html', width = 1600, height=900)


# ## ML Approaches
# there are two common ways for using text as input to machine learning algorthim Bag of words, TF-idf, here i'm goin to use TFIDF and i will try some different algorthims to see which one will gove me better accuracy.

# In[ ]:


import re
def clean(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    return text


# In[ ]:


train['Phrase'] = train['Phrase'].apply(lambda x: clean(x))
test['Phrase']=test['Phrase'].apply(lambda x: clean(x))


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
x_train, x_test, y_train, y_test = train_test_split(train['Phrase'], train['Sentiment'], train_size=0.8)
vectorizer = TfidfVectorizer().fit(x_train)
x_train_v = vectorizer.transform(x_train)
x_test_v  = vectorizer.transform(x_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from time import time
entries = []
def training():
    models = {
        "LogisticRegression": LogisticRegression(),
        "SGDClassifier": SGDClassifier(),
        "Multinomial":MultinomialNB(),
        "LinearSVC": LinearSVC(),
        "GBClassifier":GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, max_depth=1, random_state=0)
    }
    for model in models:
        print("training model"+model)
        start = time()
        models[model].fit(x_train_v, y_train)
        end = time()
        print("trained in {} secs".format(end-start))
        y_pred = models[model].predict(x_test_v)
        entries.append((model,accuracy_score(y_test, y_pred)))


# In[ ]:


training()


# In[ ]:


cv_df = pd.DataFrame(entries, columns=['model_name','accuracy'])
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


# Here Linear SVC gives us 64% 
# 
# Lets visualzie this weights as bag of words features.

# ### Visualizing scikit-learn text classification weights
# 
# lets look at what this weights interpret our test and train data

# Test data

# In[ ]:


corpus = st.CorpusFromScikit(
    X=CountVectorizer(vocabulary=vectorizer.vocabulary_).fit_transform(x_test[0:1000]),
    y=y_test[0:1000].values,
    feature_vocabulary=vectorizer.vocabulary_,
    category_names=['neg','som_neg','neu','som_pos','pos'],
    raw_texts=x_test[0:1000].values
).build()

clf=LinearSVC()
clf.fit(x_test_v,y_test)
html = st.produce_frequency_explorer(
    corpus,
    'neg',
    scores=clf.coef_[0],
    use_term_significance=False,
    terms_to_include=st.AutoTermSelector.get_selected_terms(corpus, clf.coef_[0])
)
file_name = "test_sklearn.html"
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width = 1300, height=700)


# Train data

# In[ ]:


corpus = st.CorpusFromScikit(
    X=CountVectorizer(vocabulary=vectorizer.vocabulary_).fit_transform(x_train[0:1000]),
    y=y_train[0:1000].values,
    feature_vocabulary=vectorizer.vocabulary_,
    category_names=['neg','som_neg','neu','som_pos','pos'],
    raw_texts=x_train[0:1000].values
).build()

clf=LinearSVC()
clf.fit(x_train_v,y_train)
html = st.produce_frequency_explorer(
    corpus,
    'neg',
    scores=clf.coef_[0],
    use_term_significance=False,
    terms_to_include=st.AutoTermSelector.get_selected_terms(corpus, clf.coef_[0])
)
file_name = "train_sklearn.html"
open(file_name, 'wb').write(html.encode('utf-8'))
IFrame(src=file_name, width = 1300, height=700)


# ### DL Approach
# So i have tried different approaches like LSTM,GRU,bidirectional but none of them was good enough so i will use the model mentioned in this  [kernenl](https://www.kaggle.com/parth05rohilla/bi-lstm-and-cnn-model-top-10#) which gives an very good results BTW 

# ### Keras API 
# i can make custom function that transform out text to sequence of integers but here i will use keras tokenizer api will make it much easier 

# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()
full_text=list(train['Phrase'].values) + list(test['Phrase'].values)
tokenizer.fit_on_texts(full_text)
train_seq = tokenizer.texts_to_sequences(train['Phrase'])
test_seq=tokenizer.texts_to_sequences(test['Phrase'])


# In[ ]:


voc_size=len(tokenizer.word_counts)


# In[ ]:


m=len(max(full_text, key=len))
X_train = pad_sequences(train_seq, maxlen = m)
X_test = pad_sequences(test_seq, maxlen = m)


# In[ ]:


y_train=train['Sentiment'].values
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y_train.reshape(-1, 1))


# In[ ]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping

file_path = "best_model.hdf5"
check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                              save_best_only = True, mode = "min")
early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)

def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (m,))
    x = Embedding(19479, 300)(inp)
    x1 = SpatialDropout1D(dr)(x)

    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x1)
    max_pool1_gru = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool3_gru = GlobalAveragePooling1D()(x3)
    max_pool3_gru = GlobalMaxPooling1D()(x3)
    
    x_lstm = Bidirectional(CuDNNLSTM(units, return_sequences = True))(x1)
    x1 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x1)
    max_pool1_lstm = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(32, kernel_size=2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool3_lstm = GlobalAveragePooling1D()(x3)
    max_pool3_lstm = GlobalMaxPooling1D()(x3)
    
    
    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
                    avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(128,activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(0.2)(Dense(100,activation='relu') (x))
    x = Dense(5, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    print(model.summary())
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    history = model.fit(X_train, y_ohe, batch_size = 128, epochs = 10, validation_split=0.1, 
                        verbose = 1, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model,history


# In[ ]:


model,history = build_model(lr = 1e-4, lr_d = 0, units = 128, dr = 0.5)


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:




plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


# In[ ]:


pred = model.predict(X_test, batch_size = 1024)


# In[ ]:


predictions = np.round(np.argmax(pred, axis=1)).astype(int)
sub['Sentiment'] = predictions
sub.to_csv("blend.csv", index=False)


# ## Resources
# [ScatterText doc](https://github.com/JasonKessler/scattertext)
# 
# [Amazing slides for explansion to F-scaled and other ideas](https://www.slideshare.net/JasonKessler/turning-unstructured-content-into-kernels-of-ideas/)

# 

# 

# 

# 
