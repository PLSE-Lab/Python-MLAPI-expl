#!/usr/bin/env python
# coding: utf-8

# # Document Clustering
# 
# ## Why we're here
# 
# Goals:
# 
# - Pique your curiosity about Natural Language Processing (NLP)
# - Introduce core tools in the Python NLP/Data Science ecosystem
# - Cluster some documents
# - Help your neighbours!
# - Have fun!
# 
# Non-goals:
# 
# - Finish the whole thing (but you can certainly try...)
# 
# ## Getting started
# 
# Click on the below "cell" and hit "Ctrl-Return" to run it

# In[ ]:


# Import the standard toolkit...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ...and a few NLP specific things
import spacy
from spacy import displacy
from wordcloud import WordCloud

# ...and switch on "in notebook" charts, and make them a bit bigger!
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 6)

# ...then print a silly message to make it clear we're done
print("Reticulating splines... DONE")


# ## Over to you: hipster business name generator
# 
# Generally, you're just going go through this notebook executing the code blocks as you did above.
# 
# But you'll also find a few "Over to you" sections scattered throughout. These are a prompt for you to experiment and try things out.
# 
# To get you started just follow the instructions after the big comment `#####` below...

# In[ ]:


def show_hipster_biz_name(a, b):
    from codecs import decode
    from zlib import adler32
    from IPython.display import HTML
    ns = 'mrcule/zvak/pbj/pbea/cvtrba/funpxyr/obngzna/pbyyne'
    qs = 'zbhfgnpur/nagvdhr/pebpurgrq/negvfnany'
    bs = 'pbzof/glcrjevgref/fyvccref/ancxvaf/jubyrsbbqf'
    def tr(n, c):
        c = decode(c, "rot13").split('/')
        return c[adler32(bytes(n, 'utf8')) % len(c)].title()
    n = "{} & {} {} {}".format(tr(a, ns), tr(b, ns), tr(a+b, qs), tr(b+a, bs))
    s = "font-family:serif;font-size:28pt;text-align:center;border:4px double black;padding:10px;"
    print("Your Hipster business name is:")
    display(HTML("<h1 style='{}'>{}</h1>".format(s, n)))

########################################################################################
# Edit the line below to include your name and your neighbour's name, then run the cell
########################################################################################

show_hipster_biz_name("Mac", "Dexter")


# ## Over to you: getting comfortable with the Kaggle Kernel
# 
# - Click in any cell then click the blue `+` in the toolbar to create a new cell above or below
# - Enter some Python code (if you don't know any try `print(1 + 2)` or copy your neighbour!)
# - Click the `Run` button (or Ctrl-Enter) to execute it
# - Write some code to print the result of multiplying `1337` with `1337` (Hint: use `print(...)`)

# In[ ]:


# Experiment here!
print(1337*1337)


# # Introducing SpaCy
# 
# Okay, let's do some Natural Language Processing (NLP)!
# 
# SpaCy is a relative newcomer to the NLP scene, but has made a big splash because:
# - it's fast
# - it ships with high-performance models for a few different languages
# - it's got a very nice API and excellent documentation
# 
# They also have great marketing: https://spacy.io/

# In[ ]:


# Load up the english language models... this takes a while!

nlp = spacy.load("en_core_web_lg")
print("{name}: {description}".format(**nlp.meta))


# In[ ]:


# Okay let's use SpaCy to process a simple sentence
# The fundamental operation is to create a stuctured "Doc" representation of a text. Let's take a look!

text = u"Pack my bag with twelve dozen liquor jugs."
doc = nlp(text)
doc.to_json()


# In[ ]:


# But since we're in Jupyter we can do a lot better than that!
# The "Parts of Speech" e.g. VERB are drawn from the "Universal POS Tag" vocabulary
# Find out more at http://universaldependencies.org/u/pos/

options={'jupyter': True, 'options':{'distance': 120}}
displacy.render(doc, style='dep', **options)


# In[ ]:


# Spacy also ships with an "entity recogniser" -- it's pretty good!

ghostbusters = nlp(u"In the eponymous 1984 film, New York City celebrated the Ghostbusters with a ticker tape parade.")
displacy.render(ghostbusters, style="ent", **options)


# ## Over to you: SpaCy exploration!
# 
# - How does SpaCy handle "The cat ate the fish"? Is it correct?
# - How about "The old man the boat"?
# - And "The complex houses married and single soldiers and their families"? What's going on?
# - Who are the people in "Saint John met Gina St. John in St John's Wood"?

# In[ ]:


# Use this cell to explore!
input_text = u"The cat ate the fish"
displacy.render(nlp(input_text), style="dep", **options)


# In[ ]:


input_text = u"The old man the boat"
## old is the noun, man is the verb
displacy.render(nlp(input_text), style="dep", **options)


# In[ ]:


input_text = u"The complex houses married and single soldiers and their families"
# verb is houses
displacy.render(nlp(input_text), style="dep", **options)


# In[ ]:


input_text = u"Saint John met Gina St. John in St John's Wood"
displacy.render(nlp(input_text), style="ent", **options)


# # Document vectors
# 
# For Machine Learning applications it's *often* the case that we want to process a document into a list of numbers or "vector".
# 
# It's worth noting that there are many different ways to do this. Also recent advances in "deep learning" as well as providing new ways to generate document vectors also offer ways to work more directly with the source text.
# 
# But for now...

# In[ ]:


# The larger SpaCy models contain a list of words and their corresponding vectors
# Glove vector -> word vector
print("Document vectors have {} dimensions".format(len(doc.vector)))
print("And are not normalized e.g. this has length {}".format(np.linalg.norm(doc.vector)))


# In[ ]:


# Document vectors capture an intuitive notion of similarity
# Words that appear similar contexts are considered similar

def print_comparison(a, b):
    # Create the doc objects
    a = nlp(a)
    b = nlp(b)
    # Euclidean "L2" distance
    distance = np.linalg.norm(a.vector - b.vector)
    # Cosine similarity
    similarity = a.similarity(b)
    print("-" * 80)
    print("A: {}\nB: {}\nDistance: {}\nSimilarity: {}".format(a, b, distance, similarity))

text = "The cat sat on the mat"
print_comparison(text, "The feline lay on the carpet")
print_comparison(text, "Three hundred Theban warriors died that day")
print_comparison(text, "Ceci n'est pas une pipe")


# ## Over to you: comparison shopping
# 
# - How does "cat" compare with "feline"?
# - How does "good" compare with "goods"? How does it compare with "bad"? What's going on?
# - How does "teh" compare with "the"?

# In[ ]:


# Use this cell to explore!
print_comparison("cat", "feline")
print_comparison("good", "goods")
print_comparison("good", "bad")
print_comparison("teh", "the")


# In[ ]:


# Document vectors often also have a very interesting property sometimes called "linear substructure"
# Basically you can do arithmetic with words/concepts!

def vectorize(text):
    """Get the SpaCy vector corresponding to a text"""
    return nlp(text, disable=['parser', 'tagger', 'ner']).vector

from heapq import heappush, nsmallest, nlargest

def get_top_n(target_v, n=5):
    """Figure out the top-N words most similar to the target vector"""
    heap = []
    # SpaCy has a long list of words in `vocab` which we can pick from!
    for word in nlp.vocab:
        # Filter out mixed case and uncommon terms
        if not word.is_lower or word.prob < -15:
            continue
        distance = np.linalg.norm(target_v - word.vector)
        heappush(heap, (distance, word.text))
    return nsmallest(n, heap)


PUPPY, DOG, KITTEN = [vectorize(w) for w in ("puppy", "dog", "kitten")]

get_top_n(DOG - PUPPY + KITTEN)


# In[ ]:


# We can generalize that into a cute analogy finder

def print_analogy(a, b, c):
    """A is to B as C is to ???"""
    top_n = get_top_n(vectorize(b) - vectorize(a) + vectorize(c))
    best = [w for (s,w) in top_n if w not in (a,b,c)][0]
    print("{} is to {} as {} is to {}".format(a, b, c, best))
    
print_analogy("queen", "king", "woman")


# ## Over to you: an analogy is an idea with another idea's hat on
# 
# - "Boy" is to "girl" as "prince" is to what?
# - "Red" is to "reddest" as "blue" is to what?
# - What is to "simile" as "analogy" is to "metaphor"? (Sorry for the brain strain, but it's good to stretch a bit no?)
# - Find an example that doesn't work! (It's not too hard :))
# - (Advanced) Try changing the "-15" in `get_top_n` to "-100". What effect is it having on your examples?
# - (Advanced) Could you adapt this code to find "opposites" for words?
# 

# In[ ]:


# Use this cell to explore!
print_analogy("boy", "girl", "prince")
print_analogy("red", "reddest", "blue")
print_analogy("smile", "analogy", "metaphor")
print_analogy("one", "two", "three")


# # Cleaning and visualizing documents
# 
# Okay, so that's a quick introduction to SpaCy. Let's look at some documents.
# 
# Scikit Learn (sklearn) is a brilliant python library for machine learning. You can find out more about it at http://scikit-learn.org
# 
# It also ships with some handy features for downloading and reading in some standard pedagogical datasets. Let's have a look at the newsgroups collection of documents (or "corpus") that we used Scikit Learn to download earlier.

# In[ ]:


from sklearn.datasets import fetch_20newsgroups
raw_posts = fetch_20newsgroups()

print("Number of posts: {}".format(len(raw_posts.data)))
 # Source groups are listed in `target_names`
print("Newsgroups: {}".format(raw_posts.target_names))
 # Post text is in `data`
print("Sample post text:\n{0}\n{1}\n{0}".format('-' * 80, raw_posts.data[19]))
 # Post group is encoded in `target` as an index into `target_names`
print("Sample post group: {}".format(raw_posts.target_names[raw_posts.target[19]]))


# In[ ]:


# There's quite a lot of junk in there, headers etc. Fortunately sklearn can help a bit...
# We can pass through a special argument to strip out headers, footers and inline quotes

raw_posts = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
print("Sample post text:\n{0}\n{1}\n{0}".format('-' * 80, raw_posts.data[19]))


# In[ ]:


# Another key tool in the Python ecosystem is Pandas which is a library for working with tables of data
# We're going to convert the dataset in to a Panda DataFrame for ease of manipulation
# Don't worry too much about this -- it's not really our focus today -- but if you're interested you can
# find out more at http://pandas.pydata.org/

posts = pd.DataFrame({'text': raw_posts.data, 'group': [raw_posts.target_names[t] for t in raw_posts.target]})

# Many tools in the Python ecosystem are quite tightly integrated, so once we have DataFrame we can
# do things like plot it via the standard charting tool `matplotlib` which we importes as `plt` earlier
posts['group'].value_counts().plot(kind='bar', title="Per group document counts")
plt.show()


# In[ ]:


# One way to get a handle on a collection of documents (or "corpus") is to look at a wordcloud
# Thankfully someone has written a little library to help us do that

wc = WordCloud(background_color='white', width=1000, height=400, stopwords=[])
wc.generate(" ".join(t for t in posts[posts.group == 'rec.autos'].text)).to_image()


# In[ ]:


# Oh dear that wasn't much use... of course common words completely dominate!
# These are called "stopwords". It's common (if a little controversial these days...) to filter them out
# The wordcloud library we're using supports that

from wordcloud import STOPWORDS
better_stopwords = STOPWORDS.union({'may', 'one', 'will', 'also'})
wc = WordCloud(background_color='white', width=1000, height=400, stopwords=better_stopwords)
wc.generate(" ".join(t for t in posts[posts.group == 'rec.autos'].text)).to_image()


# In[ ]:


# Okay, that's more like it! Let's eyeball all the groups

for group in raw_posts.target_names:
    print("Wordcloud for {}".format(group))
    display(wc.generate(" ".join(t for t in posts[posts.group == group].text)).to_image())


# In[ ]:


# Looks okay, but comp.os.ms-windows.misc appears to be full of garbage
# Let's cull it (rather crudely...) and take another look

posts = posts[~posts.text.str.contains("AX")]
for group in raw_posts.target_names:
    print("Wordcloud for {}".format(group))
    display(wc.generate(" ".join(t for t in posts[posts.group == group].text)).to_image())


# ## Over to you: more windows on the data
# 
# - What does the wordcloud for the whole corpus look like?
# - How many posts contain the word "window"? (Hint: look at how we removed posts containing "AX")
# - How do they split out across the groups? (Hint: look at how we drew the barchart for groups earlier)
# - (Tricky!) Can you filter out more junk posts?
# 

# In[ ]:


# Use this cell to explore!
display(wc.generate(" ".join(t for t in posts.text)).to_image())


# # Topic extraction
# 
# We're going to see if we can automatically infer a set of topics from the corpus. Obviously we'd expect these to be somehow related to the original newgroups, but perhaps there'll be some surprises?

# In[ ]:


# First let's get the documents into a suitable form
# Build a matrix by "stacking" the row vectors from SpaCy
# Takes about 20 seconds...

from sklearn.preprocessing import normalize

def vectorize(text):
    # Get the SpaCy vector -- turning off other processing to speed things up
    return nlp(text, disable=['parser', 'tagger', 'ner']).vector

# Now we stack the vectors and normalize them
# Inputs are typically called "X"
X = normalize(np.stack(vectorize(t) for t in posts.text))
print("X (the document matrix) has shape: {}".format(X.shape))
print("That means it has {} rows and {} columns".format(X.shape[0], X.shape[1]))


# ## Visualizing the vectors: 300D Glasses
# 
# Our document vectors have 300 dimensions. That's quite difficult to visualize on a 2 dimensional screen!
# 
# We're going to a use a standard technique called "Principal Components Analysis" (or PCA) to automatically reduce that to 2 dimensions, so we can get some insight into what's going on.
# 
# You can read more about PCA at https://en.wikipedia.org/wiki/Principal_component_analysis

# In[ ]:


# Scikit Learn ships with a neat PCA implementation

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)
print("X2 shape is {}".format(X2.shape))


# In[ ]:


# Okay let's take a look at it via matplotlib

def plot_groups(X, y, groups):
    for group in groups:
        plt.scatter(X[y == group, 0], X[y == group, 1], label=group, alpha=0.4)
    plt.legend()
    plt.show()
    
plot_groups(X2, posts.group, ('comp.os.ms-windows.misc', 'alt.atheism'))


# ## Clustering the documents
# 
# It looks like our vectors are doing something vaguely useful, in that there's a visual separate between groups.
# 
# Now we'll use the standard "k-means" algorithm to automatically identify clusters within the data.
# 
# You can read more about k-means at https://en.wikipedia.org/wiki/K-means_clustering
# 

# In[ ]:


CLUSTERS = 20

# First we fit the model...
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=CLUSTERS, random_state=1)
k_means.fit(X)


# In[ ]:


# Then we use it to predict clusters for each document...
# Again it's common to use yhat for a predicted value -- although we wouldn't expect these to
# correspond directly to the original groups
yhat = k_means.predict(X)

# Let's take a look at the distribution across classes
plt.hist(yhat, bins=range(CLUSTERS))
plt.show()


# In[ ]:


# To be honest that's not looking very healthy -- ideally we'd see a more even distribution
# Let's take a look at a couple of the big ones

plot_groups(X2, yhat, (1,14))


# In[ ]:


# Okay there are some definite (if rather blurry...) clusters there!
# Let's have a look at how our clusters relate to the original groups
def plot_cluster(c):
    posts[yhat == c]['group'].value_counts().plot(kind='bar', title="Cluster #{}".format(c))
    plt.show()

# Some are great matches...
plot_cluster(0)


# In[ ]:


# Some are not so great a match, but sensible (why...?)
plot_cluster(14)


# In[ ]:


# Some are just a bit random!
plot_cluster(9)


# In[ ]:


# Let's have a look at the wordclouds...
for c in range(CLUSTERS):
    print("Wordcloud for category #{}".format(c))
    display(wc.generate(" ".join(posts.text[yhat == c])).to_image())


# Again, there's some confusion, but there are some really strong clusters there! Go us! But how could we do better?
# 
# ## Over to you: cluster buster
# 
# 
# 
# - Whats going on with cluster #1?
# - Was ist der story with cluster #4?
# - Scroll way up and find where we're defining `CLUSTERS`. Try a smaller value (say 5). What effect does it have?
# - Try a larger (perhaps much larger...) value. What effect does it have?
# - (Tricky!) How could you decide automatically how many clusters to use? (Hint: take a look at the `bench_k_means()` method in http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html)

# In[ ]:


# Use this cell to explore!


# # The workshop is dead long live the workshop!
# 
# ## Feedback
# 
# I'd really love to get your feedback on this workshop (be it good, bad, pull request or bug report)! You can ping me at [joe.halliwell@gmail.com](mailto:joe.halliwell@gmail.com) or even tweet `@joehalliwell` if you're so inclined.
# 
# ## Other things to try
# 
# We've really just scratched the surface of document clustering here. If you want to dig into the topic further (and dig out further topics), you might like to start by:
# 
# - Using TFIDF vectors instead of SpaCy vectors: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# - Using Latent Dirichlet Allocation (LDA) instead of k-means: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
